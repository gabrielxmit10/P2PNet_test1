"""Validate MDC/MDC++ layout, split isolation, and frame/annotation mapping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from crowd_datasets.MDC.mdc import expand_split_clips, load_clip_points
from workflow_utils import save_json


def get_args_parser():
    parser = argparse.ArgumentParser("Validate MovingDroneCrowd dataset")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_files", nargs="+", default=["train.txt", "val.txt", "test.txt"])
    parser.add_argument(
        "--images_to_open", default=25, type=int,
        help="open this many images per split to verify readability; -1 opens all",
    )
    parser.add_argument("--output", default="")
    return parser


def inspect_split(root: Path, split_file: str, images_to_open: int):
    clips = expand_split_clips(root, split_file)
    frame_count = 0
    point_count = 0
    empty_frames = 0
    resolutions = Counter()
    opened = 0
    problems = []

    for scene, clip in clips:
        annotation_path = root / "annotations" / scene / f"{clip}.csv"
        points = load_clip_points(annotation_path)
        frame_dir = root / "frames" / scene / clip
        images = sorted(
            frame_dir.glob("*.jpg"),
            key=lambda path: (int(path.stem) if path.stem.isdigit() else 2**31 - 1, path.name),
        )
        frame_ids = set()
        for image in images:
            if not image.stem.isdigit():
                problems.append(f"Non-numeric frame filename: {image}")
                continue
            frame_id = int(image.stem) - 1
            frame_ids.add(frame_id)
            count = len(points.get(frame_id, ()))
            point_count += count
            empty_frames += int(count == 0)
            if images_to_open < 0 or opened < images_to_open:
                try:
                    with Image.open(image) as handle:
                        handle.verify()
                    with Image.open(image) as handle:
                        resolutions[handle.size] += 1
                except Exception as exc:
                    problems.append(f"Unreadable image {image}: {exc}")
                opened += 1
        missing_images = sorted(set(points) - frame_ids)
        if missing_images:
            problems.append(
                f"{annotation_path} refers to {len(missing_images)} missing frame(s), "
                f"first={missing_images[:5]}"
            )
        frame_count += len(images)

    return {
        "split_file": split_file,
        "clips": len(clips),
        "frames": frame_count,
        "head_annotations": point_count,
        "empty_frames": empty_frames,
        "images_opened": opened,
        "sampled_resolutions": {f"{w}x{h}": count for (w, h), count in resolutions.items()},
        "problems": problems,
        "clip_ids": [f"{scene}/{clip}" for scene, clip in clips],
    }


def main(args):
    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    for required in ("frames", "annotations"):
        if not (root / required).is_dir():
            raise FileNotFoundError(f"Missing required directory: {root / required}")

    splits = [inspect_split(root, name, args.images_to_open) for name in args.split_files]
    ownership = {}
    overlap = []
    for split in splits:
        for clip_id in split["clip_ids"]:
            if clip_id in ownership:
                overlap.append((clip_id, ownership[clip_id], split["split_file"]))
            else:
                ownership[clip_id] = split["split_file"]

    report = {
        "data_root": str(root),
        "splits": [{key: value for key, value in split.items() if key != "clip_ids"} for split in splits],
        "clip_overlap": overlap,
        "valid": not overlap and not any(split["problems"] for split in splits),
    }
    # ASCII escaping keeps reports printable in Windows consoles whose active
    # code page cannot represent every character in a dataset path.
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if args.output:
        save_json(Path(args.output).expanduser().resolve(), report)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
