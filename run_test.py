"""Single-image or directory inference for P2PNet."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
from PIL import Image

from workflow_utils import (
    create_model,
    draw_prediction,
    environment_report,
    list_images,
    predict_pil,
    resolve_device,
    save_json,
)


def get_args_parser():
    parser = argparse.ArgumentParser("P2PNet image/directory inference")
    parser.add_argument("--input", default="./vis/demo1.jpg", help="image or directory")
    parser.add_argument("--weight_path", required=True, help="compatible P2PNet checkpoint")
    parser.add_argument("--output_dir", default="./outputs/inference")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--backbone", default="vgg16_bn", choices=("vgg16_bn", "vgg16"))
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--line", default=2, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument(
        "--tile_size", default=1024, type=int,
        help="tile side in pixels; <=0 processes the full image at once",
    )
    parser.add_argument("--tile_overlap", default=128, type=int)
    parser.add_argument(
        "--max_size", default=0, type=int,
        help="optionally resize the longest image side before inference; 0 disables resizing",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no_visualizations", action="store_true")
    parser.add_argument("--limit", default=0, type=int, help="process only the first N images")
    return parser


def main(args):
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visualizations"
    if not args.no_visualizations:
        visual_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(
        args.weight_path,
        device=device,
        backbone=args.backbone,
        row=args.row,
        line=args.line,
    )
    images = list_images(args.input, recursive=args.recursive)
    if args.limit > 0:
        images = images[: args.limit]

    rows = []
    for index, image_path in enumerate(images, start=1):
        started = time.perf_counter()
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        prediction = predict_pil(
            model,
            image,
            device=device,
            threshold=args.threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_size=args.max_size,
        )
        elapsed = time.perf_counter() - started
        print(
            f"[{index}/{len(images)}] {image_path.name}: count={prediction.count} "
            f"tiles={prediction.tiles} time={elapsed:.2f}s"
        )
        rows.append(
            {
                "image": str(image_path),
                "predicted_count": prediction.count,
                "width": prediction.original_size[0],
                "height": prediction.original_size[1],
                "inference_width": prediction.inference_size[0],
                "inference_height": prediction.inference_size[1],
                "scale": prediction.scale,
                "tiles": prediction.tiles,
                "seconds": elapsed,
            }
        )
        save_json(
            output_dir / "points" / f"{index:06d}_{image_path.stem}.json",
            {
                **rows[-1],
                "points": [
                    {"x": float(point[0]), "y": float(point[1]), "score": float(score)}
                    for point, score in zip(prediction.points, prediction.scores)
                ],
            },
        )
        if not args.no_visualizations:
            canvas = draw_prediction(image, prediction)
            cv2.imwrite(str(visual_dir / f"{index:06d}_{image_path.stem}_pred.jpg"), canvas)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        output_dir / "run.json",
        {
            "arguments": vars(args),
            "environment": environment_report(Path(__file__).resolve().parent),
            "images_processed": len(rows),
        },
    )
    print(f"Saved inference outputs to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
