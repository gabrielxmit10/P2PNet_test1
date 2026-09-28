"""Frame-level P2PNet evaluation on MDC/MDC++."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

from crowd_datasets.MDC.mdc import MDCFrameDataset
from workflow_utils import (
    create_model,
    draw_prediction,
    environment_report,
    predict_pil,
    resolve_device,
    save_json,
)


def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate P2PNet on MDC/MDC++ frames")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_file", default="val.txt")
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--output_dir", default="./outputs/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone", default="vgg16_bn", choices=("vgg16_bn", "vgg16"))
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--line", default=2, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--tile_size", default=1024, type=int)
    parser.add_argument("--tile_overlap", default=128, type=int)
    parser.add_argument("--max_size", default=0, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_samples", default=0, type=int)
    parser.add_argument(
        "--visualize_first", default=0, type=int,
        help="save GT (green) and predictions (red) for the first N samples",
    )
    return parser


def evaluate_dataset(model, dataset, device, args, output_dir: Path | None = None):
    evaluation_started = time.perf_counter()
    errors = []
    rows = []
    visual_dir = output_dir / "visualizations" if output_dir else None
    if visual_dir and args.visualize_first > 0:
        visual_dir.mkdir(parents=True, exist_ok=True)

    # Standalone evaluation already creates an eval-mode model, but training calls this
    # function with its live train-mode model. Preserve and restore that state so VGG
    # batch-normalization statistics are not changed by validation frames.
    was_training = model.training
    model.eval()
    try:
        for index in range(len(dataset)):
            image, ground_truth, record = dataset.get_raw_sample(index)
            started = time.perf_counter()
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
            error = prediction.count - len(ground_truth)
            errors.append(error)
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "image": str(record.image_path),
                    "ground_truth_count": len(ground_truth),
                    "predicted_count": prediction.count,
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error * error,
                    "tiles": prediction.tiles,
                    "seconds": elapsed,
                }
            )
            print(
                f"[{index + 1}/{len(dataset)}] {record.sample_id}: "
                f"gt={len(ground_truth)} pred={prediction.count} error={error:+d}"
            )
            if visual_dir and index < args.visualize_first:
                canvas = draw_prediction(image, prediction, ground_truth=ground_truth)
                safe_name = record.sample_id.replace("/", "_")
                cv2.imwrite(str(visual_dir / f"{safe_name}.jpg"), canvas)
    finally:
        if was_training:
            model.train()

    error_array = np.asarray(errors, dtype=np.float64)
    wall_seconds = time.perf_counter() - evaluation_started
    inference_seconds = sum(row["seconds"] for row in rows)
    split_file = getattr(args, "split_file", getattr(args, "val_split", ""))
    frame_stride = getattr(args, "frame_stride", getattr(args, "val_frame_stride", 1))
    metrics = {
        "samples": len(rows),
        "mae": float(np.mean(np.abs(error_array))),
        "rmse": float(np.sqrt(np.mean(np.square(error_array)))),
        "mean_error": float(np.mean(error_array)),
        "threshold": args.threshold,
        "split_file": split_file,
        "frame_stride": frame_stride,
        "wall_seconds": wall_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_sample": wall_seconds / len(rows) if rows else 0.0,
        "samples_per_second": len(rows) / wall_seconds if wall_seconds > 0 else 0.0,
    }
    return metrics, rows


def main(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = create_model(
        args.weight_path,
        device=device,
        backbone=args.backbone,
        row=args.row,
        line=args.line,
    )
    dataset = MDCFrameDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        train=False,
        frame_stride=args.frame_stride,
        max_samples=args.max_samples,
    )
    metrics, rows = evaluate_dataset(model, dataset, device, args, output_dir)

    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        output_dir / "metrics.json",
        {
            **metrics,
            "arguments": vars(args),
            "environment": environment_report(Path(__file__).resolve().parent),
        },
    )
    print(f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
