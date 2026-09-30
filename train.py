"""P2PNet training with MDC/MDC++ support and resumable checkpoints."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from crowd_datasets import build_dataset
from engine import train_one_epoch
from evaluate import evaluate_dataset
from models import build_model
from workflow_utils import environment_report, extract_state_dict, resolve_device, save_json

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def get_args_parser():
    parser = argparse.ArgumentParser("P2PNet training and validation", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int, help="source frames per loader batch")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=3500, type=int)
    parser.add_argument("--lr_drop", default=3500, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float)

    parser.add_argument("--backbone", default="vgg16_bn", choices=("vgg16_bn", "vgg16"))
    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--set_cost_class", default=1.0, type=float)
    parser.add_argument("--set_cost_point", default=0.05, type=float)
    parser.add_argument("--point_loss_coef", default=0.0002, type=float)
    parser.add_argument("--eos_coef", default=0.5, type=float)
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--line", default=2, type=int)

    parser.add_argument("--dataset_file", default="MDC", choices=("MDC", "MDC++", "SHHA"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_split", default="train.txt")
    parser.add_argument("--val_split", default="val.txt")
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--num_patches", default=1, type=int)
    parser.add_argument("--scale_min", default=0.7, type=float)
    parser.add_argument("--scale_max", default=1.3, type=float)
    parser.add_argument("--flip_probability", default=0.5, type=float)
    parser.add_argument("--min_crop_points", default=0, type=int)
    parser.add_argument("--train_frame_stride", default=1, type=int)
    parser.add_argument("--val_frame_stride", default=1, type=int)
    parser.add_argument("--max_train_samples", default=0, type=int)
    parser.add_argument("--max_eval_samples", default=0, type=int)

    parser.add_argument("--output_dir", default="./outputs/train")
    parser.add_argument("--checkpoints_dir", default="")
    parser.add_argument("--tensorboard_dir", default="")
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--weights", default="", help="model-only initialization/fine-tuning checkpoint")
    parser.add_argument("--resume", default="", help="full training checkpoint")
    parser.add_argument("--strict_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true", help="evaluate the validation split and exit")
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument(
        "--early_stopping_patience", default=0, type=int,
        help="stop after this many validation checks without improvement; 0 disables",
    )
    parser.add_argument(
        "--early_stopping_min_epochs", default=0, type=int,
        help="never early-stop before this many completed epochs",
    )
    parser.add_argument(
        "--early_stopping_min_delta", default=0.0, type=float,
        help="minimum MAE decrease required to reset early-stopping patience",
    )
    parser.add_argument("--save_every", default=25, type=int, help="extra numbered checkpoint frequency; latest is always saved")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_train_batches", default=0, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--tile_size", default=1024, type=int)
    parser.add_argument("--tile_overlap", default=128, type=int)
    parser.add_argument("--max_size", default=0, type=int)
    parser.add_argument("--visualize_first", default=0, type=int)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_id", default=0, type=int, help="visible GPU when --device is auto/cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _flush_and_sync(handle):
    """Flush a history file before its atomic replacement, including on Drive mounts."""
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        # Some mounted/cloud filesystems do not expose fsync. The atomic replace
        # below still prevents a partially written file from becoming canonical.
        pass


def _read_history_jsonl(path: Path):
    """Read normal or accidentally concatenated JSON objects from history.jsonl."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records = []
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            break
        try:
            record, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Cannot recover history JSON near character {exc.pos} in {path}"
            ) from exc
        if not isinstance(record, dict) or "epoch" not in record:
            raise ValueError(f"Invalid history record in {path}: {record!r}")
        records.append(record)
    return records


def _atomic_write_history_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _flush_and_sync(handle)
    os.replace(temporary, path)


HISTORY_FIELDS = [
    "epoch_index", "epoch_number",
    "train_loss", "train_loss_ce", "train_loss_point",
    "main_lr", "backbone_lr", "next_main_lr", "next_backbone_lr",
    "optimizer_steps", "patches_seen", "source_frames_seen",
    "train_seconds", "validation_seconds", "checkpoint_seconds", "total_epoch_seconds",
    "train_peak_allocated_mb", "train_peak_reserved_mb",
    "validation_peak_allocated_mb", "validation_peak_reserved_mb",
    "validation_samples", "validation_seconds_per_sample",
    "validation_mae", "validation_rmse", "validation_mean_error",
    "validation_threshold", "validation_frame_stride",
    "new_best", "best_mae", "best_epoch_number", "bad_validation_checks",
    "early_stop_triggered",
]


def _history_csv_row(record: dict):
    train = record.get("train", {})
    validation = record.get("validation", {})
    return {
        "epoch_index": record["epoch"],
        "epoch_number": record["epoch"] + 1,
        "train_loss": train.get("loss"),
        "train_loss_ce": train.get("loss_ce"),
        "train_loss_point": train.get("loss_point"),
        "main_lr": record.get("main_lr"),
        "backbone_lr": record.get("backbone_lr"),
        "next_main_lr": record.get("next_main_lr"),
        "next_backbone_lr": record.get("next_backbone_lr"),
        "optimizer_steps": train.get("optimizer_steps"),
        "patches_seen": train.get("patches_seen"),
        "source_frames_seen": record.get("source_frames_seen"),
        "train_seconds": record.get("train_seconds"),
        "validation_seconds": record.get("validation_seconds"),
        "checkpoint_seconds": record.get("checkpoint_seconds"),
        "total_epoch_seconds": record.get("total_epoch_seconds"),
        "train_peak_allocated_mb": record.get("train_peak_allocated_mb"),
        "train_peak_reserved_mb": record.get("train_peak_reserved_mb"),
        "validation_peak_allocated_mb": record.get("validation_peak_allocated_mb"),
        "validation_peak_reserved_mb": record.get("validation_peak_reserved_mb"),
        "validation_samples": validation.get("samples"),
        "validation_seconds_per_sample": validation.get("seconds_per_sample"),
        "validation_mae": validation.get("mae"),
        "validation_rmse": validation.get("rmse"),
        "validation_mean_error": validation.get("mean_error"),
        "validation_threshold": validation.get("threshold"),
        "validation_frame_stride": validation.get("frame_stride"),
        "new_best": record.get("new_best", False),
        "best_mae": record.get("best_mae"),
        "best_epoch_number": record.get("best_epoch_number"),
        "bad_validation_checks": record.get("bad_validation_checks", 0),
        "early_stop_triggered": record.get("early_stop_triggered", False),
    }


def _atomic_write_history_csv(path: Path, records):
    """Rewrite a complete rectangular CSV; never append to a possibly partial row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_history_csv_row(record))
        _flush_and_sync(handle)
    os.replace(temporary, path)


def _persist_history(jsonl_path: Path, csv_path: Path, record: dict):
    """Upsert one epoch and atomically regenerate both durable history files."""
    records_by_epoch = {
        int(existing["epoch"]): existing
        for existing in _read_history_jsonl(jsonl_path)
    }
    records_by_epoch[int(record["epoch"])] = record
    ordered_records = [records_by_epoch[index] for index in sorted(records_by_epoch)]

    # JSONL is canonical. If CSV replacement is interrupted, the next epoch or
    # resumed run reconstructs CSV from the complete JSONL automatically.
    _atomic_write_history_jsonl(jsonl_path, ordered_records)
    _atomic_write_history_csv(csv_path, ordered_records)


def _checkpoint_payload(
    args, epoch, model, optimizer, scheduler, best_mae, best_epoch, bad_validation_checks
):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "bad_validation_checks": bad_validation_checks,
        "args": vars(args),
    }


def _save_checkpoint(payload: dict, path: Path):
    """Atomically replace a checkpoint so an interrupted save keeps the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak(device):
    if device.type != "cuda":
        return 0.0, 0.0
    _cuda_sync(device)
    divisor = 1024 ** 2
    return (
        torch.cuda.max_memory_allocated(device) / divisor,
        torch.cuda.max_memory_reserved(device) / divisor,
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = resolve_device(args.device)
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve() if args.checkpoints_dir else output_dir / "checkpoints"
    tensorboard_dir = Path(args.tensorboard_dir).expanduser().resolve() if args.tensorboard_dir else output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    if args.early_stopping_min_epochs < 0:
        raise ValueError("early_stopping_min_epochs must be nonnegative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    if args.early_stopping_patience > 0 and args.eval_freq <= 0:
        raise ValueError("early stopping requires eval_freq > 0")

    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not args.deterministic
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # A complete checkpoint already contains the backbone. Avoid a needless
    # ImageNet download before loading it.
    if args.resume or args.weights:
        args.pretrained_backbone = False

    train_set, val_set = build_dataset(args)(args.data_root)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=(len(train_set) >= args.batch_size)
    )
    data_loader_train = DataLoader(
        train_set,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn_crowd,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [p for name, p in model.named_parameters() if "backbone" not in name and p.requires_grad]},
            {
                "params": [p for name, p in model.named_parameters() if "backbone" in name and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    best_mae = float("inf")
    best_epoch = None
    bad_validation_checks = 0

    if args.weights:
        checkpoint = _torch_load(args.weights)
        model.load_state_dict(extract_state_dict(checkpoint), strict=args.strict_load)
        print(f"Loaded model initialization weights: {args.weights}")

    if args.resume:
        checkpoint = _torch_load(args.resume)
        model.load_state_dict(extract_state_dict(checkpoint), strict=args.strict_load)
        if isinstance(checkpoint, dict) and all(
            key in checkpoint for key in ("optimizer", "lr_scheduler", "epoch")
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = int(checkpoint["epoch"]) + 1
            best_mae = float(checkpoint.get("best_mae", best_mae))
            best_epoch = checkpoint.get("best_epoch")
            bad_validation_checks = int(checkpoint.get("bad_validation_checks", 0))
            print(f"Resuming after epoch {checkpoint['epoch']}: {args.resume}")
        else:
            print("Resume file had model weights only; optimizer and epoch start fresh.")

    manifest = {
        "arguments": vars(args),
        "environment": environment_report(repo_root),
        "command": [sys.executable, *sys.argv],
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "effective_patch_batch": args.batch_size * args.num_patches,
        "estimated_optimizer_steps_per_full_epoch": len(data_loader_train),
        "gpu_total_memory_mb": (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
            if device.type == "cuda" else 0.0
        ),
    }
    save_json(output_dir / "run_config.json", manifest)
    print(json.dumps(manifest, indent=2, default=str))

    if args.eval:
        metrics, _ = evaluate_dataset(model, val_set, device, args, output_dir / "validation")
        print(f"Validation MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
        return

    writer = None
    if not args.disable_tensorboard:
        if SummaryWriter is None:
            print("TensorBoard is not installed; continuing without TensorBoard logging.")
        else:
            writer = SummaryWriter(str(tensorboard_dir))

    history_path = output_dir / "history.jsonl"
    history_csv_path = output_dir / "training_history.csv"
    started = time.time()
    completed_epoch = args.start_epoch - 1
    stop_reason = "epoch_ceiling"
    for epoch in range(args.start_epoch, args.epochs):
        epoch_started = time.time()
        epoch_main_lr = optimizer.param_groups[0]["lr"]
        epoch_backbone_lr = optimizer.param_groups[1]["lr"]
        _reset_cuda_peak(device)
        _cuda_sync(device)
        train_started = time.perf_counter()
        stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            max_steps=args.max_train_batches,
            print_freq=args.print_freq,
        )
        _cuda_sync(device)
        train_seconds = time.perf_counter() - train_started
        train_peak_allocated, train_peak_reserved = _cuda_peak(device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train": stats,
            "main_lr": epoch_main_lr,
            "backbone_lr": epoch_backbone_lr,
            "next_main_lr": optimizer.param_groups[0]["lr"],
            "next_backbone_lr": optimizer.param_groups[1]["lr"],
            "source_frames_seen": stats["patches_seen"] // args.num_patches,
            "train_seconds": train_seconds,
            "validation_seconds": 0.0,
            "train_peak_allocated_mb": train_peak_allocated,
            "train_peak_reserved_mb": train_peak_reserved,
            "validation_peak_allocated_mb": 0.0,
            "validation_peak_reserved_mb": 0.0,
            "new_best": False,
            "early_stop_triggered": False,
        }
        for name, value in stats.items():
            if writer:
                writer.add_scalar(f"train/{name}", value, epoch)
        if writer:
            writer.add_scalar("learning_rate/main", epoch_main_lr, epoch)
            writer.add_scalar("learning_rate/backbone", epoch_backbone_lr, epoch)
            writer.add_scalar("timing/train_seconds", train_seconds, epoch)
            writer.add_scalar("memory/train_peak_allocated_mb", train_peak_allocated, epoch)

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            _reset_cuda_peak(device)
            _cuda_sync(device)
            validation_started = time.perf_counter()
            metrics, _ = evaluate_dataset(model, val_set, device, args)
            _cuda_sync(device)
            validation_seconds = time.perf_counter() - validation_started
            val_peak_allocated, val_peak_reserved = _cuda_peak(device)
            record["validation"] = metrics
            record["validation_seconds"] = validation_seconds
            record["validation_peak_allocated_mb"] = val_peak_allocated
            record["validation_peak_reserved_mb"] = val_peak_reserved
            if writer:
                writer.add_scalar("validation/mae", metrics["mae"], epoch)
                writer.add_scalar("validation/rmse", metrics["rmse"], epoch)
                writer.add_scalar("timing/validation_seconds", validation_seconds, epoch)
                writer.add_scalar("memory/validation_peak_allocated_mb", val_peak_allocated, epoch)
            if metrics["mae"] < best_mae - args.early_stopping_min_delta:
                best_mae = metrics["mae"]
                best_epoch = epoch
                bad_validation_checks = 0
                payload = _checkpoint_payload(
                    args, epoch, model, optimizer, scheduler,
                    best_mae, best_epoch, bad_validation_checks,
                )
                best_path = checkpoints_dir / "best_mae.pth"
                _save_checkpoint(payload, best_path)
                save_json(
                    checkpoints_dir / "best_checkpoint.json",
                    {
                        "checkpoint": str(best_path),
                        "epoch_index": epoch,
                        "epoch_number": epoch + 1,
                        "selection_metric": "validation_mae",
                        "validation": metrics,
                        "early_stopping_min_delta": args.early_stopping_min_delta,
                    },
                )
                record["new_best"] = True
            else:
                bad_validation_checks += 1
            if (
                args.early_stopping_patience > 0
                and epoch + 1 >= args.early_stopping_min_epochs
                and bad_validation_checks >= args.early_stopping_patience
            ):
                record["early_stop_triggered"] = True
            print(
                f"epoch={epoch + 1} val_mae={metrics['mae']:.4f} "
                f"val_rmse={metrics['rmse']:.4f} "
                f"patience={bad_validation_checks}/{args.early_stopping_patience or 'off'}"
            )

        checkpoint_started = time.perf_counter()
        payload = _checkpoint_payload(
            args, epoch, model, optimizer, scheduler,
            best_mae, best_epoch, bad_validation_checks,
        )
        _save_checkpoint(payload, checkpoints_dir / "latest.pth")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save_checkpoint(payload, checkpoints_dir / f"epoch_{epoch:04d}.pth")

        record["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        record["total_epoch_seconds"] = time.time() - epoch_started
        record["seconds"] = record["total_epoch_seconds"]  # backward-compatible alias
        record["best_mae"] = best_mae if np.isfinite(best_mae) else None
        record["best_epoch"] = best_epoch
        record["best_epoch_number"] = best_epoch + 1 if best_epoch is not None else None
        record["bad_validation_checks"] = bad_validation_checks
        _persist_history(history_path, history_csv_path, record)
        completed_epoch = epoch
        if writer:
            writer.add_scalar("timing/total_epoch_seconds", record["total_epoch_seconds"], epoch)
            writer.add_scalar("early_stopping/bad_validation_checks", bad_validation_checks, epoch)
            writer.flush()
        print(
            f"Finished epoch {epoch + 1} in "
            f"{datetime.timedelta(seconds=int(record['total_epoch_seconds']))} "
            f"(train={train_seconds:.1f}s, validation={record['validation_seconds']:.1f}s)"
        )
        if record["early_stop_triggered"]:
            stop_reason = "early_stopping"
            print(
                f"EARLY STOP: no validation MAE improvement greater than "
                f"{args.early_stopping_min_delta:g} for {bad_validation_checks} checks; "
                f"minimum {args.early_stopping_min_epochs} epochs satisfied."
            )
            break

    if writer:
        writer.close()
    total_training_seconds = time.time() - started
    summary = {
        "stop_reason": stop_reason,
        "completed_epoch_index": completed_epoch,
        "completed_epoch_number": completed_epoch + 1 if completed_epoch >= 0 else 0,
        "requested_epoch_ceiling": args.epochs,
        "best_mae": best_mae if np.isfinite(best_mae) else None,
        "best_epoch_index": best_epoch,
        "best_epoch_number": best_epoch + 1 if best_epoch is not None else None,
        "bad_validation_checks": bad_validation_checks,
        "total_training_seconds": total_training_seconds,
        "history_jsonl": str(history_path),
        "history_csv": str(history_csv_path),
    }
    save_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Training time {datetime.timedelta(seconds=int(total_training_seconds))}")
    print(f"Checkpoints: {checkpoints_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("P2PNet training", parents=[get_args_parser()])
    main(parser.parse_args())
