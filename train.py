"""P2PNet training with MDC/MDC++ support and resumable checkpoints."""

from __future__ import annotations

import argparse
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


def _append_jsonl(path: Path, payload: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _checkpoint_payload(args, epoch, model, optimizer, scheduler, best_mae):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "best_mae": best_mae,
        "args": vars(args),
    }


def _save_checkpoint(payload: dict, path: Path):
    """Atomically replace a checkpoint so an interrupted save keeps the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = resolve_device(args.device)
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve() if args.checkpoints_dir else output_dir / "checkpoints"
    tensorboard_dir = Path(args.tensorboard_dir).expanduser().resolve() if args.tensorboard_dir else output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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
    started = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_started = time.time()
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
        scheduler.step()
        record = {"epoch": epoch, "train": stats, "seconds": time.time() - epoch_started}
        for name, value in stats.items():
            if writer:
                writer.add_scalar(f"train/{name}", value, epoch)

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            metrics, _ = evaluate_dataset(model, val_set, device, args)
            record["validation"] = metrics
            if writer:
                writer.add_scalar("validation/mae", metrics["mae"], epoch)
                writer.add_scalar("validation/rmse", metrics["rmse"], epoch)
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                payload = _checkpoint_payload(args, epoch, model, optimizer, scheduler, best_mae)
                _save_checkpoint(payload, checkpoints_dir / "best_mae.pth")
                record["new_best"] = True
            print(f"epoch={epoch} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f}")

        payload = _checkpoint_payload(args, epoch, model, optimizer, scheduler, best_mae)
        _save_checkpoint(payload, checkpoints_dir / "latest.pth")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save_checkpoint(payload, checkpoints_dir / f"epoch_{epoch:04d}.pth")

        record["best_mae"] = best_mae
        _append_jsonl(history_path, record)
        print(
            f"Finished epoch {epoch} in {datetime.timedelta(seconds=int(time.time() - epoch_started))}"
        )

    if writer:
        writer.close()
    print(f"Training time {datetime.timedelta(seconds=int(time.time() - started))}")
    print(f"Checkpoints: {checkpoints_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("P2PNet training", parents=[get_args_parser()])
    main(parser.parse_args())
