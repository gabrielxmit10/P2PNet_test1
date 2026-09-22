"""Cheap progressive smoke test for the P2PNet + MDC workflow."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch

import util.misc as utils
from crowd_datasets.MDC.mdc import MDCFrameDataset
from models import build_model
from workflow_utils import extract_state_dict, resolve_device


def get_args_parser():
    parser = argparse.ArgumentParser("P2PNet MDC smoke test")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_file", default="train.txt")
    parser.add_argument("--weights", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop_size", default=128, type=int)
    parser.add_argument("--backbone", default="vgg16_bn", choices=("vgg16_bn", "vgg16"))
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--line", default=2, type=int)
    return parser


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main(args):
    device = resolve_device(args.device)
    dataset = MDCFrameDataset(
        args.data_root,
        split_file=args.split_file,
        train=True,
        crop_size=args.crop_size,
        num_patches=1,
        max_samples=2,
        min_crop_points=1,
        crop_attempts=50,
    )
    images, targets = dataset[0]
    samples, targets = utils.collate_fn_crowd([(images, targets)])
    print(f"Dataset sample: tensor={tuple(samples.shape)} points={len(targets[0]['point'])}")

    model_options = SimpleNamespace(
        backbone=args.backbone,
        pretrained_backbone=False,
        row=args.row,
        line=args.line,
        set_cost_class=1.0,
        set_cost_point=0.05,
        point_loss_coef=0.0002,
        eos_coef=0.5,
    )
    model, criterion = build_model(model_options, training=True)
    if args.weights:
        model.load_state_dict(extract_state_dict(_load(args.weights)), strict=True)
        print(f"Checkpoint loaded: {args.weights}")
    model.to(device)
    criterion.to(device)
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    model.train()
    outputs = model(samples)
    losses = criterion(outputs, targets)
    weighted = sum(losses[key] * criterion.weight_dict[key] for key in losses if key in criterion.weight_dict)
    if "loss_point" not in losses or "loss_point" not in criterion.weight_dict:
        raise RuntimeError("Point regression loss is not active")
    if not math.isfinite(weighted.item()):
        raise RuntimeError(f"Non-finite loss: {weighted.item()}")
    weighted.backward()
    print(
        "Forward/backward passed:",
        {name: float(value.detach().cpu()) for name, value in losses.items()},
    )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
