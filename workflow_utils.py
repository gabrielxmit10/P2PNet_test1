"""Shared checkpoint, device, tiled-inference, and reporting utilities."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from crowd_datasets.MDC.mdc import IMAGENET_MEAN, IMAGENET_STD
from models import build_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Prediction:
    points: np.ndarray
    scores: np.ndarray
    original_size: Tuple[int, int]
    inference_size: Tuple[int, int]
    scale: float
    tiles: int

    @property
    def count(self) -> int:
        return int(len(self.points))


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def model_args(
    backbone: str = "vgg16_bn",
    row: int = 2,
    line: int = 2,
    pretrained_backbone: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        backbone=backbone,
        row=row,
        line=line,
        pretrained_backbone=pretrained_backbone,
    )


def _torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state-dict or a dictionary containing one")
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("No model state-dict found in checkpoint")

    cleaned = {}
    for key, value in checkpoint.items():
        while key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, strict: bool = True):
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = _torch_load(path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, incompatible


def create_model(
    checkpoint_path: str,
    device: torch.device,
    backbone: str = "vgg16_bn",
    row: int = 2,
    line: int = 2,
    strict: bool = True,
) -> torch.nn.Module:
    args = model_args(backbone=backbone, row=row, line=line, pretrained_backbone=False)
    model = build_model(args)
    load_model_weights(model, checkpoint_path, strict=strict)
    model.to(device)
    model.eval()
    return model


def list_images(input_path: str, recursive: bool = False) -> List[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input image or directory not found: {path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found under: {path}")
    return images


def _axis_tiles(length: int, tile_size: int, overlap: int):
    if tile_size <= 0 or length <= tile_size:
        starts = [0]
        tile_size = length
    else:
        stride = tile_size - overlap
        starts = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if starts[-1] != last:
            starts.append(last)

    result = []
    for index, start in enumerate(starts):
        end = min(length, start + tile_size)
        if index == 0:
            own_start = 0.0
        else:
            previous_end = min(length, starts[index - 1] + tile_size)
            own_start = (start + previous_end) / 2.0
        if index == len(starts) - 1:
            own_end = float(length)
        else:
            next_start = starts[index + 1]
            own_end = (end + next_start) / 2.0
        result.append((start, end, own_start, own_end))
    return result


def _image_tensor(image: Image.Image, pad_multiple: int = 128) -> torch.Tensor:
    tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    height, width = tensor.shape[-2:]
    padded_height = int(math.ceil(height / pad_multiple) * pad_multiple)
    padded_width = int(math.ceil(width / pad_multiple) * pad_multiple)
    if (padded_height, padded_width) != (height, width):
        tensor = torch.nn.functional.pad(
            tensor,
            (0, padded_width - width, 0, padded_height - height),
            mode="constant",
            value=0.0,
        )
    return tensor


@torch.inference_mode()
def predict_pil(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    threshold: float = 0.5,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    max_size: int = 0,
) -> Prediction:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")
    if tile_size > 0 and not (0 <= tile_overlap < tile_size):
        raise ValueError("tile_overlap must satisfy 0 <= overlap < tile_size")

    image = image.convert("RGB")
    original_width, original_height = image.size
    scale = 1.0
    if max_size > 0 and max(original_width, original_height) > max_size:
        scale = max_size / float(max(original_width, original_height))
        resized = (
            max(1, int(round(original_width * scale))),
            max(1, int(round(original_height * scale))),
        )
        image = image.resize(resized, Image.Resampling.BILINEAR)

    width, height = image.size
    x_tiles = _axis_tiles(width, tile_size, tile_overlap)
    y_tiles = _axis_tiles(height, tile_size, tile_overlap)
    all_points: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for y_start, y_end, own_top, own_bottom in y_tiles:
        for x_start, x_end, own_left, own_right in x_tiles:
            crop = image.crop((x_start, y_start, x_end, y_end))
            tensor = _image_tensor(crop).unsqueeze(0).to(device)
            outputs = model(tensor)
            scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
            points = outputs["pred_points"][0]
            keep = scores > threshold
            points = points[keep]
            scores = scores[keep]

            crop_width, crop_height = crop.size
            valid = (
                (points[:, 0] >= 0)
                & (points[:, 0] < crop_width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < crop_height)
            )
            points = points[valid]
            scores = scores[valid]
            if points.numel() == 0:
                continue

            points = points.clone()
            points[:, 0] += x_start
            points[:, 1] += y_start
            owned = (
                (points[:, 0] >= own_left)
                & (points[:, 0] < own_right)
                & (points[:, 1] >= own_top)
                & (points[:, 1] < own_bottom)
            )
            points = points[owned]
            scores = scores[owned]
            if points.numel():
                all_points.append(points.cpu().numpy())
                all_scores.append(scores.cpu().numpy())

    if all_points:
        point_array = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
        score_array = np.concatenate(all_scores, axis=0).astype(np.float32, copy=False)
    else:
        point_array = np.empty((0, 2), dtype=np.float32)
        score_array = np.empty((0,), dtype=np.float32)

    if scale != 1.0 and len(point_array):
        point_array /= scale
    return Prediction(
        points=point_array,
        scores=score_array,
        original_size=(original_width, original_height),
        inference_size=(width, height),
        scale=scale,
        tiles=len(x_tiles) * len(y_tiles),
    )


def draw_prediction(
    image: Image.Image,
    prediction: Prediction,
    ground_truth: np.ndarray | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    radius = max(2, int(round(max(image.size) / 1000)))
    if ground_truth is not None:
        for x, y in np.asarray(ground_truth).reshape(-1, 2):
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, (0, 255, 0), -1)
    for x, y in prediction.points:
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, (0, 0, 255), -1)
    text = f"pred={prediction.count}"
    if ground_truth is not None:
        text += f" gt={len(ground_truth)}"
    cv2.putText(canvas, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment_report(repo_root: Path | None = None) -> dict:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if repo_root is not None:
        report["git_commit"] = git_revision(Path(repo_root))
    return report
