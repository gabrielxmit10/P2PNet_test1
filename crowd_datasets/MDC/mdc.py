"""MovingDroneCrowd/MovingDroneCrowd++ adapter for P2PNet.

The source dataset is never modified.  MOT-style head boxes are read directly
from each clip CSV and converted in memory to head-centre points.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MDCRecord:
    image_path: Path
    annotation_path: Path
    scene: str
    clip: str
    frame_id: int
    image_number: int
    points: np.ndarray

    @property
    def sample_id(self) -> str:
        return f"{self.scene}/{self.clip}/{self.image_path.name}"


def _natural_number(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**31 - 1, path.name


def read_split_entries(data_root: Path, split_file: str) -> List[str]:
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"MDC split file not found: {split_path}")

    entries = []
    for raw in split_path.read_text(encoding="utf-8-sig").splitlines():
        value = raw.split("#", 1)[0].strip().replace("\\", "/").strip("/")
        if value:
            entries.append(value)
    if not entries:
        raise ValueError(f"MDC split file is empty: {split_path}")
    return entries


def expand_split_clips(data_root: Path, split_file: str) -> List[Tuple[str, str]]:
    """Expand scene or scene/clip split entries into unique clip pairs."""
    frames_root = data_root / "frames"
    clips: List[Tuple[str, str]] = []
    seen = set()
    for entry in read_split_entries(data_root, split_file):
        parts = Path(entry).parts
        if len(parts) == 1:
            scene = parts[0]
            scene_dir = frames_root / scene
            if not scene_dir.is_dir():
                raise FileNotFoundError(f"Scene in {split_file} does not exist: {scene_dir}")
            clip_names = [p.name for p in scene_dir.iterdir() if p.is_dir()]
            clip_names.sort(key=lambda name: (int(name) if name.isdigit() else 2**31 - 1, name))
        elif len(parts) == 2:
            scene, clip = parts
            clip_names = [clip]
        else:
            raise ValueError(
                f"Invalid split entry {entry!r}; expected 'scene_N' or 'scene_N/clip'."
            )

        for clip in clip_names:
            pair = (scene, clip)
            clip_dir = frames_root / scene / clip
            annotation = data_root / "annotations" / scene / f"{clip}.csv"
            if not clip_dir.is_dir():
                raise FileNotFoundError(f"Frame clip directory not found: {clip_dir}")
            if not annotation.is_file():
                raise FileNotFoundError(f"Annotation file not found: {annotation}")
            if pair not in seen:
                clips.append(pair)
                seen.add(pair)
    return clips


def load_clip_points(annotation_path: Path) -> Dict[int, np.ndarray]:
    """Load frame -> head-centre points from one MOT-style annotation CSV."""
    grouped: Dict[int, List[Tuple[float, float]]] = {}
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 6:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected at least 6 columns, got {len(row)}"
                )
            try:
                frame_id = int(float(row[0]))
                x, y, width, height = (float(value) for value in row[2:6])
            except ValueError as exc:
                raise ValueError(
                    f"{annotation_path}:{line_number}: invalid numeric annotation {row[:6]}"
                ) from exc
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{annotation_path}:{line_number}: non-positive head box {row[2:6]}"
                )
            grouped.setdefault(frame_id, []).append((x + width / 2.0, y + height / 2.0))

    return {
        frame_id: np.asarray(points, dtype=np.float32).reshape(-1, 2)
        for frame_id, points in grouped.items()
    }


def build_records(
    data_root: Path,
    split_file: str,
    frame_stride: int = 1,
    max_samples: int = 0,
) -> List[MDCRecord]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    records: List[MDCRecord] = []
    for scene, clip in expand_split_clips(data_root, split_file):
        annotation_path = data_root / "annotations" / scene / f"{clip}.csv"
        points_by_frame = load_clip_points(annotation_path)
        image_dir = data_root / "frames" / scene / clip
        images = sorted(image_dir.glob("*.jpg"), key=_natural_number)
        if not images:
            raise FileNotFoundError(f"No JPG frames found in {image_dir}")

        for image_index, image_path in enumerate(images):
            if image_index % frame_stride:
                continue
            try:
                image_number = int(image_path.stem)
            except ValueError as exc:
                raise ValueError(f"MDC frame filename must be numeric: {image_path}") from exc
            frame_id = image_number - 1
            points = points_by_frame.get(frame_id, np.empty((0, 2), dtype=np.float32))
            records.append(
                MDCRecord(
                    image_path=image_path,
                    annotation_path=annotation_path,
                    scene=scene,
                    clip=clip,
                    frame_id=frame_id,
                    image_number=image_number,
                    points=points,
                )
            )
            if max_samples > 0 and len(records) >= max_samples:
                return records
    return records


def _to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class MDCFrameDataset(Dataset):
    """P2PNet frame dataset backed directly by MDC/MDC++ annotations."""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        train: bool = False,
        crop_size: int = 512,
        num_patches: int = 1,
        scale_min: float = 0.7,
        scale_max: float = 1.3,
        flip_probability: float = 0.5,
        frame_stride: int = 1,
        max_samples: int = 0,
        min_crop_points: int = 0,
        crop_attempts: int = 10,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"MDC dataset root not found: {self.data_root}")
        self.split_file = split_file
        self.train = train
        self.crop_size = int(crop_size)
        self.num_patches = int(num_patches)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.flip_probability = float(flip_probability)
        self.min_crop_points = int(min_crop_points)
        self.crop_attempts = max(1, int(crop_attempts))

        if self.crop_size < 32:
            raise ValueError("crop_size must be at least 32")
        if self.num_patches < 1:
            raise ValueError("num_patches must be at least 1")
        if not (0 < self.scale_min <= self.scale_max):
            raise ValueError("Expected 0 < scale_min <= scale_max")
        if not (0 <= self.flip_probability <= 1):
            raise ValueError("flip_probability must be between 0 and 1")

        self.records = build_records(
            self.data_root,
            split_file,
            frame_stride=frame_stride,
            max_samples=max_samples,
        )
        if not self.records:
            raise ValueError(f"No MDC frames selected by {split_file}")

    def __len__(self) -> int:
        return len(self.records)

    def get_raw_sample(self, index: int) -> Tuple[Image.Image, np.ndarray, MDCRecord]:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        return image, record.points.copy(), record

    def _make_patch(
        self,
        image: Image.Image,
        points: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        width, height = image.size
        chosen = None
        for _ in range(self.crop_attempts):
            scale = random.uniform(self.scale_min, self.scale_max)
            source_size = max(1, int(round(self.crop_size / scale)))
            source_width = min(source_size, width)
            source_height = min(source_size, height)
            left = random.randint(0, max(0, width - source_width))
            top = random.randint(0, max(0, height - source_height))
            mask = (
                (points[:, 0] >= left)
                & (points[:, 0] < left + source_width)
                & (points[:, 1] >= top)
                & (points[:, 1] < top + source_height)
            ) if len(points) else np.zeros((0,), dtype=bool)
            selected = points[mask].copy().reshape(-1, 2)
            chosen = (left, top, source_width, source_height, selected)
            if len(selected) >= self.min_crop_points:
                break

        left, top, source_width, source_height, selected = chosen
        patch = image.crop((left, top, left + source_width, top + source_height))
        if patch.size != (self.crop_size, self.crop_size):
            patch = patch.resize((self.crop_size, self.crop_size), Image.Resampling.BILINEAR)

        if len(selected):
            selected[:, 0] = (selected[:, 0] - left) * self.crop_size / source_width
            selected[:, 1] = (selected[:, 1] - top) * self.crop_size / source_height

        if random.random() < self.flip_probability:
            patch = TF.hflip(patch)
            if len(selected):
                selected[:, 0] = self.crop_size - selected[:, 0]

        return _to_normalized_tensor(patch), torch.from_numpy(selected.astype(np.float32, copy=False))

    def __getitem__(self, index: int):
        image, points, record = self.get_raw_sample(index)
        if not self.train:
            target = {
                "point": torch.from_numpy(points.astype(np.float32, copy=False)),
                "labels": torch.ones((len(points),), dtype=torch.int64),
                "image_id": torch.tensor([index], dtype=torch.int64),
            }
            return _to_normalized_tensor(image), [target]

        images = []
        targets = []
        for patch_index in range(self.num_patches):
            patch, patch_points = self._make_patch(image, points)
            images.append(patch)
            targets.append(
                {
                    "point": patch_points,
                    "labels": torch.ones((len(patch_points),), dtype=torch.int64),
                    "image_id": torch.tensor(
                        [index * self.num_patches + patch_index], dtype=torch.int64
                    ),
                }
            )
        return torch.stack(images), targets


def loading_data(data_root: str, args):
    common = dict(
        data_root=data_root,
        crop_size=args.crop_size,
    )
    train_set = MDCFrameDataset(
        split_file=args.train_split,
        train=True,
        num_patches=args.num_patches,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        flip_probability=args.flip_probability,
        max_samples=args.max_train_samples,
        min_crop_points=args.min_crop_points,
        frame_stride=args.train_frame_stride,
        **common,
    )
    val_set = MDCFrameDataset(
        split_file=args.val_split,
        train=False,
        num_patches=1,
        max_samples=args.max_eval_samples,
        frame_stride=args.val_frame_stride,
        **common,
    )
    return train_set, val_set
