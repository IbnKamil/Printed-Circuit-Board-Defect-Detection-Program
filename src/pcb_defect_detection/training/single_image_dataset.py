"""Datasets for single-image binary PCB classification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

from pcb_defect_detection.config import SUPPORTED_IMAGE_EXTENSIONS
from pcb_defect_detection.models.single_image_cnn import pil_to_rgb_array, preprocess_single_image
from pcb_defect_detection.utils.errors import DatasetError

NORMAL_FOLDER_NAMES = {"normal", "good", "ok", "no_defect", "nodefect"}
DEFECTIVE_FOLDER_NAMES = {"defective", "defect", "bad", "ng", "with_defect"}


class SingleImageFolderDataset(Dataset):
    """Binary dataset with subfolders for normal and defective PCB images."""

    def __init__(self, root: Path, image_size: int = 224) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.samples = _collect_samples(self.root)
        if not self.samples:
            raise DatasetError(
                "No training images found. Expected folders like "
                "'normal/' and 'defective/' with image files inside."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = preprocess_single_image(
                pil_to_rgb_array(image),
                image_size=self.image_size,
            )
        return tensor, torch.tensor(label, dtype=torch.long)


def split_single_image_dataset(
    dataset: SingleImageFolderDataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train, validation and test subsets."""

    if len(dataset) < 3:
        raise DatasetError("At least three images are required for train/val/test split.")
    test_size = max(1, int(round(len(dataset) * test_ratio)))
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = len(dataset) - val_size - test_size
    if train_size < 1:
        train_size = 1
        remaining = len(dataset) - train_size
        val_size = max(1, remaining // 2)
        test_size = remaining - val_size
    if test_size < 1:
        raise DatasetError("Dataset is too small to create a non-empty test split.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def describe_single_image_dataset(
    dataset: SingleImageFolderDataset,
    train_dataset: Subset | None = None,
    val_dataset: Subset | None = None,
    test_dataset: Subset | None = None,
) -> dict[str, Any]:
    """Return dataset and split counts for console logging/checkpoints."""

    summary: dict[str, Any] = {
        "dataset_root": str(dataset.root),
        "total": len(dataset),
        "total_by_class": _count_labels(dataset, range(len(dataset))),
    }
    if train_dataset is not None:
        summary["train"] = len(train_dataset)
        summary["train_by_class"] = _count_labels(dataset, train_dataset.indices)
    if val_dataset is not None:
        summary["validation"] = len(val_dataset)
        summary["validation_by_class"] = _count_labels(dataset, val_dataset.indices)
    if test_dataset is not None:
        summary["test"] = len(test_dataset)
        summary["test_by_class"] = _count_labels(dataset, test_dataset.indices)
    return summary


def _count_labels(dataset: SingleImageFolderDataset, indices) -> dict[str, int]:
    labels = Counter(dataset.samples[index][1] for index in indices)
    return {
        "normal": int(labels.get(0, 0)),
        "defective": int(labels.get(1, 0)),
    }


def _collect_samples(root: Path) -> list[tuple[Path, int]]:
    if not root.exists():
        raise DatasetError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise DatasetError(f"Expected dataset directory, got file: {root}")

    samples: list[tuple[Path, int]] = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        label = _label_from_folder(folder.name)
        if label is None:
            continue
        for image_path in folder.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    labels = {label for _path, label in samples}
    if samples and labels != {0, 1}:
        raise DatasetError(
            "Single-image training requires both classes: normal and defective."
        )
    return sorted(samples, key=lambda item: str(item[0]))


def _label_from_folder(folder_name: str) -> int | None:
    normalized = folder_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in NORMAL_FOLDER_NAMES:
        return 0
    if normalized in DEFECTIVE_FOLDER_NAMES:
        return 1
    return None
