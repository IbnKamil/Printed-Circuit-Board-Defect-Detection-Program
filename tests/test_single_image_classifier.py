from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pcb_defect_detection.models.single_image_cnn import predict_single_image
from pcb_defect_detection.training.single_image_dataset import SingleImageFolderDataset
from pcb_defect_detection.training.train_single_image_classifier import (
    train_single_image_classifier,
)


def test_single_image_dataset_loads_normal_and_defective(tmp_path: Path) -> None:
    dataset_root = _create_tiny_binary_dataset(tmp_path)

    dataset = SingleImageFolderDataset(dataset_root, image_size=32)

    assert len(dataset) == 6
    labels = [int(dataset[index][1].item()) for index in range(len(dataset))]
    assert set(labels) == {0, 1}


def test_train_and_predict_single_image_classifier(tmp_path: Path) -> None:
    dataset_root = _create_tiny_binary_dataset(tmp_path / "dataset")
    checkpoint_path = tmp_path / "single_image_model.pt"

    metrics = train_single_image_classifier(
        dataset_root=dataset_root,
        output_path=checkpoint_path,
        epochs=1,
        batch_size=2,
        image_size=32,
    )

    prediction = predict_single_image(
        np.full((64, 64, 3), (180, 40, 40), dtype=np.uint8),
        checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert "validation" in metrics
    assert "test" in metrics
    assert "dataset" in metrics
    assert "accuracy" in metrics["validation"]
    assert prediction["label"] in {"normal", "defective"}
    assert 0.0 <= prediction["confidence"] <= 1.0


def _create_tiny_binary_dataset(root: Path) -> Path:
    normal_dir = root / "normal"
    defective_dir = root / "defective"
    normal_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)

    for index in range(3):
        normal = np.full((64, 64, 3), (40, 150 + index, 60), dtype=np.uint8)
        defective = np.full((64, 64, 3), (160 + index, 40, 40), dtype=np.uint8)
        Image.fromarray(normal).save(normal_dir / f"normal_{index}.png")
        Image.fromarray(defective).save(defective_dir / f"defective_{index}.png")
    return root
