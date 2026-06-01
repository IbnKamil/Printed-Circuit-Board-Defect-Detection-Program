from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pcb_defect_detection.training.single_image_startup import ensure_single_image_model


def test_startup_trains_once_then_loads_existing_model(tmp_path: Path) -> None:
    dataset_root = _create_startup_dataset(tmp_path / "data")
    checkpoint_path = tmp_path / "model.pt"

    first_summary = ensure_single_image_model(
        checkpoint_path=checkpoint_path,
        default_dataset_root=dataset_root,
        epochs=1,
        batch_size=2,
        image_size=32,
    )
    second_summary = ensure_single_image_model(
        checkpoint_path=checkpoint_path,
        default_dataset_root=dataset_root,
        epochs=1,
        batch_size=2,
        image_size=32,
    )

    assert checkpoint_path.exists()
    assert first_summary["status"] == "trained_new_model"
    assert second_summary["status"] == "loaded_existing_model"
    assert second_summary["dataset"]["train"] > 0
    assert second_summary["dataset"]["validation"] > 0
    assert second_summary["dataset"]["test"] > 0


def _create_startup_dataset(root: Path) -> Path:
    for label, color in {"normal": (40, 140, 60), "defective": (170, 40, 40)}.items():
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            image = np.full((48, 48, 3), color, dtype=np.uint8)
            Image.fromarray(image).save(folder / f"{label}_{index}.png")
    return root
