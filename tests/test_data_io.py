from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pcb_defect_detection.data.io import load_image, save_image, validate_pair_shapes
from pcb_defect_detection.data.sample import create_synthetic_pair
from pcb_defect_detection.utils.errors import ImageLoadError, InputValidationError


def test_load_image_and_validate_pair(tmp_path: Path) -> None:
    template_path, tested_path, _ = create_synthetic_pair(tmp_path)

    template = load_image(template_path)
    tested = load_image(tested_path)

    assert template.shape == tested.shape
    assert template.ndim == 3
    validate_pair_shapes(template, tested)


def test_load_image_rejects_broken_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageLoadError):
        load_image(broken)


def test_validate_pair_shapes_rejects_mismatch(tmp_path: Path) -> None:
    first = np.zeros((10, 20, 3), dtype=np.uint8)
    second = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(InputValidationError):
        validate_pair_shapes(first, second)


def test_save_image_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "image.png"
    save_image(output_path, np.zeros((16, 16, 3), dtype=np.uint8))

    assert output_path.exists()
