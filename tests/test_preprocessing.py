from __future__ import annotations

from pathlib import Path

from pcb_defect_detection.data.io import load_image
from pcb_defect_detection.data.sample import create_synthetic_pair
from pcb_defect_detection.preprocessing import PairPreprocessor, PreprocessingConfig


def test_preprocessor_outputs_grayscale_and_difference(tmp_path: Path) -> None:
    template_path, tested_path, _ = create_synthetic_pair(tmp_path)
    preprocessor = PairPreprocessor(PreprocessingConfig(image_size=(128, 96)))

    result = preprocessor(load_image(template_path), load_image(tested_path))

    assert result.template_rgb.shape[:2] == (96, 128)
    assert result.template_gray.shape == (96, 128)
    assert result.abs_diff.shape == (96, 128)
    assert result.abs_diff.max() > 0
