"""Preprocessing transforms for aligned PCB image pairs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pcb_defect_detection.data.io import validate_pair_shapes


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable preprocessing options."""

    image_size: tuple[int, int] | None = None
    use_clahe: bool = True
    denoise: bool = True
    blur_kernel: int = 3


@dataclass
class PreprocessedPair:
    """Preprocessed outputs used by detectors and visualizers."""

    template_rgb: np.ndarray
    tested_rgb: np.ndarray
    template_gray: np.ndarray
    tested_gray: np.ndarray
    abs_diff: np.ndarray


class PairPreprocessor:
    """Prepare a DeepPCB template/tested pair for comparison."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def __call__(self, template_rgb: np.ndarray, tested_rgb: np.ndarray) -> PreprocessedPair:
        validate_pair_shapes(template_rgb, tested_rgb)
        template = self._resize_if_needed(template_rgb)
        tested = self._resize_if_needed(tested_rgb)
        validate_pair_shapes(template, tested)

        template_gray = self._to_normalized_gray(template)
        tested_gray = self._to_normalized_gray(tested)
        abs_diff = cv2.absdiff(template_gray, tested_gray)
        return PreprocessedPair(
            template_rgb=template,
            tested_rgb=tested,
            template_gray=template_gray,
            tested_gray=tested_gray,
            abs_diff=abs_diff,
        )

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if self.config.image_size is None:
            return image
        width, height = self.config.image_size
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _to_normalized_gray(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if self.config.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        if self.config.denoise:
            kernel = max(1, self.config.blur_kernel)
            if kernel % 2 == 0:
                kernel += 1
            gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        return gray


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float32 [0, 1]."""

    return image.astype(np.float32) / 255.0
