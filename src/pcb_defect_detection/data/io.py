"""Image input/output and validation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pcb_defect_detection.config import SUPPORTED_IMAGE_EXTENSIONS
from pcb_defect_detection.utils.errors import ImageLoadError, InputValidationError


def validate_image_path(path: Path) -> Path:
    """Validate that a path points to a supported image file."""

    path = Path(path)
    if not path.exists():
        raise InputValidationError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"Expected image file, got directory: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    return path


def load_image(path: Path, color: bool = True) -> np.ndarray:
    """Load an image with OpenCV and raise a readable error on failure."""

    path = validate_image_path(path)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ImageLoadError(f"Could not decode image file: {path}")
    if color:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save RGB or grayscale image to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        encoded = image
    success = cv2.imwrite(str(path), encoded)
    if not success:
        raise ImageLoadError(f"Could not save image to: {path}")


def validate_pair_shapes(template: np.ndarray, tested: np.ndarray) -> None:
    """Ensure DeepPCB pair images are aligned by shape."""

    if template.shape[:2] != tested.shape[:2]:
        raise InputValidationError(
            "Template and tested images must have identical height and width. "
            f"Got {template.shape[:2]} and {tested.shape[:2]}."
        )


def list_images(directory: Path) -> list[Path]:
    """List supported images in a directory, non-recursively."""

    directory = Path(directory)
    if not directory.exists():
        raise InputValidationError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise InputValidationError(f"Expected directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
