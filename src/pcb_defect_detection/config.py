"""Project-level constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEEPPCB_CLASS_NAMES: dict[int, str] = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "pin_hole",
    5: "spurious_copper",
}

DEEPPCB_CLASS_TO_ID: dict[str, int] = {
    name: class_id for class_id, name in DEEPPCB_CLASS_NAMES.items()
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
