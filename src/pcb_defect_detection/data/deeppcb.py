"""DeepPCB manifest and annotation parsing."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pcb_defect_detection.config import DEEPPCB_CLASS_NAMES, SUPPORTED_IMAGE_EXTENSIONS
from pcb_defect_detection.utils.errors import DatasetError
from pcb_defect_detection.utils.schemas import Annotation, BoundingBox, ImagePair

_TEMPLATE_TOKENS = ("temp", "template")
_TEST_TOKENS = ("test", "tested")
_ANNOTATION_EXTENSIONS = {".txt", ".csv"}


def load_manifest(path: Path, split: str | None = None) -> list[ImagePair]:
    """Load a CSV manifest with template_path,test_path,annotation_path,split."""

    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Manifest does not exist: {path}")

    pairs: list[ImagePair] = []
    with path.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"template_path", "test_path"}
        if not required.issubset(reader.fieldnames or set()):
            raise DatasetError(
                "Manifest must contain columns: template_path,test_path. "
                "Optional: annotation_path,split."
            )
        for row in reader:
            row_split = row.get("split") or None
            if split and row_split != split:
                continue
            annotation_value = row.get("annotation_path") or ""
            pairs.append(
                ImagePair(
                    template_path=_resolve_relative(path.parent, row["template_path"]),
                    test_path=_resolve_relative(path.parent, row["test_path"]),
                    annotation_path=(
                        _resolve_relative(path.parent, annotation_value)
                        if annotation_value
                        else None
                    ),
                    split=row_split,
                )
            )

    if not pairs:
        raise DatasetError(f"No image pairs found in manifest: {path}")
    return pairs


def discover_pairs(root: Path, split: str | None = None) -> list[ImagePair]:
    """Discover DeepPCB-like pairs recursively by filename conventions."""

    root = Path(root)
    search_root = root / split if split and (root / split).exists() else root
    if not search_root.exists():
        raise DatasetError(f"Dataset directory does not exist: {search_root}")

    image_paths = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    templates = {
        _pair_key(path): path
        for path in image_paths
        if _has_token(path.stem, _TEMPLATE_TOKENS)
    }
    tests = {
        _pair_key(path): path for path in image_paths if _has_token(path.stem, _TEST_TOKENS)
    }

    pairs: list[ImagePair] = []
    for key, template_path in templates.items():
        test_path = tests.get(key)
        if test_path is None:
            continue
        annotation_path = _find_annotation(test_path)
        pairs.append(
            ImagePair(
                template_path=template_path,
                test_path=test_path,
                annotation_path=annotation_path,
                split=split,
            )
        )

    if not pairs:
        raise DatasetError(
            f"No DeepPCB-style pairs were discovered under {search_root}. "
            "Use a CSV manifest if your filenames use a different convention."
        )
    return sorted(pairs, key=lambda pair: str(pair.test_path))


def parse_annotation_file(path: Path) -> list[Annotation]:
    """Parse DeepPCB bounding-box annotations from txt/csv files."""

    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Annotation file does not exist: {path}")

    annotations: list[Annotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part for part in re.split(r"[\s,;]+", stripped) if part]
        if len(parts) < 5:
            raise DatasetError(f"Invalid annotation at {path}:{line_number}: {line}")
        values = [int(float(value)) for value in parts[:5]]
        bbox, class_id = _parse_annotation_values(values)
        if class_id not in DEEPPCB_CLASS_NAMES:
            raise DatasetError(
                f"Unknown DeepPCB class id {class_id} at {path}:{line_number}. "
                f"Expected 0-5 or 1-6."
            )
        annotations.append(
            Annotation(
                bbox=bbox,
                label=DEEPPCB_CLASS_NAMES[class_id],
                class_id=class_id,
            )
        )
    return annotations


def write_manifest(pairs: list[ImagePair], path: Path) -> None:
    """Write image pairs to a CSV manifest."""

    if not pairs:
        raise DatasetError("Cannot write empty manifest.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["template_path", "test_path", "annotation_path", "split"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "template_path": str(pair.template_path),
                    "test_path": str(pair.test_path),
                    "annotation_path": str(pair.annotation_path or ""),
                    "split": pair.split or "",
                }
            )


def _resolve_relative(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _has_token(stem: str, tokens: tuple[str, ...]) -> bool:
    normalized = stem.lower().replace("-", "_")
    return any(token in normalized.split("_") or normalized.endswith(token) for token in tokens)


def _pair_key(path: Path) -> str:
    stem = path.stem.lower().replace("-", "_")
    for token in (*_TEMPLATE_TOKENS, *_TEST_TOKENS):
        stem = stem.replace(f"_{token}", "").replace(f"{token}_", "")
        if stem.endswith(token):
            stem = stem[: -len(token)]
    return re.sub(r"[_\W]+", "", stem)


def _find_annotation(test_path: Path) -> Path | None:
    candidates = []
    for suffix in ("_not", "_anno", "_annotation", "_label"):
        candidates.extend(test_path.with_name(test_path.stem + suffix + ext) for ext in _ANNOTATION_EXTENSIONS)
    candidates.extend(test_path.with_suffix(ext) for ext in _ANNOTATION_EXTENSIONS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_annotation_values(values: list[int]) -> tuple[BoundingBox, int]:
    first, second, third, fourth, fifth = values

    coord_first_valid = _is_supported_class_id(fifth) and first < third and second < fourth
    class_first_valid = _is_supported_class_id(first) and second < fourth and third < fifth

    if coord_first_valid:
        class_id = _normalize_class_id(fifth)
        coordinates = [first, second, third, fourth]
    elif class_first_valid:
        class_id = _normalize_class_id(first)
        coordinates = [second, third, fourth, fifth]
    else:
        class_id = _normalize_class_id(fifth)
        coordinates = [first, second, third, fourth]

    x_min, y_min, x_max, y_max = coordinates
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    return BoundingBox(x_min, y_min, x_max, y_max), class_id


def _is_supported_class_id(value: int) -> bool:
    return value == 0 or 1 <= value <= 6


def _normalize_class_id(value: int) -> int:
    if value == 0:
        return 0
    if 1 <= value <= 6:
        return value - 1
    return value
