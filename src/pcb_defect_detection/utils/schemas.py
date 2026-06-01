"""Typed data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_min", int(self.x_min))
        object.__setattr__(self, "y_min", int(self.y_min))
        object.__setattr__(self, "x_max", int(self.x_max))
        object.__setattr__(self, "y_max", int(self.y_max))

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clip(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x_min=max(0, min(self.x_min, width - 1)),
            y_min=max(0, min(self.y_min, height - 1)),
            x_max=max(1, min(self.x_max, width)),
            y_max=max(1, min(self.y_max, height)),
        )

    def expand(self, margin: int, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin,
        ).clip(width, height)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass
class DefectDetection:
    """Single localized defect prediction."""

    bbox: BoundingBox
    label: str
    confidence: float
    area: int
    mean_difference: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = asdict(self.bbox)
        data["confidence"] = float(self.confidence)
        data["area"] = int(self.area)
        data["mean_difference"] = float(self.mean_difference)
        data["metadata"] = _json_safe(self.metadata)
        return data


@dataclass
class ImagePair:
    """Paths to a DeepPCB-style template/tested image pair."""

    template_path: Path
    test_path: Path
    annotation_path: Path | None = None
    split: str | None = None


@dataclass
class Annotation:
    """Ground-truth DeepPCB annotation."""

    bbox: BoundingBox
    label: str
    class_id: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = asdict(self.bbox)
        data["confidence"] = float(self.confidence)
        data["area"] = int(self.area)
        data["mean_difference"] = float(self.mean_difference)
        data["metadata"] = _json_safe(self.metadata)
        return data


@dataclass
class PipelineResult:
    """Full inference result for one PCB pair."""

    has_defect: bool
    conclusion: str
    detections: list[DefectDetection]
    image_shape: tuple[int, int]
    template_path: str | None = None
    test_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_defect": bool(self.has_defect),
            "conclusion": self.conclusion,
            "image_shape": [int(value) for value in self.image_shape],
            "template_path": self.template_path,
            "test_path": self.test_path,
            "detections": [detection.to_dict() for detection in self.detections],
        }


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalar-like values to JSON-serializable Python values."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
