"""Defect detection algorithms."""

from pcb_defect_detection.detection.classical import (
    ClassicalDetectorConfig,
    ClassicalPairDetector,
)
from pcb_defect_detection.detection.postprocessing import iou

__all__ = ["ClassicalDetectorConfig", "ClassicalPairDetector", "iou"]
