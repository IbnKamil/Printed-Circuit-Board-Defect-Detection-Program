"""Reference-based classical CV detector for DeepPCB pairs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pcb_defect_detection.detection.postprocessing import (
    filter_by_confidence,
    merge_overlapping_detections,
)
from pcb_defect_detection.preprocessing.transforms import PreprocessedPair
from pcb_defect_detection.utils.schemas import BoundingBox, DefectDetection


@dataclass(frozen=True)
class ClassicalDetectorConfig:
    """Parameters of the pair-difference detector."""

    diff_threshold: int = 22
    min_component_area: int = 18
    max_component_area_ratio: float = 0.05
    morphology_kernel: int = 3
    confidence_threshold: float = 0.25
    merge_iou_threshold: float = 0.20


class ClassicalPairDetector:
    """Detect defects by comparing aligned template and tested PCB images."""

    def __init__(self, config: ClassicalDetectorConfig | None = None) -> None:
        self.config = config or ClassicalDetectorConfig()

    def detect(self, pair: PreprocessedPair) -> tuple[list[DefectDetection], np.ndarray]:
        """Return defect detections and binary difference mask."""

        mask = self._build_mask(pair.abs_diff)
        detections = self._components_to_detections(pair, mask)
        detections = merge_overlapping_detections(
            detections,
            iou_threshold=self.config.merge_iou_threshold,
        )
        detections = filter_by_confidence(detections, self.config.confidence_threshold)
        return detections, mask

    def _build_mask(self, abs_diff: np.ndarray) -> np.ndarray:
        _, fixed_mask = cv2.threshold(
            abs_diff,
            self.config.diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        otsu_threshold, otsu_mask = cv2.threshold(
            abs_diff,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        if otsu_threshold > self.config.diff_threshold:
            mask = cv2.bitwise_or(fixed_mask, otsu_mask)
        else:
            mask = fixed_mask

        kernel_size = max(1, self.config.morphology_kernel)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _components_to_detections(
        self,
        pair: PreprocessedPair,
        mask: np.ndarray,
    ) -> list[DefectDetection]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        image_area = mask.shape[0] * mask.shape[1]
        max_area = image_area * self.config.max_component_area_ratio
        detections: list[DefectDetection] = []

        for label_id in range(1, num_labels):
            x, y, width, height, area = stats[label_id]
            if area < self.config.min_component_area or area > max_area:
                continue
            bbox = BoundingBox(x, y, x + width, y + height)
            component_mask = (labels == label_id).astype(np.uint8)
            detection = self._classify_component(pair, bbox, component_mask, int(area))
            detections.append(detection)
        return detections

    def _classify_component(
        self,
        pair: PreprocessedPair,
        bbox: BoundingBox,
        component_mask: np.ndarray,
        area: int,
    ) -> DefectDetection:
        x_min, y_min, x_max, y_max = bbox.to_xyxy()
        region_template = pair.template_gray[y_min:y_max, x_min:x_max]
        region_tested = pair.tested_gray[y_min:y_max, x_min:x_max]
        region_diff = pair.abs_diff[y_min:y_max, x_min:x_max]
        region_mask = component_mask[y_min:y_max, x_min:x_max].astype(bool)

        mean_difference = float(region_diff[region_mask].mean()) if region_mask.any() else 0.0
        signed_delta = (
            region_tested.astype(np.float32) - region_template.astype(np.float32)
        )
        mean_signed_delta = float(signed_delta[region_mask].mean()) if region_mask.any() else 0.0
        density = area / max(1, bbox.area)
        aspect_ratio = bbox.width / max(1, bbox.height)
        confidence = self._confidence(mean_difference, density, area)
        label = self._heuristic_label(
            mean_signed_delta=mean_signed_delta,
            density=density,
            aspect_ratio=aspect_ratio,
            area=area,
        )

        return DefectDetection(
            bbox=bbox,
            label=label,
            confidence=confidence,
            area=area,
            mean_difference=mean_difference,
            metadata={
                "mean_signed_delta": mean_signed_delta,
                "density": density,
                "aspect_ratio": aspect_ratio,
            },
        )

    @staticmethod
    def _confidence(mean_difference: float, density: float, area: int) -> float:
        diff_score = min(1.0, mean_difference / 90.0)
        density_score = min(1.0, max(0.05, density))
        area_score = min(1.0, area / 250.0)
        return float(0.55 * diff_score + 0.25 * density_score + 0.20 * area_score)

    @staticmethod
    def _heuristic_label(
        mean_signed_delta: float,
        density: float,
        aspect_ratio: float,
        area: int,
    ) -> str:
        """Approximate a DeepPCB class from local pair-difference features."""

        elongated = aspect_ratio > 2.8 or aspect_ratio < 0.36
        compact = 0.65 <= density <= 1.0

        if mean_signed_delta > 8:
            if elongated:
                return "short"
            if compact and area < 180:
                return "spur"
            return "spurious_copper"

        if mean_signed_delta < -8:
            if elongated:
                return "open"
            if area < 160:
                return "pin_hole"
            return "mousebite"

        if elongated:
            return "open"
        return "spurious_copper"
