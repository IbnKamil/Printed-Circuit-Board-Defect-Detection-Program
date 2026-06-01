"""Postprocessing for defect proposals."""

from __future__ import annotations

from pcb_defect_detection.utils.schemas import BoundingBox, DefectDetection


def iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute intersection over union for two boxes."""

    x_left = max(box_a.x_min, box_b.x_min)
    y_top = max(box_a.y_min, box_b.y_min)
    x_right = min(box_a.x_max, box_b.x_max)
    y_bottom = min(box_a.y_max, box_b.y_max)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = box_a.area + box_b.area - intersection
    return intersection / union if union else 0.0


def merge_overlapping_detections(
    detections: list[DefectDetection],
    iou_threshold: float = 0.25,
) -> list[DefectDetection]:
    """Merge highly-overlapping proposals using confidence priority."""

    if not detections:
        return []

    sorted_detections = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[DefectDetection] = []
    for detection in sorted_detections:
        matched_index = next(
            (
                index
                for index, kept_detection in enumerate(kept)
                if iou(detection.bbox, kept_detection.bbox) >= iou_threshold
            ),
            None,
        )
        if matched_index is None:
            kept.append(detection)
        else:
            kept[matched_index] = _merge_pair(kept[matched_index], detection)
    return sorted(kept, key=lambda item: item.bbox.area, reverse=True)


def filter_by_confidence(
    detections: list[DefectDetection],
    threshold: float,
) -> list[DefectDetection]:
    """Drop low-confidence detections."""

    return [detection for detection in detections if detection.confidence >= threshold]


def _merge_pair(first: DefectDetection, second: DefectDetection) -> DefectDetection:
    bbox = BoundingBox(
        x_min=min(first.bbox.x_min, second.bbox.x_min),
        y_min=min(first.bbox.y_min, second.bbox.y_min),
        x_max=max(first.bbox.x_max, second.bbox.x_max),
        y_max=max(first.bbox.y_max, second.bbox.y_max),
    )
    chosen = first if first.confidence >= second.confidence else second
    total_area = max(1, first.area + second.area)
    mean_difference = (
        first.mean_difference * first.area + second.mean_difference * second.area
    ) / total_area
    return DefectDetection(
        bbox=bbox,
        label=chosen.label,
        confidence=max(first.confidence, second.confidence),
        area=first.area + second.area,
        mean_difference=mean_difference,
        metadata={**first.metadata, **second.metadata, "merged": True},
    )
