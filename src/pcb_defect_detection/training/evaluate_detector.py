"""Evaluation utilities for the pair-difference detector."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from pcb_defect_detection.config import DEEPPCB_CLASS_NAMES
from pcb_defect_detection.data.deeppcb import parse_annotation_file
from pcb_defect_detection.detection.postprocessing import iou
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.utils.reporting import save_csv_report, save_json_report
from pcb_defect_detection.utils.schemas import Annotation, DefectDetection, ImagePair


def evaluate_detector(
    pairs: list[ImagePair],
    output_dir: Path,
    iou_threshold: float = 0.3,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate detector by greedy IoU matching to ground-truth boxes."""

    annotated_pairs = [pair for pair in pairs if pair.annotation_path is not None]
    if not annotated_pairs:
        raise ValueError("Evaluation requires pairs with annotation_path.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PCBDefectPipeline(checkpoint_path=checkpoint_path)

    true_labels: list[str] = []
    predicted_labels: list[str] = []
    matched_ious: list[float] = []
    total_gt = 0
    total_pred = 0
    matched_count = 0

    for index, pair in enumerate(annotated_pairs):
        result, _artifacts = pipeline.predict_paths(pair.template_path, pair.test_path)
        annotations = parse_annotation_file(pair.annotation_path)
        total_gt += len(annotations)
        total_pred += len(result.detections)
        matches = _match_predictions(result.detections, annotations, iou_threshold)
        matched_count += len(matches)
        for detection, annotation, overlap in matches:
            predicted_labels.append(detection.label)
            true_labels.append(annotation.label)
            matched_ious.append(overlap)

        pair_dir = output_dir / f"pair_{index:04d}"
        save_json_report(result, pair_dir / "report.json")
        save_csv_report(result, pair_dir / "report.csv")

    precision = matched_count / total_pred if total_pred else 0.0
    recall = matched_count / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    labels = list(DEEPPCB_CLASS_NAMES.values())
    metrics = {
        "pairs": len(annotated_pairs),
        "total_ground_truth": total_gt,
        "total_predictions": total_pred,
        "matched_predictions": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
        if true_labels
        else {},
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=labels,
        ).tolist()
        if true_labels
        else [],
        "labels": labels,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def _match_predictions(
    detections: list[DefectDetection],
    annotations: list[Annotation],
    iou_threshold: float,
) -> list[tuple[DefectDetection, Annotation, float]]:
    candidate_matches: list[tuple[float, int, int]] = []
    for pred_index, detection in enumerate(detections):
        for gt_index, annotation in enumerate(annotations):
            overlap = iou(detection.bbox, annotation.bbox)
            if overlap >= iou_threshold:
                candidate_matches.append((overlap, pred_index, gt_index))

    matches: list[tuple[DefectDetection, Annotation, float]] = []
    used_predictions: set[int] = set()
    used_annotations: set[int] = set()
    for overlap, pred_index, gt_index in sorted(candidate_matches, reverse=True):
        if pred_index in used_predictions or gt_index in used_annotations:
            continue
        used_predictions.add(pred_index)
        used_annotations.add(gt_index)
        matches.append((detections[pred_index], annotations[gt_index], overlap))
    return matches
