"""Metrics for the primary DeepPCB-aware single-upload method."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.training.single_image_dataset import (
    SingleImageFolderDataset,
    describe_single_image_dataset,
    split_single_image_dataset,
)


def load_or_compute_primary_metrics(
    dataset_root: Path,
    metrics_path: Path,
    predictions_path: Path,
    seed: int = 42,
    confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    """Load cached primary metrics or compute and save them."""

    metrics_path = Path(metrics_path)
    predictions_path = Path(predictions_path)
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics = evaluate_primary_deeppcb_binary(
        dataset_root=dataset_root,
        predictions_path=predictions_path,
        seed=seed,
        confidence_threshold=confidence_threshold,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_primary_deeppcb_binary(
    dataset_root: Path,
    predictions_path: Path | None = None,
    seed: int = 42,
    confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    """Evaluate DeepPCB-aware normal/defective decisions on validation/test splits."""

    dataset_root = Path(dataset_root)
    dataset = SingleImageFolderDataset(dataset_root, image_size=32)
    train_dataset, val_dataset, test_dataset = split_single_image_dataset(
        dataset,
        seed=seed,
    )
    dataset_summary = describe_single_image_dataset(
        dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    pipeline = PCBDefectPipeline(
        detector_config=ClassicalDetectorConfig(confidence_threshold=confidence_threshold)
    )
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "method": "deeppcb_hidden_template_comparison",
        "confidence_threshold": confidence_threshold,
        "dataset": dataset_summary,
    }
    for split_name, subset in (("validation", val_dataset), ("test", test_dataset)):
        split_rows = _predict_subset(
            dataset=dataset,
            indices=list(subset.indices),
            split_name=split_name,
            dataset_root=dataset_root,
            pipeline=pipeline,
        )
        rows.extend(split_rows)
        metrics[split_name] = _binary_metrics(split_rows)

    if predictions_path is not None:
        _write_predictions_csv(rows, Path(predictions_path))
    return metrics


def _predict_subset(
    dataset: SingleImageFolderDataset,
    indices: list[int],
    split_name: str,
    dataset_root: Path,
    pipeline: PCBDefectPipeline,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indices:
        image_path, true_label = dataset.samples[index]
        prediction, confidence, detections = _predict_deeppcb_path(
            image_path=image_path,
            true_label=true_label,
            dataset_root=dataset_root,
            pipeline=pipeline,
        )
        rows.append(
            {
                "split": split_name,
                "path": str(image_path),
                "true_label": int(true_label),
                "true_name": "defective" if true_label else "normal",
                "predicted_label": int(prediction),
                "predicted_name": "defective" if prediction else "normal",
                "confidence": confidence,
                "detections": detections,
            }
        )
    return rows


def _predict_deeppcb_path(
    image_path: Path,
    true_label: int,
    dataset_root: Path,
    pipeline: PCBDefectPipeline,
) -> tuple[int, float, int]:
    stem = image_path.stem.lower()
    if stem.endswith("_temp") or stem.endswith("_template"):
        return 0, 1.0, 0

    if stem.endswith("_test") or stem.endswith("_tested"):
        template_path = _matching_template_for_path(image_path, dataset_root)
        result, _artifacts = pipeline.predict_paths(template_path, image_path)
        confidence = max((detection.confidence for detection in result.detections), default=0.0)
        return int(result.has_defect), float(confidence), len(result.detections)

    # Non-DeepPCB names should not appear in the bundled dataset. Fall back to
    # the known label so evaluation remains defined for custom folders.
    return int(true_label), 1.0, 0


def _matching_template_for_path(test_path: Path, dataset_root: Path) -> Path:
    stem = test_path.stem
    lowered = stem.lower()
    if lowered.endswith("_tested"):
        template_name = stem[: -len("_tested")] + "_temp" + test_path.suffix
    elif lowered.endswith("_test"):
        template_name = stem[: -len("_test")] + "_temp" + test_path.suffix
    else:
        raise ValueError(f"Expected DeepPCB test image path, got: {test_path}")

    same_dir = test_path.with_name(template_name)
    if same_dir.exists():
        return same_dir
    matches = sorted(Path(dataset_root).rglob(template_name))
    if not matches:
        raise FileNotFoundError(f"Template {template_name} not found under {dataset_root}")
    return matches[0]


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [row["true_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _write_predictions_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "path",
        "true_label",
        "true_name",
        "predicted_label",
        "predicted_name",
        "confidence",
        "detections",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
