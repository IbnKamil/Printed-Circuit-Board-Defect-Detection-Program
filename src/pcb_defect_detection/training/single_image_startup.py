"""Startup helpers for Streamlit single-image model management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pcb_defect_detection.training.train_single_image_classifier import (
    load_single_image_training_summary,
    train_single_image_classifier,
)


def ensure_single_image_model(
    checkpoint_path: Path,
    default_dataset_root: Path,
    epochs: int = 8,
    batch_size: int = 16,
    image_size: int = 224,
) -> dict[str, Any]:
    """Load an existing model summary or train a model from a local dataset."""

    checkpoint_path = Path(checkpoint_path)
    dataset_root = Path(
        os.environ.get("PCB_SINGLE_IMAGE_DATASET_DIR", str(default_dataset_root))
    )

    if checkpoint_path.exists():
        summary = load_single_image_training_summary(checkpoint_path)
        summary["status"] = "loaded_existing_model"
        print(format_single_image_startup_report(summary, dataset_root))
        return summary

    print("[PCB Streamlit] Single-image model checkpoint was not found.")
    print(f"[PCB Streamlit] Expected checkpoint: {checkpoint_path}")
    print(f"[PCB Streamlit] Trying to train from local dataset: {dataset_root}")
    metrics = train_single_image_classifier(
        dataset_root=dataset_root,
        output_path=checkpoint_path,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
    )
    summary = {
        "status": "trained_new_model",
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "dataset": metrics.get("dataset", {}),
        "image_size": image_size,
    }
    print(format_single_image_startup_report(summary, dataset_root))
    return summary


def format_single_image_startup_report(
    summary: dict[str, Any],
    dataset_root: Path,
) -> str:
    """Format startup information as a readable PyCharm console report."""

    checkpoint_path = summary.get("checkpoint_path") or "not found"
    image_size = summary.get("image_size") or "unknown"
    metrics = summary.get("metrics", {}) or {}
    dataset = summary.get("dataset") or metrics.get("dataset", {}) or {}
    lines = [
        "",
        "========== PCB STREAMLIT STARTUP REPORT =========",
        "PRIMARY METHOD",
        "  Mode       : DeepPCB-aware one-image upload",
        "  *_temp.jpg : NORMAL, template image is known defect-free",
        "  *_test.jpg : compared with matching hidden *_temp.jpg template",
        "  Note       : This is the method used for DeepPCB images in the UI.",
        "",
        "DATASET",
        f"  DeepPCB root : {dataset_root}",
        _format_dataset_counts(dataset),
        "",
        "FALLBACK CNN CHECKPOINT",
        f"  Path       : {checkpoint_path}",
        f"  Image size : {image_size}",
    ]

    if checkpoint_path == "not found":
        lines.extend(
            [
                "  Status     : fallback CNN is absent",
                "  Impact     : DeepPCB *_temp/*_test images still work via template lookup.",
            ]
        )
    else:
        lines.extend(
            [
                "  Status     : fallback CNN is available",
                "  Usage      : only for non-DeepPCB filenames; not used for *_temp/*_test.",
                _format_metrics_table(metrics),
                "  Interpretation:",
                "    Low fallback CNN accuracy is expected on DeepPCB because tiny defects",
                "    are hard to classify from a whole image without the reference template.",
                "    The Streamlit DeepPCB workflow uses hidden template comparison instead.",
            ]
        )

    lines.append("==================================================")
    return "\n".join(lines)


def _format_dataset_counts(dataset: dict[str, Any]) -> str:
    if not dataset:
        return "  Counts     : no dataset counts stored in checkpoint"

    rows = [
        ("total", dataset.get("total"), dataset.get("total_by_class", {})),
        ("train", dataset.get("train"), dataset.get("train_by_class", {})),
        ("validation", dataset.get("validation"), dataset.get("validation_by_class", {})),
        ("test", dataset.get("test"), dataset.get("test_by_class", {})),
    ]
    lines = ["  Split        Total    Normal    Defective"]
    for name, total, by_class in rows:
        if total is None:
            continue
        normal = by_class.get("normal", "-") if isinstance(by_class, dict) else "-"
        defective = by_class.get("defective", "-") if isinstance(by_class, dict) else "-"
        lines.append(f"  {name:<11} {int(total):>5} {str(normal):>9} {str(defective):>12}")
    return "\n".join(lines)


def _format_metrics_table(metrics: dict[str, Any]) -> str:
    validation = metrics.get("validation", {}) if isinstance(metrics, dict) else {}
    test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    if not validation and not test:
        return "  Metrics    : no validation/test metrics stored in checkpoint"

    lines = ["", "  Split          Loss   Accuracy  Precision     Recall         F1"]
    for name, values in (("validation", validation), ("test", test)):
        if not values:
            continue
        lines.append(
            "  "
            f"{name:<10} "
            f"{_fmt(values.get('loss')):>8} "
            f"{_fmt(values.get('accuracy')):>10} "
            f"{_fmt(values.get('precision')):>10} "
            f"{_fmt(values.get('recall')):>10} "
            f"{_fmt(values.get('f1')):>10}"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
