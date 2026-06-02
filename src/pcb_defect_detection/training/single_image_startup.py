"""Startup report formatting for Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_single_image_startup_report(
    dataset_root: Path,
    primary_metrics: dict[str, Any] | None = None,
) -> str:
    """Format primary DeepPCB-aware metrics as a PyCharm console report."""

    primary_metrics = primary_metrics or {}
    dataset = primary_metrics.get("dataset", {}) or {}
    lines = [
        "",
        "========== PCB STREAMLIT STARTUP REPORT =========",
        "PRIMARY METHOD",
        "  Mode       : DeepPCB-aware one-image upload",
        "  *_temp.jpg : NORMAL, template image is known defect-free",
        "  *_test.jpg : compared with matching hidden *_temp.jpg template",
        "  Note       : This is the only method used by the Streamlit UI.",
        "",
        "PRIMARY METHOD METRICS",
        _format_primary_metrics_table(primary_metrics),
        "",
        "DATASET",
        f"  DeepPCB root : {dataset_root}",
        _format_dataset_counts(dataset),
        "",
        "NOTES",
        "  CNN fallback has been removed from the Streamlit workflow/report.",
        "  Metrics above evaluate the actual UI behavior on validation/test splits.",
        "==================================================",
    ]
    return "\n".join(lines)


def _format_primary_metrics_table(primary_metrics: dict[str, Any]) -> str:
    if not primary_metrics:
        return "  Metrics    : not computed yet"

    lines = ["  Split        Samples   Accuracy  Precision     Recall         F1     TN     FP     FN     TP"]
    for split_name in ("validation", "test"):
        values = primary_metrics.get(split_name, {})
        if not values:
            continue
        lines.append(
            "  "
            f"{split_name:<10} "
            f"{int(values.get('samples', 0)):>7} "
            f"{_fmt(values.get('accuracy')):>10} "
            f"{_fmt(values.get('precision')):>10} "
            f"{_fmt(values.get('recall')):>10} "
            f"{_fmt(values.get('f1')):>10} "
            f"{int(values.get('true_negative', 0)):>6} "
            f"{int(values.get('false_positive', 0)):>6} "
            f"{int(values.get('false_negative', 0)):>6} "
            f"{int(values.get('true_positive', 0)):>6}"
        )
    lines.extend(
        [
            "  These metrics evaluate the actual Streamlit DeepPCB workflow:",
            "  *_temp -> normal, *_test -> hidden template comparison.",
        ]
    )
    return "\n".join(lines)


def _format_dataset_counts(dataset: dict[str, Any]) -> str:
    if not dataset:
        return "  Counts     : no dataset counts stored in metrics cache"

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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
