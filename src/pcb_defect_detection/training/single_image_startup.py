"""Startup helpers for Streamlit single-image model management."""

from __future__ import annotations

import json
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
        _print_startup_summary(summary)
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
    _print_startup_summary(summary)
    return summary


def _print_startup_summary(summary: dict[str, Any]) -> None:
    print("\n========== PCB SINGLE-IMAGE MODEL SUMMARY ==========")
    print(f"Status: {summary.get('status', 'unknown')}")
    print(f"Checkpoint: {summary.get('checkpoint_path', 'unknown')}")
    print(f"Image size: {summary.get('image_size', 'unknown')}")
    print("\nDataset counts:")
    dataset = summary.get("dataset", {})
    if dataset:
        print(json.dumps(dataset, indent=2, ensure_ascii=False))
    else:
        print("No dataset counts stored in checkpoint.")
    print("\nMetrics:")
    metrics = summary.get("metrics", {})
    if metrics:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        print("No metrics stored in checkpoint.")
    print("====================================================\n")
