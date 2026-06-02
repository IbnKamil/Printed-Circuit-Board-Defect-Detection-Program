from __future__ import annotations

from pathlib import Path

from pcb_defect_detection.training.single_image_startup import (
    format_single_image_startup_report,
)


def test_startup_report_is_readable_and_labels_fallback_metrics() -> None:
    summary = {
        "checkpoint_path": "outputs/single_image_model.pt",
        "image_size": 128,
        "dataset": {
            "total": 3001,
            "total_by_class": {"normal": 1501, "defective": 1500},
            "train": 2101,
            "train_by_class": {"normal": 1055, "defective": 1046},
            "validation": 450,
            "validation_by_class": {"normal": 226, "defective": 224},
            "test": 450,
            "test_by_class": {"normal": 220, "defective": 230},
        },
        "metrics": {
            "validation": {
                "loss": 0.6886,
                "accuracy": 0.5422,
                "precision": 0.5473,
                "recall": 0.4642,
                "f1": 0.5024,
            },
            "test": {
                "loss": 0.6915,
                "accuracy": 0.5488,
                "precision": 0.5818,
                "recall": 0.4173,
                "f1": 0.4860,
            },
        },
    }

    primary_metrics = {
        "validation": {
            "samples": 450,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_negative": 226,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 224,
        },
        "test": {
            "samples": 450,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_negative": 220,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 230,
        },
    }

    report = format_single_image_startup_report(
        summary,
        Path("data/deeppcb/PCBData"),
        primary_metrics,
    )

    assert "PRIMARY METHOD" in report
    assert "PRIMARY METHOD METRICS" in report
    assert "FALLBACK CNN CHECKPOINT" in report
    assert "1.000" in report
    assert "not used for *_temp/*_test" in report
    assert "validation" in report
    assert "0.542" in report
    assert "Low fallback CNN accuracy is expected" in report
