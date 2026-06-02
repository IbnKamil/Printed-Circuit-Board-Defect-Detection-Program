"""Report serialization utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pcb_defect_detection.utils.schemas import PipelineResult


def save_json_report(result: PipelineResult, path: Path) -> None:
    """Save pipeline result as pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_csv_report(result: PipelineResult, path: Path) -> None:
    """Save detections as CSV for spreadsheet-friendly inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "label",
                "confidence",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "area",
                "mean_difference",
            ],
        )
        writer.writeheader()
        for detection in result.detections:
            bbox = detection.bbox
            writer.writerow(
                {
                    "label": detection.label,
                    "confidence": f"{detection.confidence:.4f}",
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "area": detection.area,
                    "mean_difference": f"{detection.mean_difference:.4f}",
                }
            )
