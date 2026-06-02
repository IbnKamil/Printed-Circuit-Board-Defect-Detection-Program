from __future__ import annotations

from pathlib import Path

from pcb_defect_detection.data.sample import create_synthetic_pair
from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference import PCBDefectPipeline
from pcb_defect_detection.utils.reporting import save_csv_report, save_json_report
from pcb_defect_detection.visualization import difference_heatmap, draw_detections, mask_to_rgb


def test_pipeline_detects_synthetic_defects(tmp_path: Path) -> None:
    template_path, tested_path, _ = create_synthetic_pair(tmp_path)
    pipeline = PCBDefectPipeline(
        detector_config=ClassicalDetectorConfig(confidence_threshold=0.1)
    )

    result, artifacts = pipeline.predict_paths(template_path, tested_path)

    assert result.has_defect is True
    assert result.detections
    assert artifacts["mask"].sum() > 0
    assert all(detection.confidence >= 0.1 for detection in result.detections)


def test_reports_and_visualizations_are_saved(tmp_path: Path) -> None:
    template_path, tested_path, _ = create_synthetic_pair(tmp_path / "sample")
    pipeline = PCBDefectPipeline(
        detector_config=ClassicalDetectorConfig(confidence_threshold=0.1)
    )
    result, artifacts = pipeline.predict_paths(template_path, tested_path)

    save_json_report(result, tmp_path / "out" / "report.json")
    save_csv_report(result, tmp_path / "out" / "report.csv")
    overlay = draw_detections(artifacts["tested_rgb"], result.detections)
    heatmap = difference_heatmap(artifacts["abs_diff"])
    mask = mask_to_rgb(artifacts["mask"])

    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "report.csv").exists()
    assert overlay.shape == artifacts["tested_rgb"].shape
    assert heatmap.shape == artifacts["tested_rgb"].shape
    assert mask.shape == artifacts["tested_rgb"].shape
