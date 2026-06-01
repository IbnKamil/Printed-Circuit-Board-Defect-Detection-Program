"""High-level inference pipeline for one DeepPCB image pair."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pcb_defect_detection.data.io import load_image
from pcb_defect_detection.detection.classical import (
    ClassicalDetectorConfig,
    ClassicalPairDetector,
)
from pcb_defect_detection.models.patch_cnn import classify_patch, load_patch_classifier
from pcb_defect_detection.preprocessing.transforms import PairPreprocessor, PreprocessingConfig
from pcb_defect_detection.utils.schemas import DefectDetection, PipelineResult


class PCBDefectPipeline:
    """Run validation, preprocessing, detection and optional CNN classification."""

    def __init__(
        self,
        preprocessing_config: PreprocessingConfig | None = None,
        detector_config: ClassicalDetectorConfig | None = None,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.preprocessor = PairPreprocessor(preprocessing_config)
        self.detector = ClassicalPairDetector(detector_config)
        self.device = device
        self.patch_classifier = (
            load_patch_classifier(checkpoint_path, device=device) if checkpoint_path else None
        )

    def predict_paths(self, template_path: Path, test_path: Path) -> tuple[PipelineResult, dict[str, np.ndarray]]:
        """Run inference from image paths."""

        template = load_image(template_path)
        tested = load_image(test_path)
        result, artifacts = self.predict_arrays(template, tested)
        result.template_path = str(template_path)
        result.test_path = str(test_path)
        return result, artifacts

    def predict_arrays(
        self,
        template_rgb: np.ndarray,
        tested_rgb: np.ndarray,
    ) -> tuple[PipelineResult, dict[str, np.ndarray]]:
        """Run inference from already-loaded RGB arrays."""

        preprocessed = self.preprocessor(template_rgb, tested_rgb)
        detections, mask = self.detector.detect(preprocessed)
        if self.patch_classifier is not None and detections:
            detections = self._refine_labels_with_cnn(preprocessed, detections)

        has_defect = bool(detections)
        conclusion = "DEFECT DETECTED" if has_defect else "NO DEFECT DETECTED"
        result = PipelineResult(
            has_defect=has_defect,
            conclusion=conclusion,
            detections=detections,
            image_shape=preprocessed.tested_rgb.shape[:2],
        )
        artifacts = {
            "template_rgb": preprocessed.template_rgb,
            "tested_rgb": preprocessed.tested_rgb,
            "template_gray": preprocessed.template_gray,
            "tested_gray": preprocessed.tested_gray,
            "abs_diff": preprocessed.abs_diff,
            "mask": mask,
        }
        return result, artifacts

    def _refine_labels_with_cnn(
        self,
        preprocessed,
        detections: list[DefectDetection],
    ) -> list[DefectDetection]:
        refined: list[DefectDetection] = []
        for detection in detections:
            label, confidence = classify_patch(
                self.patch_classifier,
                preprocessed.template_gray,
                preprocessed.tested_gray,
                detection.bbox,
                device=self.device,
            )
            refined.append(
                DefectDetection(
                    bbox=detection.bbox,
                    label=label,
                    confidence=max(detection.confidence, confidence),
                    area=detection.area,
                    mean_difference=detection.mean_difference,
                    metadata={**detection.metadata, "classifier": "patch_cnn"},
                )
            )
        return refined
