"""Single-upload inference with DeepPCB-aware template lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcb_defect_detection.data.io import load_image
from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.utils.errors import InputValidationError
from pcb_defect_detection.utils.schemas import DefectDetection
from pcb_defect_detection.visualization import draw_detections


@dataclass
class SingleUploadResult:
    """Result for the simplified one-image Streamlit workflow."""

    label: str
    is_defective: bool
    confidence: float
    method: str
    message: str
    overlay_rgb: np.ndarray
    detections: list[DefectDetection]


def predict_uploaded_pcb_image(
    image_rgb: np.ndarray,
    filename: str,
    dataset_root: Path,
    confidence_threshold: float = 0.80,
) -> SingleUploadResult:
    """Classify one uploaded PCB image using DeepPCB-aware logic.

    The UI accepts one file, but DeepPCB itself is reference-based. Therefore:
    - ``*_temp`` is treated as the known defect-free template and returned normal;
    - ``*_test`` is compared with the matching hidden ``*_temp`` from dataset_root.

    Other filenames are rejected instead of using a weak whole-image CNN fallback.
    """

    stem = _normalized_stem(filename)
    if _is_template_name(stem):
        return SingleUploadResult(
            label="normal",
            is_defective=False,
            confidence=1.0,
            method="deeppcb_template_filename",
            message="Файл похож на DeepPCB template (*_temp): эталонная плата считается нормальной.",
            overlay_rgb=image_rgb,
            detections=[],
        )

    if _is_test_name(stem):
        template_path = find_matching_template(filename, dataset_root)
        template_rgb = load_image(template_path)
        detector_config = ClassicalDetectorConfig(confidence_threshold=confidence_threshold)
        pipeline = PCBDefectPipeline(detector_config=detector_config)
        result, _artifacts = pipeline.predict_arrays(template_rgb, image_rgb)
        overlay = draw_detections(image_rgb, result.detections)
        confidence = max((detection.confidence for detection in result.detections), default=0.0)
        return SingleUploadResult(
            label="defective" if result.has_defect else "normal",
            is_defective=result.has_defect,
            confidence=confidence if result.has_defect else 1.0,
            method="deeppcb_hidden_template_comparison",
            message=f"Найден соответствующий template: {template_path.name}. Выполнено reference-based сравнение.",
            overlay_rgb=overlay,
            detections=result.detections,
        )

    raise InputValidationError(
        "Файл не похож на DeepPCB-изображение. "
        "Загрузите изображение с исходным именем вида *_test.jpg или *_temp.jpg."
    )


def find_matching_template(filename: str, dataset_root: Path) -> Path:
    """Find the matching DeepPCB template path for an uploaded *_test file."""

    template_stem = _to_template_stem(filename)
    uploaded_suffix = Path(_normalized_name(filename)).suffix
    candidate_patterns: list[str] = []
    if uploaded_suffix:
        candidate_patterns.append(template_stem + uploaded_suffix)
    candidate_patterns.append(template_stem + ".*")

    for pattern in candidate_patterns:
        matches = sorted(Path(dataset_root).rglob(pattern))
        if matches:
            return matches[0]

    raise InputValidationError(
        f"Не найден template '{template_stem}.*' внутри {dataset_root}. "
        "Убедитесь, что датасет DeepPCB добавлен в data/deeppcb/PCBData."
    )


def _is_template_name(stem: str) -> bool:
    return stem.endswith("_temp") or stem.endswith("_template")


def _is_test_name(stem: str) -> bool:
    return stem.endswith("_test") or stem.endswith("_tested")


def _to_template_stem(filename: str) -> str:
    stem = _normalized_stem(filename)
    image_id = _extract_deeppcb_id(stem)
    if image_id is not None:
        return f"{image_id}_temp"

    if stem.endswith("_tested"):
        return stem[: -len("_tested")] + "_temp"
    if stem.endswith("_test"):
        return stem[: -len("_test")] + "_temp"
    raise InputValidationError(f"Expected DeepPCB *_test filename, got: {filename}")


def _normalized_name(filename: str) -> str:
    return str(filename).replace("\\", "/").split("/")[-1]


def _normalized_stem(filename: str) -> str:
    return Path(_normalized_name(filename)).stem.lower()


def _extract_deeppcb_id(stem: str) -> str | None:
    match = re.search(r"(\d+)(?:_test|_tested|_temp|_template)$", stem.lower())
    return match.group(1) if match else None
