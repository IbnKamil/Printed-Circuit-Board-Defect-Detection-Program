"""Single-upload inference with DeepPCB-aware template lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcb_defect_detection.data.io import load_image
from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.models.single_image_cnn import predict_single_image
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
    checkpoint_path: Path | None = None,
    confidence_threshold: float = 0.80,
) -> SingleUploadResult:
    """Classify one uploaded PCB image.

    For DeepPCB filenames, the function keeps the UI single-image while using the
    correct reference-based formulation internally:
    - ``*_temp`` is a known defect-free template and is returned as normal;
    - ``*_test`` is compared against the matching ``*_temp`` found in dataset_root.

    If the filename is not DeepPCB-like, an optional trained single-image CNN is
    used as a fallback.
    """

    stem = Path(filename).stem.lower()
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

    if checkpoint_path and Path(checkpoint_path).exists():
        prediction = predict_single_image(image_rgb, checkpoint_path)
        return SingleUploadResult(
            label=str(prediction["label"]),
            is_defective=bool(prediction["is_defective"]),
            confidence=float(prediction["confidence"]),
            method="single_image_cnn_fallback",
            message=(
                "Файл не похож на DeepPCB *_temp/*_test. Использован fallback CNN; "
                "для высокой точности лучше использовать изображения DeepPCB с исходными именами."
            ),
            overlay_rgb=image_rgb,
            detections=[],
        )

    raise InputValidationError(
        "Не удалось определить paired template для этого файла. "
        "Для надежной проверки загрузите DeepPCB-файл с именем вида *_test.jpg "
        "или *_temp.jpg. Для произвольных фото нужна отдельно обученная single-image CNN."
    )


def find_matching_template(filename: str, dataset_root: Path) -> Path:
    """Find the matching DeepPCB template path for an uploaded *_test file."""

    uploaded_name = Path(filename).name
    template_name = _to_template_name(uploaded_name)
    matches = sorted(Path(dataset_root).rglob(template_name))
    if not matches:
        raise InputValidationError(
            f"Не найден template '{template_name}' внутри {dataset_root}. "
            "Убедитесь, что датасет DeepPCB добавлен в data/deeppcb/PCBData."
        )
    return matches[0]


def _is_template_name(stem: str) -> bool:
    return stem.endswith("_temp") or stem.endswith("_template")


def _is_test_name(stem: str) -> bool:
    return stem.endswith("_test") or stem.endswith("_tested")


def _to_template_name(filename: str) -> str:
    path = Path(filename)
    stem = path.stem
    lowered = stem.lower()
    if lowered.endswith("_tested"):
        template_stem = stem[: -len("_tested")] + "_temp"
    elif lowered.endswith("_test"):
        template_stem = stem[: -len("_test")] + "_temp"
    else:
        raise InputValidationError(f"Expected DeepPCB *_test filename, got: {filename}")
    return template_stem + path.suffix
