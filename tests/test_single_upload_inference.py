from __future__ import annotations

from pathlib import Path

from pcb_defect_detection.data.io import load_image, save_image
from pcb_defect_detection.data.sample import create_synthetic_pair
from pcb_defect_detection.inference.single_image import (
    find_matching_template,
    predict_uploaded_pcb_image,
)


def test_single_upload_marks_deeppcb_template_as_normal(tmp_path: Path) -> None:
    template_path, _test_path = _create_deeppcb_named_pair(tmp_path)
    template = load_image(template_path)

    result = predict_uploaded_pcb_image(
        image_rgb=template,
        filename=template_path.name,
        dataset_root=tmp_path,
    )

    assert result.is_defective is False
    assert result.confidence == 1.0
    assert result.method == "deeppcb_template_filename"


def test_single_upload_uses_hidden_template_for_deeppcb_test(tmp_path: Path) -> None:
    template_path, test_path = _create_deeppcb_named_pair(tmp_path)
    tested = load_image(test_path)

    matched_template = find_matching_template(test_path.name, tmp_path)
    result = predict_uploaded_pcb_image(
        image_rgb=tested,
        filename=test_path.name,
        dataset_root=tmp_path,
        confidence_threshold=0.1,
    )

    assert matched_template == template_path
    assert result.is_defective is True
    assert result.detections
    assert result.method == "deeppcb_hidden_template_comparison"


def test_template_lookup_ignores_windows_path_and_extra_prefix(tmp_path: Path) -> None:
    template_path, _test_path = _create_deeppcb_named_pair(tmp_path, image_id="20085007")

    matched_template = find_matching_template(
        r"C:\fakepath\D20085007_test.jpg",
        tmp_path,
    )

    assert matched_template == template_path


def _create_deeppcb_named_pair(root: Path, image_id: str = "00041000") -> tuple[Path, Path]:
    source_template, source_tested, _annotation = create_synthetic_pair(root / "source")
    group = root / "group00041" / "00041"
    group.mkdir(parents=True)
    template_path = group / f"{image_id}_temp.png"
    test_path = group / f"{image_id}_test.png"
    save_image(template_path, load_image(source_template))
    save_image(test_path, load_image(source_tested))
    return template_path, test_path
