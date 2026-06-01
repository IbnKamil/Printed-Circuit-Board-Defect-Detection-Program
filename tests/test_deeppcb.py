from __future__ import annotations

from pathlib import Path

import pytest

from pcb_defect_detection.data.deeppcb import load_manifest, parse_annotation_file
from pcb_defect_detection.data.sample import create_synthetic_pair
from pcb_defect_detection.utils.errors import DatasetError


def test_parse_annotation_file_supports_deeppcb_labels(tmp_path: Path) -> None:
    _template_path, _tested_path, annotation_path = create_synthetic_pair(tmp_path)

    annotations = parse_annotation_file(annotation_path)

    assert len(annotations) == 4
    assert {annotation.label for annotation in annotations} >= {"open", "short"}


def test_load_manifest_rejects_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("template_path,test_path,annotation_path,split\n", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_manifest(manifest)


def test_load_manifest_filters_split(tmp_path: Path) -> None:
    template_path, tested_path, annotation_path = create_synthetic_pair(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "template_path,test_path,annotation_path,split\n"
        f"{template_path.name},{tested_path.name},{annotation_path.name},train\n",
        encoding="utf-8",
    )

    pairs = load_manifest(manifest, split="train")

    assert len(pairs) == 1
    assert pairs[0].template_path.exists()
