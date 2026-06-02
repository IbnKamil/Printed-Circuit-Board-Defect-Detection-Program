from __future__ import annotations

from pathlib import Path

from pcb_defect_detection.data.io import load_image, save_image
from pcb_defect_detection.data.sample import create_synthetic_pair
from pcb_defect_detection.evaluation import evaluate_primary_deeppcb_binary


def test_primary_deeppcb_metrics_use_hidden_template_comparison(tmp_path: Path) -> None:
    dataset_root = tmp_path / "deeppcb"
    for index in range(3):
        source_template, source_tested, _annotation = create_synthetic_pair(
            tmp_path / f"source_{index}",
            seed=index,
        )
        group = dataset_root / f"group{index:05d}" / f"{index:05d}"
        group.mkdir(parents=True)
        save_image(group / f"{index:05d}_temp.png", load_image(source_template))
        save_image(group / f"{index:05d}_test.png", load_image(source_tested))

    metrics = evaluate_primary_deeppcb_binary(
        dataset_root=dataset_root,
        predictions_path=tmp_path / "predictions.csv",
        confidence_threshold=0.1,
    )

    assert metrics["validation"]["accuracy"] >= 0.5
    assert metrics["test"]["accuracy"] >= 0.5
    assert (tmp_path / "predictions.csv").exists()
    assert metrics["dataset"]["total"] == 6
