"""CLI inference for one DeepPCB pair or a CSV manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.data.deeppcb import load_manifest
from pcb_defect_detection.data.io import save_image
from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.utils.reporting import save_csv_report, save_json_report
from pcb_defect_detection.visualization import (
    difference_heatmap,
    draw_detections,
    mask_to_rgb,
    side_by_side,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PCB defects from template/tested pairs.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="CSV manifest for batch inference.")
    input_group.add_argument("--template", type=Path, help="Template image path.")
    parser.add_argument("--test", type=Path, help="Tested PCB image path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional PatchCNN checkpoint.")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--diff-threshold", type=int, default=22)
    parser.add_argument("--save-intermediate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_config = ClassicalDetectorConfig(
        confidence_threshold=args.confidence_threshold,
        diff_threshold=args.diff_threshold,
    )
    pipeline = PCBDefectPipeline(
        detector_config=detector_config,
        checkpoint_path=args.checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        pairs = load_manifest(args.manifest)
        summary = []
        for index, pair in enumerate(pairs):
            pair_output_dir = args.output_dir / f"pair_{index:04d}"
            result = _run_one_pair(
                pipeline,
                pair.template_path,
                pair.test_path,
                pair_output_dir,
                args.save_intermediate,
            )
            summary.append(result.to_dict())
        (args.output_dir / "batch_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Processed {len(summary)} pairs. Results saved to {args.output_dir}")
    else:
        if args.test is None:
            raise SystemExit("--test is required when --template is used.")
        result = _run_one_pair(
            pipeline,
            args.template,
            args.test,
            args.output_dir,
            args.save_intermediate,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def _run_one_pair(
    pipeline: PCBDefectPipeline,
    template_path: Path,
    test_path: Path,
    output_dir: Path,
    save_intermediate: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    result, artifacts = pipeline.predict_paths(template_path, test_path)
    overlay = draw_detections(artifacts["tested_rgb"], result.detections)
    heatmap = difference_heatmap(artifacts["abs_diff"])
    comparison = side_by_side(artifacts["template_rgb"], artifacts["tested_rgb"], overlay)

    save_json_report(result, output_dir / "report.json")
    save_csv_report(result, output_dir / "report.csv")
    save_image(output_dir / "overlay.png", overlay)
    save_image(output_dir / "heatmap.png", heatmap)
    save_image(output_dir / "comparison.png", comparison)
    if save_intermediate:
        save_image(output_dir / "mask.png", mask_to_rgb(artifacts["mask"]))
        save_image(output_dir / "template_gray.png", artifacts["template_gray"])
        save_image(output_dir / "tested_gray.png", artifacts["tested_gray"])
    return result


if __name__ == "__main__":
    main()
