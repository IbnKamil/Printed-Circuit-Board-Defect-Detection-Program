"""CLI for evaluating pair-based detector on annotated DeepPCB pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.data.deeppcb import discover_pairs, load_manifest
from pcb_defect_detection.training import evaluate_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCB defect detector.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest:
        pairs = load_manifest(args.manifest, split=args.split)
    else:
        pairs = discover_pairs(args.dataset_root, split=args.split)
    metrics = evaluate_detector(
        pairs=pairs,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
