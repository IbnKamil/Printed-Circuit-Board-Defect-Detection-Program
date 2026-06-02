"""CLI for training the optional DeepPCB patch classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.data.deeppcb import load_manifest
from pcb_defect_detection.training import train_patch_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCNN on DeepPCB annotations.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_pairs = load_manifest(args.manifest, split="train")
    val_pairs = load_manifest(args.manifest, split="val")
    metrics = train_patch_classifier(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
