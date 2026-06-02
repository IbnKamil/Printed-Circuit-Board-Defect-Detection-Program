"""Generate a synthetic DeepPCB-like pair for demos and tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.data.sample import create_synthetic_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic PCB pair.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/sample"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path, tested_path, annotation_path = create_synthetic_pair(
        args.output_dir,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"Template:    {template_path}")
    print(f"Tested:      {tested_path}")
    print(f"Annotation:  {annotation_path}")


if __name__ == "__main__":
    main()
