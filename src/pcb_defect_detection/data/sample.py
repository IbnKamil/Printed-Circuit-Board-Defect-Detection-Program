"""Synthetic sample generation for demos and tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pcb_defect_detection.data.io import save_image


def create_synthetic_pair(
    output_dir: Path,
    width: int = 512,
    height: int = 384,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Create a simple PCB-like template/tested pair with several defects.

    The generated data is synthetic and intended only for smoke tests and demos.
    """

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = np.full((height, width, 3), (24, 92, 48), dtype=np.uint8)

    # Copper traces and pads.
    copper = (206, 166, 54)
    for y in range(70, height - 50, 70):
        cv2.line(template, (40, y), (width - 40, y), copper, 10)
    for x in range(70, width - 40, 85):
        cv2.line(template, (x, 35), (x, height - 35), copper, 8)
    for x in range(70, width - 40, 85):
        for y in range(70, height - 50, 70):
            cv2.circle(template, (x, y), 18, copper, -1)
            cv2.circle(template, (x, y), 8, (18, 70, 38), -1)

    noise = rng.normal(0, 3, template.shape).astype(np.int16)
    template = np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    tested = template.copy()

    annotations = []

    # Open circuit: remove a trace section.
    cv2.rectangle(tested, (158, 136), (205, 153), (24, 92, 48), -1)
    annotations.append((158, 136, 205, 153, 1))

    # Short: add copper bridge.
    cv2.rectangle(tested, (320, 67), (340, 142), copper, -1)
    annotations.append((320, 67, 340, 142, 2))

    # Mousebite: small missing bites at pad edge.
    for center in [(250, 140), (259, 151), (242, 153)]:
        cv2.circle(tested, center, 8, (24, 92, 48), -1)
    annotations.append((232, 131, 268, 162, 3))

    # Spurious copper: isolated blob.
    cv2.ellipse(tested, (430, 260), (24, 12), 20, 0, 360, copper, -1)
    annotations.append((404, 244, 456, 276, 6))

    template_path = output_dir / "template.png"
    tested_path = output_dir / "tested.png"
    annotation_path = output_dir / "annotations.txt"
    save_image(template_path, template)
    save_image(tested_path, tested)
    annotation_path.write_text(
        "\n".join(" ".join(map(str, row)) for row in annotations) + "\n",
        encoding="utf-8",
    )
    return template_path, tested_path, annotation_path
