"""Visualization helpers for defect detection outputs."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pcb_defect_detection.utils.schemas import DefectDetection

_LABEL_COLORS = {
    "open": (239, 83, 80),
    "short": (66, 165, 245),
    "mousebite": (255, 202, 40),
    "spur": (171, 71, 188),
    "pin_hole": (102, 187, 106),
    "spurious_copper": (255, 112, 67),
}


def draw_detections(
    image_rgb: np.ndarray,
    detections: list[DefectDetection],
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on an RGB image."""

    canvas = image_rgb.copy()
    for detection in detections:
        color = _LABEL_COLORS.get(detection.label, (255, 0, 0))
        x_min, y_min, x_max, y_max = detection.bbox.to_xyxy()
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, thickness)
        text = f"{detection.label} {detection.confidence:.2f}"
        y_text = max(15, y_min - 6)
        cv2.rectangle(canvas, (x_min, y_text - 14), (x_min + 190, y_text + 4), color, -1)
        cv2.putText(
            canvas,
            text,
            (x_min + 3, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to RGB for saving/display."""

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def difference_heatmap(abs_diff: np.ndarray) -> np.ndarray:
    """Create a colorful heatmap from absolute image difference."""

    normalized = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX)
    colored_bgr = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def side_by_side(
    template_rgb: np.ndarray,
    tested_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
) -> np.ndarray:
    """Combine template, tested and overlay panels into one image."""

    return np.concatenate([template_rgb, tested_rgb, overlay_rgb], axis=1)


def save_training_curves(history: list[dict[str, float]], output_path) -> None:
    """Save loss/F1 curves from the training loop."""

    if not history:
        return
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    plt.plot(epochs, [row["val_f1"] for row in history], label="val_macro_f1")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
