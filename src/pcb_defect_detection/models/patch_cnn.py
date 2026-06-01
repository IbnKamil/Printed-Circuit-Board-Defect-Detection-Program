"""Compact CNN for classifying defect patches from template/tested pairs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from pcb_defect_detection.config import DEEPPCB_CLASS_NAMES
from pcb_defect_detection.utils.errors import ModelLoadError
from pcb_defect_detection.utils.schemas import BoundingBox


class PatchCNN(nn.Module):
    """Small 2-channel CNN for DeepPCB defect type classification."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_patch_tensor(
    template_gray: np.ndarray,
    tested_gray: np.ndarray,
    bbox: BoundingBox,
    image_size: int = 64,
    margin: int = 8,
) -> torch.Tensor:
    """Crop an aligned pair patch and convert it to a model tensor."""

    height, width = template_gray.shape[:2]
    crop_box = bbox.expand(margin=margin, width=width, height=height)
    x_min, y_min, x_max, y_max = crop_box.to_xyxy()
    template_crop = template_gray[y_min:y_max, x_min:x_max]
    tested_crop = tested_gray[y_min:y_max, x_min:x_max]
    template_crop = cv2.resize(template_crop, (image_size, image_size))
    tested_crop = cv2.resize(tested_crop, (image_size, image_size))
    stacked = np.stack([template_crop, tested_crop], axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def load_patch_classifier(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> PatchCNN:
    """Load a PatchCNN checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ModelLoadError(f"Model checkpoint does not exist: {checkpoint_path}")

    model = PatchCNN(num_classes=len(DEEPPCB_CLASS_NAMES))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 - convert low-level torch errors.
        raise ModelLoadError(f"Could not load model checkpoint: {checkpoint_path}") from exc
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def classify_patch(
    model: PatchCNN,
    template_gray: np.ndarray,
    tested_gray: np.ndarray,
    bbox: BoundingBox,
    device: torch.device | str = "cpu",
    image_size: int = 64,
) -> tuple[str, float]:
    """Predict a DeepPCB class for one localized defect bbox."""

    tensor = make_patch_tensor(template_gray, tested_gray, bbox, image_size=image_size)
    logits = model(tensor.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    confidence, class_id = torch.max(probabilities, dim=0)
    return DEEPPCB_CLASS_NAMES[int(class_id.item())], float(confidence.item())
