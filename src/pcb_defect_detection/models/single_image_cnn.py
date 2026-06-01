"""Binary CNN for single-image PCB defect classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from pcb_defect_detection.utils.errors import ModelLoadError

SINGLE_IMAGE_CLASS_NAMES = {0: "normal", 1: "defective"}


class SingleImageCNN(nn.Module):
    """Small CNN that predicts whether one PCB image is normal or defective."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def preprocess_single_image(image_rgb: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Convert an RGB image into a normalized model tensor."""

    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to RGB NumPy array."""

    return np.asarray(image.convert("RGB"))


def load_single_image_classifier(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[SingleImageCNN, int]:
    """Load a trained single-image classifier and return model, image_size."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ModelLoadError(f"Single-image model checkpoint does not exist: {checkpoint_path}")

    model = SingleImageCNN(num_classes=len(SINGLE_IMAGE_CLASS_NAMES))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        image_size = int(checkpoint.get("image_size", 224))
    except Exception as exc:  # noqa: BLE001 - convert low-level torch errors.
        raise ModelLoadError(f"Could not load single-image checkpoint: {checkpoint_path}") from exc
    model.to(device)
    model.eval()
    return model, image_size


@torch.no_grad()
def predict_single_image(
    image_rgb: np.ndarray,
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> dict[str, float | str | bool]:
    """Predict whether a single PCB image is defective."""

    model, image_size = load_single_image_classifier(checkpoint_path, device=device)
    tensor = preprocess_single_image(image_rgb, image_size=image_size)
    logits = model(tensor.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    defective_probability = float(probabilities[1].item())
    class_id = int(torch.argmax(probabilities).item())
    label = SINGLE_IMAGE_CLASS_NAMES[class_id]
    confidence = float(probabilities[class_id].item())
    return {
        "label": label,
        "is_defective": label == "defective",
        "confidence": confidence,
        "defective_probability": defective_probability,
    }
