"""Training loop for the single-image binary PCB classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from pcb_defect_detection.models.single_image_cnn import (
    SINGLE_IMAGE_CLASS_NAMES,
    SingleImageCNN,
)
from pcb_defect_detection.training.single_image_dataset import (
    SingleImageFolderDataset,
    describe_single_image_dataset,
    split_single_image_dataset,
)
from pcb_defect_detection.utils.seed import set_seed


def train_single_image_classifier(
    dataset_root: Path,
    output_path: Path,
    epochs: int = 8,
    batch_size: int = 16,
    image_size: int = 224,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and save a binary normal/defective PCB classifier."""

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SingleImageFolderDataset(dataset_root, image_size=image_size)
    train_dataset, val_dataset, test_dataset = split_single_image_dataset(dataset, seed=seed)
    dataset_summary = describe_single_image_dataset(
        dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SingleImageCNN(num_classes=len(SINGLE_IMAGE_CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1.0
    best_payload: dict[str, Any] = {}
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = _evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            test_loss, test_metrics = _evaluate(model, test_loader, criterion, device)
            best_payload = {
                "validation": {"loss": val_loss, **val_metrics},
                "test": {"loss": test_loss, **test_metrics},
                "dataset": dataset_summary,
                "epochs_trained": epoch,
                "device": device,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": SINGLE_IMAGE_CLASS_NAMES,
                    "image_size": image_size,
                    "metrics": best_payload,
                    "dataset": dataset_summary,
                },
                output_path,
            )

    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_payload


def load_single_image_training_summary(checkpoint_path: Path) -> dict[str, Any]:
    """Load metrics and dataset counts stored in a single-image checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    dataset = checkpoint.get("dataset", metrics.get("dataset", {}))
    return {
        "checkpoint_path": str(checkpoint_path),
        "image_size": int(checkpoint.get("image_size", 224)),
        "metrics": metrics,
        "dataset": dataset,
    }


def _train_one_epoch(
    model: SingleImageCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_samples += inputs.size(0)
    return total_loss / max(1, total_samples)


@torch.no_grad()
def _evaluate(
    model: SingleImageCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * inputs.size(0)
        total_samples += inputs.size(0)
        targets_all.extend(targets.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_all,
        predictions_all,
        average="binary",
        zero_division=0,
    )
    return total_loss / max(1, total_samples), {
        "accuracy": float(accuracy_score(targets_all, predictions_all)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
