"""Training loop for the optional DeepPCB patch classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from pcb_defect_detection.config import DEEPPCB_CLASS_NAMES
from pcb_defect_detection.models.patch_cnn import PatchCNN
from pcb_defect_detection.training.dataset import DeepPCBPatchDataset
from pcb_defect_detection.utils.schemas import ImagePair
from pcb_defect_detection.utils.seed import set_seed
from pcb_defect_detection.visualization.draw import save_training_curves


def train_patch_classifier(
    train_pairs: list[ImagePair],
    val_pairs: list[ImagePair],
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    image_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, float]:
    """Train PatchCNN and save the best validation checkpoint."""

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DeepPCBPatchDataset(train_pairs, image_size=image_size)
    val_dataset = DeepPCBPatchDataset(val_pairs, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PatchCNN(num_classes=len(DEEPPCB_CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = _evaluate_classifier(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = {"val_loss": val_loss, **val_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": DEEPPCB_CLASS_NAMES,
                    "image_size": image_size,
                    "epoch": epoch,
                    "metrics": best_metrics,
                },
                output_dir / "best_model.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    save_training_curves(history, output_dir / "training_curves.png")
    return best_metrics


def _train_one_epoch(
    model: PatchCNN,
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
def _evaluate_classifier(
    model: PatchCNN,
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
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(targets_all, predictions_all)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return total_loss / max(1, total_samples), metrics
