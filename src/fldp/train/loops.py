"""Baseline PyTorch training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainMetrics:
    """Aggregate metrics from one local training pass."""

    loss: float
    num_examples: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate metrics from centralized evaluation."""

    loss: float
    accuracy: float
    num_examples: int


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    *,
    device: str = "cpu",
) -> TrainMetrics:
    """Train a model for one epoch with cross-entropy loss."""

    torch, nn = _require_torch()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.to(device)
    model.train()

    total_loss = 0.0
    total_examples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu().item())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("dataloader produced no examples")

    return TrainMetrics(loss=total_loss / total_examples, num_examples=total_examples)


def evaluate(model: Any, dataloader: Any, *, device: str = "cpu") -> EvaluationMetrics:
    """Evaluate a classifier with cross-entropy loss and accuracy."""

    torch, nn = _require_torch()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)

            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == targets).sum().cpu().item())
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().cpu().item())
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("dataloader produced no examples")

    return EvaluationMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        num_examples=total_examples,
    )


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        raise RuntimeError("PyTorch is required for training loops") from exc
    return torch, nn
