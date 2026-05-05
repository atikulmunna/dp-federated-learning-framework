"""Small CNN model factories for MNIST and CIFAR-10."""

from __future__ import annotations

from typing import Any


def create_mnist_cnn(*, num_classes: int = 10) -> Any:
    """Create the small MNIST CNN specified for smoke tests."""

    torch, nn = _require_torch()

    class MnistCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return MnistCNN()


def create_cifar10_cnn(*, num_classes: int = 10) -> Any:
    """Create a compact CIFAR-10 CNN without BatchNorm."""

    torch, nn = _require_torch()

    class Cifar10CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 96, kernel_size=3, padding=1),
                nn.GroupNorm(8, 96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return Cifar10CNN()


def count_trainable_parameters(model: Any) -> int:
    """Count trainable parameters for a torch module-like object."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        raise RuntimeError("PyTorch is required to create models") from exc
    return torch, nn
