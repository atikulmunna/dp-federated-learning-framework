"""Flower NumPyClient adapters for PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fldp.data import make_dataloader
from fldp.train import evaluate, get_model_parameters, set_model_parameters, train_one_epoch

try:
    from flwr.client import NumPyClient
except Exception as exc:  # pragma: no cover - depends on optional Flower install.
    raise RuntimeError("Flower is required for TorchFlowerClient") from exc


ModelFactory = Callable[[], Any]


@dataclass(frozen=True)
class FlowerClientConfig:
    """Local training settings for a Flower client."""

    batch_size: int
    local_epochs: int
    learning_rate: float
    device: str = "cpu"
    seed: int = 0


class TorchFlowerClient(NumPyClient):
    """Flower NumPyClient backed by the project's PyTorch train/eval loops."""

    def __init__(
        self,
        *,
        client_id: int,
        model_factory: ModelFactory,
        train_dataset: Any,
        eval_dataset: Any,
        config: FlowerClientConfig,
    ) -> None:
        self.client_id = client_id
        self.model_factory = model_factory
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.model = model_factory()

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        return list(get_model_parameters(self.model))

    def fit(self, parameters: list[Any], config: dict[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
        torch = _require_torch()
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        batch_size = int(config.get("batch_size", self.config.batch_size))
        server_round = int(config.get("server_round", 0))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        train_metrics = None
        for epoch in range(local_epochs):
            train_loader = make_dataloader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                seed=self.config.seed + server_round * 10_000 + self.client_id * 100 + epoch,
            )
            train_metrics = train_one_epoch(
                self.model,
                train_loader,
                optimizer,
                device=self.config.device,
            )

        if train_metrics is None:  # pragma: no cover - config validation prevents this in normal use.
            raise ValueError("local_epochs must be positive")

        return (
            list(get_model_parameters(self.model)),
            len(self.train_dataset),
            {
                "client_id": str(self.client_id),
                "train_loss": train_metrics.loss,
            },
        )

    def evaluate(self, parameters: list[Any], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        set_model_parameters(self.model, parameters)
        batch_size = int(config.get("batch_size", self.config.batch_size))
        eval_loader = make_dataloader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        metrics = evaluate(self.model, eval_loader, device=self.config.device)
        return (
            metrics.loss,
            metrics.num_examples,
            {
                "accuracy": metrics.accuracy,
                "client_id": str(self.client_id),
            },
        )

def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local ML runtime.
        raise RuntimeError("PyTorch is required for TorchFlowerClient") from exc
    return torch
