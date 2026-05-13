"""PyTorch FedAvg and DP-FedAvg smoke experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from fldp.artifacts import RunPaths, append_round_metrics, write_accountant_trace
from fldp.data import (
    IndexedSubset,
    iid_partition,
    load_vision_dataset,
    make_client_datasets,
    make_dataloader,
    make_train_validation_split,
)
from fldp.experiments.config import TorchDPFedAvgSmokeConfig, TorchFedAvgSmokeConfig
from fldp.models import create_mnist_cnn
from fldp.privacy import PrivacyAccountant
from fldp.strategies import ClientUpdate, aggregate_dpfedavg, aggregate_fedavg, sample_cohort
from fldp.strategies.parameters import l2_norm
from fldp.train import evaluate, get_model_parameters, set_model_parameters, train_one_epoch


def run_torch_fedavg_smoke(
    config: TorchFedAvgSmokeConfig,
    *,
    paths: RunPaths,
    seed: int,
) -> dict[str, object]:
    """Run a small PyTorch FedAvg smoke experiment."""

    torch = _require_torch()
    rng = np.random.default_rng(seed)
    client_datasets, validation_loader = _prepare_torch_data(config, seed=seed)
    global_model = _create_torch_smoke_model(config)
    server_parameters = list(get_model_parameters(global_model))
    final_metrics: dict[str, float] = {}

    for round_index in range(1, config.num_rounds + 1):
        cohort = sample_cohort(
            num_clients=config.num_clients,
            cohort_size=config.cohort_size,
            rng=rng,
        )
        updates = _train_client_updates(
            config=config,
            client_datasets=client_datasets,
            cohort=cohort,
            server_parameters=server_parameters,
            round_index=round_index,
            seed=seed,
            torch_module=torch,
        )
        result = aggregate_fedavg(
            server_parameters,
            updates,
            weighted_by_examples=config.weighted_by_examples,
        )
        server_parameters = result.parameters
        set_model_parameters(global_model, server_parameters)
        validation_metrics = evaluate(global_model, validation_loader, device=config.device)
        final_metrics = {
            "train_loss": result.metrics["train_loss"],
            "validation_loss": validation_metrics.loss,
            "validation_accuracy": validation_metrics.accuracy,
            "average_delta_l2": l2_norm(result.average_delta),
            "parameter_l2": l2_norm(server_parameters),
        }

        append_round_metrics(
            paths,
            {
                "round": round_index,
                "cohort": list(cohort),
                "cohort_size": result.cohort_size,
                "num_examples": result.num_examples,
                **final_metrics,
            },
        )

    return {
        "mode": "torch_fedavg_smoke",
        "status": "completed",
        "dataset": config.dataset,
        "model": config.model,
        "rounds_completed": config.num_rounds,
        "num_clients": config.num_clients,
        "cohort_size": config.cohort_size,
        "final_metrics": final_metrics,
    }


def run_torch_dpfedavg_smoke(
    config: TorchDPFedAvgSmokeConfig,
    *,
    paths: RunPaths,
    seed: int,
) -> dict[str, object]:
    """Run a small PyTorch DP-FedAvg smoke experiment."""

    torch = _require_torch()
    base = config.base
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)
    accountant = PrivacyAccountant()
    client_datasets, validation_loader = _prepare_torch_data(base, seed=seed)
    global_model = _create_torch_smoke_model(base)
    server_parameters = list(get_model_parameters(global_model))
    final_metrics: dict[str, float] = {}

    for round_index in range(1, base.num_rounds + 1):
        cohort = sample_cohort(
            num_clients=base.num_clients,
            cohort_size=base.cohort_size,
            rng=rng,
        )
        updates = _train_client_updates(
            config=base,
            client_datasets=client_datasets,
            cohort=cohort,
            server_parameters=server_parameters,
            round_index=round_index,
            seed=seed,
            torch_module=torch,
        )
        result = aggregate_dpfedavg(
            server_parameters,
            updates,
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier,
            accountant=accountant,
            num_total_clients=base.num_clients,
            noise_rng=noise_rng,
            delta=config.delta,
        )
        server_parameters = result.parameters
        set_model_parameters(global_model, server_parameters)
        validation_metrics = evaluate(global_model, validation_loader, device=base.device)
        final_metrics = {
            "train_loss": _mean_update_metric(updates, "train_loss"),
            "validation_loss": validation_metrics.loss,
            "validation_accuracy": validation_metrics.accuracy,
            "epsilon_cumulative": result.epsilon if result.epsilon is not None else float("nan"),
            "sample_rate": result.sample_rate,
            "clip_norm": config.clip_norm,
            "noise_multiplier": config.noise_multiplier,
            "noise_std": result.noise_std,
            "pre_clip_l2_mean": float(np.mean(result.pre_clip_norms)),
            "pre_clip_l2_max": float(np.max(result.pre_clip_norms)),
            "clip_scale_min": float(np.min(result.clip_scales)),
            "average_clipped_delta_l2": l2_norm(result.average_clipped_delta),
            "noised_delta_l2": l2_norm(result.noised_delta),
            "parameter_l2": l2_norm(server_parameters),
        }

        append_round_metrics(
            paths,
            {
                "round": round_index,
                "cohort": list(cohort),
                "cohort_size": result.cohort_size,
                "num_examples": sum(update.num_examples for update in updates),
                **final_metrics,
            },
        )

    write_accountant_trace(
        paths,
        accountant,
        delta=config.delta,
        privacy_unit=config.privacy_unit,
    )

    return {
        "mode": "torch_dpfedavg_smoke",
        "status": "completed",
        "dataset": base.dataset,
        "model": base.model,
        "rounds_completed": base.num_rounds,
        "num_clients": base.num_clients,
        "cohort_size": base.cohort_size,
        "privacy_unit": config.privacy_unit,
        "delta": config.delta,
        "epsilon_cumulative_final": accountant.get_epsilon(delta=config.delta),
        "clip_norm": config.clip_norm,
        "noise_multiplier": config.noise_multiplier,
        "final_metrics": final_metrics,
    }


def _prepare_torch_data(config: TorchFedAvgSmokeConfig, *, seed: int) -> tuple[Any, Any]:
    dataset = _load_torch_smoke_dataset(config, seed=seed)
    if config.max_samples is not None:
        dataset = IndexedSubset(dataset, np.arange(min(config.max_samples, len(dataset))))

    split = make_train_validation_split(
        len(dataset),
        validation_fraction=config.validation_fraction,
        seed=seed,
    )
    partition = iid_partition(
        len(split.train_indices),
        config.num_clients,
        seed=seed,
        min_client_size=config.client_min_size,
    )
    client_datasets = make_client_datasets(dataset, split, partition)
    validation_loader = make_dataloader(
        client_datasets.validation,
        batch_size=config.batch_size,
        shuffle=False,
    )
    return client_datasets, validation_loader


def _train_client_updates(
    *,
    config: TorchFedAvgSmokeConfig,
    client_datasets: Any,
    cohort: tuple[int, ...],
    server_parameters: list[np.ndarray],
    round_index: int,
    seed: int,
    torch_module: Any,
) -> list[ClientUpdate]:
    updates = []
    for client_id in cohort:
        client_model = _create_torch_smoke_model(config)
        set_model_parameters(client_model, server_parameters)
        optimizer = torch_module.optim.SGD(client_model.parameters(), lr=config.learning_rate)
        train_metrics = None
        for _ in range(config.local_epochs):
            train_loader = make_dataloader(
                client_datasets.clients[client_id],
                batch_size=config.batch_size,
                shuffle=True,
                seed=seed + round_index * 10_000 + client_id,
            )
            train_metrics = train_one_epoch(client_model, train_loader, optimizer, device=config.device)
        if train_metrics is None:  # pragma: no cover - local_epochs validation prevents this.
            raise RuntimeError("local training did not run")

        updates.append(
            ClientUpdate(
                client_id=client_id,
                initial_parameters=tuple(array.copy() for array in server_parameters),
                updated_parameters=get_model_parameters(client_model),
                num_examples=len(client_datasets.clients[client_id]),
                metrics={"train_loss": train_metrics.loss},
            )
        )
    return updates


def _load_torch_smoke_dataset(config: TorchFedAvgSmokeConfig, *, seed: int) -> Any:
    if config.dataset == "mnist":
        return load_vision_dataset("mnist", train=True, download=config.download)

    torch = _require_torch()
    from torch.utils.data import TensorDataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    features = torch.randn(
        config.synthetic_num_samples,
        1,
        28,
        28,
        generator=generator,
    )
    labels = torch.arange(config.synthetic_num_samples) % config.synthetic_num_classes
    return TensorDataset(features, labels.long())


def _create_torch_smoke_model(config: TorchFedAvgSmokeConfig) -> Any:
    if config.model == "mnist_cnn":
        return create_mnist_cnn(num_classes=config.synthetic_num_classes if config.dataset == "synthetic" else 10)
    raise ValueError(f"unsupported model: {config.model}")


def _mean_update_metric(updates: list[ClientUpdate], metric_name: str) -> float:
    values = [
        float(update.metrics[metric_name])
        for update in updates
        if update.metrics and metric_name in update.metrics
    ]
    if not values:
        raise ValueError(f"metric not found on updates: {metric_name}")
    return sum(values) / len(values)


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        raise RuntimeError("PyTorch is required for torch smoke experiments") from exc
    return torch
