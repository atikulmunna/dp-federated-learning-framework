"""Runnable experiment driver skeleton."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fldp.artifacts import (
    RunPaths,
    append_round_metrics,
    collect_environment_metadata,
    create_run_dir,
    write_config,
    write_metadata,
    write_summary,
)
from fldp.data import (
    IndexedSubset,
    iid_partition,
    load_vision_dataset,
    make_client_datasets,
    make_dataloader,
    make_train_validation_split,
)
from fldp.models import create_mnist_cnn
from fldp.strategies import ClientUpdate, aggregate_fedavg, sample_cohort
from fldp.strategies.parameters import l2_norm
from fldp.train import evaluate, get_model_parameters, seed_everything, set_model_parameters, train_one_epoch


@dataclass(frozen=True)
class RunConfig:
    """Run-level configuration shared by all experiment modes."""

    output_dir: Path
    seed: int
    id: str | None = None


@dataclass(frozen=True)
class DryRunFedAvgConfig:
    """Synthetic FedAvg configuration used to exercise the stack."""

    num_rounds: int
    num_clients: int
    cohort_size: int
    parameter_dim: int
    client_examples: int
    client_update_scale: float
    weighted_by_examples: bool = False


@dataclass(frozen=True)
class TorchFedAvgSmokeConfig:
    """Small PyTorch FedAvg configuration for end-to-end smoke runs."""

    dataset: str
    model: str
    num_rounds: int
    num_clients: int
    cohort_size: int
    batch_size: int
    local_epochs: int
    learning_rate: float
    validation_fraction: float
    max_samples: int | None = None
    synthetic_num_samples: int = 120
    synthetic_num_classes: int = 10
    client_min_size: int = 1
    download: bool = False
    device: str = "cpu"
    weighted_by_examples: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    run: RunConfig
    mode: str
    dryrun_fedavg: DryRunFedAvgConfig
    torch_fedavg_smoke: TorchFedAvgSmokeConfig | None
    raw: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("config file must contain a mapping")

    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Parse a raw mapping into typed config objects."""

    run_raw = _required_mapping(raw, "run")
    experiment_raw = _required_mapping(raw, "experiment")
    mode = str(experiment_raw.get("mode", "dryrun_fedavg"))
    if mode not in {"dryrun_fedavg", "torch_fedavg_smoke"}:
        raise ValueError(f"unsupported experiment mode: {mode}")

    run = RunConfig(
        output_dir=Path(str(run_raw.get("output_dir", "runs"))),
        seed=_positive_int(run_raw.get("seed"), "run.seed", allow_zero=True),
        id=None if run_raw.get("id") is None else str(run_raw["id"]),
    )
    dryrun = _parse_dryrun_config(experiment_raw)
    torch_smoke = _parse_torch_smoke_config(experiment_raw) if mode == "torch_fedavg_smoke" else None

    return ExperimentConfig(
        run=run,
        mode=mode,
        dryrun_fedavg=dryrun,
        torch_fedavg_smoke=torch_smoke,
        raw=dict(raw),
    )


def run_experiment(config: ExperimentConfig, *, repo_path: str | Path = ".") -> RunPaths:
    """Run the configured experiment and write artifacts."""

    seed_everything(config.run.seed)
    paths = create_run_dir(config.run.output_dir, run_id=config.run.id)
    write_config(paths, config.raw)
    write_metadata(
        paths,
        {
            **collect_environment_metadata(repo_path=repo_path),
            "seed": config.run.seed,
            "mode": config.mode,
        },
    )

    if config.mode == "dryrun_fedavg":
        summary = run_dryrun_fedavg(config.dryrun_fedavg, paths=paths, seed=config.run.seed)
    elif config.mode == "torch_fedavg_smoke":
        if config.torch_fedavg_smoke is None:
            raise ValueError("torch_fedavg_smoke config is required")
        summary = run_torch_fedavg_smoke(config.torch_fedavg_smoke, paths=paths, seed=config.run.seed)
    else:  # pragma: no cover - parse_config prevents this branch.
        raise ValueError(f"unsupported experiment mode: {config.mode}")

    write_summary(paths, summary)
    return paths


def run_dryrun_fedavg(config: DryRunFedAvgConfig, *, paths: RunPaths, seed: int) -> dict[str, Any]:
    """Run a deterministic synthetic FedAvg workflow."""

    rng = np.random.default_rng(seed)
    server_parameters = [np.zeros(config.parameter_dim, dtype=np.float64)]
    final_metrics: dict[str, float] = {}

    for round_index in range(1, config.num_rounds + 1):
        cohort = sample_cohort(
            num_clients=config.num_clients,
            cohort_size=config.cohort_size,
            rng=rng,
        )
        updates = [
            _synthetic_client_update(
                client_id=client_id,
                round_index=round_index,
                server_parameters=server_parameters,
                config=config,
            )
            for client_id in cohort
        ]
        result = aggregate_fedavg(
            server_parameters,
            updates,
            weighted_by_examples=config.weighted_by_examples,
        )
        server_parameters = result.parameters
        update_norm = l2_norm(result.average_delta)
        parameter_norm = l2_norm(server_parameters)
        final_metrics = {
            "synthetic_loss": result.metrics["synthetic_loss"],
            "average_delta_l2": update_norm,
            "parameter_l2": parameter_norm,
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
        "mode": "dryrun_fedavg",
        "status": "completed",
        "rounds_completed": config.num_rounds,
        "num_clients": config.num_clients,
        "cohort_size": config.cohort_size,
        "parameter_dim": config.parameter_dim,
        "final_metrics": final_metrics,
    }


def run_torch_fedavg_smoke(
    config: TorchFedAvgSmokeConfig,
    *,
    paths: RunPaths,
    seed: int,
) -> dict[str, Any]:
    """Run a small PyTorch FedAvg smoke experiment."""

    torch = _require_torch()
    rng = np.random.default_rng(seed)
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

    global_model = _create_torch_smoke_model(config)
    server_parameters = list(get_model_parameters(global_model))
    final_metrics: dict[str, float] = {}

    for round_index in range(1, config.num_rounds + 1):
        cohort = sample_cohort(
            num_clients=config.num_clients,
            cohort_size=config.cohort_size,
            rng=rng,
        )
        updates = []
        for client_id in cohort:
            client_model = _create_torch_smoke_model(config)
            set_model_parameters(client_model, server_parameters)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=config.learning_rate)
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Run FL-DP experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--repo-path", default=".", help="Repository path for git metadata.")
    args = parser.parse_args(argv)

    paths = run_experiment(load_config(args.config), repo_path=args.repo_path)
    print(paths.root)
    return 0


def _synthetic_client_update(
    *,
    client_id: int,
    round_index: int,
    server_parameters: list[np.ndarray],
    config: DryRunFedAvgConfig,
) -> ClientUpdate:
    direction = np.full(config.parameter_dim, client_id + 1, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    round_scale = config.client_update_scale / round_index
    delta = direction * round_scale
    synthetic_loss = 1.0 / round_index + 0.01 * client_id
    return ClientUpdate(
        client_id=client_id,
        initial_parameters=tuple(array.copy() for array in server_parameters),
        updated_parameters=tuple(array + delta for array in server_parameters),
        num_examples=config.client_examples,
        metrics={"synthetic_loss": synthetic_loss},
    )


def _parse_dryrun_config(experiment_raw: Mapping[str, Any]) -> DryRunFedAvgConfig:
    dryrun = DryRunFedAvgConfig(
        num_rounds=_positive_int(experiment_raw.get("num_rounds", 1), "experiment.num_rounds"),
        num_clients=_positive_int(experiment_raw.get("num_clients", 2), "experiment.num_clients"),
        cohort_size=_positive_int(experiment_raw.get("cohort_size", 1), "experiment.cohort_size"),
        parameter_dim=_positive_int(experiment_raw.get("parameter_dim", 1), "experiment.parameter_dim"),
        client_examples=_positive_int(experiment_raw.get("client_examples", 1), "experiment.client_examples"),
        client_update_scale=_positive_float(
            experiment_raw.get("client_update_scale", 0.1),
            "experiment.client_update_scale",
        ),
        weighted_by_examples=bool(experiment_raw.get("weighted_by_examples", False)),
    )
    if dryrun.cohort_size > dryrun.num_clients:
        raise ValueError("experiment.cohort_size cannot exceed experiment.num_clients")
    return dryrun


def _parse_torch_smoke_config(experiment_raw: Mapping[str, Any]) -> TorchFedAvgSmokeConfig:
    max_samples_raw = experiment_raw.get("max_samples")
    config = TorchFedAvgSmokeConfig(
        dataset=str(experiment_raw.get("dataset", "synthetic")),
        model=str(experiment_raw.get("model", "mnist_cnn")),
        num_rounds=_positive_int(experiment_raw.get("num_rounds"), "experiment.num_rounds"),
        num_clients=_positive_int(experiment_raw.get("num_clients"), "experiment.num_clients"),
        cohort_size=_positive_int(experiment_raw.get("cohort_size"), "experiment.cohort_size"),
        batch_size=_positive_int(experiment_raw.get("batch_size"), "experiment.batch_size"),
        local_epochs=_positive_int(experiment_raw.get("local_epochs"), "experiment.local_epochs"),
        learning_rate=_positive_float(experiment_raw.get("learning_rate"), "experiment.learning_rate"),
        validation_fraction=_fraction(experiment_raw.get("validation_fraction"), "experiment.validation_fraction"),
        max_samples=None if max_samples_raw is None else _positive_int(max_samples_raw, "experiment.max_samples"),
        synthetic_num_samples=_positive_int(
            experiment_raw.get("synthetic_num_samples", 120),
            "experiment.synthetic_num_samples",
        ),
        synthetic_num_classes=_positive_int(
            experiment_raw.get("synthetic_num_classes", 10),
            "experiment.synthetic_num_classes",
        ),
        client_min_size=_positive_int(experiment_raw.get("client_min_size", 1), "experiment.client_min_size"),
        download=bool(experiment_raw.get("download", False)),
        device=str(experiment_raw.get("device", "cpu")),
        weighted_by_examples=bool(experiment_raw.get("weighted_by_examples", False)),
    )
    if config.dataset not in {"synthetic", "mnist"}:
        raise ValueError("experiment.dataset must be synthetic or mnist")
    if config.model != "mnist_cnn":
        raise ValueError("experiment.model must be mnist_cnn")
    if config.cohort_size > config.num_clients:
        raise ValueError("experiment.cohort_size cannot exceed experiment.num_clients")
    return config


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


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _fraction(value: Any, name: str) -> float:
    parsed = _positive_float(value, name)
    if not 0 < parsed < 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        raise RuntimeError("PyTorch is required for torch_fedavg_smoke") from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
