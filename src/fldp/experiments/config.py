"""Experiment configuration parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
class TorchDPFedAvgSmokeConfig:
    """Small PyTorch DP-FedAvg configuration for smoke runs."""

    base: TorchFedAvgSmokeConfig
    clip_norm: float
    noise_multiplier: float
    delta: float
    privacy_unit: str = "client"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    run: RunConfig
    mode: str
    dryrun_fedavg: DryRunFedAvgConfig
    torch_fedavg_smoke: TorchFedAvgSmokeConfig | None
    torch_dpfedavg_smoke: TorchDPFedAvgSmokeConfig | None
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
    if mode not in {"dryrun_fedavg", "torch_fedavg_smoke", "torch_dpfedavg_smoke"}:
        raise ValueError(f"unsupported experiment mode: {mode}")

    run = RunConfig(
        output_dir=Path(str(run_raw.get("output_dir", "runs"))),
        seed=_positive_int(run_raw.get("seed"), "run.seed", allow_zero=True),
        id=None if run_raw.get("id") is None else str(run_raw["id"]),
    )
    dryrun = _parse_dryrun_config(experiment_raw)
    torch_smoke = _parse_torch_smoke_config(experiment_raw) if mode == "torch_fedavg_smoke" else None
    torch_dp_smoke = _parse_torch_dp_smoke_config(experiment_raw) if mode == "torch_dpfedavg_smoke" else None

    return ExperimentConfig(
        run=run,
        mode=mode,
        dryrun_fedavg=dryrun,
        torch_fedavg_smoke=torch_smoke,
        torch_dpfedavg_smoke=torch_dp_smoke,
        raw=dict(raw),
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


def _parse_torch_dp_smoke_config(experiment_raw: Mapping[str, Any]) -> TorchDPFedAvgSmokeConfig:
    base = _parse_torch_smoke_config(experiment_raw)
    privacy_unit = str(experiment_raw.get("privacy_unit", "client"))
    if privacy_unit != "client":
        raise ValueError("experiment.privacy_unit must be client")
    return TorchDPFedAvgSmokeConfig(
        base=base,
        clip_norm=_positive_float(experiment_raw.get("clip_norm"), "experiment.clip_norm"),
        noise_multiplier=_positive_float(
            experiment_raw.get("noise_multiplier"),
            "experiment.noise_multiplier",
        ),
        delta=_fraction(experiment_raw.get("delta"), "experiment.delta"),
        privacy_unit=privacy_unit,
    )


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
