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
from fldp.strategies import ClientUpdate, aggregate_fedavg, sample_cohort
from fldp.strategies.parameters import l2_norm
from fldp.train import seed_everything


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
class ExperimentConfig:
    """Top-level experiment configuration."""

    run: RunConfig
    mode: str
    dryrun_fedavg: DryRunFedAvgConfig
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
    if mode != "dryrun_fedavg":
        raise ValueError(f"unsupported experiment mode: {mode}")

    run = RunConfig(
        output_dir=Path(str(run_raw.get("output_dir", "runs"))),
        seed=_positive_int(run_raw.get("seed"), "run.seed", allow_zero=True),
        id=None if run_raw.get("id") is None else str(run_raw["id"]),
    )
    dryrun = DryRunFedAvgConfig(
        num_rounds=_positive_int(experiment_raw.get("num_rounds"), "experiment.num_rounds"),
        num_clients=_positive_int(experiment_raw.get("num_clients"), "experiment.num_clients"),
        cohort_size=_positive_int(experiment_raw.get("cohort_size"), "experiment.cohort_size"),
        parameter_dim=_positive_int(experiment_raw.get("parameter_dim"), "experiment.parameter_dim"),
        client_examples=_positive_int(experiment_raw.get("client_examples"), "experiment.client_examples"),
        client_update_scale=_positive_float(
            experiment_raw.get("client_update_scale"),
            "experiment.client_update_scale",
        ),
        weighted_by_examples=bool(experiment_raw.get("weighted_by_examples", False)),
    )
    if dryrun.cohort_size > dryrun.num_clients:
        raise ValueError("experiment.cohort_size cannot exceed experiment.num_clients")

    return ExperimentConfig(run=run, mode=mode, dryrun_fedavg=dryrun, raw=dict(raw))


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


if __name__ == "__main__":
    raise SystemExit(main())
