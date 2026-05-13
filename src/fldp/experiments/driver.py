"""Experiment CLI and dispatcher."""

from __future__ import annotations

import argparse
from pathlib import Path

from fldp.artifacts import (
    RunPaths,
    collect_environment_metadata,
    create_run_dir,
    write_config,
    write_metadata,
    write_summary,
)
from fldp.experiments.config import ExperimentConfig, load_config, parse_config
from fldp.experiments.dryrun import run_dryrun_fedavg
from fldp.experiments.torch_smoke import run_torch_dpfedavg_smoke, run_torch_fedavg_smoke
from fldp.train import seed_everything

__all__ = [
    "ExperimentConfig",
    "load_config",
    "main",
    "parse_config",
    "run_experiment",
]


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
    elif config.mode == "torch_dpfedavg_smoke":
        if config.torch_dpfedavg_smoke is None:
            raise ValueError("torch_dpfedavg_smoke config is required")
        summary = run_torch_dpfedavg_smoke(config.torch_dpfedavg_smoke, paths=paths, seed=config.run.seed)
    else:  # pragma: no cover - parse_config prevents this branch.
        raise ValueError(f"unsupported experiment mode: {config.mode}")

    write_summary(paths, summary)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Run FL-DP experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--repo-path", default=".", help="Repository path for git metadata.")
    args = parser.parse_args(argv)

    paths = run_experiment(load_config(args.config), repo_path=args.repo_path)
    print(paths.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
