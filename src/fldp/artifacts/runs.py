"""Stable on-disk run artifact layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import yaml

from fldp.privacy import PrivacyAccountant


@dataclass(frozen=True)
class RunPaths:
    """Canonical file paths for one experiment run."""

    root: Path
    config: Path
    metadata: Path
    metrics: Path
    accountant_trace: Path
    summary: Path


def create_run_dir(base_dir: str | Path, *, run_id: str | None = None) -> RunPaths:
    """Create a new run directory and return its artifact paths."""

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    selected_run_id = run_id or _timestamp_run_id()
    run_root = base_path / selected_run_id
    run_root.mkdir(parents=False, exist_ok=False)

    return RunPaths(
        root=run_root,
        config=run_root / "config.yaml",
        metadata=run_root / "metadata.json",
        metrics=run_root / "metrics.jsonl",
        accountant_trace=run_root / "accountant_trace.json",
        summary=run_root / "summary.json",
    )


def write_config(paths: RunPaths, config: Mapping[str, Any]) -> None:
    """Write the immutable run configuration as YAML."""

    _ensure_run_dir(paths)
    with paths.config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _to_plain_data(config),
            handle,
            sort_keys=True,
            allow_unicode=False,
        )


def write_metadata(paths: RunPaths, metadata: Mapping[str, Any]) -> None:
    """Write environment and provenance metadata as JSON."""

    _write_json(paths.metadata, metadata)


def write_summary(paths: RunPaths, summary: Mapping[str, Any]) -> None:
    """Write a final run summary as JSON."""

    _write_json(paths.summary, summary)


def append_round_metrics(paths: RunPaths, metrics: Mapping[str, Any]) -> None:
    """Append one JSONL row of per-round metrics."""

    _ensure_run_dir(paths)
    with paths.metrics.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_plain_data(metrics), sort_keys=True))
        handle.write("\n")


def write_accountant_trace(
    paths: RunPaths,
    accountant: PrivacyAccountant,
    *,
    delta: float,
    privacy_unit: str,
) -> None:
    """Persist the accountant curve and cumulative final epsilon."""

    trace = {
        "accountant": "rdp_sampled_gaussian",
        "privacy_unit": privacy_unit,
        "delta": delta,
        "epsilon_cumulative_final": accountant.get_epsilon(delta=delta),
        "orders": accountant.orders,
        "rdp": accountant.get_rdp(),
        "steps": [
            {"round": index + 1, "noise_multiplier": sigma, "sample_rate": sample_rate}
            for index, (sigma, sample_rate) in enumerate(accountant.steps)
        ],
    }
    _write_json(paths.accountant_trace, trace)


def collect_environment_metadata(
    *,
    repo_path: str | Path = ".",
    package_names: Sequence[str] = ("numpy", "PyYAML", "pytest", "torch", "torchvision", "opacus", "flwr"),
) -> dict[str, Any]:
    """Collect reproducibility metadata for a run."""

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit_hash(Path(repo_path)),
        "git_dirty": _git_is_dirty(Path(repo_path)),
        "packages": {
            package_name: _package_version(package_name)
            for package_name in package_names
        },
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact into dictionaries."""

    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_plain_data(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_data(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_run_dir(paths: RunPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit_hash(repo_path: Path) -> str | None:
    return _run_git(repo_path, "rev-parse", "HEAD")


def _git_is_dirty(repo_path: Path) -> bool | None:
    output = _run_git(repo_path, "status", "--short")
    return None if output is None else bool(output.strip())


def _run_git(repo_path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
