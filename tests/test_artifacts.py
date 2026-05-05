import json

import numpy as np
import pytest
import yaml

from fldp.artifacts import (
    append_round_metrics,
    collect_environment_metadata,
    create_run_dir,
    read_jsonl,
    write_accountant_trace,
    write_config,
    write_metadata,
    write_summary,
)
from fldp.privacy import PrivacyAccountant


def test_create_run_dir_returns_stable_artifact_paths(tmp_path) -> None:
    paths = create_run_dir(tmp_path, run_id="mnist_seed_1")

    assert paths.root == tmp_path / "mnist_seed_1"
    assert paths.config.name == "config.yaml"
    assert paths.metadata.name == "metadata.json"
    assert paths.metrics.name == "metrics.jsonl"
    assert paths.accountant_trace.name == "accountant_trace.json"
    assert paths.summary.name == "summary.json"

    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, run_id="mnist_seed_1")


def test_write_config_serializes_yaml_with_numpy_values(tmp_path) -> None:
    paths = create_run_dir(tmp_path, run_id="run")

    write_config(
        paths,
        {
            "dataset": "mnist",
            "seed": np.int64(7),
            "epsilon_targets": np.array([1.0, 2.0]),
        },
    )

    loaded = yaml.safe_load(paths.config.read_text(encoding="utf-8"))

    assert loaded == {
        "dataset": "mnist",
        "seed": 7,
        "epsilon_targets": [1.0, 2.0],
    }


def test_metadata_and_summary_are_pretty_json(tmp_path) -> None:
    paths = create_run_dir(tmp_path, run_id="run")

    write_metadata(paths, {"seed": 3, "package_versions": {"numpy": np.__version__}})
    write_summary(paths, {"accuracy": np.float64(0.91)})

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))

    assert metadata["seed"] == 3
    assert summary["accuracy"] == pytest.approx(0.91)
    assert paths.metadata.read_text(encoding="utf-8").endswith("\n")


def test_append_round_metrics_writes_jsonl_rows(tmp_path) -> None:
    paths = create_run_dir(tmp_path, run_id="run")

    append_round_metrics(paths, {"round": 1, "accuracy": np.float64(0.8)})
    append_round_metrics(paths, {"round": 2, "accuracy": np.float64(0.85)})

    assert read_jsonl(paths.metrics) == [
        {"round": 1, "accuracy": 0.8},
        {"round": 2, "accuracy": 0.85},
    ]


def test_write_accountant_trace_records_cumulative_final_epsilon(tmp_path) -> None:
    paths = create_run_dir(tmp_path, run_id="run")
    accountant = PrivacyAccountant(orders=[2, 4])
    accountant.step(noise_multiplier=1.0, sample_rate=0.1)
    accountant.step(noise_multiplier=1.5, sample_rate=0.1)

    write_accountant_trace(paths, accountant, delta=1e-5, privacy_unit="client")

    trace = json.loads(paths.accountant_trace.read_text(encoding="utf-8"))

    assert trace["accountant"] == "rdp_sampled_gaussian"
    assert trace["privacy_unit"] == "client"
    assert trace["delta"] == 1e-5
    assert trace["epsilon_cumulative_final"] == pytest.approx(accountant.get_epsilon(delta=1e-5))
    assert trace["orders"] == [2.0, 4.0]
    assert len(trace["rdp"]) == 2
    assert trace["steps"] == [
        {"round": 1, "noise_multiplier": 1.0, "sample_rate": 0.1},
        {"round": 2, "noise_multiplier": 1.5, "sample_rate": 0.1},
    ]


def test_collect_environment_metadata_handles_git_and_missing_packages(tmp_path) -> None:
    metadata = collect_environment_metadata(
        repo_path=tmp_path,
        package_names=("numpy", "definitely-missing-package"),
    )

    assert metadata["python"]
    assert metadata["platform"]
    assert metadata["git_commit"] is None
    assert metadata["git_dirty"] is None
    assert metadata["packages"]["numpy"] is not None
    assert metadata["packages"]["definitely-missing-package"] is None
