import json
from pathlib import Path

import pytest
import yaml

from fldp.artifacts import read_jsonl
from fldp.experiments.driver import load_config, main, parse_config, run_experiment


def _write_config(tmp_path: Path, *, run_id: str = "dryrun") -> Path:
    config = {
        "run": {
            "id": run_id,
            "output_dir": str(tmp_path / "runs"),
            "seed": 123,
        },
        "experiment": {
            "mode": "dryrun_fedavg",
            "num_rounds": 3,
            "num_clients": 6,
            "cohort_size": 2,
            "parameter_dim": 4,
            "client_examples": 5,
            "client_update_scale": 0.1,
            "weighted_by_examples": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_torch_smoke_config(tmp_path: Path, *, run_id: str = "torch-smoke") -> Path:
    config = {
        "run": {
            "id": run_id,
            "output_dir": str(tmp_path / "runs"),
            "seed": 321,
        },
        "experiment": {
            "mode": "torch_fedavg_smoke",
            "dataset": "synthetic",
            "model": "mnist_cnn",
            "synthetic_num_samples": 48,
            "synthetic_num_classes": 10,
            "validation_fraction": 0.25,
            "num_rounds": 2,
            "num_clients": 4,
            "cohort_size": 2,
            "batch_size": 8,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "client_min_size": 1,
            "device": "cpu",
            "weighted_by_examples": False,
        },
    }
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_torch_dp_smoke_config(tmp_path: Path, *, run_id: str = "torch-dp-smoke") -> Path:
    config = {
        "run": {
            "id": run_id,
            "output_dir": str(tmp_path / "runs"),
            "seed": 654,
        },
        "experiment": {
            "mode": "torch_dpfedavg_smoke",
            "dataset": "synthetic",
            "model": "mnist_cnn",
            "synthetic_num_samples": 48,
            "synthetic_num_classes": 10,
            "validation_fraction": 0.25,
            "num_rounds": 2,
            "num_clients": 4,
            "cohort_size": 2,
            "batch_size": 8,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "client_min_size": 1,
            "device": "cpu",
            "weighted_by_examples": False,
            "clip_norm": 1.0,
            "noise_multiplier": 1.25,
            "delta": 1e-5,
            "privacy_unit": "client",
        },
    }
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_config_parses_dryrun_yaml(tmp_path) -> None:
    config_path = _write_config(tmp_path)

    config = load_config(config_path)

    assert config.mode == "dryrun_fedavg"
    assert config.run.id == "dryrun"
    assert config.run.seed == 123
    assert config.dryrun_fedavg.num_rounds == 3
    assert config.dryrun_fedavg.cohort_size == 2


def test_parse_config_rejects_invalid_cohort_size() -> None:
    with pytest.raises(ValueError, match="cohort_size"):
        parse_config(
            {
                "run": {"seed": 1},
                "experiment": {
                    "mode": "dryrun_fedavg",
                    "num_rounds": 1,
                    "num_clients": 2,
                    "cohort_size": 3,
                    "parameter_dim": 4,
                    "client_examples": 5,
                    "client_update_scale": 0.1,
                },
            }
        )


def test_load_config_parses_torch_smoke_yaml(tmp_path) -> None:
    config = load_config(_write_torch_smoke_config(tmp_path))

    assert config.mode == "torch_fedavg_smoke"
    assert config.torch_fedavg_smoke is not None
    assert config.torch_fedavg_smoke.dataset == "synthetic"
    assert config.torch_fedavg_smoke.num_rounds == 2
    assert config.torch_fedavg_smoke.batch_size == 8


def test_load_config_parses_torch_dp_smoke_yaml(tmp_path) -> None:
    config = load_config(_write_torch_dp_smoke_config(tmp_path))

    assert config.mode == "torch_dpfedavg_smoke"
    assert config.torch_dpfedavg_smoke is not None
    assert config.torch_dpfedavg_smoke.base.dataset == "synthetic"
    assert config.torch_dpfedavg_smoke.clip_norm == 1.0
    assert config.torch_dpfedavg_smoke.noise_multiplier == 1.25
    assert config.torch_dpfedavg_smoke.delta == 1e-5
    assert config.torch_dpfedavg_smoke.privacy_unit == "client"


def test_parse_config_rejects_unknown_torch_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        parse_config(
            {
                "run": {"seed": 1},
                "experiment": {
                    "mode": "torch_fedavg_smoke",
                    "dataset": "unknown",
                    "model": "mnist_cnn",
                    "num_rounds": 1,
                    "num_clients": 2,
                    "cohort_size": 1,
                    "batch_size": 4,
                    "local_epochs": 1,
                    "learning_rate": 0.01,
                    "validation_fraction": 0.2,
                },
            }
        )


def test_parse_config_rejects_non_client_dp_privacy_unit() -> None:
    with pytest.raises(ValueError, match="privacy_unit"):
        parse_config(
            {
                "run": {"seed": 1},
                "experiment": {
                    "mode": "torch_dpfedavg_smoke",
                    "dataset": "synthetic",
                    "model": "mnist_cnn",
                    "num_rounds": 1,
                    "num_clients": 2,
                    "cohort_size": 1,
                    "batch_size": 4,
                    "local_epochs": 1,
                    "learning_rate": 0.01,
                    "validation_fraction": 0.2,
                    "clip_norm": 1.0,
                    "noise_multiplier": 1.0,
                    "delta": 1e-5,
                    "privacy_unit": "sample",
                },
            }
        )


def test_run_experiment_writes_dryrun_artifacts(tmp_path) -> None:
    config = load_config(_write_config(tmp_path))

    paths = run_experiment(config, repo_path=tmp_path)

    assert paths.root.exists()
    assert paths.config.exists()
    assert paths.metadata.exists()
    assert paths.metrics.exists()
    assert paths.summary.exists()
    assert not paths.accountant_trace.exists()

    metrics = read_jsonl(paths.metrics)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))

    assert len(metrics) == 3
    assert [row["round"] for row in metrics] == [1, 2, 3]
    assert all(row["cohort_size"] == 2 for row in metrics)
    assert all(row["num_examples"] == 10 for row in metrics)
    assert all(len(row["cohort"]) == 2 for row in metrics)
    assert summary["status"] == "completed"
    assert summary["rounds_completed"] == 3
    assert summary["final_metrics"]["parameter_l2"] == metrics[-1]["parameter_l2"]
    assert metadata["seed"] == 123
    assert metadata["mode"] == "dryrun_fedavg"


def test_run_experiment_writes_torch_smoke_artifacts(tmp_path) -> None:
    config = load_config(_write_torch_smoke_config(tmp_path))

    paths = run_experiment(config, repo_path=tmp_path)

    metrics = read_jsonl(paths.metrics)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))

    assert len(metrics) == 2
    assert [row["round"] for row in metrics] == [1, 2]
    assert all(row["cohort_size"] == 2 for row in metrics)
    assert all(row["num_examples"] > 0 for row in metrics)
    assert all(0 <= row["validation_accuracy"] <= 1 for row in metrics)
    assert all(row["validation_loss"] > 0 for row in metrics)
    assert summary["mode"] == "torch_fedavg_smoke"
    assert summary["status"] == "completed"
    assert summary["dataset"] == "synthetic"
    assert summary["rounds_completed"] == 2
    assert summary["final_metrics"]["validation_accuracy"] == metrics[-1]["validation_accuracy"]


def test_run_experiment_writes_torch_dp_smoke_artifacts_and_accountant_trace(tmp_path) -> None:
    config = load_config(_write_torch_dp_smoke_config(tmp_path))

    paths = run_experiment(config, repo_path=tmp_path)

    metrics = read_jsonl(paths.metrics)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    trace = json.loads(paths.accountant_trace.read_text(encoding="utf-8"))

    assert len(metrics) == 2
    assert [row["round"] for row in metrics] == [1, 2]
    assert all(row["cohort_size"] == 2 for row in metrics)
    assert all(row["sample_rate"] == 0.5 for row in metrics)
    assert all(row["clip_norm"] == 1.0 for row in metrics)
    assert all(row["noise_multiplier"] == 1.25 for row in metrics)
    assert all(row["noise_std"] == 0.625 for row in metrics)
    assert all(row["epsilon_cumulative"] > 0 for row in metrics)
    assert all(row["pre_clip_l2_max"] >= row["pre_clip_l2_mean"] for row in metrics)
    assert all(0 < row["clip_scale_min"] <= 1 for row in metrics)
    assert summary["mode"] == "torch_dpfedavg_smoke"
    assert summary["privacy_unit"] == "client"
    assert summary["epsilon_cumulative_final"] == metrics[-1]["epsilon_cumulative"]
    assert trace["privacy_unit"] == "client"
    assert trace["delta"] == 1e-5
    assert trace["epsilon_cumulative_final"] == summary["epsilon_cumulative_final"]
    assert trace["steps"] == [
        {"round": 1, "noise_multiplier": 1.25, "sample_rate": 0.5},
        {"round": 2, "noise_multiplier": 1.25, "sample_rate": 0.5},
    ]


def test_run_experiment_is_deterministic_for_same_config_and_seed(tmp_path) -> None:
    first_config = load_config(_write_config(tmp_path, run_id="first"))
    second_config = load_config(_write_config(tmp_path, run_id="second"))

    first = run_experiment(first_config, repo_path=tmp_path)
    second = run_experiment(second_config, repo_path=tmp_path)

    assert read_jsonl(first.metrics) == read_jsonl(second.metrics)
    first_summary = json.loads(first.summary.read_text(encoding="utf-8"))
    second_summary = json.loads(second.summary.read_text(encoding="utf-8"))
    assert first_summary == second_summary


def test_torch_smoke_run_is_deterministic_for_same_config_and_seed(tmp_path) -> None:
    first_config = load_config(_write_torch_smoke_config(tmp_path, run_id="first-torch"))
    second_config = load_config(_write_torch_smoke_config(tmp_path, run_id="second-torch"))

    first = run_experiment(first_config, repo_path=tmp_path)
    second = run_experiment(second_config, repo_path=tmp_path)

    assert read_jsonl(first.metrics) == read_jsonl(second.metrics)


def test_torch_dp_smoke_run_is_deterministic_for_same_config_and_seed(tmp_path) -> None:
    first_config = load_config(_write_torch_dp_smoke_config(tmp_path, run_id="first-dp"))
    second_config = load_config(_write_torch_dp_smoke_config(tmp_path, run_id="second-dp"))

    first = run_experiment(first_config, repo_path=tmp_path)
    second = run_experiment(second_config, repo_path=tmp_path)

    assert read_jsonl(first.metrics) == read_jsonl(second.metrics)
    first_trace = json.loads(first.accountant_trace.read_text(encoding="utf-8"))
    second_trace = json.loads(second.accountant_trace.read_text(encoding="utf-8"))
    assert first_trace == second_trace


def test_driver_main_prints_run_directory(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)

    exit_code = main(["--config", str(config_path), "--repo-path", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(tmp_path / "runs" / "dryrun") in captured.out
