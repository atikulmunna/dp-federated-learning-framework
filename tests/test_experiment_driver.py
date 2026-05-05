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


def test_run_experiment_is_deterministic_for_same_config_and_seed(tmp_path) -> None:
    first_config = load_config(_write_config(tmp_path, run_id="first"))
    second_config = load_config(_write_config(tmp_path, run_id="second"))

    first = run_experiment(first_config, repo_path=tmp_path)
    second = run_experiment(second_config, repo_path=tmp_path)

    assert read_jsonl(first.metrics) == read_jsonl(second.metrics)
    first_summary = json.loads(first.summary.read_text(encoding="utf-8"))
    second_summary = json.loads(second.summary.read_text(encoding="utf-8"))
    assert first_summary == second_summary


def test_driver_main_prints_run_directory(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)

    exit_code = main(["--config", str(config_path), "--repo-path", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(tmp_path / "runs" / "dryrun") in captured.out
