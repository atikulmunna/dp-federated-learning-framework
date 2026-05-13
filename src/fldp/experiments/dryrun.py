"""Synthetic NumPy FedAvg dry-run experiment."""

from __future__ import annotations

import numpy as np

from fldp.artifacts import RunPaths, append_round_metrics
from fldp.experiments.config import DryRunFedAvgConfig
from fldp.strategies import ClientUpdate, aggregate_fedavg, sample_cohort
from fldp.strategies.parameters import l2_norm


def run_dryrun_fedavg(config: DryRunFedAvgConfig, *, paths: RunPaths, seed: int) -> dict[str, object]:
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
        final_metrics = {
            "synthetic_loss": result.metrics["synthetic_loss"],
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
        "mode": "dryrun_fedavg",
        "status": "completed",
        "rounds_completed": config.num_rounds,
        "num_clients": config.num_clients,
        "cohort_size": config.cohort_size,
        "parameter_dim": config.parameter_dim,
        "final_metrics": final_metrics,
    }


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
