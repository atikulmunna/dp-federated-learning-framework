"""Plain FedAvg aggregation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fldp.strategies.parameters import ArrayList, apply_delta, mean_arrays, params_to_delta


@dataclass(frozen=True)
class ClientUpdate:
    """A client result returned to the server after local training."""

    client_id: int
    initial_parameters: tuple[NDArray[np.floating], ...]
    updated_parameters: tuple[NDArray[np.floating], ...]
    num_examples: int
    metrics: dict[str, float] | None = None

    def delta(self) -> ArrayList:
        return params_to_delta(self.initial_parameters, self.updated_parameters)


@dataclass(frozen=True)
class AggregationResult:
    """Server-side result after applying FedAvg."""

    parameters: ArrayList
    average_delta: ArrayList
    cohort_size: int
    num_examples: int
    metrics: dict[str, float]


def aggregate_fedavg(
    server_parameters: Sequence[NDArray[np.floating]],
    updates: Sequence[ClientUpdate],
    *,
    weighted_by_examples: bool = False,
) -> AggregationResult:
    """Aggregate client updates and apply the average delta to the server model."""

    if len(updates) == 0:
        raise ValueError("updates must not be empty")

    for update in updates:
        if update.num_examples <= 0:
            raise ValueError("client updates must contain at least one example")

    deltas = [update.delta() for update in updates]
    weights = [update.num_examples for update in updates] if weighted_by_examples else None
    average_delta = mean_arrays(deltas, weights=weights)
    parameters = apply_delta(server_parameters, average_delta)

    return AggregationResult(
        parameters=parameters,
        average_delta=average_delta,
        cohort_size=len(updates),
        num_examples=sum(update.num_examples for update in updates),
        metrics=_aggregate_metrics(updates, weighted_by_examples=weighted_by_examples),
    )


def sample_cohort(
    *,
    num_clients: int,
    cohort_size: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Sample a client cohort uniformly without replacement."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")
    if cohort_size > num_clients:
        raise ValueError("cohort_size cannot exceed num_clients")

    sampled = rng.choice(num_clients, size=cohort_size, replace=False)
    return tuple(int(client_id) for client_id in sampled)


def _aggregate_metrics(
    updates: Sequence[ClientUpdate],
    *,
    weighted_by_examples: bool,
) -> dict[str, float]:
    metric_names = sorted(
        {
            metric_name
            for update in updates
            if update.metrics
            for metric_name in update.metrics
        }
    )
    if not metric_names:
        return {}

    aggregated: dict[str, float] = {}
    for metric_name in metric_names:
        values = [
            (float(update.metrics[metric_name]), update.num_examples)
            for update in updates
            if update.metrics and metric_name in update.metrics
        ]
        if weighted_by_examples:
            total_weight = sum(weight for _, weight in values)
            aggregated[metric_name] = sum(value * weight for value, weight in values) / total_weight
        else:
            aggregated[metric_name] = sum(value for value, _ in values) / len(values)
    return aggregated
