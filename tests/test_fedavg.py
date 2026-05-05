import numpy as np
import pytest

from fldp.strategies import ClientUpdate, aggregate_fedavg, sample_cohort


def _update(
    client_id: int,
    delta: float,
    *,
    num_examples: int = 1,
    loss: float | None = None,
) -> ClientUpdate:
    metrics = None if loss is None else {"loss": loss}
    return ClientUpdate(
        client_id=client_id,
        initial_parameters=(np.array([1.0, 2.0]), np.array([[3.0]])),
        updated_parameters=(np.array([1.0 + delta, 2.0 + delta]), np.array([[3.0 + delta]])),
        num_examples=num_examples,
        metrics=metrics,
    )


def test_aggregate_fedavg_uniformly_averages_client_deltas() -> None:
    server_parameters = [np.array([1.0, 2.0]), np.array([[3.0]])]

    result = aggregate_fedavg(
        server_parameters,
        [_update(0, 1.0), _update(1, 3.0)],
    )

    assert result.cohort_size == 2
    assert result.num_examples == 2
    assert np.allclose(result.average_delta[0], [2.0, 2.0])
    assert np.allclose(result.average_delta[1], [[2.0]])
    assert np.allclose(result.parameters[0], [3.0, 4.0])
    assert np.allclose(result.parameters[1], [[5.0]])


def test_aggregate_fedavg_can_weight_by_examples() -> None:
    server_parameters = [np.array([0.0])]

    result = aggregate_fedavg(
        server_parameters,
        [
            ClientUpdate(0, (np.array([0.0]),), (np.array([1.0]),), 1, {"loss": 2.0}),
            ClientUpdate(1, (np.array([0.0]),), (np.array([3.0]),), 3, {"loss": 6.0}),
        ],
        weighted_by_examples=True,
    )

    assert np.allclose(result.average_delta[0], [2.5])
    assert np.allclose(result.parameters[0], [2.5])
    assert result.metrics["loss"] == pytest.approx(5.0)


def test_aggregate_fedavg_uses_unweighted_metric_mean_by_default() -> None:
    server_parameters = [np.array([0.0])]

    result = aggregate_fedavg(
        server_parameters,
        [
            ClientUpdate(0, (np.array([0.0]),), (np.array([1.0]),), 1, {"loss": 2.0}),
            ClientUpdate(1, (np.array([0.0]),), (np.array([3.0]),), 3, {"loss": 6.0}),
        ],
    )

    assert result.metrics["loss"] == pytest.approx(4.0)


def test_aggregate_fedavg_rejects_empty_or_invalid_updates() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_fedavg([np.array([0.0])], [])

    with pytest.raises(ValueError, match="at least one example"):
        aggregate_fedavg(
            [np.array([0.0])],
            [ClientUpdate(0, (np.array([0.0]),), (np.array([1.0]),), 0)],
        )


def test_sample_cohort_is_deterministic_and_without_replacement() -> None:
    first = sample_cohort(num_clients=10, cohort_size=4, rng=np.random.default_rng(5))
    second = sample_cohort(num_clients=10, cohort_size=4, rng=np.random.default_rng(5))

    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
    assert all(0 <= client_id < 10 for client_id in first)


def test_sample_cohort_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="num_clients"):
        sample_cohort(num_clients=0, cohort_size=1, rng=np.random.default_rng(1))

    with pytest.raises(ValueError, match="cohort_size"):
        sample_cohort(num_clients=2, cohort_size=0, rng=np.random.default_rng(1))

    with pytest.raises(ValueError, match="exceed"):
        sample_cohort(num_clients=2, cohort_size=3, rng=np.random.default_rng(1))
