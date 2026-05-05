import numpy as np
import pytest

from fldp.privacy import PrivacyAccountant
from fldp.strategies import ClientUpdate, aggregate_dpfedavg


def _client_update(client_id: int, delta: np.ndarray, *, num_examples: int = 1) -> ClientUpdate:
    initial = (np.zeros_like(delta, dtype=np.float64),)
    updated = (np.asarray(delta, dtype=np.float64),)
    return ClientUpdate(
        client_id=client_id,
        initial_parameters=initial,
        updated_parameters=updated,
        num_examples=num_examples,
    )


def test_dpfedavg_clips_client_deltas_before_averaging() -> None:
    accountant = PrivacyAccountant(orders=[2, 4])

    result = aggregate_dpfedavg(
        [np.array([0.0, 0.0])],
        [
            _client_update(0, np.array([3.0, 4.0])),
            _client_update(1, np.array([1.0, 0.0])),
        ],
        clip_norm=2.0,
        noise_multiplier=1.0,
        accountant=accountant,
        num_total_clients=10,
        noise_rng=np.random.default_rng(0),
        delta=1e-5,
    )

    assert result.pre_clip_norms == pytest.approx((5.0, 1.0))
    assert result.clip_scales == pytest.approx((0.4, 1.0))
    assert np.allclose(result.average_clipped_delta[0], [1.1, 0.8])
    assert result.sample_rate == pytest.approx(0.2)
    assert result.noise_std == pytest.approx(1.0)
    assert result.epsilon is not None


def test_dpfedavg_adds_deterministic_gaussian_noise_with_seeded_rng() -> None:
    kwargs = dict(
        server_parameters=[np.array([0.0, 0.0])],
        updates=[
            _client_update(0, np.array([1.0, 0.0])),
            _client_update(1, np.array([0.0, 1.0])),
        ],
        clip_norm=2.0,
        noise_multiplier=0.5,
        num_total_clients=4,
    )

    first = aggregate_dpfedavg(
        **kwargs,
        accountant=PrivacyAccountant(orders=[2, 4]),
        noise_rng=np.random.default_rng(9),
    )
    second = aggregate_dpfedavg(
        **kwargs,
        accountant=PrivacyAccountant(orders=[2, 4]),
        noise_rng=np.random.default_rng(9),
    )

    assert np.allclose(first.noised_delta[0], second.noised_delta[0])
    assert not np.allclose(first.noised_delta[0], first.average_clipped_delta[0])


def test_dpfedavg_steps_accountant_once_with_computed_sample_rate() -> None:
    accountant = PrivacyAccountant(orders=[2, 4])

    aggregate_dpfedavg(
        [np.array([0.0])],
        [_client_update(0, np.array([1.0])), _client_update(1, np.array([2.0]))],
        clip_norm=1.0,
        noise_multiplier=1.25,
        accountant=accountant,
        num_total_clients=8,
        noise_rng=np.random.default_rng(1),
    )

    assert accountant.num_steps == 1
    assert accountant.steps == ((1.25, 0.25),)


def test_dpfedavg_updates_server_parameters_with_noised_delta() -> None:
    result = aggregate_dpfedavg(
        [np.array([10.0, 20.0])],
        [_client_update(0, np.array([1.0, 2.0]))],
        clip_norm=10.0,
        noise_multiplier=0.1,
        accountant=PrivacyAccountant(orders=[2, 4]),
        num_total_clients=1,
        noise_rng=np.random.default_rng(3),
    )

    assert np.allclose(result.parameters[0], np.array([10.0, 20.0]) + result.noised_delta[0])


def test_dpfedavg_rejects_invalid_privacy_parameters() -> None:
    valid_update = _client_update(0, np.array([1.0]))

    with pytest.raises(ValueError, match="clip_norm"):
        aggregate_dpfedavg(
            [np.array([0.0])],
            [valid_update],
            clip_norm=0,
            noise_multiplier=1,
            accountant=PrivacyAccountant(),
            num_total_clients=1,
            noise_rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="noise_multiplier"):
        aggregate_dpfedavg(
            [np.array([0.0])],
            [valid_update],
            clip_norm=1,
            noise_multiplier=0,
            accountant=PrivacyAccountant(),
            num_total_clients=1,
            noise_rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="cohort size"):
        aggregate_dpfedavg(
            [np.array([0.0])],
            [valid_update, _client_update(1, np.array([1.0]))],
            clip_norm=1,
            noise_multiplier=1,
            accountant=PrivacyAccountant(),
            num_total_clients=1,
            noise_rng=np.random.default_rng(1),
        )
