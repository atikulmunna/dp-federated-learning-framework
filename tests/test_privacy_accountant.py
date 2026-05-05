import math

import numpy as np
import pytest

from fldp.privacy import PrivacyAccountant, find_noise_multiplier


def test_accountant_matches_full_participation_gaussian_rdp() -> None:
    accountant = PrivacyAccountant(orders=[2, 4, 8])

    accountant.step(noise_multiplier=2.0, sample_rate=1.0)

    assert np.allclose(accountant.get_rdp(), [0.25, 0.5, 1.0])


def test_accountant_composes_steps_and_resets() -> None:
    accountant = PrivacyAccountant(orders=[2, 4])

    accountant.step(noise_multiplier=1.0, sample_rate=1.0)
    accountant.step(noise_multiplier=1.0, sample_rate=1.0)

    assert accountant.num_steps == 2
    assert np.allclose(accountant.get_rdp(), [2.0, 4.0])

    accountant.reset()

    assert accountant.num_steps == 0
    assert np.allclose(accountant.get_rdp(), [0.0, 0.0])


def test_subsampling_reports_smaller_epsilon_than_full_participation() -> None:
    sampled = PrivacyAccountant(orders=range(2, 16))
    full = PrivacyAccountant(orders=range(2, 16))

    for _ in range(10):
        sampled.step(noise_multiplier=1.0, sample_rate=0.1)
        full.step(noise_multiplier=1.0, sample_rate=1.0)

    assert sampled.get_epsilon(delta=1e-5) < full.get_epsilon(delta=1e-5)


def test_epsilon_conversion_uses_best_order() -> None:
    accountant = PrivacyAccountant(orders=[2, 8])
    accountant.step(noise_multiplier=2.0, sample_rate=1.0)

    result = accountant.get_epsilon_result(delta=1e-5)

    expected = min(
        2 / (2 * 2.0**2) + math.log(1e5) / (2 - 1),
        8 / (2 * 2.0**2) + math.log(1e5) / (8 - 1),
    )
    assert result.epsilon == pytest.approx(expected)
    assert result.order == 8


def test_accountant_rejects_invalid_parameters() -> None:
    accountant = PrivacyAccountant()

    with pytest.raises(ValueError, match="noise_multiplier"):
        accountant.step(noise_multiplier=0, sample_rate=0.1)

    with pytest.raises(ValueError, match="sample_rate"):
        accountant.step(noise_multiplier=1, sample_rate=0)

    with pytest.raises(ValueError, match="delta"):
        accountant.get_epsilon(delta=1)

    with pytest.raises(ValueError, match="integer"):
        PrivacyAccountant(orders=[1.5, 2])


def test_find_noise_multiplier_hits_target_epsilon() -> None:
    sigma = find_noise_multiplier(
        target_epsilon=4.0,
        delta=1e-5,
        sample_rate=0.1,
        steps=20,
        orders=range(2, 32),
        tolerance=1e-3,
    )

    accountant = PrivacyAccountant(orders=range(2, 32))
    for _ in range(20):
        accountant.step(noise_multiplier=sigma, sample_rate=0.1)

    assert accountant.get_epsilon(delta=1e-5) == pytest.approx(4.0, abs=1e-3)
