"""Inverse routines for target privacy budgets."""

from __future__ import annotations

from collections.abc import Sequence

from fldp.privacy.accountant import DEFAULT_ORDERS, PrivacyAccountant


def find_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    orders: Sequence[float] = DEFAULT_ORDERS,
    tolerance: float = 1e-3,
    max_iterations: int = 80,
) -> float:
    """Find the Gaussian noise multiplier needed for a target final epsilon."""

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    low = 1e-6
    high = 1.0
    while _epsilon_for_sigma(
        high,
        delta=delta,
        sample_rate=sample_rate,
        steps=steps,
        orders=orders,
    ) > target_epsilon:
        high *= 2
        if high > 1e6:
            raise RuntimeError("could not bracket target_epsilon")

    for _ in range(max_iterations):
        mid = (low + high) / 2
        epsilon = _epsilon_for_sigma(
            mid,
            delta=delta,
            sample_rate=sample_rate,
            steps=steps,
            orders=orders,
        )
        if abs(epsilon - target_epsilon) <= tolerance:
            return mid
        if epsilon > target_epsilon:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def _epsilon_for_sigma(
    sigma: float,
    *,
    delta: float,
    sample_rate: float,
    steps: int,
    orders: Sequence[float],
) -> float:
    accountant = PrivacyAccountant(orders=orders)
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta)
