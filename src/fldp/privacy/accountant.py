"""RDP accountant for the sampled Gaussian mechanism."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_ORDERS: tuple[float, ...] = tuple(float(order) for order in range(2, 65))


@dataclass(frozen=True)
class EpsilonResult:
    """Converted epsilon and the RDP order that minimized it."""

    epsilon: float
    order: float


class PrivacyAccountant:
    """Compose sampled Gaussian RDP events across federation rounds."""

    def __init__(self, orders: Sequence[float] = DEFAULT_ORDERS) -> None:
        if len(orders) == 0:
            raise ValueError("orders must not be empty")
        order_array = np.asarray(orders, dtype=np.float64)
        if np.any(order_array <= 1):
            raise ValueError("all RDP orders must be greater than 1")
        if np.any(order_array != np.floor(order_array)):
            raise ValueError("the pure accountant currently supports integer RDP orders only")

        self._orders = order_array
        self._rdp = np.zeros_like(order_array, dtype=np.float64)
        self._steps: list[tuple[float, float]] = []

    @property
    def orders(self) -> NDArray[np.float64]:
        return self._orders.copy()

    @property
    def steps(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._steps)

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    def step(self, *, noise_multiplier: float, sample_rate: float) -> None:
        """Compose one sampled Gaussian mechanism event."""

        _validate_event(noise_multiplier, sample_rate)
        event_rdp = np.array(
            [
                _compute_sampled_gaussian_rdp(
                    order=int(order),
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                )
                for order in self._orders
            ],
            dtype=np.float64,
        )
        self._rdp += event_rdp
        self._steps.append((float(noise_multiplier), float(sample_rate)))

    def get_rdp(self) -> NDArray[np.float64]:
        """Return the composed RDP curve."""

        return self._rdp.copy()

    def get_epsilon(self, *, delta: float) -> float:
        """Return cumulative epsilon at the configured delta."""

        return self.get_epsilon_result(delta=delta).epsilon

    def get_epsilon_result(self, *, delta: float) -> EpsilonResult:
        """Return cumulative epsilon and the best RDP order."""

        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1")
        epsilons = self._rdp + log(1 / delta) / (self._orders - 1)
        best = int(np.argmin(epsilons))
        return EpsilonResult(epsilon=float(epsilons[best]), order=float(self._orders[best]))

    def reset(self) -> None:
        """Clear all composed events."""

        self._rdp.fill(0.0)
        self._steps.clear()


def _validate_event(noise_multiplier: float, sample_rate: float) -> None:
    if noise_multiplier <= 0 or not isfinite(noise_multiplier):
        raise ValueError("noise_multiplier must be positive and finite")
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")


def _compute_sampled_gaussian_rdp(
    *,
    order: int,
    noise_multiplier: float,
    sample_rate: float,
) -> float:
    if sample_rate == 1:
        return order / (2 * noise_multiplier**2)

    log_terms = []
    for i in range(order + 1):
        log_coef = _log_binomial(order, i)
        log_prob = i * log(sample_rate) + (order - i) * log(1 - sample_rate)
        log_moment = (i * i - i) / (2 * noise_multiplier**2)
        log_terms.append(log_coef + log_prob + log_moment)

    return _logsumexp(log_terms) / (order - 1)


def _log_binomial(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _logsumexp(values: Sequence[float]) -> float:
    max_value = max(values)
    if max_value == -float("inf"):
        return max_value
    return max_value + log(sum(exp(value - max_value) for value in values))
