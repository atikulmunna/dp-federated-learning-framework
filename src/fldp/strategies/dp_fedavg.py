"""DP-FedAvg aggregation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fldp.privacy import PrivacyAccountant
from fldp.strategies.fedavg import ClientUpdate
from fldp.strategies.parameters import (
    ArrayList,
    add_gaussian_noise,
    apply_delta,
    clip_by_l2_norm,
    l2_norm,
    mean_arrays,
)


@dataclass(frozen=True)
class DPFedAvgResult:
    """Server-side result after one DP-FedAvg aggregation round."""

    parameters: ArrayList
    average_clipped_delta: ArrayList
    noised_delta: ArrayList
    cohort_size: int
    num_total_clients: int
    sample_rate: float
    noise_std: float
    pre_clip_norms: tuple[float, ...]
    clip_scales: tuple[float, ...]
    epsilon: float | None


def aggregate_dpfedavg(
    server_parameters: Sequence[NDArray[np.floating]],
    updates: Sequence[ClientUpdate],
    *,
    clip_norm: float,
    noise_multiplier: float,
    accountant: PrivacyAccountant,
    num_total_clients: int,
    noise_rng: np.random.Generator,
    delta: float | None = None,
) -> DPFedAvgResult:
    """Clip, average, noise, and account for one DP-FedAvg round."""

    if len(updates) == 0:
        raise ValueError("updates must not be empty")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if num_total_clients <= 0:
        raise ValueError("num_total_clients must be positive")
    if len(updates) > num_total_clients:
        raise ValueError("cohort size cannot exceed num_total_clients")
    for update in updates:
        if update.num_examples <= 0:
            raise ValueError("client updates must contain at least one example")

    raw_deltas = [update.delta() for update in updates]
    pre_clip_norms = tuple(l2_norm(delta) for delta in raw_deltas)
    clip_scales = tuple(min(1.0, clip_norm / (norm + 1e-12)) for norm in pre_clip_norms)
    clipped_deltas = [clip_by_l2_norm(delta, clip_norm) for delta in raw_deltas]

    average_clipped_delta = mean_arrays(clipped_deltas)
    cohort_size = len(updates)
    sample_rate = cohort_size / num_total_clients
    noise_std = noise_multiplier * clip_norm / cohort_size
    noised_delta = add_gaussian_noise(average_clipped_delta, std=noise_std, rng=noise_rng)

    accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    epsilon = accountant.get_epsilon(delta=delta) if delta is not None else None

    return DPFedAvgResult(
        parameters=apply_delta(server_parameters, noised_delta),
        average_clipped_delta=average_clipped_delta,
        noised_delta=noised_delta,
        cohort_size=cohort_size,
        num_total_clients=num_total_clients,
        sample_rate=sample_rate,
        noise_std=noise_std,
        pre_clip_norms=pre_clip_norms,
        clip_scales=clip_scales,
        epsilon=epsilon,
    )
