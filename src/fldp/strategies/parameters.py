"""Utilities for manipulating model parameter arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


ArrayList = list[NDArray[np.floating]]


def params_to_delta(
    initial: Sequence[NDArray[np.floating]],
    updated: Sequence[NDArray[np.floating]],
) -> ArrayList:
    """Return ``updated - initial`` for a model parameter list."""

    _validate_compatible(initial, updated)
    return [np.asarray(new) - np.asarray(old) for old, new in zip(initial, updated)]


def apply_delta(
    initial: Sequence[NDArray[np.floating]],
    delta: Sequence[NDArray[np.floating]],
) -> ArrayList:
    """Apply a parameter delta to an initial parameter list."""

    _validate_compatible(initial, delta)
    return [np.asarray(param) + np.asarray(update) for param, update in zip(initial, delta)]


def l2_norm(arrays: Sequence[NDArray[np.floating]]) -> float:
    """Compute the global L2 norm across a list of arrays."""

    _validate_non_empty(arrays)
    squared_sum = sum(float(np.sum(np.asarray(array, dtype=np.float64) ** 2)) for array in arrays)
    return float(np.sqrt(squared_sum))


def clip_by_l2_norm(
    arrays: Sequence[NDArray[np.floating]],
    clip_norm: float,
) -> ArrayList:
    """Clip a list of arrays to a global L2 norm."""

    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    norm = l2_norm(arrays)
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return [np.asarray(array) * scale for array in arrays]


def mean_arrays(
    array_lists: Sequence[Sequence[NDArray[np.floating]]],
    *,
    weights: Sequence[float] | None = None,
) -> ArrayList:
    """Compute a uniform or weighted mean over compatible array lists."""

    if len(array_lists) == 0:
        raise ValueError("array_lists must not be empty")
    reference = array_lists[0]
    _validate_non_empty(reference)
    for arrays in array_lists[1:]:
        _validate_compatible(reference, arrays)

    normalized_weights = _normalize_weights(len(array_lists), weights)
    result: ArrayList = []
    for layer_index in range(len(reference)):
        layer = sum(
            normalized_weights[item_index] * np.asarray(arrays[layer_index])
            for item_index, arrays in enumerate(array_lists)
        )
        result.append(layer)
    return result


def add_gaussian_noise(
    arrays: Sequence[NDArray[np.floating]],
    *,
    std: float,
    rng: np.random.Generator,
) -> ArrayList:
    """Add independent Gaussian noise to each array."""

    if std < 0:
        raise ValueError("std must be non-negative")
    _validate_non_empty(arrays)
    return [
        np.asarray(array) + rng.normal(loc=0.0, scale=std, size=np.asarray(array).shape)
        for array in arrays
    ]


def _normalize_weights(count: int, weights: Sequence[float] | None) -> NDArray[np.float64]:
    if weights is None:
        return np.full(count, 1 / count, dtype=np.float64)

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (count,):
        raise ValueError("weights must match array_lists length")
    if np.any(weight_array < 0):
        raise ValueError("weights must be non-negative")
    total = float(weight_array.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return weight_array / total


def _validate_compatible(
    left: Sequence[NDArray[np.floating]],
    right: Sequence[NDArray[np.floating]],
) -> None:
    _validate_non_empty(left)
    _validate_non_empty(right)
    if len(left) != len(right):
        raise ValueError("array lists must have the same length")
    for left_array, right_array in zip(left, right):
        if np.asarray(left_array).shape != np.asarray(right_array).shape:
            raise ValueError("array shapes must match")


def _validate_non_empty(arrays: Sequence[NDArray[np.floating]]) -> None:
    if len(arrays) == 0:
        raise ValueError("array list must not be empty")
