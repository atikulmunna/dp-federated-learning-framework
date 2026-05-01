"""Client partitioning utilities for federated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


IndexArray = NDArray[np.int64]


@dataclass(frozen=True)
class PartitionResult:
    """A deterministic partition of dataset indices across clients."""

    clients: tuple[IndexArray, ...]
    redraws: int = 0

    @property
    def num_clients(self) -> int:
        return len(self.clients)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(client) for client in self.clients)


def iid_partition(
    num_samples: int,
    num_clients: int,
    *,
    seed: int,
    min_client_size: int = 0,
) -> PartitionResult:
    """Split sample indices evenly after a deterministic shuffle."""

    _validate_common(num_clients, min_client_size)
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if num_samples < num_clients * min_client_size:
        raise ValueError("num_samples cannot satisfy min_client_size")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples).astype(np.int64)
    clients = tuple(split.astype(np.int64) for split in np.array_split(indices, num_clients))

    if any(len(client) < min_client_size for client in clients):
        raise ValueError("num_samples cannot satisfy min_client_size")

    return PartitionResult(clients=clients)


def dirichlet_partition(
    labels: Iterable[int],
    num_clients: int,
    *,
    alpha: float,
    seed: int,
    min_client_size: int = 1,
    max_redraws: int = 100,
) -> PartitionResult:
    """Partition indices by drawing a client distribution for each class."""

    _validate_common(num_clients, min_client_size)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if max_redraws < 0:
        raise ValueError("max_redraws must be non-negative")

    label_array = _as_label_array(labels)
    if len(label_array) < num_clients * min_client_size:
        raise ValueError("labels cannot satisfy min_client_size")

    rng = np.random.default_rng(seed)
    classes = np.unique(label_array)

    for redraw in range(max_redraws + 1):
        client_chunks: list[list[int]] = [[] for _ in range(num_clients)]

        for class_id in classes:
            class_indices = np.flatnonzero(label_array == class_id).astype(np.int64)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(class_indices), proportions)

            start = 0
            for client_id, count in enumerate(counts):
                stop = start + int(count)
                if stop > start:
                    client_chunks[client_id].extend(class_indices[start:stop].tolist())
                start = stop

        clients = tuple(np.array(chunk, dtype=np.int64) for chunk in client_chunks)
        if all(len(client) >= min_client_size for client in clients):
            for client in clients:
                rng.shuffle(client)
            return PartitionResult(clients=clients, redraws=redraw)

    raise RuntimeError(
        "could not draw a Dirichlet partition satisfying min_client_size "
        f"after {max_redraws + 1} attempts"
    )


def class_histograms(
    labels: Iterable[int],
    partition: PartitionResult,
    *,
    num_classes: int | None = None,
) -> NDArray[np.int64]:
    """Return a clients-by-classes count matrix."""

    label_array = _as_label_array(labels)
    if num_classes is None:
        num_classes = int(label_array.max()) + 1 if len(label_array) else 0
    if num_classes < 0:
        raise ValueError("num_classes must be non-negative")

    histograms = np.zeros((partition.num_clients, num_classes), dtype=np.int64)
    for client_id, indices in enumerate(partition.clients):
        _validate_indices(indices, len(label_array))
        if len(indices) == 0:
            continue
        histograms[client_id] = np.bincount(
            label_array[indices],
            minlength=num_classes,
        )[:num_classes]

    return histograms


def class_entropy(histograms: NDArray[np.integer]) -> NDArray[np.float64]:
    """Compute base-2 class entropy for each client histogram row."""

    counts = np.asarray(histograms, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("histograms must be a 2D array")

    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log2(probabilities), 0.0)
    return -terms.sum(axis=1)


def validate_complete_coverage(partition: PartitionResult, num_samples: int) -> None:
    """Raise if a partition omits or duplicates any sample index."""

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if partition.num_clients == 0:
        raise ValueError("partition must contain at least one client")

    concatenated = np.concatenate(partition.clients) if partition.clients else np.array([], dtype=np.int64)
    if len(concatenated) != num_samples:
        raise ValueError("partition does not contain exactly num_samples indices")
    _validate_indices(concatenated, num_samples)

    unique = np.unique(concatenated)
    if len(unique) != num_samples:
        raise ValueError("partition contains duplicate indices")
    if not np.array_equal(unique, np.arange(num_samples, dtype=np.int64)):
        raise ValueError("partition does not cover all sample indices")


def _validate_common(num_clients: int, min_client_size: int) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if min_client_size < 0:
        raise ValueError("min_client_size must be non-negative")


def _as_label_array(labels: Iterable[int]) -> IndexArray:
    label_array = np.asarray(list(labels), dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(label_array) and label_array.min() < 0:
        raise ValueError("labels must be non-negative integers")
    return label_array


def _validate_indices(indices: IndexArray, num_samples: int) -> None:
    if len(indices) == 0:
        return
    if indices.min() < 0 or indices.max() >= num_samples:
        raise ValueError("partition index out of bounds")
