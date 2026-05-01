import numpy as np
import pytest

from fldp.data.partitions import (
    PartitionResult,
    class_entropy,
    class_histograms,
    dirichlet_partition,
    iid_partition,
    validate_complete_coverage,
)


def test_iid_partition_is_deterministic_and_balanced() -> None:
    first = iid_partition(103, 10, seed=42)
    second = iid_partition(103, 10, seed=42)

    assert first.sizes == second.sizes
    assert all(np.array_equal(a, b) for a, b in zip(first.clients, second.clients))
    assert max(first.sizes) - min(first.sizes) <= 1
    validate_complete_coverage(first, 103)


def test_iid_partition_rejects_impossible_min_size() -> None:
    with pytest.raises(ValueError, match="min_client_size"):
        iid_partition(9, 10, seed=1, min_client_size=1)


def test_dirichlet_partition_is_deterministic_and_complete() -> None:
    labels = np.repeat(np.arange(10), 50)

    first = dirichlet_partition(labels, 20, alpha=0.5, seed=7, min_client_size=1)
    second = dirichlet_partition(labels, 20, alpha=0.5, seed=7, min_client_size=1)

    assert first.redraws == second.redraws
    assert all(np.array_equal(a, b) for a, b in zip(first.clients, second.clients))
    assert min(first.sizes) >= 1
    validate_complete_coverage(first, len(labels))


def test_dirichlet_partition_creates_skewed_clients_for_small_alpha() -> None:
    labels = np.repeat(np.arange(10), 100)

    partition = dirichlet_partition(labels, 20, alpha=0.05, seed=11, min_client_size=1)
    histograms = class_histograms(labels, partition, num_classes=10)
    dominant_fraction = (histograms.max(axis=1) / histograms.sum(axis=1)).max()

    assert dominant_fraction > 0.70


def test_dirichlet_partition_rejects_invalid_parameters() -> None:
    labels = [0, 1, 2]

    with pytest.raises(ValueError, match="alpha"):
        dirichlet_partition(labels, 2, alpha=0, seed=1)

    with pytest.raises(ValueError, match="min_client_size"):
        dirichlet_partition(labels, 4, alpha=0.5, seed=1, min_client_size=1)


def test_class_histograms_and_entropy() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    partition = PartitionResult(
        clients=(
            np.array([0, 2, 4], dtype=np.int64),
            np.array([1, 3, 5], dtype=np.int64),
        )
    )

    histograms = class_histograms(labels, partition, num_classes=3)

    assert histograms.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert np.allclose(class_entropy(histograms), np.log2(3))


def test_validate_complete_coverage_rejects_duplicates_and_omissions() -> None:
    duplicate = PartitionResult(
        clients=(
            np.array([0, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        )
    )

    with pytest.raises(ValueError, match="duplicate|cover"):
        validate_complete_coverage(duplicate, 4)
