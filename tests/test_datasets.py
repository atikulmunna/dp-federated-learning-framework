import numpy as np
import pytest

from fldp.data.datasets import (
    IndexedSubset,
    load_vision_dataset,
    make_client_datasets,
    make_dataloader,
    make_train_validation_split,
)
from fldp.data.partitions import iid_partition


class ToyDataset:
    def __init__(self, size: int) -> None:
        self.items = [(index, index % 3) for index in range(size)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.items[index]


def test_train_validation_split_is_deterministic_and_disjoint() -> None:
    first = make_train_validation_split(100, validation_fraction=0.1, seed=12)
    second = make_train_validation_split(100, validation_fraction=0.1, seed=12)

    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.validation_indices, second.validation_indices)
    assert len(first.validation_indices) == 10
    assert set(first.train_indices).isdisjoint(set(first.validation_indices))
    assert sorted(np.concatenate([first.train_indices, first.validation_indices])) == list(range(100))


def test_train_validation_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        make_train_validation_split(10, validation_fraction=0, seed=1)

    with pytest.raises(ValueError, match="validation_fraction"):
        make_train_validation_split(10, validation_fraction=1, seed=1)


def test_indexed_subset_maps_to_original_dataset() -> None:
    dataset = ToyDataset(5)
    subset = IndexedSubset(dataset, [4, 1, 3])

    assert len(subset) == 3
    assert subset[0] == (4, 1)
    assert subset[1] == (1, 1)


def test_indexed_subset_rejects_out_of_bounds_indices() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        IndexedSubset(ToyDataset(3), [0, 3])


def test_make_client_datasets_uses_train_relative_partition_indices() -> None:
    dataset = ToyDataset(20)
    split = make_train_validation_split(20, validation_fraction=0.25, seed=5)
    partition = iid_partition(len(split.train_indices), 3, seed=8, min_client_size=1)

    client_datasets = make_client_datasets(dataset, split, partition)
    client_items = [item for client in client_datasets.clients for item in client]
    client_indices = sorted(index for index, _ in client_items)

    assert client_datasets.num_clients == 3
    assert client_indices == sorted(split.train_indices.tolist())
    assert [item[0] for item in client_datasets.validation] == split.validation_indices.tolist()


def test_make_dataloader_reports_missing_torch_cleanly() -> None:
    try:
        loader = make_dataloader(ToyDataset(4), batch_size=2, shuffle=False, seed=1)
    except RuntimeError as exc:
        assert "PyTorch is required" in str(exc)
    else:
        assert len(list(loader)) == 2


def test_load_vision_dataset_rejects_unknown_dataset_without_torchvision() -> None:
    with pytest.raises((ValueError, RuntimeError)):
        load_vision_dataset("unknown", train=True)
