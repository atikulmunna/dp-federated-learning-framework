"""Dataset helpers and deterministic split construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from fldp.data.partitions import PartitionResult, validate_complete_coverage


IndexArray = NDArray[np.int64]


class SizedDataset(Protocol):
    """Minimal dataset protocol shared by torchvision and test datasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


@dataclass(frozen=True)
class TrainValidationSplit:
    """Stable train/validation index split derived from a training dataset."""

    train_indices: IndexArray
    validation_indices: IndexArray


class IndexedSubset:
    """A small dependency-free equivalent of torch.utils.data.Subset."""

    def __init__(self, dataset: SizedDataset, indices: Sequence[int] | IndexArray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        _validate_subset_indices(self.indices, len(dataset))

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> Any:
        return self.dataset[int(self.indices[index])]


@dataclass(frozen=True)
class ClientDatasets:
    """Client-local datasets plus the shared validation dataset."""

    clients: tuple[IndexedSubset, ...]
    validation: IndexedSubset

    @property
    def num_clients(self) -> int:
        return len(self.clients)


def make_train_validation_split(
    num_samples: int,
    *,
    validation_fraction: float,
    seed: int,
) -> TrainValidationSplit:
    """Create deterministic disjoint train and validation indices."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    validation_size = int(round(num_samples * validation_fraction))
    validation_size = min(max(validation_size, 1), num_samples - 1)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_samples).astype(np.int64)
    validation_indices = np.sort(shuffled[:validation_size]).astype(np.int64)
    train_indices = np.sort(shuffled[validation_size:]).astype(np.int64)

    return TrainValidationSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def make_client_datasets(
    dataset: SizedDataset,
    split: TrainValidationSplit,
    partition: PartitionResult,
) -> ClientDatasets:
    """Build client subsets from train-relative partition indices."""

    validate_complete_coverage(partition, len(split.train_indices))

    clients = tuple(
        IndexedSubset(dataset, split.train_indices[client_indices])
        for client_indices in partition.clients
    )
    validation = IndexedSubset(dataset, split.validation_indices)

    return ClientDatasets(clients=clients, validation=validation)


def make_dataloader(
    dataset: SizedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    num_workers: int = 0,
) -> Any:
    """Create a PyTorch DataLoader lazily, keeping imports optional for tests."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover - exercised in ML environment setup
        raise RuntimeError("PyTorch is required to create DataLoader instances") from exc

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def load_vision_dataset(
    name: str,
    *,
    root: str | Path = "data",
    train: bool,
    download: bool = False,
) -> SizedDataset:
    """Load a supported torchvision dataset with the default tensor transform."""

    dataset_name = name.lower()
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - exercised in ML environment setup
        raise RuntimeError("torchvision is required to load vision datasets") from exc

    transform = transforms.ToTensor()
    root_path = str(root)

    if dataset_name == "mnist":
        return datasets.MNIST(root=root_path, train=train, download=download, transform=transform)
    if dataset_name in {"cifar10", "cifar-10"}:
        return datasets.CIFAR10(root=root_path, train=train, download=download, transform=transform)

    raise ValueError(f"unsupported dataset: {name}")


def _validate_subset_indices(indices: IndexArray, dataset_size: int) -> None:
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if len(indices) == 0:
        return
    if indices.min() < 0 or indices.max() >= dataset_size:
        raise ValueError("subset index out of bounds")
