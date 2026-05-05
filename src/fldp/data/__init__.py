"""Dataset loading and client partitioning utilities."""

from fldp.data.datasets import (
    ClientDatasets,
    IndexedSubset,
    TrainValidationSplit,
    load_vision_dataset,
    make_client_datasets,
    make_dataloader,
    make_train_validation_split,
)
from fldp.data.partitions import (
    PartitionResult,
    class_entropy,
    class_histograms,
    dirichlet_partition,
    iid_partition,
    validate_complete_coverage,
)

__all__ = [
    "ClientDatasets",
    "IndexedSubset",
    "PartitionResult",
    "TrainValidationSplit",
    "class_entropy",
    "class_histograms",
    "dirichlet_partition",
    "iid_partition",
    "load_vision_dataset",
    "make_client_datasets",
    "make_dataloader",
    "make_train_validation_split",
    "validate_complete_coverage",
]
