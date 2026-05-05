"""Federated aggregation strategies and parameter utilities."""

from fldp.strategies.dp_fedavg import DPFedAvgResult, aggregate_dpfedavg
from fldp.strategies.fedavg import (
    AggregationResult,
    ClientUpdate,
    aggregate_fedavg,
    sample_cohort,
)
from fldp.strategies.parameters import (
    ArrayList,
    add_gaussian_noise,
    apply_delta,
    clip_by_l2_norm,
    l2_norm,
    mean_arrays,
    params_to_delta,
)

__all__ = [
    "AggregationResult",
    "ArrayList",
    "ClientUpdate",
    "DPFedAvgResult",
    "add_gaussian_noise",
    "aggregate_dpfedavg",
    "aggregate_fedavg",
    "apply_delta",
    "clip_by_l2_norm",
    "l2_norm",
    "mean_arrays",
    "params_to_delta",
    "sample_cohort",
]
