"""Bridge between PyTorch modules and NumPy strategy primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_model_parameters(model: Any) -> tuple[NDArray[np.floating], ...]:
    """Return trainable and non-trainable model state tensors as NumPy arrays."""

    return tuple(
        tensor.detach().cpu().numpy().copy()
        for tensor in model.state_dict().values()
    )


def set_model_parameters(model: Any, parameters: Sequence[NDArray[np.floating]]) -> None:
    """Load NumPy arrays into a model state dict in order."""

    torch = _require_torch()
    state_dict = model.state_dict()
    if len(parameters) != len(state_dict):
        raise ValueError("parameter count does not match model state")

    updated_state = {}
    for (name, tensor), array in zip(state_dict.items(), parameters):
        if tuple(array.shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for parameter {name}")
        updated_state[name] = torch.as_tensor(array, dtype=tensor.dtype, device=tensor.device)
    model.load_state_dict(updated_state)


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        raise RuntimeError("PyTorch is required for model parameter conversion") from exc
    return torch
