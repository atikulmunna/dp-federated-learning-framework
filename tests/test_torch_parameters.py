import numpy as np
import pytest

from fldp.models import create_mnist_cnn
from fldp.train import get_model_parameters, set_model_parameters


def _torch_or_skip():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"PyTorch is not available in this environment: {exc}")
    return torch


def test_model_parameter_round_trip() -> None:
    torch = _torch_or_skip()
    model = create_mnist_cnn()
    parameters = get_model_parameters(model)
    shifted = tuple(array + 0.5 for array in parameters)

    set_model_parameters(model, shifted)
    loaded = get_model_parameters(model)

    assert all(np.allclose(left, right) for left, right in zip(shifted, loaded))
    output = model(torch.zeros(2, 1, 28, 28))
    assert tuple(output.shape) == (2, 10)


def test_set_model_parameters_rejects_mismatched_count() -> None:
    model = create_mnist_cnn()
    parameters = get_model_parameters(model)

    with pytest.raises(ValueError, match="count"):
        set_model_parameters(model, parameters[:-1])


def test_set_model_parameters_rejects_mismatched_shape() -> None:
    model = create_mnist_cnn()
    parameters = list(get_model_parameters(model))
    parameters[0] = np.zeros((1,), dtype=parameters[0].dtype)

    with pytest.raises(ValueError, match="shape mismatch"):
        set_model_parameters(model, parameters)
