import pytest

from fldp.models import count_trainable_parameters, create_cifar10_cnn, create_mnist_cnn


def _torch_or_skip():
    try:
        import torch
        from torch import nn
    except Exception as exc:
        pytest.skip(f"PyTorch is not available in this environment: {exc}")
    return torch, nn


def test_mnist_cnn_forward_shape_and_size() -> None:
    torch, _ = _torch_or_skip()
    model = create_mnist_cnn()

    output = model(torch.zeros(4, 1, 28, 28))

    assert tuple(output.shape) == (4, 10)
    assert 20_000 <= count_trainable_parameters(model) <= 120_000


def test_cifar10_cnn_forward_shape_size_and_no_batchnorm() -> None:
    torch, nn = _torch_or_skip()
    model = create_cifar10_cnn()

    output = model(torch.zeros(4, 3, 32, 32))

    assert tuple(output.shape) == (4, 10)
    assert 250_000 <= count_trainable_parameters(model) <= 1_000_000
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
