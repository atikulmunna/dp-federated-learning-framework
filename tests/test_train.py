import random

import numpy as np
import pytest

from fldp.models import create_mnist_cnn
from fldp.train import evaluate, seed_everything, train_one_epoch


def _torch_or_skip():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        pytest.skip(f"PyTorch is not available in this environment: {exc}")
    return torch, DataLoader, TensorDataset


def test_seed_everything_controls_python_and_numpy_rngs() -> None:
    seed_everything(123)
    first_random = random.random()
    first_numpy = np.random.rand(3)

    seed_everything(123)

    assert random.random() == first_random
    assert np.allclose(np.random.rand(3), first_numpy)


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)


def test_train_one_epoch_and_evaluate_smoke() -> None:
    torch, DataLoader, TensorDataset = _torch_or_skip()
    seed_everything(9)

    features = torch.randn(12, 1, 28, 28)
    targets = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(features, targets), batch_size=4, shuffle=False)
    model = create_mnist_cnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_metrics = train_one_epoch(model, loader, optimizer)
    eval_metrics = evaluate(model, loader)

    assert train_metrics.num_examples == 12
    assert train_metrics.loss > 0
    assert eval_metrics.num_examples == 12
    assert eval_metrics.loss > 0
    assert 0 <= eval_metrics.accuracy <= 1
