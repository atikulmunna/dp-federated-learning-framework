import pytest

from fldp.models import create_mnist_cnn

pytest.importorskip("flwr")

from fldp.flower import FlowerClientConfig, TorchFlowerClient  # noqa: E402


def _torch_or_skip():
    try:
        import torch
        from torch.utils.data import TensorDataset
    except Exception as exc:
        pytest.skip(f"PyTorch is not available in this environment: {exc}")
    return torch, TensorDataset


def _dataset(size: int):
    torch, TensorDataset = _torch_or_skip()
    generator = torch.Generator()
    generator.manual_seed(123)
    features = torch.randn(size, 1, 28, 28, generator=generator)
    labels = torch.arange(size) % 10
    return TensorDataset(features, labels.long())


def test_torch_flower_client_exposes_numpyclient_contract() -> None:
    client = TorchFlowerClient(
        client_id=3,
        model_factory=create_mnist_cnn,
        train_dataset=_dataset(12),
        eval_dataset=_dataset(8),
        config=FlowerClientConfig(batch_size=4, local_epochs=1, learning_rate=0.01, seed=7),
    )

    parameters = client.get_parameters({})
    updated_parameters, num_examples, metrics = client.fit(
        parameters,
        {"server_round": 1},
    )
    loss, eval_examples, eval_metrics = client.evaluate(updated_parameters, {})

    assert len(updated_parameters) == len(parameters)
    assert num_examples == 12
    assert metrics["client_id"] == "3"
    assert metrics["train_loss"] > 0
    assert loss > 0
    assert eval_examples == 8
    assert eval_metrics["client_id"] == "3"
    assert 0 <= eval_metrics["accuracy"] <= 1


def test_torch_flower_client_fit_is_deterministic_for_same_seed() -> None:
    client_a = TorchFlowerClient(
        client_id=1,
        model_factory=create_mnist_cnn,
        train_dataset=_dataset(12),
        eval_dataset=_dataset(8),
        config=FlowerClientConfig(batch_size=4, local_epochs=1, learning_rate=0.01, seed=11),
    )
    client_b = TorchFlowerClient(
        client_id=1,
        model_factory=create_mnist_cnn,
        train_dataset=_dataset(12),
        eval_dataset=_dataset(8),
        config=FlowerClientConfig(batch_size=4, local_epochs=1, learning_rate=0.01, seed=11),
    )

    parameters = client_a.get_parameters({})
    updated_a, _, metrics_a = client_a.fit(parameters, {"server_round": 2})
    updated_b, _, metrics_b = client_b.fit(parameters, {"server_round": 2})

    assert metrics_a == metrics_b
    assert all((left == right).all() for left, right in zip(updated_a, updated_b))
