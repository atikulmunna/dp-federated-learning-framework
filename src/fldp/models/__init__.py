"""Model factories used by FL experiments."""

from fldp.models.cnn import count_trainable_parameters, create_cifar10_cnn, create_mnist_cnn

__all__ = [
    "count_trainable_parameters",
    "create_cifar10_cnn",
    "create_mnist_cnn",
]
