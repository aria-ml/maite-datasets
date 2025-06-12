"""Module for MAITE compliant Image Classification datasets."""

from maite_datasets.image_classification._cifar10 import CIFAR10
from maite_datasets.image_classification._mnist import MNIST
from maite_datasets.image_classification._ships import Ships

__all__ = [
    "CIFAR10",
    "MNIST",
    "Ships",
]
