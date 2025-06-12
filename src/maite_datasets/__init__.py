"""Module for MAITE compliant Computer Vision datasets."""

from maite_datasets._antiuav import AntiUAVDetection
from maite_datasets._cifar10 import CIFAR10
from maite_datasets._milco import MILCO
from maite_datasets._mnist import MNIST
from maite_datasets._seadrone import SeaDrone
from maite_datasets._ships import Ships
from maite_datasets._voc import VOCDetection, VOCSegmentation

__all__ = [
    "MNIST",
    "Ships",
    "CIFAR10",
    "AntiUAVDetection",
    "MILCO",
    "SeaDrone",
    "VOCDetection",
    "VOCSegmentation",
]
