"""Module for MAITE compliant Object Detection datasets."""

from maite_datasets.object_detection._antiuav import AntiUAVDetection
from maite_datasets.object_detection._milco import MILCO
from maite_datasets.object_detection._seadrone import SeaDrone
from maite_datasets.object_detection._voc import VOCDetection

__all__ = [
    "AntiUAVDetection",
    "MILCO",
    "SeaDrone",
    "VOCDetection",
]

import importlib.util

if importlib.util.find_spec("torch") is not None:
    from maite_datasets.object_detection._voc_torch import VOCDetectionTorch

    __all__ += ["VOCDetectionTorch"]
