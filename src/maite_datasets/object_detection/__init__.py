"""Module for MAITE compliant Object Detection datasets."""

from maite_datasets.object_detection._antiuav import AntiUAVDetection
from maite_datasets.object_detection._coco import COCODatasetReader
from maite_datasets.object_detection._milco import MILCO
from maite_datasets.object_detection._seadrone import SeaDrone
from maite_datasets.object_detection._voc import VOCDetection
from maite_datasets.object_detection._yolo import YOLODatasetReader

__all__ = [
    "AntiUAVDetection",
    "MILCO",
    "SeaDrone",
    "VOCDetection",
    "COCODatasetReader",
    "YOLODatasetReader",
]
