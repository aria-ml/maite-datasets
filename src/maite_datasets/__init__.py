"""Module for MAITE compliant Computer Vision datasets."""

from maite_datasets._builder import to_image_classification_dataset, to_object_detection_dataset
from maite_datasets._collate import collate_as_torch, collate_as_numpy, collate_as_list
from maite_datasets._validate import validate_dataset
from maite_datasets._reader._factory import create_dataset_reader
from maite_datasets._reader._coco import COCODatasetReader
from maite_datasets._reader._yolo import YOLODatasetReader

__all__ = [
    "collate_as_list",
    "collate_as_numpy",
    "collate_as_torch",
    "create_dataset_reader",
    "to_image_classification_dataset",
    "to_object_detection_dataset",
    "validate_dataset",
    "COCODatasetReader",
    "YOLODatasetReader",
]
