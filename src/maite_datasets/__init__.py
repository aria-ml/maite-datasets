"""Module for MAITE compliant Computer Vision datasets."""

from maite_datasets._builder import to_image_classification_dataset, to_object_detection_dataset
from maite_datasets._collate import collate_as_list, collate_as_numpy, collate_as_torch
from maite_datasets._reader import create_dataset_reader
from maite_datasets._validate import validate_dataset

__all__ = [
    "collate_as_list",
    "collate_as_numpy",
    "collate_as_torch",
    "create_dataset_reader",
    "to_image_classification_dataset",
    "to_object_detection_dataset",
    "validate_dataset",
]
