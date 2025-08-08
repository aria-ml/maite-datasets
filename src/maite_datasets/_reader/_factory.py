from __future__ import annotations

import logging
from pathlib import Path

from maite_datasets._reader._base import BaseDatasetReader
from maite_datasets._reader._yolo import YOLODatasetReader
from maite_datasets._reader._coco import COCODatasetReader

_logger = logging.getLogger(__name__)


def create_dataset_reader(dataset_path: str | Path, format_hint: str | None = None) -> BaseDatasetReader:
    """
    Factory function to create appropriate dataset reader based on directory structure.

    Parameters
    ----------
    dataset_path : str or Path
        Root directory containing dataset files
    format_hint : str or None, default None
        Format hint ("coco" or "yolo"). If None, auto-detects based on file structure

    Returns
    -------
    BaseDatasetReader
        Appropriate reader instance for the detected format

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported
    """
    dataset_path = Path(dataset_path)

    if format_hint:
        format_hint = format_hint.lower()
        if format_hint == "coco":
            return COCODatasetReader(dataset_path)
        elif format_hint == "yolo":
            return YOLODatasetReader(dataset_path)
        else:
            raise ValueError(f"Unsupported format hint: {format_hint}")

    # Auto-detect format
    has_annotations_json = (dataset_path / "annotations.json").exists()
    has_labels_dir = (dataset_path / "labels").exists()

    if has_annotations_json and not has_labels_dir:
        _logger.info(f"Detected COCO format for {dataset_path}")
        return COCODatasetReader(dataset_path)
    elif has_labels_dir and not has_annotations_json:
        _logger.info(f"Detected YOLO format for {dataset_path}")
        return YOLODatasetReader(dataset_path)
    elif has_annotations_json and has_labels_dir:
        raise ValueError(
            f"Ambiguous format in {dataset_path}: both annotations.json and labels/ exist. "
            "Use format_hint parameter to specify format."
        )
    else:
        raise ValueError(
            f"Cannot detect dataset format in {dataset_path}. "
            "Expected either annotations.json (COCO) or labels/ directory (YOLO)."
        )
