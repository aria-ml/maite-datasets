from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od

_logger = logging.getLogger(__name__)

_TDataset = TypeVar("_TDataset", ic.Dataset, od.Dataset)


class BaseDatasetReader(Generic[_TDataset], ABC):
    """
    Abstract base class for object detection dataset readers.

    Provides common functionality for dataset path handling, validation,
    and dataset creation while allowing format-specific implementations.

    Parameters
    ----------
    dataset_path : str or Path
        Root directory containing dataset files
    dataset_id : str or None, default None
        Dataset identifier. If None, uses dataset_path name
    """

    def __init__(self, dataset_path: str | Path, dataset_id: str | None = None) -> None:
        self.dataset_path: Path = Path(dataset_path)
        self.dataset_id: str = dataset_id or self.dataset_path.name

        # Basic path validation
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        # Format-specific initialization
        self._initialize_format_specific()

    @abstractmethod
    def _initialize_format_specific(self) -> None:
        """Initialize format-specific components (annotations, classes, etc.)."""
        pass

    @abstractmethod
    def create_dataset(self) -> _TDataset:
        """Create the format-specific dataset implementation."""
        pass

    @abstractmethod
    def _validate_format_specific(self) -> tuple[list[str], dict[str, Any]]:
        """Validate format-specific structure and return issues and stats."""
        pass

    @property
    @abstractmethod
    def index2label(self) -> dict[int, str]:
        """Mapping from class index to class name."""
        pass

    def _validate_images_directory(self) -> tuple[list[str], dict[str, Any]]:
        """Validate images directory and return issues and stats."""
        issues = []
        stats = {}

        images_path = self.dataset_path / "images"
        if not images_path.exists():
            issues.append("Missing images/ directory")
            return issues, stats

        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))

        stats["num_images"] = len(image_files)
        if len(image_files) == 0:
            issues.append("No image files found in images/ directory")

        return issues, stats

    def validate_structure(self) -> dict[str, Any]:
        """
        Validate dataset directory structure and return diagnostic information.

        Returns
        -------
        dict[str, Any]
            Validation results containing:
            - is_valid: bool indicating if structure is valid
            - issues: list of validation issues found
            - stats: dict with dataset statistics
        """
        # Validate images directory (common to all formats)
        issues, stats = self._validate_images_directory()

        # Format-specific validation
        format_issues, format_stats = self._validate_format_specific()
        issues.extend(format_issues)
        stats.update(format_stats)

        return {"is_valid": len(issues) == 0, "issues": issues, "stats": stats}


def create_dataset_reader(
    dataset_path: str | Path, format_hint: str | None = None
) -> BaseDatasetReader[ic.Dataset] | BaseDatasetReader[od.Dataset]:
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
    from maite_datasets.object_detection._coco import COCODatasetReader
    from maite_datasets.object_detection._yolo import YOLODatasetReader

    dataset_path = Path(dataset_path)

    if format_hint:
        format_hint = format_hint.lower()
        if format_hint == "coco":
            return COCODatasetReader(dataset_path)
        if format_hint == "yolo":
            return YOLODatasetReader(dataset_path)
        raise ValueError(f"Unsupported format hint: {format_hint}")

    # Auto-detect format
    has_annotations_json = (dataset_path / "annotations.json").exists()
    has_labels_dir = (dataset_path / "labels").exists()

    if has_annotations_json and not has_labels_dir:
        _logger.info(f"Detected COCO format for {dataset_path}")
        return COCODatasetReader(dataset_path)
    if has_labels_dir and not has_annotations_json:
        _logger.info(f"Detected YOLO format for {dataset_path}")
        return YOLODatasetReader(dataset_path)
    if has_annotations_json and has_labels_dir:
        raise ValueError(
            f"Ambiguous format in {dataset_path}: both annotations.json and labels/ exist. "
            "Use format_hint parameter to specify format."
        )
    raise ValueError(
        f"Cannot detect dataset format in {dataset_path}. "
        "Expected either annotations.json (COCO) or labels/ directory (YOLO)."
    )
