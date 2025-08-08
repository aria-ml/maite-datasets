from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

import numpy as np

from maite_datasets._protocols import ArrayLike, ObjectDetectionDataset

_logger = logging.getLogger(__name__)


class _ObjectDetectionTarget:
    """Internal implementation of ObjectDetectionTarget protocol."""

    def __init__(self, boxes: ArrayLike, labels: ArrayLike, scores: ArrayLike) -> None:
        self._boxes = np.asarray(boxes)
        self._labels = np.asarray(labels)
        self._scores = np.asarray(scores)

    @property
    def boxes(self) -> ArrayLike:
        return self._boxes

    @property
    def labels(self) -> ArrayLike:
        return self._labels

    @property
    def scores(self) -> ArrayLike:
        return self._scores


class BaseDatasetReader(ABC):
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
        self.dataset_path = Path(dataset_path)
        self._dataset_id = dataset_id or self.dataset_path.name

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
    def _create_dataset_implementation(self) -> ObjectDetectionDataset:
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

    def get_dataset(self) -> ObjectDetectionDataset:
        """
        Get dataset conforming to MAITE ObjectDetectionDataset protocol.

        Returns
        -------
        ObjectDetectionDataset
            Dataset instance with MAITE-compatible interface
        """
        return self._create_dataset_implementation()
