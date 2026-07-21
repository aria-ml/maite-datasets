from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od

from maite_datasets._base import ReaderTransforms

_logger = logging.getLogger(__name__)

_TDataset = TypeVar("_TDataset", ic.Dataset, od.Dataset)

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_IMAGES_DIR = "images"
DEFAULT_LABELS_DIR = "labels"
DEFAULT_CLASSES_FILE = "classes.txt"
DEFAULT_ANNOTATION_FILE = "annotations.json"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :meth:`BaseDatasetReader.validate_structure`.

    Truthy when the structure is valid, so ``if reader.validate_structure():``
    reads naturally.
    """

    issues: list[str]
    """Human-readable descriptions of every problem found."""
    stats: dict[str, Any]
    """Counts describing the dataset, keyed by format-specific names."""

    def __bool__(self) -> bool:
        return not self.issues


def _read_index2label(path: Path) -> dict[int, str]:
    """Read a one-class-per-line file into an index -> class name mapping."""
    with open(path) as f:
        return dict(enumerate(line.strip() for line in f if line.strip()))


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

    _image_extensions: set[str] = set(DEFAULT_IMAGE_EXTENSIONS)

    def __init__(self, dataset_path: str | Path, dataset_id: str | None = None) -> None:
        self.dataset_path: Path = Path(dataset_path)
        self.dataset_id: str = dataset_id or self.dataset_path.name
        self._index2label: dict[int, str] = {}

        # Basic path validation
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        # Format-specific initialization
        self._initialize_format_specific()

    @property
    def index2label(self) -> dict[int, str]:
        """Mapping from class index to class name."""
        return self._index2label

    @classmethod
    @abstractmethod
    def can_read(cls, dataset_path: str | Path) -> bool:
        """Whether this reader recognizes the layout at `dataset_path`.

        Used by :func:`create_dataset_reader` to auto-detect the format, so each
        format owns the knowledge of what its own directory layouts look like.
        """
        pass

    @abstractmethod
    def _initialize_format_specific(self) -> None:
        """Initialize format-specific components (annotations, classes, etc.)."""
        pass

    @abstractmethod
    def create_dataset(self, lazy: bool = False, transforms: ReaderTransforms = None) -> _TDataset:
        """Create the format-specific dataset implementation.

        Parameters
        ----------
        lazy : bool, default False
            When True, the returned dataset defers per-item image decode
            until first numpy access (see :class:`maite_datasets._lazy.LazyArray`).
        transforms : ReaderTransforms, default None
            Optional image-only or datum-tuple transform(s) applied to each datum
            on access, following the same pipeline as the downloadable datasets.
        """
        pass

    @abstractmethod
    def _validate_format_specific(self) -> tuple[list[str], dict[str, Any]]:
        """Validate format-specific structure and return issues and stats."""
        pass

    def _image_directories(self) -> list[Path]:
        """Directories holding this reader's images.

        Overridden by formats that resolve their image directories from config
        (split layouts, custom directory names) rather than the default location.
        """
        return [self.dataset_path / DEFAULT_IMAGES_DIR]

    def _iter_images(self, images_path: Path) -> Iterator[Path]:
        """Yield the supported image files in `images_path` with a single directory scan."""
        return (p for p in images_path.iterdir() if p.suffix.lower() in self._image_extensions)

    def _display(self, path: Path) -> str:
        """Render `path` relative to the dataset root when possible, for messages."""
        try:
            return str(path.relative_to(self.dataset_path))
        except ValueError:
            return str(path)

    def _validate_images_directory(self) -> tuple[list[str], dict[str, Any]]:
        """Validate images directories and return issues and stats."""
        issues: list[str] = []
        num_images = 0

        for images_path in self._image_directories():
            if not images_path.exists():
                issues.append(f"Missing {self._display(images_path)}/ directory")
                continue
            count = sum(1 for _ in self._iter_images(images_path))
            if count == 0:
                issues.append(f"No image files found in {self._display(images_path)}/ directory")
            num_images += count

        return issues, {"num_images": num_images}

    def validate_structure(self) -> ValidationResult:
        """
        Validate dataset directory structure and return diagnostic information.

        Returns
        -------
        ValidationResult
            Truthy when valid; carries `issues` and `stats`.
        """
        # Validate images directory (common to all formats)
        issues, stats = self._validate_images_directory()

        # Format-specific validation. Shared keys win so a format cannot silently
        # redefine what the common layer already measured.
        format_issues, format_stats = self._validate_format_specific()
        issues.extend(format_issues)

        return ValidationResult(issues, {**format_stats, **stats})


def _readers() -> dict[str, type[BaseDatasetReader[od.Dataset]]]:
    """Registry of supported formats, imported lazily to avoid an import cycle."""
    from maite_datasets.object_detection._coco import COCODatasetReader
    from maite_datasets.object_detection._yolo import YOLODatasetReader

    return {"coco": COCODatasetReader, "yolo": YOLODatasetReader}


def create_dataset_reader(
    dataset_path: str | Path, format_hint: str | None = None, **kwargs: Any
) -> BaseDatasetReader[od.Dataset]:
    """
    Factory function to create appropriate dataset reader based on directory structure.

    Parameters
    ----------
    dataset_path : str or Path
        Root directory containing dataset files
    format_hint : str or None, default None
        Format hint ("coco" or "yolo"). If None, auto-detects by asking each reader
        whether it recognizes the layout.
    **kwargs : Any
        Additional keyword arguments forwarded to the selected reader's constructor,
        e.g. `image_set` or `data_yaml` for YOLO, `annotation_file` for COCO.

    Returns
    -------
    BaseDatasetReader
        Appropriate reader instance for the detected format

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported
    """
    readers = _readers()
    dataset_path = Path(dataset_path)

    if format_hint:
        reader = readers.get(format_hint.lower())
        if reader is None:
            raise ValueError(f"Unsupported format hint: {format_hint}")
        return reader(dataset_path, **kwargs)

    matches = [name for name, reader in readers.items() if reader.can_read(dataset_path)]
    if len(matches) == 1:
        _logger.info(f"Detected {matches[0].upper()} format for {dataset_path}")
        return readers[matches[0]](dataset_path, **kwargs)
    if matches:
        raise ValueError(
            f"Ambiguous format in {dataset_path}: {', '.join(sorted(matches))} layouts both match. "
            "Use format_hint parameter to specify format."
        )
    raise ValueError(f"Cannot detect dataset format in {dataset_path}. Expected one of: {', '.join(sorted(readers))}.")
