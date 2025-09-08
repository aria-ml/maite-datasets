"""Dataset reader for YOLO detection format."""

from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od
import numpy as np
from maite.protocols import DatasetMetadata, DatumMetadata
from PIL import Image

from maite_datasets._base import BaseDataset, ObjectDetectionTargetTuple
from maite_datasets._reader import BaseDatasetReader


class YOLODatasetReader(BaseDatasetReader[od.Dataset]):
    """
    YOLO format dataset reader conforming to MAITE protocols.

    Reads YOLO format object detection datasets from disk and provides
    MAITE-compatible interface.

    Directory Structure Requirements
    --------------------------------
    ```
    dataset_root/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── labels/
    │   ├── image1.txt    # YOLO format annotations
    │   ├── image2.txt
    │   └── ...
    ├── classes.txt       # Required: one class name per line
    └── data.yaml         # Optional: dataset metadata
    ```

    YOLO Format Specifications
    --------------------------
    Label file format (one line per object):
    ```
    class_id center_x center_y width height
    0 0.5 0.3 0.2 0.4
    1 0.7 0.8 0.1 0.2
    ```
    All YOLO coordinates are normalized to [0, 1] relative to image dimensions.

    classes.txt format (required, one class per line, ordered by index):
    ```
    person
    bicycle
    car
    motorcycle
    ```

    Parameters
    ----------
    dataset_path : str or Path
        Root directory containing YOLO dataset files
    images_dir : str, default "images"
        Name of directory containing images
    labels_dir : str, default "labels"
        Name of directory containing YOLO label files
    classes_file : str, default "classes.txt"
        File containing class names (one per line)
    dataset_id : str or None, default None
        Dataset identifier. If None, uses dataset_path name
    image_extensions : list[str], default [".jpg", ".jpeg", ".png", ".bmp"]
        Supported image file extensions

    Notes
    -----
    YOLO label files should contain one line per object:
    `class_id center_x center_y width height`

    All coordinates should be normalized to [0, 1] relative to image dimensions.
    Coordinates are converted to absolute pixel values and MAITE format (x1, y1, x2, y2).
    """

    def __init__(
        self,
        dataset_path: str | Path,
        images_dir: str = "images",
        labels_dir: str = "labels",
        classes_file: str = "classes.txt",
        dataset_id: str | None = None,
        image_extensions: list[str] | None = None,
    ) -> None:
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        self._classes_file = classes_file

        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        self._image_extensions = [ext.lower() for ext in image_extensions]

        # Initialize base class
        super().__init__(dataset_path, dataset_id)

    def _initialize_format_specific(self) -> None:
        """Initialize YOLO-specific components."""
        self._images_path = self.dataset_path / self._images_dir
        self._labels_path = self.dataset_path / self._labels_dir
        self._classes_path = self.dataset_path / self._classes_file

        if not self._images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self._images_path}")
        if not self._labels_path.exists():
            raise FileNotFoundError(f"Labels directory not found: {self._labels_path}")
        if not self._classes_path.exists():
            raise FileNotFoundError(f"Classes file not found: {self._classes_path}")

        self._load_class_names()
        self._find_image_files()

    @property
    def index2label(self) -> dict[int, str]:
        """Mapping from class index to class name."""
        return self._index2label

    def create_dataset(self) -> od.Dataset:
        """Create YOLO dataset implementation."""
        return YOLODataset(self)

    def _validate_format_specific(self) -> tuple[list[str], dict[str, Any]]:
        """Validate YOLO format specific files and structure."""
        issues = []
        stats = {}

        # Check labels directory
        labels_path = self.dataset_path / self._labels_dir
        if not labels_path.exists():
            issues.append(f"Missing {self._labels_dir}/ directory")
        else:
            label_files = list(labels_path.glob("*.txt"))
            stats["num_label_files"] = len(label_files)
            if len(label_files) == 0:
                issues.append(f"No label files found in {self._labels_dir}/ directory")
            else:
                # Validate label file format (sample check)
                label_issues = self._validate_yolo_label_format(labels_path)
                issues.extend(label_issues)

        # Check required classes.txt
        classes_path = self.dataset_path / self._classes_file
        if not classes_path.exists():
            issues.append(f"Missing required {self._classes_file} file")
        else:
            try:
                with open(classes_path) as f:
                    class_lines = [line.strip() for line in f if line.strip()]
                stats["num_classes"] = len(class_lines)
                if len(class_lines) == 0:
                    issues.append(f"{self._classes_file} is empty")
            except Exception as e:
                issues.append(f"Error reading {self._classes_file}: {e}")

        return issues, stats

    def _validate_yolo_label_format(self, labels_path: Path) -> list[str]:
        """Validate YOLO label file format (sample check)."""
        issues = []
        label_files = list(labels_path.glob("*.txt"))

        if not label_files:
            return issues

        label_files.sort()
        sample_label = label_files[0]
        try:
            with open(sample_label) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(
                            f"Invalid YOLO format in {sample_label.name} line {line_num}: "
                            f"expected 5 values, got {len(parts)}"
                        )
                        break

                    try:
                        coords = [float(x) for x in parts[1:]]
                        if not all(0 <= coord <= 1 for coord in coords):
                            issues.append(f"Coordinates out of range [0,1] in {sample_label.name} line {line_num}")
                            break
                    except ValueError:
                        issues.append(f"Invalid numeric values in {sample_label.name} line {line_num}")
                        break
        except Exception as e:
            issues.append(f"Error validating label file {sample_label.name}: {e}")

        return issues

    def _load_class_names(self) -> None:
        """Load class names from classes file."""
        with open(self._classes_path) as f:
            class_names = [line.strip() for line in f if line.strip()]
        self._index2label = dict(enumerate(class_names))

    def _find_image_files(self) -> None:
        """Find all valid image files."""
        self._image_files = []
        for ext in self._image_extensions:
            self._image_files.extend(self._images_path.glob(f"*{ext}"))
        self._image_files.sort()

        if not self._image_files:
            raise ValueError(f"No image files found in {self._images_path}")


class YOLODataset(BaseDataset):
    """Internal YOLO dataset implementation."""

    def __init__(self, reader: YOLODatasetReader) -> None:
        self._reader = reader

        self.root = reader.dataset_path
        self.images_path = reader._images_path
        self.annotation_path = reader._labels_path
        self.size = len(reader._image_files)
        self.classes = reader.index2label
        self.metadata = DatasetMetadata(
            id=self._reader.dataset_id,
            index2label=self._reader.index2label,
        )

    def __len__(self) -> int:
        return len(self._reader._image_files)

    def __getitem__(self, index: int) -> tuple[od.InputType, od.ObjectDetectionTarget, DatumMetadata]:
        image_path = self._reader._image_files[index]

        # Load image
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        img_height, img_width = image.shape[:2]
        image = np.transpose(image, (2, 0, 1))  # Convert to CHW format

        # Load corresponding label file
        label_path = self._reader._labels_path / f"{image_path.stem}.txt"

        annotation_metadata = []
        if label_path.exists():
            boxes = []
            labels = []

            with open(label_path) as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue

                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])

                    # Convert normalized YOLO format to absolute pixel coordinates
                    x1 = (center_x - width / 2) * img_width
                    y1 = (center_y - height / 2) * img_height
                    x2 = (center_x + width / 2) * img_width
                    y2 = (center_y + height / 2) * img_height

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

                    # Store original YOLO format coordinates in metadata
                    ann_meta = {
                        "line_number": line_num + 1,
                        "class_id": class_id,
                        "yolo_center_x": center_x,
                        "yolo_center_y": center_y,
                        "yolo_width": width,
                        "yolo_height": height,
                        "absolute_bbox": [x1, y1, x2, y2],
                    }
                    annotation_metadata.append(ann_meta)

            if boxes:
                boxes = np.array(boxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
                scores = np.ones(len(labels), dtype=np.float32)  # Ground truth scores
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                labels = np.empty(0, dtype=np.int64)
                scores = np.empty(0, dtype=np.float32)
        else:
            # No label file - empty annotations
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty(0, dtype=np.int64)
            scores = np.empty(0, dtype=np.float32)

        target = ObjectDetectionTargetTuple(boxes, labels, scores)

        # Create comprehensive datum metadata
        datum_metadata = DatumMetadata(
            **{
                "id": f"{self._reader.dataset_id}_{image_path.stem}",
                # Image-level metadata
                "file_name": image_path.name,
                "file_path": str(image_path),
                "width": img_width,
                "height": img_height,
                # Label file metadata
                "label_file": label_path.name if label_path.exists() else None,
                "label_file_exists": label_path.exists(),
                # Annotation metadata
                "annotations": annotation_metadata,
                "num_annotations": len(annotation_metadata),
            }
        )

        return image, target, datum_metadata
