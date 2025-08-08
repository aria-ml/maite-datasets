"""Dataset reader for COCO detection format."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from maite_datasets._protocols import DatasetMetadata, DatumMetadata, ObjectDetectionDataset, ObjectDetectionDatum
from maite_datasets._reader._base import _ObjectDetectionTarget, BaseDatasetReader

_logger = logging.getLogger(__name__)


class COCODatasetReader(BaseDatasetReader):
    """
    COCO format dataset reader conforming to MAITE protocols.

    Reads COCO format object detection datasets from disk and provides
    MAITE-compatible interface.

    Directory Structure Requirements
    --------------------------------
    ```
    dataset_root/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── annotations.json  # COCO format annotation file
    └── classes.txt       # Optional: one class name per line
    ```

    COCO Format Specifications
    --------------------------
    annotations.json structure:
    ```json
    {
      "images": [
        {
          "id": 1,
          "file_name": "image1.jpg",
          "width": 640,
          "height": 480
        }
      ],
      "annotations": [
        {
          "id": 1,
          "image_id": 1,
          "category_id": 1,
          "bbox": [100, 50, 200, 150],  // [x, y, width, height]
          "area": 30000
        }
      ],
      "categories": [
        {
          "id": 1,
          "name": "person"
        }
      ]
    }
    ```

    classes.txt format (optional, one class per line, ordered by index):
    ```
    person
    bicycle
    car
    motorcycle
    ```

    Parameters
    ----------
    dataset_path : str or Path
        Root directory containing COCO dataset files
    annotation_file : str, default "annotations.json"
        Name of COCO annotation JSON file
    images_dir : str, default "images"
        Name of directory containing images
    classes_file : str or None, default "classes.txt"
        Optional file containing class names (one per line)
        If None, uses category names from COCO annotations
    dataset_id : str or None, default None
        Dataset identifier. If None, uses dataset_path name

    Notes
    -----
    COCO annotations should follow standard COCO format with:
    - "images": list of image metadata
    - "annotations": list of bounding box annotations
    - "categories": list of category definitions

    Bounding boxes are converted from COCO format (x, y, width, height)
    to MAITE format (x1, y1, x2, y2).
    """

    def __init__(
        self,
        dataset_path: str | Path,
        annotation_file: str = "annotations.json",
        images_dir: str = "images",
        classes_file: str | None = "classes.txt",
        dataset_id: str | None = None,
    ) -> None:
        self.annotation_file = annotation_file
        self.images_dir = images_dir
        self.classes_file = classes_file

        # Initialize base class
        super().__init__(dataset_path, dataset_id)

    def _initialize_format_specific(self) -> None:
        """Initialize COCO-specific components."""
        self.images_path = self.dataset_path / self.images_dir
        self.annotation_path = self.dataset_path / self.annotation_file
        self.classes_path = self.dataset_path / self.classes_file if self.classes_file else None

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")

        self._load_annotations()

    @property
    def index2label(self) -> dict[int, str]:
        """Mapping from class index to class name."""
        return self._index2label

    def _create_dataset_implementation(self) -> ObjectDetectionDataset:
        """Create COCO dataset implementation."""
        return _COCODataset(self)

    def _validate_format_specific(self) -> tuple[list[str], dict[str, Any]]:
        """Validate COCO format specific files and structure."""
        issues = []
        stats = {}

        annotation_path = self.dataset_path / self.annotation_file
        if not annotation_path.exists():
            issues.append(f"Missing {self.annotation_file} file")
            return issues, stats

        try:
            with open(annotation_path) as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON in {self.annotation_file}: {e}")
            return issues, stats

        # Check required keys
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in coco_data:
                issues.append(f"Missing required key '{key}' in {self.annotation_file}")
            else:
                stats[f"num_{key}"] = len(coco_data[key])

        # Check optional classes.txt
        if self.classes_file:
            classes_path = self.dataset_path / self.classes_file
            if classes_path.exists():
                try:
                    with open(classes_path) as f:
                        class_lines = [line.strip() for line in f if line.strip()]
                    stats["num_class_names"] = len(class_lines)
                except Exception as e:
                    issues.append(f"Error reading {self.classes_file}: {e}")

        return issues, stats

    def _load_annotations(self) -> None:
        """Load and parse COCO annotations."""
        with open(self.annotation_path) as f:
            self.coco_data = json.load(f)

        # Build mappings
        self.image_id_to_info = {img["id"]: img for img in self.coco_data["images"]}
        self.category_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(self.coco_data["categories"])}

        # Group annotations by image
        self.image_id_to_annotations: dict[int, list[dict[str, Any]]] = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # Load class names
        if self.classes_path and self.classes_path.exists():
            with open(self.classes_path) as f:
                class_names = [line.strip() for line in f if line.strip()]
        else:
            class_names = [cat["name"] for cat in self.coco_data["categories"]]

        self._index2label = {idx: name for idx, name in enumerate(class_names)}


class _COCODataset:
    """Internal COCO dataset implementation."""

    def __init__(self, reader: COCODatasetReader) -> None:
        self.reader = reader
        self.image_ids = list(reader.image_id_to_info.keys())

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            id=self.reader._dataset_id,
            index2label=self.reader.index2label,
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> ObjectDetectionDatum:
        image_id = self.image_ids[index]
        image_info = self.reader.image_id_to_info[image_id]

        # Load image
        image_path = self.reader.images_path / image_info["file_name"]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = np.transpose(image, (2, 0, 1))  # Convert to CHW format

        # Get annotations for this image
        annotations = self.reader.image_id_to_annotations.get(image_id, [])

        if annotations:
            boxes = []
            labels = []
            annotation_metadata = []

            for ann in annotations:
                # Convert COCO bbox (x, y, w, h) to (x1, y1, x2, y2)
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])

                # Map category_id to class index
                cat_idx = self.reader.category_id_to_idx[ann["category_id"]]
                labels.append(cat_idx)

                # Collect annotation metadata
                ann_meta = {
                    "annotation_id": ann["id"],
                    "category_id": ann["category_id"],
                    "area": ann.get("area", 0),
                    "iscrowd": ann.get("iscrowd", 0),
                }
                # Add any additional fields from annotation
                for key, value in ann.items():
                    if key not in ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]:
                        ann_meta[f"ann_{key}"] = value
                annotation_metadata.append(ann_meta)

            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            scores = np.ones(len(labels), dtype=np.float32)  # Ground truth scores
        else:
            # Empty annotations
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty(0, dtype=np.int64)
            scores = np.empty(0, dtype=np.float32)
            annotation_metadata = []

        target = _ObjectDetectionTarget(boxes, labels, scores)

        # Create comprehensive datum metadata
        datum_metadata = DatumMetadata(
            id=f"{self.reader._dataset_id}_{image_id}",
            # Image-level metadata
            coco_image_id=image_id,
            file_name=image_info["file_name"],
            width=image_info["width"],
            height=image_info["height"],
            # Optional COCO image fields
            **{key: value for key, value in image_info.items() if key not in ["id", "file_name", "width", "height"]},
            # Annotation metadata
            annotations=annotation_metadata,
            num_annotations=len(annotations),
        )

        return image, target, datum_metadata
