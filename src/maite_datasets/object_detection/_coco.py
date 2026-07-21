"""Dataset reader for COCO detection format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import maite.protocols.object_detection as od
import numpy as np
from maite.protocols import DatumMetadata
from numpy.typing import NDArray

from maite_datasets._base import (
    BaseReaderDataset,
    LazyAnnotations,
    ObjectDetectionTargetTuple,
    ReaderTransforms,
)
from maite_datasets._bbox import BoundingBoxFormat, convert_to_xyxy_array
from maite_datasets._reader import (
    DEFAULT_ANNOTATION_FILE,
    DEFAULT_CLASSES_FILE,
    DEFAULT_IMAGES_DIR,
    BaseDatasetReader,
    _read_index2label,
)
from maite_datasets.protocols import Array


class TargetIndex(NamedTuple):
    """Every image's boxes and labels, stored flat with per-image offsets.

    Image `i` owns rows ``offsets[i]:offsets[i + 1]``; slicing yields numpy views,
    so per-datum access allocates nothing.
    """

    boxes: NDArray[np.float32]
    """(total_boxes, 4) absolute x1, y1, x2, y2."""
    labels: NDArray[np.int64]
    """(total_boxes,) class index per box."""
    offsets: NDArray[np.int64]
    """(num_images + 1,) start of each image's rows."""

    def rows(self, index: int) -> slice:
        """Row range owned by image `index`."""
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


class COCODatasetReader(BaseDatasetReader[od.Dataset]):
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
        annotation_file: str = DEFAULT_ANNOTATION_FILE,
        images_dir: str = DEFAULT_IMAGES_DIR,
        classes_file: str | None = DEFAULT_CLASSES_FILE,
        dataset_id: str | None = None,
    ) -> None:
        self._annotation_file: str = annotation_file
        self._images_dir: str = images_dir
        self._classes_file: str | None = classes_file

        # Resolved by _initialize_format_specific(), which the base class calls once
        # dataset_path has been validated.
        self._images_path: Path = Path()
        self._annotation_path: Path = Path()
        self._classes_path: Path | None = None
        self._coco_data: dict[str, list[dict[str, Any]]] = {}
        self._image_ids: list[int] = []
        self._image_id_to_info: dict[int, dict[str, Any]] = {}
        self._category_id_to_idx: dict[int, int] = {}
        self._image_id_to_annotations: dict[int, list[dict[str, Any]]] = {}
        self._target_index: TargetIndex = TargetIndex(
            np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
        )

        # Initialize base class
        super().__init__(dataset_path, dataset_id)

    @classmethod
    def can_read(cls, dataset_path: str | Path) -> bool:
        """True when `dataset_path` holds a COCO annotation file.

        Recognizes both the flat `annotations.json` and the upstream
        `annotations/*.json` layout.
        """
        path = Path(dataset_path)
        return (path / DEFAULT_ANNOTATION_FILE).exists() or any((path / "annotations").glob("*.json"))

    def _initialize_format_specific(self) -> None:
        """Initialize COCO-specific components."""
        self._images_path = self.dataset_path / self._images_dir
        self._annotation_path = self.dataset_path / self._annotation_file
        self._classes_path = self.dataset_path / self._classes_file if self._classes_file else None

        if not self._annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self._annotation_path}")
        if not self._images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self._images_path}")

        self._load_annotations()

    @property
    def images_path(self) -> Path:
        """Directory holding the image files referenced by the annotations."""
        return self._images_path

    @property
    def annotation_path(self) -> Path:
        """Path to the COCO annotation JSON file."""
        return self._annotation_path

    @property
    def image_ids(self) -> list[int]:
        """COCO image ids in dataset order."""
        return self._image_ids

    def image_info(self, image_id: int) -> dict[str, Any]:
        """The `images` record for `image_id`."""
        return self._image_id_to_info[image_id]

    def annotations_for(self, image_id: int) -> list[dict[str, Any]]:
        """Annotation records belonging to `image_id`, empty when it has none."""
        return self._image_id_to_annotations.get(image_id, [])

    def category_index(self, category_id: int) -> int:
        """Class index for a COCO `category_id`."""
        return self._category_id_to_idx[category_id]

    @property
    def target_index(self) -> TargetIndex:
        """All boxes and labels, converted once at init and sliced per image."""
        return self._target_index

    def create_dataset(self, lazy: bool = False, transforms: ReaderTransforms = None) -> od.Dataset:
        """Create COCO dataset implementation.

        Parameters
        ----------
        lazy : bool, default False
            When True, each item's image is returned as a :class:`LazyArray`
            that defers PIL decode until first numpy access.
        transforms : ReaderTransforms, default None
            Optional image-only or datum-tuple transform(s) applied to each datum.
        """
        return COCODataset(self, lazy=lazy, transforms=transforms)

    def _image_directories(self) -> list[Path]:
        return [self._images_path]

    def _validate_format_specific(self) -> tuple[list[str], dict[str, Any]]:
        """Validate COCO format specific files and structure.

        Reports on the annotations already parsed at init rather than re-reading
        them, so the diagnostics always describe what the reader actually loaded.
        """
        issues: list[str] = []
        stats: dict[str, Any] = {
            "num_image_records": len(self._image_id_to_info),
            "num_annotations": sum(len(anns) for anns in self._image_id_to_annotations.values()),
            "num_categories": len(self._category_id_to_idx),
            "num_class_names": len(self._index2label),
        }

        for key in ("images", "annotations", "categories"):
            if key not in self._coco_data:
                issues.append(f"Missing required key '{key}' in {self._annotation_file}")

        missing = [
            info["file_name"]
            for info in self._image_id_to_info.values()
            if not (self._images_path / info["file_name"]).exists()
        ]
        if missing:
            issues.append(f"{len(missing)} annotated image file(s) not found, e.g. {missing[0]}")

        return issues, stats

    def _load_annotations(self) -> None:
        """Load and parse COCO annotations."""
        try:
            with open(self._annotation_path) as f:
                self._coco_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {self._annotation_file}: {e.msg}", e.doc, e.pos) from e

        # Build mappings
        self._image_id_to_info = {img["id"]: img for img in self._coco_data.get("images", [])}
        self._image_ids = list(self._image_id_to_info)
        self._category_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(self._coco_data.get("categories", []))}

        # Group annotations by image
        for ann in self._coco_data.get("annotations", []):
            self._image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

        # Load class names, preferring the classes file over the embedded category names
        self._index2label = (
            _read_index2label(self._classes_path)
            if self._classes_path and self._classes_path.exists()
            else dict(enumerate(str(cat["name"]) for cat in self._coco_data.get("categories", [])))
        )

        self._target_index = self._build_target_index()

    def _build_target_index(self) -> TargetIndex:
        """Convert every annotation to its final box and label exactly once.

        COCO boxes are already in absolute pixels, so nothing about them depends on
        the image being loaded - the conversion belongs here, not in the hot path.
        """
        boxes: list[list[float]] = []
        labels: list[int] = []
        offsets: list[int] = [0]

        for image_id in self._image_ids:
            for ann in self._image_id_to_annotations.get(image_id, []):
                boxes.append(ann["bbox"])
                labels.append(self._category_id_to_idx[ann["category_id"]])
            offsets.append(len(labels))

        return TargetIndex(
            convert_to_xyxy_array(np.array(boxes, dtype=np.float64).reshape(-1, 4), BoundingBoxFormat.XYWH).astype(
                np.float32
            ),
            np.array(labels, dtype=np.int64),
            np.array(offsets, dtype=np.int64),
        )


class COCODataset(BaseReaderDataset[od.ObjectDetectionTarget]):
    """Internal COCO dataset implementation.

    Parameters
    ----------
    reader : COCODatasetReader
        Reader providing image paths and parsed annotations.
    lazy : bool, default False
        When True, the image element of each datum is returned as a
        :class:`LazyArray` that defers PIL decode until first numpy access.
        Useful for metadata-only iteration over large image folders.
    transforms : ReaderTransforms, default None
        Optional image-only or datum-tuple transform(s) applied to each datum
        on access via the inherited transform pipeline.
    """

    _RESERVED_ANNOTATION_KEYS = frozenset({"id", "image_id", "category_id", "bbox", "area", "iscrowd"})

    def __init__(self, reader: COCODatasetReader, lazy: bool = False, transforms: ReaderTransforms = None) -> None:
        super().__init__(reader, len(reader.image_ids), lazy, transforms)
        self._reader: COCODatasetReader = reader
        self.images_path: Path = reader.images_path
        self.annotation_path: Path = reader.annotation_path

    def __getitem__(self, index: int) -> tuple[Array, od.ObjectDetectionTarget, DatumMetadata]:
        image_id = self._reader.image_ids[index]
        image_info = self._reader.image_info(image_id)

        # Load image (lazy when self.lazy)
        image_path = self._reader.images_path / image_info["file_name"]
        image = self._get_image(image_path)

        target_index = self._reader.target_index
        rows = target_index.rows(index)
        labels = target_index.labels[rows]

        target = cast(
            od.ObjectDetectionTarget,
            ObjectDetectionTargetTuple(
                target_index.boxes[rows],
                labels,
                np.ones(len(labels), dtype=np.float32),  # Ground truth scores
            ),
        )

        # Per-box records, materialized only if the caller reads them
        annotations = self._reader.annotations_for(image_id)
        annotation_metadata = LazyAnnotations(
            len(annotations),
            lambda: [
                {
                    "annotation_id": ann["id"],
                    "category_id": ann["category_id"],
                    "area": ann.get("area", 0),
                    "iscrowd": ann.get("iscrowd", 0),
                    # Carry any non-standard fields through
                    **{f"ann_{k}": v for k, v in ann.items() if k not in self._RESERVED_ANNOTATION_KEYS},
                }
                for ann in annotations
            ],
        )

        # Create comprehensive datum metadata
        datum_metadata = DatumMetadata(
            **{
                "id": f"{self._reader.dataset_id}_{image_id}",
                # Image-level metadata
                "coco_image_id": image_id,
                "file_name": image_info["file_name"],
                "width": image_info["width"],
                "height": image_info["height"],
                # Optional COCO image fields
                **{
                    key: value for key, value in image_info.items() if key not in ["id", "file_name", "width", "height"]
                },
                # Annotation metadata
                "annotations": annotation_metadata,
                "num_annotations": len(annotations),
            }
        )

        return self._transform((image, target, datum_metadata))
