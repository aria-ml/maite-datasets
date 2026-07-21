"""Dataset reader for YOLO detection format."""

from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import maite.protocols.object_detection as od
import numpy as np
import yaml
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
    DEFAULT_CLASSES_FILE,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_IMAGES_DIR,
    DEFAULT_LABELS_DIR,
    BaseDatasetReader,
    _read_index2label,
)
from maite_datasets.protocols import Array


@dataclass(frozen=True)
class SplitDirs:
    """One resolved pair of image and label directories.

    Pairing the two in a single record makes an images/labels mismatch
    unrepresentable, and lets every consumer work off absolute paths regardless
    of which construction mode produced them.
    """

    name: str
    """Split name ("train", "val", "test"), or "" for the flat single-split layout."""
    images: Path
    labels: Path

    def __str__(self) -> str:
        return f"{self.name} split" if self.name else "dataset"


class LabelIndex(NamedTuple):
    """Every image's parsed YOLO annotations, stored flat with per-image offsets.

    Label files are immutable once the reader has indexed them, so they are parsed
    exactly once instead of on every datum access. Image `i` owns rows
    ``offsets[i]:offsets[i + 1]``; slicing yields numpy views, so per-datum access
    allocates nothing.
    """

    boxes: NDArray[np.float64]
    """(total_boxes, 4) normalized center_x, center_y, width, height."""
    class_ids: NDArray[np.int64]
    """(total_boxes,) class index per box."""
    line_numbers: NDArray[np.int32]
    """(total_boxes,) 1-based line the box was read from, for metadata."""
    offsets: NDArray[np.int64]
    """(num_images + 1,) start of each image's rows."""

    def rows(self, index: int) -> slice:
        """Row range owned by image `index`."""
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


def _parse_label_files(label_files: list[Path | None]) -> LabelIndex:
    """Parse every label file once into a flat :class:`LabelIndex`.

    Malformed lines are skipped, matching the leniency the per-datum parser had;
    `validate_structure` is what reports them.
    """
    boxes: list[tuple[float, float, float, float]] = []
    class_ids: list[int] = []
    line_numbers: list[int] = []
    offsets: list[int] = [0]

    for label_path in label_files:
        if label_path is not None:
            with open(label_path) as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_ids.append(int(parts[0]))
                    boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                    line_numbers.append(line_num)
        offsets.append(len(class_ids))

    return LabelIndex(
        np.array(boxes, dtype=np.float64).reshape(-1, 4),
        np.array(class_ids, dtype=np.int64),
        np.array(line_numbers, dtype=np.int32),
        np.array(offsets, dtype=np.int64),
    )


def _validate_label_format(label_file: Path) -> list[str]:
    """Validate a single YOLO label file, returning a list of issues found."""
    issues: list[str] = []
    try:
        with open(label_file) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.split()
                if not parts:
                    continue

                if len(parts) != 5:
                    issues.append(
                        f"Invalid YOLO format in {label_file.name} line {line_num}: expected 5 values, got {len(parts)}"
                    )
                    break

                try:
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    issues.append(f"Invalid numeric values in {label_file.name} line {line_num}")
                    break

                if not all(0 <= coord <= 1 for coord in coords):
                    issues.append(f"Coordinates out of range [0,1] in {label_file.name} line {line_num}")
                    break
    except OSError as e:
        issues.append(f"Error validating label file {label_file.name}: {e}")

    return issues


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
    ├── classes.txt       # Optional/Required: one class name per line
    └── data.yaml         # Optional/Required: dataset metadata
    ```
    Must have either a classes.txt file or a data.yaml file with classes found under the "names" heading

    YOLO Format Specifications
    --------------------------
    Label file format (one line per object):
    ```
    class_id center_x center_y width height
    0 0.5 0.3 0.2 0.4
    1 0.7 0.8 0.1 0.2
    ```
    All YOLO coordinates are normalized to [0, 1] relative to image dimensions.

    classes.txt format (one class per line, ordered by index, either this or a data.yaml with classes is required):
    ```
    person
    bicycle
    car
    motorcycle
    ```

    data.yaml format:
    ```
    path: path/to/data/folder                       # ignored
    train: train/images                             # path to train images folder from path directory
    val: val/images                                 # optional, path to validation images folder from path directory
    test: test/images                               # optional, path to test images folder from path directory

    nc: 4
    names: [person, bicycle, car, motorcycle]       # either index ordered list or dictionary
    # names:                                        # dictionary format for classes, shown for example
    #   0: person
    #   1: bicycle
    #   2: car
    #   3: motorcycle

    # additional dataset metadata - optional
    collection_date: 07/04/2026
    ```

    Parameters
    ----------
    dataset_path : str, Path
        Root directory containing YOLO dataset files
    data_yaml : str or None, default "data.yaml"
        File containing dataset structure, read when it exists and `images_dir` was not given.
        Label directories are derived from the split image directories by swapping the
        "images" path component for "labels"; pass `images_dir` and `labels_dir` instead
        for layouts that do not follow that convention.
    images_dir : str or None, default None
        Name of directory containing images. Supplying this selects the flat layout and
        ignores `data_yaml`. Defaults to "images".
    labels_dir : str or None, default None
        Name of directory containing YOLO label files. Defaults to "labels".
    classes_file : str or None, default None
        File containing class names (one per line). Defaults to "classes.txt".
        Unused when the class names come from `data_yaml`.
    dataset_id : str or None, default None
        Dataset identifier. If None, uses dataset_path name
    image_extensions : list[str], default [".jpg", ".jpeg", ".png", ".bmp"]
        Supported image file extensions
    image_set : "train", "val", "test" or "base", default "train"
        Split to read from `data_yaml`; "base" concatenates every split it defines.

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
        data_yaml: str | None = "data.yaml",
        images_dir: str | None = None,
        labels_dir: str | None = None,
        classes_file: str | None = None,
        dataset_id: str | None = None,
        image_extensions: list[str] | None = None,
        image_set: Literal["train", "val", "test", "base"] = "train",
    ) -> None:
        self._data_yaml: str | None = data_yaml
        self._images_dir: str | None = images_dir
        self._labels_dir: str | None = labels_dir
        self._classes_file: str | None = classes_file
        self.image_set = image_set
        self._image_extensions: set[str] = {ext.lower() for ext in image_extensions or DEFAULT_IMAGE_EXTENSIONS}

        # Resolved by _initialize_format_specific(), which the base class calls once
        # dataset_path has been validated.
        self._classes_path: Path | None = None
        self._splits: list[SplitDirs] = []
        self._image_files: list[Path] = []
        self._labels: list[Path | None] = []
        self._label_index: LabelIndex = _parse_label_files([])

        super().__init__(dataset_path, dataset_id)

    @classmethod
    def can_read(cls, dataset_path: str | Path) -> bool:
        """True when `dataset_path` holds a YOLO layout.

        Recognizes both the flat `labels/` layout and any data.yaml-driven split
        layout, where the labels directories live under the splits rather than the root.
        """
        path = Path(dataset_path)
        return (path / "data.yaml").exists() or (path / DEFAULT_LABELS_DIR).is_dir()

    def _initialize_format_specific(self) -> None:
        """Resolve split directories and class names, then index the data files."""
        self._resolve_sources()
        self._find_data_files()

    @property
    def splits(self) -> list[SplitDirs]:
        """Resolved (images, labels) directory pairs backing this reader."""
        return self._splits

    @property
    def image_files(self) -> list[Path]:
        """Image paths in dataset order."""
        return self._image_files

    @property
    def label_files(self) -> list[Path | None]:
        """Label path for each image, or None where the image has no label file."""
        return self._labels

    @property
    def label_index(self) -> LabelIndex:
        """All annotations, parsed once at init and sliced per image."""
        return self._label_index

    def create_dataset(self, lazy: bool = False, transforms: ReaderTransforms = None) -> od.Dataset:
        """Create YOLO dataset implementation.

        Parameters
        ----------
        lazy : bool, default False
            When True, each item's image is returned as a :class:`LazyArray`
            that defers PIL decode until first numpy access.
        transforms : ReaderTransforms, default None
            Optional image-only or datum-tuple transform(s) applied to each datum.
        """
        return YOLODataset(self, lazy=lazy, transforms=transforms)

    def _resolve_sources(self) -> None:
        """Resolve split directories and class names from data.yaml or the flat layout.

        Explicit `images_dir` wins; otherwise data.yaml is used when present; otherwise
        the flat `images/` + `labels/` + `classes.txt` layout applies.
        """
        yaml_path = self.dataset_path / self._data_yaml if self._data_yaml else None
        if self._images_dir is None and yaml_path is not None and yaml_path.exists():
            self._load_data_yaml(yaml_path)
            return

        images = self.dataset_path / (self._images_dir or DEFAULT_IMAGES_DIR)
        labels = self.dataset_path / (self._labels_dir or DEFAULT_LABELS_DIR)
        self._splits = [SplitDirs("", images, labels)]

        self._classes_path = self.dataset_path / (self._classes_file or DEFAULT_CLASSES_FILE)
        if not self._classes_path.exists():
            raise FileNotFoundError(f"Classes file not found: {self._classes_path}")
        self._index2label = _read_index2label(self._classes_path)

    def _load_data_yaml(self, yaml_path: Path) -> None:
        """Populate class names and split directories from a data.yaml file."""
        with open(yaml_path) as file:
            config: dict[str, Any] = yaml.safe_load(file)

        names = config.get("names")
        if names is None:
            raise KeyError(
                f"Yolo data yaml file does not contain class information {config}. "
                "Classes must be listed under the 'names' heading."
            )
        self._index2label = (
            dict(enumerate(str(name) for name in names))
            if isinstance(names, (list, tuple))
            else {int(index): str(name) for index, name in names.items()}
        )

        splits = ("train", "val", "test") if self.image_set == "base" else (self.image_set,)
        for split in splits:
            split_dir = config.get(split)
            if split_dir is None:
                if self.image_set == "base":
                    continue
                raise KeyError(f"'{split}' not found in {yaml_path.name}")
            images = self._resolve(split_dir)
            self._splits.append(SplitDirs(split, images, self._labels_dir_for(images)))

        if not self._splits:
            raise KeyError(f"No 'train', 'val' or 'test' entries found in {yaml_path.name}")

    def _resolve(self, directory: str | Path) -> Path:
        """Resolve a configured directory against the dataset root."""
        path = Path(directory)
        return path if path.is_absolute() else self.dataset_path / path

    @staticmethod
    def _labels_dir_for(images: Path) -> Path:
        """Derive the labels directory by swapping the last "images" path component.

        Covers both conventional YOLO layouts - `<split>/images` and `images/<split>` -
        and fails loudly rather than silently pointing at a directory that does not exist.
        """
        parts = list(images.parts)
        for i in reversed(range(len(parts))):
            if parts[i].lower() == DEFAULT_IMAGES_DIR:
                parts[i] = DEFAULT_LABELS_DIR
                return Path(*parts)
        raise ValueError(
            f"Cannot derive a labels directory from {images}: no 'images' path component. "
            "Pass images_dir and labels_dir explicitly instead of using data.yaml."
        )

    def _find_data_files(self) -> None:
        """Find all valid image files and their corresponding label files."""
        for split in self._splits:
            if not split.images.exists():
                raise FileNotFoundError(f"Images directory not found: {split.images}")
            if not split.labels.exists():
                raise FileNotFoundError(f"Labels directory not found: {split.labels}")

            split_images = sorted(self._iter_images(split.images))
            if not split_images:
                raise ValueError(f"No image files found in {split.images}")

            # Pair by stem instead of relying on parallel sort order matching 1:1.
            labels_by_stem = {p.stem: p for p in split.labels.glob("*.txt")}
            self._image_files.extend(split_images)
            self._labels.extend(labels_by_stem.get(p.stem) for p in split_images)

        self._label_index = _parse_label_files(self._labels)

    def _image_directories(self) -> list[Path]:
        return [split.images for split in self._splits]

    def _validate_format_specific(self) -> tuple[list[str], dict[str, Any]]:
        """Validate YOLO format specific files and structure."""
        issues: list[str] = []
        stats: dict[str, Any] = {"num_label_files": 0, "num_classes": len(self._index2label)}

        for split in self._splits:
            if not split.labels.exists():
                issues.append(f"Missing {self._display(split.labels)}/ directory for {split}")
                continue

            label_files = sorted(split.labels.glob("*.txt"))
            stats["num_label_files"] += len(label_files)
            if not label_files:
                issues.append(f"No label files found in {self._display(split.labels)}/ directory")
            else:
                # Sample check - validating every file would be O(dataset) on every call.
                issues.extend(_validate_label_format(label_files[0]))

        if not self._index2label:
            issues.append("No class names found in classes file or data yaml")

        return issues, stats


class YOLODataset(BaseReaderDataset[od.ObjectDetectionTarget]):
    """Internal YOLO dataset implementation.

    Parameters
    ----------
    reader : YOLODatasetReader
        Reader providing image paths and parsed label files.
    lazy : bool, default False
        When True, the image element of each datum is returned as a
        :class:`LazyArray` that defers PIL decode until first numpy access.
        Box scaling uses the cheap PIL header probe so OD targets resolve
        without pixel decode.
    transforms : ReaderTransforms, default None
        Optional image-only or datum-tuple transform(s) applied to each datum
        on access via the inherited transform pipeline.
    """

    def __init__(self, reader: YOLODatasetReader, lazy: bool = False, transforms: ReaderTransforms = None) -> None:
        super().__init__(reader, len(reader.image_files), lazy, transforms)
        self._reader: YOLODatasetReader = reader
        self.images_path: list[Path] = [split.images for split in reader.splits]
        self.annotation_path: list[Path] = [split.labels for split in reader.splits]
        # Image dimensions are fixed per file but cost a header read to learn, so
        # remember them the first time each index is touched rather than every epoch.
        self._shapes: list[tuple[int, int] | None] = [None] * self.size

    def _dimensions(self, index: int, image_path: Path, image: Array) -> tuple[int, int]:
        """Height and width of the image at `index`, probed at most once."""
        shape = self._shapes[index]
        if shape is None:
            _, height, width = self._get_shape(image_path, image)
            shape = self._shapes[index] = (height, width)
        return shape

    def __getitem__(self, index: int) -> tuple[Array, od.ObjectDetectionTarget, DatumMetadata]:
        image_path = self._reader.image_files[index]
        label_path = self._reader.label_files[index]

        # Load image (lazy when self.lazy); shape probed from the header without decode.
        image = self._get_image(image_path)
        img_height, img_width = self._dimensions(index, image_path, image)

        label_index = self._reader.label_index
        rows = label_index.rows(index)
        normalized = label_index.boxes[rows]
        class_ids = label_index.class_ids[rows]
        boxes = convert_to_xyxy_array(normalized, BoundingBoxFormat.NORMALIZED_CXCYWH, (img_height, img_width))

        target = cast(
            od.ObjectDetectionTarget,
            ObjectDetectionTargetTuple(
                boxes.astype(np.int32),
                class_ids,
                np.ones(len(class_ids), dtype=np.float32),  # Ground truth scores
            ),
        )

        # Original YOLO coordinates, materialized only if the caller reads them
        annotations = LazyAnnotations(
            len(class_ids),
            lambda: [
                {
                    "line_number": int(line_number),
                    "class_id": int(class_id),
                    "yolo_center_x": float(center_x),
                    "yolo_center_y": float(center_y),
                    "yolo_width": float(width),
                    "yolo_height": float(height),
                    "absolute_bbox": [float(v) for v in box],
                }
                for (center_x, center_y, width, height), class_id, line_number, box in zip(
                    normalized, class_ids, label_index.line_numbers[rows], boxes
                )
            ],
        )

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
                "label_file": label_path.name if label_path is not None else None,
                "label_file_exists": label_path is not None,
                # Annotation metadata
                "annotations": annotations,
                "num_annotations": len(annotations),
            }
        )

        return self._transform((image, target, datum_metadata))
