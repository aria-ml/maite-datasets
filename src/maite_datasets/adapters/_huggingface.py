from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, TypeAlias, overload

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
import numpy as np
from maite.protocols import DatasetMetadata, DatumMetadata

from maite_datasets._base import BaseDataset, NumpyArray, ObjectDetectionTargetTuple
from maite_datasets._bbox import BoundingBoxFormat, convert_to_xyxy, detect_bbox_format
from maite_datasets.protocols import HFArray, HFClassLabel, HFDataset, HFImage, HFList, HFValue
from maite_datasets.wrappers._torch import TTarget

# Constants for image processing
MAX_VALID_CHANNELS = 10

FeatureDict: TypeAlias = Mapping[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class HFDatasetInfo:
    image_key: str


@dataclass
class HFImageClassificationDatasetInfo(HFDatasetInfo):
    label_key: str


@dataclass
class HFObjectDetectionDatasetInfo(HFDatasetInfo):
    objects_key: str
    bbox_key: str
    label_key: str


class HFBaseDataset(BaseDataset[NumpyArray, TTarget]):
    """Base wrapper for Hugging Face datasets, handling common logic."""

    def __init__(self, hf_dataset: HFDataset, image_key: str, known_keys: set[str]) -> None:
        self.source = hf_dataset
        self._image_key = image_key

        # Add dataset metadata
        dataset_info_dict = hf_dataset.info.__dict__
        if "id" in dataset_info_dict:
            dataset_info_dict["datasetinfo_id"] = dataset_info_dict.pop("id")
        self._metadata_id = dataset_info_dict["dataset_name"]
        self._metadata_dict = dataset_info_dict

        # Pre-validate features and cache metadata keys
        self._validate_features(hf_dataset.features)
        self._scalar_meta_keys = self._extract_scalar_meta_keys(hf_dataset.features, known_keys)

        # Cache for image conversions
        self._image_cache: dict[int, np.ndarray] = {}

    def _validate_features(self, features: FeatureDict) -> None:
        """Pre-validate all features during initialization."""
        if self._image_key not in features:
            raise ValueError(f"Image key '{self._image_key}' not found in dataset features.")

        if not isinstance(features[self._image_key], (HFImage, HFArray)):
            raise TypeError(f"Image feature '{self._image_key}' must be HFImage or HFArray.")

    def _extract_scalar_meta_keys(self, features: FeatureDict, known_keys: set[str]) -> list[str]:
        """Extract scalar metadata keys during initialization."""
        return [key for key, feature in features.items() if key not in known_keys and isinstance(feature, HFValue)]

    def __len__(self) -> int:
        return len(self.source)

    def _get_base_metadata(self, index: int) -> DatumMetadata:
        """Extract base metadata for a datum."""
        item = self.source[index]
        datum_metadata: DatumMetadata = {"id": index}
        for key in self._scalar_meta_keys:
            datum_metadata[key] = item[key]
        return datum_metadata

    @lru_cache(maxsize=64)  # Cache image conversions
    def _get_image(self, index: int) -> np.ndarray:
        """Get and process image with caching and optimized conversions."""
        # Convert to numpy array only once
        raw_image = self.source[index][self._image_key]
        image = np.asarray(raw_image)

        # Handle different image formats efficiently
        if image.ndim == 2:
            # Grayscale: HW -> CHW
            image = image[np.newaxis, ...]  # More efficient than expand_dims
        elif image.ndim == 3:
            # Check if we need to transpose from HWC to CHW
            if image.shape[-1] < image.shape[-3] and image.shape[-1] <= MAX_VALID_CHANNELS:
                # HWC -> CHW using optimized transpose
                image = np.transpose(image, (2, 0, 1))
            elif image.shape[0] > MAX_VALID_CHANNELS:
                raise ValueError(
                    f"Image at index {index} has invalid channel configuration. "
                    f"Expected channels to be less than {MAX_VALID_CHANNELS}, got shape {image.shape}"
                )
        else:
            raise ValueError(
                f"Image at index {index} has unsupported dimensionality. "
                f"Expected 2D or 3D, got {image.ndim}D with shape {image.shape}"
            )

        if image.ndim != 3:
            raise ValueError(f"Image processing failed for index {index}. Final shape: {image.shape}")

        return image


class HFImageClassificationDataset(HFBaseDataset[NumpyArray], ic.Dataset):
    """Wraps a Hugging Face dataset to comply with the ImageClassificationDataset protocol."""

    def __init__(self, hf_dataset: HFDataset, image_key: str, label_key: str) -> None:
        super().__init__(hf_dataset, image_key, known_keys={image_key, label_key})
        self._label_key = label_key

        # Pre-validate label feature
        label_feature = hf_dataset.features[self._label_key]
        if not isinstance(label_feature, HFClassLabel):
            raise TypeError(
                f"Label feature '{self._label_key}' must be a datasets.ClassLabel, got {type(label_feature).__name__}."
            )

        self._num_classes: int = label_feature.num_classes

        # Pre-compute one-hot identity matrix for efficient encoding
        self._one_hot_matrix = np.eye(self._num_classes, dtype=np.float32)

        # Enhanced metadata with validation
        self.metadata: DatasetMetadata = DatasetMetadata(
            id=self._metadata_id, index2label=dict(enumerate(label_feature.names)), **self._metadata_dict
        )

    def __getitem__(self, index: int) -> tuple[NumpyArray, NumpyArray, DatumMetadata]:
        if not 0 <= index < len(self.source):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.source)}")

        # Process image
        image = self._get_image(index)
        label_int = self.source[index][self._label_key]

        # Process target
        if not 0 <= label_int < self._num_classes:
            raise ValueError(f"Label {label_int} at index {index} is out of range [0, {self._num_classes})")
        one_hot_label = self._one_hot_matrix[label_int]

        # Process metadata
        datum_metadata = self._get_base_metadata(index)

        return image, one_hot_label, datum_metadata


class HFObjectDetectionDataset(HFBaseDataset[ObjectDetectionTargetTuple], od.Dataset):
    """Wraps a Hugging Face dataset to comply with the ObjectDetectionDataset protocol."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        image_key: str,
        objects_key: str,
        bbox_key: str,
        label_key: str,
        bbox_format: Literal["xyxy", "xywh", "yolo", "auto"] = "auto",
    ) -> None:
        super().__init__(hf_dataset, image_key, known_keys={image_key, objects_key})
        self._objects_key = objects_key
        self._bbox_key = bbox_key
        self._label_key = label_key

        # Pre-validate and extract object features
        self._object_meta_keys = self._validate_and_extract_object_features(hf_dataset.features)

        # Validate and extract label information
        label_feature = self._extract_label_feature(hf_dataset.features)
        self.metadata: DatasetMetadata = DatasetMetadata(
            id=self._metadata_id, index2label=dict(enumerate(label_feature.names)), **self._metadata_dict
        )

        # Detect bounding box format during initialization
        if bbox_format == "xyxy":
            self._bbox_format = BoundingBoxFormat.XYXY
        elif bbox_format == "xywh":
            self._bbox_format = BoundingBoxFormat.XYWH
        elif bbox_format == "yolo":
            self._bbox_format = BoundingBoxFormat.NORMALIZED_CXCYWH
        else:
            self._bbox_format = self._detect_bbox_format()
            logger.info(f"Detected bounding box format: {self._bbox_format}")

    def _detect_bbox_format(self) -> BoundingBoxFormat:
        """
        Detect the bounding box format by sampling from the dataset.

        Returns
        -------
        BoundingBoxFormat
            The detected format of the bounding boxes.
        """
        # Sample up to 50 items for format detection to balance accuracy and performance
        max_samples = min(50, len(self.source))
        sample_indices = np.linspace(0, len(self.source) - 1, max_samples, dtype=int).tolist()

        all_bboxes = []
        image_shapes = []

        for idx in sample_indices:
            try:
                # Get image for dimension info
                image = self._get_image(idx)
                image_shapes.append(image.shape)

                # Get bounding boxes
                objects = self.source[idx][self._objects_key]
                boxes = objects[self._bbox_key]

                # Convert to the expected format for detection
                if boxes:  # Only add if there are boxes
                    # Convert each box to tuple format
                    formatted_boxes = []
                    for box in boxes:
                        if len(box) == 4:
                            formatted_boxes.append(tuple(float(coord) for coord in box))
                    all_bboxes.append(formatted_boxes)
                else:
                    all_bboxes.append([])  # Empty list for no boxes

            except Exception as e:
                logger.warning(f"Failed to process sample {idx} for bbox format detection: {e}")
                continue

        if not any(all_bboxes) or not image_shapes:
            logger.warning("No valid bounding boxes found for format detection, defaulting to XYXY")
            return BoundingBoxFormat.XYXY

        # Detect format
        detected_format = detect_bbox_format(all_bboxes, image_shapes)

        if detected_format == BoundingBoxFormat.UNKNOWN:
            logger.warning("Could not detect bounding box format, defaulting to XYXY")
            return BoundingBoxFormat.XYXY

        return detected_format

    def _convert_bboxes_to_xyxy(self, boxes: list[list[float]], image_shape: tuple[int, int, int]) -> list[list[float]]:
        """
        Convert bounding boxes to XYXY format.

        Parameters
        ----------
        boxes : list[list[float]]
            List of bounding boxes in the detected format.
        image_shape : tuple[int, int, int]
            Shape of the image in CHW format.

        Returns
        -------
        list[list[float]]
            List of bounding boxes converted to XYXY format.
        """
        if not boxes:
            return boxes

        converted_boxes = []
        for box in boxes:
            if len(box) != 4:
                logger.warning(f"Skipping invalid bounding box with {len(box)} coordinates: {box}")
                continue

            try:
                # Convert to XYXY and add to list
                xyxy_coords = convert_to_xyxy(
                    box[0], box[1], box[2], box[3], bbox_format=self._bbox_format, image_shape=image_shape
                )
                converted_boxes.append(list(xyxy_coords))

            except Exception as e:
                logger.warning(f"Failed to convert bounding box {box}: {e}")
                continue

        return converted_boxes

    def _validate_and_extract_object_features(self, features: FeatureDict) -> list[str]:
        """Validate objects feature and extract metadata keys."""
        objects_feature = features[self._objects_key]

        # Determine the structure and get inner features
        if isinstance(objects_feature, HFList):  # list(dict) case
            if not isinstance(objects_feature.feature, dict):
                raise TypeError(f"Objects feature '{self._objects_key}' with list type must contain dict features.")
            inner_feature_dict = objects_feature.feature
        elif isinstance(objects_feature, dict):  # dict(list) case
            inner_feature_dict = objects_feature
        else:
            raise TypeError(
                f"Objects feature '{self._objects_key}' must be a list or dict, got {type(objects_feature).__name__}."
            )

        # Validate required keys exist
        required_keys = {self._bbox_key, self._label_key}
        missing_keys = required_keys - set(inner_feature_dict.keys())
        if missing_keys:
            raise ValueError(f"Objects feature '{self._objects_key}' missing required keys: {missing_keys}")

        # Extract object metadata keys
        known_inner_keys = {self._bbox_key, self._label_key}
        return [
            key
            for key, feature in inner_feature_dict.items()
            if key not in known_inner_keys and isinstance(feature, (HFValue, HFList))
        ]

    def _extract_label_feature(self, features: FeatureDict) -> HFClassLabel:
        """Extract and validate the label feature."""
        objects_feature = features[self._objects_key]

        inner_features = objects_feature.feature if isinstance(objects_feature, HFList) else objects_feature
        label_feature_container = inner_features[self._label_key]
        label_feature = (
            label_feature_container.feature
            if isinstance(label_feature_container.feature, HFClassLabel)
            else label_feature_container
        )

        if not isinstance(label_feature, HFClassLabel):
            raise TypeError(
                f"Label '{self._label_key}' in '{self._objects_key}' must be a ClassLabel, "
                f"got {type(label_feature).__name__}."
            )

        return label_feature

    def __getitem__(self, index: int) -> tuple[NumpyArray, ObjectDetectionTargetTuple, DatumMetadata]:
        if not 0 <= index < len(self.source):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.source)}")

        # Process image
        image = self._get_image(index)
        objects = self.source[index][self._objects_key]

        # Process target - convert bboxes to XYXY format
        raw_boxes = objects[self._bbox_key]
        converted_boxes = self._convert_bboxes_to_xyxy(raw_boxes, image.shape)

        labels = objects[self._label_key]
        scores = np.ones_like(labels, dtype=np.float32)
        target = ObjectDetectionTargetTuple(converted_boxes, labels, scores)

        # Process metadata
        datum_metadata = self._get_base_metadata(index)
        self._add_object_metadata(objects, datum_metadata)

        return image, target, datum_metadata

    def _add_object_metadata(self, objects: dict[str, Any], datum_metadata: DatumMetadata) -> None:
        """Efficiently add object metadata to datum metadata."""
        if not objects[self._bbox_key]:  # No objects
            return

        num_objects = len(objects[self._bbox_key])

        for key in self._object_meta_keys:
            value = objects[key]
            if isinstance(value, list):
                if len(value) == num_objects:
                    datum_metadata[key] = value
                else:
                    raise ValueError(
                        f"Object metadata '{key}' length {len(value)} doesn't match number of objects {num_objects}"
                    )
            else:
                datum_metadata[key] = [value] * num_objects

    @property
    def bbox_format(self) -> BoundingBoxFormat:
        """Get the detected bounding box format."""
        return self._bbox_format


def is_bbox(feature: Any) -> bool:
    """Check if feature represents bounding box data with proper type validation."""
    if not isinstance(feature, HFList):
        return False

    # Handle nested list structure
    bbox_candidate = feature.feature if isinstance(feature.feature, HFList) else feature

    return (
        isinstance(bbox_candidate, HFList)
        and bbox_candidate.length == 4
        and isinstance(bbox_candidate.feature, HFValue)
        and any(dtype in bbox_candidate.feature.dtype for dtype in ["float", "int"])
    )


def is_label(feature: Any) -> bool:
    """Check if feature represents label data with proper type validation."""
    target_feature = feature.feature if isinstance(feature, HFList) else feature
    return isinstance(target_feature, HFClassLabel)


def find_od_keys(feature: Any) -> tuple[str | None, str | None]:
    """Helper to find bbox and label keys for object detection with improved logic."""
    if not ((isinstance(feature, HFList) and isinstance(feature.feature, dict)) or isinstance(feature, dict)):
        return None, None

    inner_features: FeatureDict = feature.feature if isinstance(feature, HFList) else feature

    bbox_key = label_key = None

    for inner_name, inner_feature in inner_features.items():
        if bbox_key is None and is_bbox(inner_feature):
            bbox_key = inner_name
        if label_key is None and is_label(inner_feature):
            label_key = inner_name

        # Early exit if both found
        if bbox_key and label_key:
            break

    return bbox_key, label_key


def get_dataset_info(dataset: HFDataset) -> HFDatasetInfo:
    """Extract dataset information with improved validation and error messages."""
    features = dataset.features
    image_key = label_key = objects_key = bbox_key = None

    # More efficient feature detection
    for name, feature in features.items():
        if image_key is None and isinstance(feature, (HFImage, HFArray)):
            image_key = name
        elif label_key is None and isinstance(feature, HFClassLabel):
            label_key = name
        elif objects_key is None:
            temp_bbox, temp_label = find_od_keys(feature)
            if temp_bbox and temp_label:
                objects_key, bbox_key, label_key = name, temp_bbox, temp_label

    if not image_key:
        available_features = list(features.keys())
        raise ValueError(
            f"No image key found in dataset. Available features: {available_features}. "
            f"Expected HFImage or HFArray type."
        )

    # Return appropriate dataset info based on detected features
    if objects_key and bbox_key and label_key:
        return HFObjectDetectionDatasetInfo(image_key, objects_key, bbox_key, label_key)
    if label_key:
        return HFImageClassificationDatasetInfo(image_key, label_key)
    return HFDatasetInfo(image_key)


@overload
def from_huggingface(
    dataset: HFDataset,
    task: Literal["image_classification"],
    bbox_format: Literal["xyxy", "xywh", "yolo", "auto"] = "auto",
) -> HFImageClassificationDataset: ...


@overload
def from_huggingface(
    dataset: HFDataset,
    task: Literal["object_detection"],
    bbox_format: Literal["xyxy", "xywh", "yolo", "auto"] = "auto",
) -> HFObjectDetectionDataset: ...


@overload
def from_huggingface(
    dataset: HFDataset,
    task: Literal["auto"] = "auto",
    bbox_format: Literal["xyxy", "xywh", "yolo", "auto"] = "auto",
) -> HFObjectDetectionDataset | HFImageClassificationDataset: ...


def from_huggingface(
    dataset: HFDataset,
    task: Literal["image_classification", "object_detection", "auto"] = "auto",
    bbox_format: Literal["xyxy", "xywh", "yolo", "auto"] = "auto",
) -> HFObjectDetectionDataset | HFImageClassificationDataset:
    """Create appropriate dataset wrapper with enhanced error handling."""
    info = get_dataset_info(dataset)

    if isinstance(info, HFImageClassificationDatasetInfo):
        if task in ("image_classification", "auto"):
            return HFImageClassificationDataset(dataset, info.image_key, info.label_key)
        if task == "object_detection":
            raise ValueError(
                f"Task mismatch: requested 'object_detection' but dataset appears to be "
                f"image classification. Detected features: image='{info.image_key}', "
                f"label='{info.label_key}'"
            )

    elif isinstance(info, HFObjectDetectionDatasetInfo):
        if task in ("object_detection", "auto"):
            return HFObjectDetectionDataset(
                dataset, info.image_key, info.objects_key, info.bbox_key, info.label_key, bbox_format
            )
        if task == "image_classification":
            raise ValueError(
                f"Task mismatch: requested 'image_classification' but dataset appears to be "
                f"object detection. Detected features: image='{info.image_key}', "
                f"objects='{info.objects_key}'"
            )

    # Enhanced error message for auto-detection failure
    available_features = list(dataset.features.keys())
    feature_types = {k: type(v).__name__ for k, v in dataset.features.items()}

    raise ValueError(
        f"Could not automatically determine task for requested type '{task}'. "
        f"Detected info: {info}. Available features: {available_features}. "
        f"Feature types: {feature_types}. Ensure dataset has proper image and label/objects features."
    )
