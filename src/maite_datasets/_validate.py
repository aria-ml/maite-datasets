from __future__ import annotations

__all__ = []

from collections.abc import Iterable, Sequence, Sized
from typing import Any, Literal

import numpy as np
from maite.protocols.object_detection import ObjectDetectionTarget

from maite_datasets.protocols import Array


class ValidationMessages:
    DATASET_SIZED = "Dataset must be sized."
    DATASET_INDEXABLE = "Dataset must be indexable."
    DATASET_NONEMPTY = "Dataset must be non-empty."
    DATASET_METADATA = "Dataset must have a 'metadata' attribute."
    DATASET_METADATA_TYPE = "Dataset metadata must be a dictionary."
    DATASET_METADATA_FORMAT = "Dataset metadata must contain an 'id' key."
    DATUM_TYPE = "Dataset datum must be a tuple."
    DATUM_FORMAT = "Dataset datum must contain 3 elements: image, target, metadata."
    DATUM_IMAGE_TYPE = "Images must be 3-dimensional arrays."
    DATUM_IMAGE_FORMAT = "Images must be in CHW format."
    DATUM_TARGET_IC_TYPE = "ImageClassificationDataset targets must be one-dimensional arrays."
    DATUM_TARGET_IC_FORMAT = "ImageClassificationDataset targets must be one-hot encoded or pseudo-probabilities."
    DATUM_TARGET_OD_TYPE = "ObjectDetectionDataset targets must be have 'boxes', 'labels' and 'scores'."
    DATUM_TARGET_OD_LABELS_TYPE = "ObjectDetectionTarget labels must be one-dimensional (N,) arrays."
    DATUM_TARGET_OD_BOXES_TYPE = "ObjectDetectionTarget boxes must be two-dimensional (N, 4) arrays in xxyy format."
    DATUM_TARGET_OD_SCORES_TYPE = "ObjectDetectionTarget scores must be one (N,) or two-dimensional (N, M) arrays."
    DATUM_VIDEO_TYPE = "MultiObjectTracking inputs must be an iterable VideoStream of frames."
    DATUM_VIDEO_FORMAT = "VideoStream frames must expose 3-dimensional (C, H, W) pixels."
    DATUM_TARGET_MOT_TYPE = "MultiObjectTrackingDataset targets must have a 'frame_tracks' attribute."
    DATUM_TARGET_MOT_FRAMES_TYPE = "MultiObjectTrackingTarget frame_tracks must be a sequence of frame targets."
    DATUM_TARGET_MOT_FRAME_TYPE = (
        "MultiObjectTracking frame targets must have 'boxes', 'labels', 'scores' and 'track_ids'."
    )
    DATUM_TARGET_MOT_BOXES_TYPE = "MultiObjectTracking frame boxes must be two-dimensional (N, 4) xxyy arrays."
    DATUM_TARGET_MOT_TRACK_IDS_TYPE = (
        "MultiObjectTracking frame track_ids must be one-dimensional (N,) integer arrays (-1 for untracked)."
    )
    DATUM_TARGET_TYPE = "Target is not a valid ImageClassification, ObjectDetection or MultiObjectTracking target."
    DATUM_METADATA_TYPE = "Datum metadata must be a dictionary."
    DATUM_METADATA_FORMAT = "Datum metadata must contain an 'id' key."


def _validate_dataset_type(dataset: Any) -> list[str]:
    issues = []
    is_sized = isinstance(dataset, Sized)
    is_indexable = hasattr(dataset, "__getitem__")
    if not is_sized:
        issues.append(ValidationMessages.DATASET_SIZED)
    if not is_indexable:
        issues.append(ValidationMessages.DATASET_INDEXABLE)
    if is_sized and len(dataset) == 0:
        issues.append(ValidationMessages.DATASET_NONEMPTY)
    return issues


def _validate_id_mapping(value: Any, type_msg: str, format_msg: str) -> list[str]:
    """Return a single message if ``value`` is not a dict, or lacks an 'id' key."""
    if not isinstance(value, dict):
        return [type_msg]
    if "id" not in value:
        return [format_msg]
    return []


def _validate_dataset_metadata(dataset: Any) -> list[str]:
    # Report a single message for the most specific root cause rather than stacking
    # the missing/type/format messages for one underlying problem.
    if not hasattr(dataset, "metadata"):
        return [ValidationMessages.DATASET_METADATA]
    return _validate_id_mapping(
        dataset.metadata, ValidationMessages.DATASET_METADATA_TYPE, ValidationMessages.DATASET_METADATA_FORMAT
    )


def _validate_datum_type(datum: Any) -> list[str]:
    issues = []
    if not isinstance(datum, tuple):
        issues.append(ValidationMessages.DATUM_TYPE)
    if datum is None or isinstance(datum, Sized) and len(datum) != 3:
        issues.append(ValidationMessages.DATUM_FORMAT)
    return issues


def _validate_datum_image(image: Any) -> list[str]:
    issues = []
    if not isinstance(image, Array) or len(image.shape) != 3:
        issues.append(ValidationMessages.DATUM_IMAGE_TYPE)
    if (
        not isinstance(image, Array)
        or len(image.shape) == 3
        and (image.shape[0] > image.shape[1] or image.shape[0] > image.shape[2])
    ):
        issues.append(ValidationMessages.DATUM_IMAGE_FORMAT)
    return issues


def _validate_datum_target_ic(target: Any) -> list[str]:
    issues = []
    if not isinstance(target, Array) or len(target.shape) != 1:
        issues.append(ValidationMessages.DATUM_TARGET_IC_TYPE)
    if target is None or sum(target) > 1 + 1e-6 or sum(target) < 1 - 1e-6:
        issues.append(ValidationMessages.DATUM_TARGET_IC_FORMAT)
    return issues


def _safe_shape(value: Any) -> tuple[int, ...] | None:
    """Return the array shape of ``value``, or ``None`` if it is missing or not array-like.

    A ``None`` attribute or a ragged/non-coercible value would otherwise make
    ``np.asarray`` raise, crashing validation instead of reporting an issue.
    """
    if value is None:
        return None
    try:
        return np.asarray(value).shape
    except (ValueError, TypeError):
        return None


def _validate_datum_target_od(target: Any) -> list[str]:
    if not isinstance(target, ObjectDetectionTarget):
        return [ValidationMessages.DATUM_TARGET_OD_TYPE]
    issues = []
    labels_shape = _safe_shape(target.labels)
    boxes_shape = _safe_shape(target.boxes)
    scores_shape = _safe_shape(target.scores)
    if labels_shape is None or len(labels_shape) != 1:
        issues.append(ValidationMessages.DATUM_TARGET_OD_LABELS_TYPE)
    if boxes_shape is None or len(boxes_shape) != 2 or boxes_shape[1] != 4:
        issues.append(ValidationMessages.DATUM_TARGET_OD_BOXES_TYPE)
    if scores_shape is None or len(scores_shape) not in (1, 2):
        issues.append(ValidationMessages.DATUM_TARGET_OD_SCORES_TYPE)
    return issues


def _validate_datum_video(video_input: Any) -> list[str]:
    # A VideoStream is an iterable of frames, not a single (C, H, W) image array.
    if isinstance(video_input, Array) or not isinstance(video_input, (Sequence, Iterable)):
        return [ValidationMessages.DATUM_VIDEO_TYPE]
    # Only peek at a frame when the stream is a re-indexable Sequence; consuming a
    # one-shot iterator (or decoding a lazy video stream) just to validate is wasteful.
    if isinstance(video_input, Sequence) and len(video_input) > 0:
        pixels = getattr(video_input[0], "pixels", video_input[0])
        if not isinstance(pixels, Array) or len(pixels.shape) != 3:
            return [ValidationMessages.DATUM_VIDEO_FORMAT]
    return []


def _validate_datum_target_mot(target: Any) -> list[str]:
    issues = []
    frame_tracks = getattr(target, "frame_tracks", None)
    if frame_tracks is None:
        issues.append(ValidationMessages.DATUM_TARGET_MOT_TYPE)
        return issues
    if not isinstance(frame_tracks, Sequence):
        issues.append(ValidationMessages.DATUM_TARGET_MOT_FRAMES_TYPE)
        return issues
    for frame in frame_tracks:
        if not all(hasattr(frame, attr) for attr in ("boxes", "labels", "scores", "track_ids")):
            issues.append(ValidationMessages.DATUM_TARGET_MOT_FRAME_TYPE)
            break
        boxes = np.asarray(frame.boxes)
        if boxes.ndim != 2 or (boxes.size and boxes.shape[1] != 4):
            issues.append(ValidationMessages.DATUM_TARGET_MOT_BOXES_TYPE)
            break
        track_ids = np.asarray(frame.track_ids)
        if track_ids.ndim != 1 or (track_ids.size and not np.issubdtype(track_ids.dtype, np.integer)):
            issues.append(ValidationMessages.DATUM_TARGET_MOT_TRACK_IDS_TYPE)
            break
    return issues


def _detect_target_type(target: Any) -> Literal["ic", "od", "mot", "auto"]:
    if isinstance(target, Array):
        return "ic"
    if hasattr(target, "frame_tracks"):
        return "mot"
    if isinstance(target, ObjectDetectionTarget):
        return "od"
    return "auto"


def _validate_datum_target(target: Any, target_type: Literal["ic", "od", "mot", "auto"]) -> list[str]:
    issues = []
    target_type = _detect_target_type(target) if target_type == "auto" else target_type
    if target_type == "ic":
        issues.extend(_validate_datum_target_ic(target))
    elif target_type == "od":
        issues.extend(_validate_datum_target_od(target))
    elif target_type == "mot":
        issues.extend(_validate_datum_target_mot(target))
    else:
        issues.append(ValidationMessages.DATUM_TARGET_TYPE)
    return issues


def _validate_datum_metadata(metadata: Any) -> list[str]:
    return _validate_id_mapping(
        metadata, ValidationMessages.DATUM_METADATA_TYPE, ValidationMessages.DATUM_METADATA_FORMAT
    )


def validate_dataset(dataset: Any, dataset_type: Literal["ic", "od", "mot", "auto"] = "auto") -> None:
    """
    Validate a dataset for compliance with MAITE protocol.

    Parameters
    ----------
    dataset: Any
        Dataset to validate.
    dataset_type: "ic", "od", "mot", or "auto", default "auto"
        Dataset type, if known.

    Raises
    ------
    ValueError
        Raises exception if dataset is invalid with a list of validation issues.
    """
    issues = []
    issues.extend(_validate_dataset_type(dataset))
    datum = None if issues else dataset[0]  # type: ignore
    issues.extend(_validate_dataset_metadata(dataset))
    issues.extend(_validate_datum_type(datum))

    is_seq = isinstance(datum, Sequence)
    datum_len = len(datum) if is_seq else 0
    input_data = datum[0] if is_seq and datum_len > 0 else None
    target = datum[1] if is_seq and datum_len > 1 else None
    metadata = datum[2] if is_seq and datum_len > 2 else None

    # Multi-object tracking inputs are VideoStreams, not (C, H, W) image arrays,
    # so the input is validated against the video contract for that task.
    effective_type = _detect_target_type(target) if dataset_type == "auto" else dataset_type
    if effective_type == "mot":
        issues.extend(_validate_datum_video(input_data))
    else:
        issues.extend(_validate_datum_image(input_data))
    issues.extend(_validate_datum_target(target, dataset_type))
    issues.extend(_validate_datum_metadata(metadata))

    if issues:
        raise ValueError("Dataset validation issues found:\n - " + "\n - ".join(issues))
