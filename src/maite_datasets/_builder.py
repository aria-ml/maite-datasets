from __future__ import annotations

__all__ = []

from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    cast,
)

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
import numpy as np
from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata

from maite_datasets._base import MultiObjectTrackingTargetTuple, SingleFrameObjectTrackingTargetTuple
from maite_datasets._video import frames_to_stream
from maite_datasets.protocols import Array

if TYPE_CHECKING:
    import maite.protocols.multiobject_tracking as mot


def _ensure_id(index: int, metadata: dict[str, Any]) -> DatumMetadata:
    return DatumMetadata(**({"id": index, **metadata} if "id" not in metadata else metadata))


def _validate_metadata_length(
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
    dataset_len: int,
    noun: str,
) -> None:
    """Raise if per-datum ``metadata`` does not line up with ``dataset_len`` items."""
    if metadata is not None and (
        len(metadata) != dataset_len
        if isinstance(metadata, Sequence)
        else any(
            not isinstance(metadatum, Sequence) or len(metadatum) != dataset_len for metadatum in metadata.values()
        )
    ):
        raise ValueError(f"Number of metadata ({len(metadata)}) does not match number of {noun} ({dataset_len}).")


def _validate_data(
    datum_type: Literal["ic", "od"],
    images: Array | Sequence[Array],
    labels: Array | Sequence[int] | Sequence[Array] | Sequence[Sequence[int]],
    bboxes: Array | Sequence[Array] | Sequence[Sequence[Array]] | Sequence[Sequence[Sequence[float]]] | None,
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
) -> None:
    # Validate inputs
    dataset_len = len(images)

    if not isinstance(images, (Sequence, Array)) or len(images[0].shape) != 3:
        raise ValueError("Images must be a sequence or array of 3 dimensional arrays (H, W, C).")
    if len(labels) != dataset_len:
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of images ({dataset_len}).")
    if bboxes is not None and len(bboxes) != dataset_len:
        raise ValueError(f"Number of bboxes ({len(bboxes)}) does not match number of images ({dataset_len}).")
    _validate_metadata_length(metadata, dataset_len, "images")

    if datum_type == "ic":
        if not isinstance(labels, (Sequence, Array)) or not isinstance(labels[0], (int, SupportsInt)):
            raise TypeError("Labels must be a sequence of integers for image classification.")
    elif datum_type == "od":
        _validate_od_labels_and_bboxes(labels, bboxes)
    else:
        raise ValueError(f"Unknown datum type '{datum_type}'. Must be 'ic' or 'od'.")


def _validate_od_labels_and_bboxes(
    labels: Array | Sequence[int] | Sequence[Array] | Sequence[Sequence[int]],
    bboxes: Array | Sequence[Array] | Sequence[Sequence[Array]] | Sequence[Sequence[Sequence[float]]] | None,
) -> None:
    if not isinstance(labels, (Sequence, Array)) or not isinstance(labels[0], (Sequence, Array)):
        raise TypeError("Labels must be a sequence of sequences of integers for object detection.")
    # Find the first non-empty label sequence to validate the inner element type.
    nested_labels = cast(Sequence[Sequence[Any]], labels)
    first_label = next((label for label in nested_labels if len(label) > 0), None)
    if first_label is not None and not isinstance(first_label[0], (int, SupportsInt)):
        raise TypeError("Labels must be a sequence of sequences of integers for object detection.")
    if bboxes is None or not isinstance(bboxes, (Sequence, Array)) or not isinstance(bboxes[0], (Sequence, Array)):
        raise TypeError("Boxes must be a sequence of sequences of (x0, y0, x1, y1) for object detection.")
    # Find the first box across all images to validate the box format.
    first_box = next((cast(Sequence[Any], box) for image_bboxes in bboxes for box in image_bboxes), None)
    if first_box is not None and (
        not isinstance(first_box, (Sequence, Array))
        or not isinstance(first_box[0], (float, SupportsFloat))
        or not len(first_box) == 4
    ):
        raise TypeError("Boxes must be a sequence of sequences of (x0, y0, x1, y1) for object detection.")


def _listify_metadata(
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
) -> Sequence[dict[str, Any]] | None:
    if isinstance(metadata, dict):
        return [{k: v[i] for k, v in metadata.items()} for i in range(len(next(iter(metadata.values()))))]
    return metadata


def _find_max(arr: ArrayLike) -> Any:
    if not isinstance(arr, (bytes, str)) and isinstance(arr, (Iterable, Sequence, Array)):
        nested = [x for x in [_find_max(x) for x in arr] if x is not None]
        return max(nested) if len(nested) > 0 else None
    return arr


_TLabels = TypeVar("_TLabels", Sequence[int], Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]])


class BaseAnnotatedDataset(Generic[_TLabels]):
    metadata: DatasetMetadata

    def __init__(
        self,
        datum_type: Literal["ic", "od", "mot"],
        data: Array | Sequence[Array],
        labels: _TLabels,
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        self._classes = classes if classes is not None else [str(i) for i in range(_find_max(labels) + 1)]
        self._index2label = dict(enumerate(self._classes))
        self._data = data
        self._labels = labels
        self._metadata = metadata
        unit = "video" if datum_type == "mot" else "image"
        self._id = name or f"{len(self._data)}_{unit}_{len(self._index2label)}_class_{datum_type}_dataset"
        self.metadata = DatasetMetadata(id=self._id, index2label=self._index2label)

    def __len__(self) -> int:
        return len(self._data)


class CustomImageClassificationDataset(BaseAnnotatedDataset[Sequence[int]], ic.Dataset):
    def __init__(
        self,
        images: Array | Sequence[Array],
        labels: Array | Sequence[int],
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            "ic",
            images,
            np.asarray(labels).tolist() if isinstance(labels, Array) else labels,
            metadata,
            classes,
        )
        if name is not None:
            self.__name__ = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = name

    def __getitem__(self, idx: int, /) -> tuple[Array, Array, DatumMetadata]:
        one_hot = [0.0] * len(self._index2label)
        one_hot[self._labels[idx]] = 1.0
        return (
            self._data[idx],
            np.asarray(one_hot),
            _ensure_id(idx, self._metadata[idx] if self._metadata is not None else {}),
        )


class CustomObjectDetectionTarget:
    def __init__(
        self,
        labels: Sequence[int],
        bboxes: Sequence[Sequence[float]],
        class_count: int,
    ) -> None:
        self._labels = labels
        self._bboxes = bboxes
        # Build one independent row per detection; `[[0.0] * n] * m` would alias a
        # single inner list across all rows, corrupting scores for >1 detection.
        one_hot = [[0.0] * class_count for _ in range(len(labels))]
        for i, label in enumerate(labels):
            one_hot[i][label] = 1.0
        self._scores = one_hot

    @property
    def labels(self) -> Sequence[int]:
        return self._labels

    @property
    def boxes(self) -> Sequence[Sequence[float]]:
        return self._bboxes

    @property
    def scores(self) -> Sequence[Sequence[float]]:
        return self._scores


class CustomObjectDetectionDataset(BaseAnnotatedDataset[Sequence[Sequence[int]]], od.Dataset):
    def __init__(
        self,
        images: Array | Sequence[Array],
        labels: Array | Sequence[Array] | Sequence[Sequence[int]],
        bboxes: Array | Sequence[Array] | Sequence[Sequence[Array]] | Sequence[Sequence[Sequence[float]]],
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            "od",
            images,
            [np.asarray(label).tolist() if isinstance(label, Array) else label for label in labels],
            metadata,
            classes,
        )
        if name is not None:
            self.__name__ = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = name
        self._bboxes = [
            [np.asarray(box).tolist() if isinstance(box, Array) else box for box in bbox] for bbox in bboxes
        ]

    def __getitem__(self, idx: int, /) -> tuple[Array, CustomObjectDetectionTarget, DatumMetadata]:
        return (
            self._data[idx],
            CustomObjectDetectionTarget(self._labels[idx], self._bboxes[idx], len(self._classes)),
            _ensure_id(idx, self._metadata[idx] if self._metadata is not None else {}),
        )


def to_image_classification_dataset(
    images: Array | Sequence[Array],
    labels: Array | Sequence[int],
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
    classes: Sequence[str] | None,
    name: str | None = None,
) -> ic.Dataset:
    """
    Helper function to create custom image classification Dataset classes.

    Parameters
    ----------
    images : Array | Sequence[Array]
        The images to use in the dataset.
    labels : Array | Sequence[int]
        The labels to use in the dataset.
    metadata : Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None
        The metadata to use in the dataset.
    classes : Sequence[str] | None
        The classes to use in the dataset.

    Returns
    -------
    Dataset
    """
    _validate_data("ic", images, labels, None, metadata)
    return CustomImageClassificationDataset(images, labels, _listify_metadata(metadata), classes, name)


def to_object_detection_dataset(
    images: Array | Sequence[Array],
    labels: Array | Sequence[Array] | Sequence[Sequence[int]],
    bboxes: Array | Sequence[Array] | Sequence[Sequence[Array]] | Sequence[Sequence[Sequence[float]]],
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
    classes: Sequence[str] | None,
    name: str | None = None,
) -> od.Dataset:
    """
    Helper function to create custom object detection Dataset classes.

    Parameters
    ----------
    images : Array | Sequence[Array]
        The images to use in the dataset.
    labels : Array | Sequence[Array] | Sequence[Sequence[int]]
        The labels to use in the dataset.
    bboxes : Array | Sequence[Array] | Sequence[Sequence[Array]] | Sequence[Sequence[Sequence[float]]]
        The bounding boxes (x0,y0,x1,y0) to use in the dataset.
    metadata : Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None
        The metadata to use in the dataset.
    classes : Sequence[str] | None
        The classes to use in the dataset.

    Returns
    -------
    Dataset
    """
    _validate_data("od", images, labels, bboxes, metadata)
    return CustomObjectDetectionDataset(images, labels, bboxes, _listify_metadata(metadata), classes, name)


def _validate_mot_data(
    videos: Sequence[Any],
    labels: Sequence[Sequence[Sequence[int]]],
    bboxes: Sequence[Sequence[Sequence[Sequence[float]]]],
    track_ids: Sequence[Sequence[Sequence[int]]],
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
) -> None:
    dataset_len = len(videos)
    if not isinstance(videos, (Sequence, Array)) or dataset_len == 0:
        raise ValueError("Videos must be a non-empty sequence of frame sequences.")
    for label, seq in (("labels", labels), ("bboxes", bboxes), ("track_ids", track_ids)):
        if len(seq) != dataset_len:
            raise ValueError(f"Number of {label} ({len(seq)}) does not match number of videos ({dataset_len}).")
    _validate_metadata_length(metadata, dataset_len, "videos")

    # Per-video and per-frame structural consistency: one annotation list per frame,
    # and equal-length labels/bboxes/track_ids per frame (one entry per detection).
    for i, frames in enumerate(videos):
        n_frames = len(frames)
        if not len(labels[i]) == len(bboxes[i]) == len(track_ids[i]) == n_frames:
            raise ValueError(
                f"Video {i}: labels, bboxes and track_ids must each provide one entry per frame ({n_frames})."
            )
        for f in range(n_frames):
            if not len(labels[i][f]) == len(bboxes[i][f]) == len(track_ids[i][f]):
                raise ValueError(
                    f"Video {i} frame {f}: labels, bboxes and track_ids must have equal length (one per detection)."
                )


def _as_video_stream(video: Any) -> Any:
    """Pass through frames that already satisfy the VideoFrame protocol; wrap raw arrays."""
    if isinstance(video, Sequence) and (len(video) == 0 or not hasattr(video[0], "pixels")):
        return frames_to_stream(video)
    return video


def _build_mot_target(
    bboxes: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[Sequence[int]],
    track_ids: Sequence[Sequence[int]],
) -> MultiObjectTrackingTargetTuple:
    """Build one video's per-frame tracking target from its raw annotation lists."""
    frame_tracks = [
        SingleFrameObjectTrackingTargetTuple(
            boxes=np.asarray(boxes_f, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels_f, dtype=np.int64),
            scores=np.ones(len(labels_f), dtype=np.float32),
            track_ids=np.asarray(track_ids_f, dtype=np.int64),
        )
        for boxes_f, labels_f, track_ids_f in zip(bboxes, labels, track_ids)
    ]
    return MultiObjectTrackingTargetTuple(frame_tracks=frame_tracks)


class CustomMultiObjectTrackingDataset(BaseAnnotatedDataset[Sequence[Sequence[Sequence[int]]]]):
    def __init__(
        self,
        videos: Sequence[Any],
        labels: Sequence[Sequence[Sequence[int]]],
        bboxes: Sequence[Sequence[Sequence[Sequence[float]]]],
        track_ids: Sequence[Sequence[Sequence[int]]],
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        super().__init__("mot", videos, labels, metadata, classes)
        if name is not None:
            self.__name__ = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = name
        # The in-memory inputs are already fully materialized, so normalize streams
        # and build targets once here rather than repeating the work on each access.
        self._videos = [_as_video_stream(video) for video in videos]
        self._targets = [_build_mot_target(bboxes[i], labels[i], track_ids[i]) for i in range(len(videos))]

    def __getitem__(self, idx: int, /) -> tuple[Any, MultiObjectTrackingTargetTuple, DatumMetadata]:
        return (
            self._videos[idx],
            self._targets[idx],
            _ensure_id(idx, self._metadata[idx] if self._metadata is not None else {}),
        )


def to_multi_object_tracking_dataset(
    videos: Sequence[Any],
    labels: Sequence[Sequence[Sequence[int]]],
    bboxes: Sequence[Sequence[Sequence[Sequence[float]]]],
    track_ids: Sequence[Sequence[Sequence[int]]],
    metadata: Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None,
    classes: Sequence[str] | None,
    name: str | None = None,
) -> mot.Dataset:
    """
    Helper function to create custom multi-object tracking Dataset classes.

    Each datum is a video: an input stream of frames, a per-frame set of tracked
    detections, and datum-level metadata. ``labels``, ``bboxes`` and ``track_ids``
    are nested per video, then per frame, then per detection.

    Parameters
    ----------
    videos : Sequence[Sequence[Any]]
        The videos to use in the dataset. Each video is a sequence of frames, where
        a frame is either a ``VideoFrame`` (anything exposing ``pixels``) or a raw
        ``(C, H, W)`` array (wrapped into a ``VideoFrame`` with synthesized timing).
    labels : Sequence[Sequence[Sequence[int]]]
        Per video, per frame, per detection integer class labels.
    bboxes : Sequence[Sequence[Sequence[Sequence[float]]]]
        Per video, per frame, per detection bounding boxes in (x0, y0, x1, y1) format.
    track_ids : Sequence[Sequence[Sequence[int]]]
        Per video, per frame, per detection track identifiers (>= 0, -1 for untracked).
    metadata : Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | None
        The datum-level metadata to use in the dataset.
    classes : Sequence[str] | None
        The classes to use in the dataset.

    Returns
    -------
    Dataset
    """
    _validate_mot_data(videos, labels, bboxes, track_ids, metadata)
    # Structurally compliant with mot.Dataset without importing the protocol at
    # runtime (keeps the maite floor low for ic/od-only users); cast for typing.
    return cast(
        "mot.Dataset",
        CustomMultiObjectTrackingDataset(videos, labels, bboxes, track_ids, _listify_metadata(metadata), classes, name),
    )
