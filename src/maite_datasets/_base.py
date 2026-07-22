from __future__ import annotations

__all__ = []

import inspect
import warnings
from abc import abstractmethod
from collections import namedtuple
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable, Generic, Literal, NamedTuple, Protocol, TypeAlias, TypeVar, cast

import numpy as np
import tifffile as tif
from maite.protocols import DatasetMetadata, DatumMetadata
from maite.protocols import image_classification as ic
from maite.protocols import object_detection as od
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from maite_datasets._fileio import ResourcePart, _download_part, _print
from maite_datasets._lazy import TIFF_EXTENSIONS, LazyArray, chw_loaders, tiff_chw_load, tiff_chw_shape
from maite_datasets.protocols import Array

_T_co = TypeVar("_T_co", covariant=True)
_TArray = TypeVar("_TArray", bound=Array)
_TTarget = TypeVar("_TTarget")
_TODTarget = TypeVar("_TODTarget", bound=od.ObjectDetectionTarget)
_TRawTarget = TypeVar(
    "_TRawTarget",
    Sequence[int],
    Sequence[str],
    Sequence[tuple[list[int], list[list[float]]]],
)
_TAnnotation = TypeVar("_TAnnotation", int, str, tuple[list[int], list[list[float]]])

# Transform shapes accepted by every dataset. An image-only transform receives just
# the image; a datum transform receives the whole (image, target, metadata) tuple and
# is distinguished at runtime by "tuple" appearing in its parameter annotation.
ImageTransform: TypeAlias = Callable[[_TArray], _TArray]
DatumTransform: TypeAlias = Callable[[tuple[_TArray, _TTarget, DatumMetadata]], tuple[_TArray, _TTarget, DatumMetadata]]
DatasetTransform: TypeAlias = ImageTransform[_TArray] | DatumTransform[_TArray, _TTarget]
DatasetTransforms: TypeAlias = (
    DatasetTransform[_TArray, _TTarget] | Sequence[DatasetTransform[_TArray, _TTarget]] | None
)

# Object detection targets. These use the functional ``namedtuple`` factory function so
# the tuple works with TorchVision
ObjectDetectionTargetTuple = namedtuple("ObjectDetectionTargetTuple", ["boxes", "labels", "scores"])


# Multi-object tracking targets. A SingleFrameObjectTrackingTargetTuple is an
# ObjectDetectionTargetTuple plus per-detection track_ids; a
# MultiobjectTrackingTargetTuple holds the per-frame targets for one video as
# its ``frame_tracks`` field. Both satisfy the MAITE multiobject_tracking
# protocols structurally via attribute access. These use class-based NamedTuple
# (rather than the functional ``namedtuple``) so the exported fields carry type
# annotations.
class SingleFrameObjectTrackingTargetTuple(NamedTuple):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike
    track_ids: ArrayLike


class MultiobjectTrackingTargetTuple(NamedTuple):
    frame_tracks: Sequence[SingleFrameObjectTrackingTargetTuple]


class BaseDatasetMixin(Generic[_TArray]):
    index2label: dict[int, str]

    def _as_array(self, raw: list[Any]) -> _TArray: ...
    def _one_hot_encode(self, value: int | list[int]) -> _TArray: ...
    def _read_file(self, path: str) -> _TArray: ...
    def _read_shape(self, path: str) -> tuple[int, ...]: ...


class Dataset(Generic[_T_co]):
    """Abstract generic base class for PyTorch style Dataset"""

    def __getitem__(self, index: int) -> _T_co: ...


class BaseDataset(Dataset[tuple[_TArray, _TTarget, DatumMetadata]]):
    metadata: DatasetMetadata

    def __init__(self, transforms: DatasetTransforms[_TArray, _TTarget]) -> None:
        self.transforms: list[DatumTransform[_TArray, _TTarget]] = []
        self._image_transforms: list[ImageTransform[_TArray]] = []
        self._tuple_only_transforms: list[DatumTransform[_TArray, _TTarget]] = []
        self._lazy: bool = False
        transforms = transforms if isinstance(transforms, Sequence) else [transforms] if transforms else []
        for transform in transforms:
            sig = inspect.signature(transform)
            if len(sig.parameters) != 1:
                warnings.warn(f"Dropping unrecognized transform: {str(transform)}")
            elif "tuple" in str(sig.parameters.values()):
                transform = cast(
                    Callable[
                        [tuple[_TArray, _TTarget, DatumMetadata]],
                        tuple[_TArray, _TTarget, DatumMetadata],
                    ],
                    transform,
                )
                self.transforms.append(transform)
                self._tuple_only_transforms.append(transform)
            else:
                transform = cast(Callable[[_TArray], _TArray], transform)
                self._image_transforms.append(transform)
                self.transforms.append(self._wrap_transform(transform))

    @property
    def lazy(self) -> bool:
        """Whether ``__getitem__`` returns a :class:`LazyArray` for the image."""
        return self._lazy

    @lazy.setter
    def lazy(self, value: bool) -> None:
        if value and getattr(self, "_tuple_only_transforms", None):
            warnings.warn(
                "lazy=True with tuple-style transforms forces image materialization "
                "on each access; only image-only transforms remain deferred.",
                UserWarning,
                stacklevel=2,
            )
        self._lazy = bool(value)

    def _wrap_transform(
        self, transform: Callable[[_TArray], _TArray]
    ) -> Callable[
        [tuple[_TArray, _TTarget, DatumMetadata]],
        tuple[_TArray, _TTarget, DatumMetadata],
    ]:
        def wrapper(
            datum: tuple[_TArray, _TTarget, DatumMetadata],
        ) -> tuple[_TArray, _TTarget, DatumMetadata]:
            image, target, metadata = datum
            return (transform(image), target, metadata)

        return wrapper

    def _transform(self, datum: tuple[_TArray, _TTarget, DatumMetadata]) -> tuple[_TArray, _TTarget, DatumMetadata]:
        """Apply the transform pipeline.

        Eager (``self._lazy=False``): runs ``self.transforms`` in order.

        Lazy + image-only transforms only: returns the datum unchanged; the
        image-only transforms ride along inside the :class:`LazyArray`'s
        ``pending`` and execute at materialization.

        Lazy + tuple-style transforms present: materializes the image and runs
        the full ``self.transforms`` pipeline. Tuple transforms see a real
        ndarray (not a ``LazyArray``), matching the lazy-setter warning's
        promise that tuple transforms force materialization.
        """
        if not self._lazy:
            for transform in self.transforms:
                datum = transform(datum)
            return datum
        if not self._tuple_only_transforms:
            return datum
        img, target, meta = datum
        datum = (cast(_TArray, np.asarray(img)), target, meta)
        for transform in self.transforms:
            datum = transform(datum)
        return datum

    def __len__(self) -> int: ...

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{self.__class__.__name__.replace('Dataset', '')} Dataset"
        sep = "-" * len(title)
        attrs = [
            f"{' '.join(w.capitalize() for w in k.split('_'))}: {v}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        ]
        return f"{title}\n{sep}{nt}{nt.join(attrs)}"


def _dataset_dir(root: Path, name: str) -> Path:
    """Resolve (and create) the per-dataset folder named after `name` under `root`.

    `root` is used as-is when it already points at that folder, so passing either the
    parent or the dataset folder itself works.
    """
    dataset_dir = root if root.stem.lower() == name.lower() else root / name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _merge_datum_metadata(datum_metadata: dict[str, list[Any]], addition: dict[str, Any]) -> None:
    """Append `addition`'s per-datum lists onto `datum_metadata` in place.

    Loaders that accumulate several resources/groups into one dataset must extend the
    existing lists; ``dict.update`` would replace them, leaving metadata shorter than
    ``_filepaths`` and desynchronized from it.
    """
    for key, val in addition.items():
        datum_metadata.setdefault(str(key), []).extend(val)


class BaseDownloadedDataset(
    BaseDataset[_TArray, _TTarget],
    Generic[_TArray, _TTarget, _TRawTarget, _TAnnotation],
):
    """
    Base class for internet downloaded datasets.
    """

    # Each subclass should override the attributes below. ``_resources`` lists the
    # *parts* a dataset is assembled from -- a split, a year, an archive chunk -- and
    # each part carries the interchangeable mirrors it can be fetched from. Datasets
    # published in one place have a single mirror; the nesting is what keeps "which
    # piece of the dataset" and "where to get that piece" from being the same axis.
    _resources: list[ResourcePart]
    _resource_index: int = 0
    index2label: dict[int, str]

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val", "test", "operational", "base"] = "train",
        transforms: DatasetTransforms[_TArray, _TTarget] = None,
        download: bool = False,
        verbose: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(transforms)
        self.lazy = lazy
        self._root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        self.image_set = image_set
        self._verbose = verbose

        # Internal Attributes
        self._download = download
        self._filepaths: list[str]
        self._targets: _TRawTarget
        self._datum_metadata: dict[str, list[Any]]
        self._resource: ResourcePart = self._resources[self._resource_index]
        self._label2index = {v: k for k, v in self.index2label.items()}

        self.metadata: DatasetMetadata = DatasetMetadata(
            **{
                "id": self._unique_id(),
                "index2label": self.index2label,
                "split": self.image_set,
            }
        )

        # Load the data
        self.path: Path = self._get_dataset_dir()
        self._filepaths, self._targets, self._datum_metadata = self._load_data()
        self.size: int = len(self._filepaths)

    @property
    def label2index(self) -> dict[str, int]:
        return self._label2index

    def __iter__(self) -> Iterator[tuple[_TArray, _TTarget, DatumMetadata]]:
        for i in range(len(self)):
            yield self[i]

    def _get_dataset_dir(self) -> Path:
        # Create a designated folder for this dataset (named after the class)
        return _dataset_dir(self._root, self.__class__.__name__)

    def _unique_id(self) -> str:
        return f"{self.__class__.__name__}_{self.image_set}"

    def _download_part(self, part: ResourcePart, directory: Path | None = None) -> None:
        """Fetch `part` into `directory` (the dataset folder by default), trying each mirror."""
        _download_part(part, self.path if directory is None else directory, self._root, self._download, self._verbose)

    def _load_data(self) -> tuple[list[str], _TRawTarget, dict[str, Any]]:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        _print(f"Determining if {self._resource.name} needs to be downloaded.", self._verbose)

        try:
            result = self._load_data_inner()
            _print("No download needed, loaded data successfully.", self._verbose)
        except FileNotFoundError:
            self._download_part(self._resource)
            result = self._load_data_inner()
        return result

    @abstractmethod
    def _load_data_inner(self) -> tuple[list[str], _TRawTarget, dict[str, Any]]: ...

    def _to_datum_metadata(self, index: int, metadata: dict[str, Any]) -> DatumMetadata:
        _id = metadata.pop("id", index)
        return DatumMetadata(id=_id, **metadata)

    def _get_image(
        self,
        path: str,
        extra_pending: Sequence[Callable[[NDArray[Any]], NDArray[Any]]] | None = None,
    ) -> _TArray:
        """Return the image for ``path``. Yields a :class:`LazyArray` when ``self._lazy``.

        ``extra_pending`` holds per-datum image operations the dataset itself needs
        (e.g. cropping to one annotated object). They run before any user transform
        and are always scheduled, so a dataset that reshapes its images stays lazy
        instead of forcing a decode in ``__getitem__``.

        User image-only transforms are only attached when the pipeline has no
        tuple-style transforms; otherwise ``_transform`` materializes and runs
        ``self.transforms`` (which already includes the wrapped image-only
        transforms), and attaching them to ``pending`` would double-apply.
        """
        mixin = cast(BaseDatasetMixin[_TArray], self)
        user_pending = [] if self._tuple_only_transforms else self._image_transforms
        pending = [*(extra_pending or []), *user_pending]
        if self._lazy:
            return cast(
                _TArray,
                LazyArray(
                    path,
                    loader=cast(Callable[[str], NDArray[Any]], mixin._read_file),
                    shape_loader=mixin._read_shape,
                    pending=cast(
                        "Sequence[Callable[[NDArray[Any]], NDArray[Any]]] | None",
                        pending or None,
                    ),
                ),
            )
        image = mixin._read_file(path)
        for transform in extra_pending or []:
            image = cast(_TArray, transform(cast(NDArray[Any], image)))
        return image

    def __len__(self) -> int:
        return self.size


class BaseICDataset(
    BaseDownloadedDataset[_TArray, _TArray, list[int], int],
    BaseDatasetMixin[_TArray],
    BaseDataset[_TArray, _TArray],
    ic.Dataset,
):
    """
    Base class for image classification datasets.
    """

    def __getitem__(self, index: int) -> tuple[_TArray, _TArray, DatumMetadata]:
        """
        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        tuple[TArray, TArray, DatumMetadata]
            Image, target, datum_metadata - where target is one-hot encoding of class.
        """
        # Get the associated label and score
        label = self._targets[index]
        score = self._one_hot_encode(label)
        # Get the image
        img = self._get_image(self._filepaths[index])

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}

        return self._transform((img, score, self._to_datum_metadata(index, img_metadata)))


class BaseODDataset(
    BaseDownloadedDataset[_TArray, _TODTarget, _TRawTarget, _TAnnotation],
    BaseDatasetMixin[_TArray],
    BaseDataset[_TArray, _TODTarget],
    od.Dataset,
):
    """
    Base class for object detection datasets.
    """

    _bboxes_per_size: bool = False

    def __getitem__(self, index: int) -> tuple[_TArray, _TODTarget, DatumMetadata]:
        """
        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        tuple[TArray, ObjectDetectionTarget, DatumMetadata]
            Image, target, datum_metadata - target.boxes returns boxes in x0, y0, x1, y1 format
        """
        # Grab the bounding boxes and labels from the annotations
        annotation = cast(_TAnnotation, self._targets[index])
        boxes, labels, additional_metadata = self._read_annotations(annotation)
        # Get the image (LazyArray when self._lazy)
        img = self._get_image(self._filepaths[index])
        img_size = img.shape
        # Adjust labels if necessary
        scaled_boxes: Any = boxes
        if self._bboxes_per_size and boxes:
            scale = np.asarray([[img_size[1], img_size[2], img_size[1], img_size[2]]])
            scaled_boxes = (np.asarray(boxes) * scale).astype(np.int32)
        # Create the Object Detection Target
        target = ObjectDetectionTargetTuple(
            self._as_array(scaled_boxes), self._as_array(labels), self._one_hot_encode(labels)
        )
        # Cast target explicitly to ODTarget as namedtuple does not provide any typing metadata
        target = cast(_TODTarget, target)

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}
        img_metadata = img_metadata | additional_metadata

        return self._transform((img, target, self._to_datum_metadata(index, img_metadata)))

    @abstractmethod
    def _read_annotations(self, annotation: _TAnnotation) -> tuple[list[list[float]], list[int], dict[str, Any]]: ...


NumpyArray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


class NumpyObjectDetectionTarget(od.ObjectDetectionTarget, Protocol):
    @property
    def boxes(self) -> NumpyArray: ...
    @property
    def labels(self) -> NumpyArray: ...
    @property
    def scores(self) -> NumpyArray: ...


class BaseDatasetNumpyMixin(BaseDatasetMixin[NumpyArray]):
    def _as_array(self, raw: list[Any]) -> NumpyArray:
        return np.asarray(raw)

    def _one_hot_encode(self, value: int | list[int]) -> NumpyArray:
        if isinstance(value, int):
            encoded = np.zeros(len(self.index2label))
            encoded[value] = 1
        else:
            encoded = np.zeros((len(value), len(self.index2label)))
            encoded[np.arange(len(value)), value] = 1
        return encoded

    def _read_file(self, path: str) -> NumpyArray:
        ext = Path(path).suffix.lower()
        if ext in TIFF_EXTENSIONS:
            return tiff_chw_load(path)
        return np.array(Image.open(path)).transpose(2, 0, 1)

    def _read_shape(self, path: str) -> tuple[int, ...]:
        """Read (C, H, W) without decoding pixel data.

        Falls back to decoding the file via ``_read_file`` when the path is
        not a real image file (e.g. integer-index strings used by in-memory
        datasets like MNIST/CIFAR), or when TIFF axis metadata can't be
        interpreted cheaply.
        """
        ext = Path(path).suffix.lower()
        try:
            if ext in TIFF_EXTENSIONS:
                channels, h, w = tiff_chw_shape(path)
            else:
                with Image.open(path) as im:
                    channels = len(im.getbands())
                    w, h = im.size
        except (FileNotFoundError, OSError, ValueError, tif.TiffFileError):
            return self._read_file(path).shape
        return (channels, h, w)


NumpyImageTransform: TypeAlias = ImageTransform[NumpyArray]
NumpyImageClassificationDatumTransform: TypeAlias = DatumTransform[NumpyArray, NumpyArray]
NumpyObjectDetectionDatumTransform: TypeAlias = DatumTransform[NumpyArray, NumpyObjectDetectionTarget]
NumpyImageClassificationTransform: TypeAlias = DatasetTransform[NumpyArray, NumpyArray]
NumpyObjectDetectionTransform: TypeAlias = DatasetTransform[NumpyArray, NumpyObjectDetectionTarget]

# Transform type accepted by reader-created datasets. Reader images satisfy the
# ``Array`` protocol (numpy ndarray when eager, LazyArray when lazy), so the alias
# is expressed against ``Array`` rather than a concrete array type.
ReaderTransform: TypeAlias = DatasetTransform[Array, Any]
ReaderTransforms: TypeAlias = DatasetTransforms[Array, Any]


class LazyAnnotations(Sequence[dict[str, Any]]):
    """Per-box annotation metadata that is only materialized when read.

    Behaves like the ``list[dict]`` it replaces - indexing, iteration, ``len`` and
    equality all work - but a training loop that never touches the ``annotations``
    key of a datum's metadata pays nothing for it. ``len`` is answered from the
    known count without building anything.
    """

    __slots__ = ("_build", "_count", "_records")

    def __init__(self, count: int, build: Callable[[], list[dict[str, Any]]]) -> None:
        self._count = count
        self._build = build
        self._records: list[dict[str, Any]] | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        """The materialized annotation dicts, built on first access."""
        if self._records is None:
            self._records = self._build()
        return self._records

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: Any) -> Any:
        return self.records[index]

    def __eq__(self, other: object) -> bool:
        return self.records == (other.records if isinstance(other, LazyAnnotations) else other)

    def __repr__(self) -> str:
        state = "built" if self._records is not None else "pending"
        return f"LazyAnnotations({self._count} annotations, {state})"


class BaseReaderDataset(BaseDataset[Array, _TTarget]):
    """Shared implementation for datasets backed by a :class:`BaseDatasetReader`.

    Owns the reader handle, the public attribute block that :meth:`BaseDataset.__str__`
    renders, and the single eager-vs-lazy image load, so every reader format decodes
    the same way and schedules pending image transforms identically.

    Parameters
    ----------
    reader : BaseDatasetReader
        Reader providing the resolved paths and parsed annotations.
    size : int
        Number of data points the reader indexed.
    lazy : bool, default False
        When True, the image element of each datum is returned as a
        :class:`LazyArray` that defers decode until first numpy access.
    transforms : ReaderTransforms, default None
        Optional image-only or datum-tuple transform(s) applied to each datum
        on access via the inherited transform pipeline.
    """

    def __init__(
        self,
        reader: Any,
        size: int,
        lazy: bool = False,
        transforms: ReaderTransforms = None,
    ) -> None:
        super().__init__(transforms)
        self._reader = reader
        self.lazy = lazy

        self.path: Path = reader.dataset_path
        self.size: int = size
        self.classes: dict[int, str] = reader.index2label
        self.metadata: DatasetMetadata = DatasetMetadata(
            id=reader.dataset_id,
            index2label=reader.index2label,
        )

    def __len__(self) -> int:
        return self.size

    def _get_image(self, path: Path) -> Array:
        """Return the image at `path`, deferring decode when ``self.lazy``.

        Pending image-only transforms ride along inside the :class:`LazyArray` so
        ``_transform``'s lazy short-circuit still applies them at materialization.
        """
        loader, shape_loader = chw_loaders(path)
        if not self._lazy:
            return loader(path)
        pending = None if self._tuple_only_transforms else self._image_transforms or None
        return LazyArray(
            str(path),
            loader=cast(Callable[[str], NDArray[Any]], loader),
            shape_loader=cast(Callable[[str], tuple[int, ...]], shape_loader),
            pending=cast("Sequence[Callable[[NDArray[Any]], NDArray[Any]]] | None", pending),
        )

    def _get_shape(self, path: Path, image: Array) -> tuple[int, ...]:
        """Shape of the stored image, without forcing a decode in lazy mode.

        ``LazyArray.shape`` materializes once transforms are pending, so probe the
        header directly instead; both modes then size boxes against the on-disk
        image, before any transform runs.
        """
        if not self._lazy:
            return image.shape
        _, shape_loader = chw_loaders(path)
        return shape_loader(path)
