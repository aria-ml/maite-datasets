from __future__ import annotations

__all__ = []

import inspect
import warnings
from abc import abstractmethod
from collections import namedtuple
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable, Generic, Literal, NamedTuple, Protocol, TypeVar, cast

import numpy as np
from maite.protocols import DatasetMetadata, DatumMetadata
from maite.protocols import image_classification as ic
from maite.protocols import object_detection as od
from numpy.typing import NDArray
from PIL import Image

from maite_datasets._fileio import _ensure_exists
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

ObjectDetectionTargetTuple = namedtuple("ObjectDetectionTargetTuple", ["boxes", "labels", "scores"])


class BaseDatasetMixin(Generic[_TArray]):
    index2label: dict[int, str]

    def _as_array(self, raw: list[Any]) -> _TArray: ...
    def _one_hot_encode(self, value: int | list[int]) -> _TArray: ...
    def _read_file(self, path: str) -> _TArray: ...


class Dataset(Generic[_T_co]):
    """Abstract generic base class for PyTorch style Dataset"""

    def __getitem__(self, index: int) -> _T_co: ...
    def __add__(self, other: Dataset[_T_co]) -> Dataset[_T_co]: ...


class BaseDataset(Dataset[tuple[_TArray, _TTarget, DatumMetadata]]):
    metadata: DatasetMetadata

    def __init__(
        self,
        transforms: Callable[[_TArray], _TArray]
        | Callable[
            [tuple[_TArray, _TTarget, DatumMetadata]],
            tuple[_TArray, _TTarget, DatumMetadata],
        ]
        | Sequence[
            Callable[[_TArray], _TArray]
            | Callable[
                [tuple[_TArray, _TTarget, DatumMetadata]],
                tuple[_TArray, _TTarget, DatumMetadata],
            ]
        ]
        | None,
    ) -> None:
        self.transforms: Sequence[
            Callable[
                [tuple[_TArray, _TTarget, DatumMetadata]],
                tuple[_TArray, _TTarget, DatumMetadata],
            ]
        ] = []
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
            else:
                transform = cast(Callable[[_TArray], _TArray], transform)
                self.transforms.append(self._wrap_transform(transform))

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
        """Function to transform the image prior to returning based on parameters passed in."""
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


class DataLocation(NamedTuple):
    url: str
    filename: str
    md5: bool
    checksum: str


class BaseDownloadedDataset(
    BaseDataset[_TArray, _TTarget],
    Generic[_TArray, _TTarget, _TRawTarget, _TAnnotation],
):
    """
    Base class for internet downloaded datasets.
    """

    # Each subclass should override the attributes below.
    # Each resource tuple must contain:
    #    'url': str, the URL to download from
    #    'filename': str, the name of the file once downloaded
    #    'md5': boolean, True if it's the checksum value is md5
    #    'checksum': str, the associated checksum for the downloaded file
    _resources: list[DataLocation]
    _resource_index: int = 0
    index2label: dict[int, str]

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val", "test", "operational", "base"] = "train",
        transforms: Callable[[_TArray], _TArray]
        | Callable[
            [tuple[_TArray, _TTarget, DatumMetadata]],
            tuple[_TArray, _TTarget, DatumMetadata],
        ]
        | Sequence[
            Callable[[_TArray], _TArray]
            | Callable[
                [tuple[_TArray, _TTarget, DatumMetadata]],
                tuple[_TArray, _TTarget, DatumMetadata],
            ]
        ]
        | None = None,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(transforms)
        self._root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        self.image_set = image_set
        self._verbose = verbose

        # Internal Attributes
        self._download = download
        self._filepaths: list[str]
        self._targets: _TRawTarget
        self._datum_metadata: dict[str, list[Any]]
        self._resource: DataLocation = self._resources[self._resource_index]
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
        if self._root.stem.lower() == self.__class__.__name__.lower():
            dataset_dir: Path = self._root
        else:
            dataset_dir: Path = self._root / self.__class__.__name__.lower()
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _unique_id(self) -> str:
        return f"{self.__class__.__name__}_{self.image_set}"

    def _load_data(self) -> tuple[list[str], _TRawTarget, dict[str, Any]]:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        if self._verbose:
            print(f"Determining if {self._resource.filename} needs to be downloaded.")

        try:
            result = self._load_data_inner()
            if self._verbose:
                print("No download needed, loaded data successfully.")
        except FileNotFoundError:
            _ensure_exists(*self._resource, self.path, self._root, self._download, self._verbose)
            result = self._load_data_inner()
        return result

    @abstractmethod
    def _load_data_inner(self) -> tuple[list[str], _TRawTarget, dict[str, Any]]: ...

    def _to_datum_metadata(self, index: int, metadata: dict[str, Any]) -> DatumMetadata:
        _id = metadata.pop("id", index)
        return DatumMetadata(id=_id, **metadata)

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
        img = self._read_file(self._filepaths[index])

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
        # Get the image
        img = self._read_file(self._filepaths[index])
        img_size = img.shape
        # Adjust labels if necessary
        if self._bboxes_per_size and boxes:
            boxes = boxes * np.asarray([[img_size[1], img_size[2], img_size[1], img_size[2]]])
        # Create the Object Detection Target
        target = ObjectDetectionTargetTuple(self._as_array(boxes), self._as_array(labels), self._one_hot_encode(labels))
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
        return np.array(Image.open(path)).transpose(2, 0, 1)


NumpyImageTransform = Callable[[NumpyArray], NumpyArray]
NumpyImageClassificationDatumTransform = Callable[
    [tuple[NumpyArray, NumpyArray, DatumMetadata]],
    tuple[NumpyArray, NumpyArray, DatumMetadata],
]
NumpyObjectDetectionDatumTransform = Callable[
    [tuple[NumpyArray, NumpyObjectDetectionTarget, DatumMetadata]],
    tuple[NumpyArray, NumpyObjectDetectionTarget, DatumMetadata],
]
NumpyImageClassificationTransform = NumpyImageTransform | NumpyImageClassificationDatumTransform
NumpyObjectDetectionTransform = NumpyImageTransform | NumpyObjectDetectionDatumTransform
