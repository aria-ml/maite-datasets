from __future__ import annotations

__all__ = []

from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterator, Literal, NamedTuple, Sequence, TypeVar, cast

import numpy as np

from maite_datasets._fileio import _ensure_exists
from maite_datasets._protocols import Array, Transform
from maite_datasets._types import (
    AnnotatedDataset,
    DatasetMetadata,
    DatumMetadata,
    ImageClassificationDataset,
    ObjectDetectionDataset,
    ObjectDetectionTarget,
)

_TArray = TypeVar("_TArray", bound=Array)
_TTarget = TypeVar("_TTarget")
_TRawTarget = TypeVar(
    "_TRawTarget",
    Sequence[int],
    Sequence[str],
    Sequence[tuple[list[int], list[list[float]]]],
)
_TAnnotation = TypeVar("_TAnnotation", int, str, tuple[list[int], list[list[float]]])


def _to_datum_metadata(index: int, metadata: dict[str, Any]) -> DatumMetadata:
    _id = metadata.pop("id", index)
    return DatumMetadata(id=_id, **metadata)


class DataLocation(NamedTuple):
    url: str
    filename: str
    md5: bool
    checksum: str


class BaseDatasetMixin(Generic[_TArray]):
    index2label: dict[int, str]

    def _as_array(self, raw: list[Any]) -> _TArray: ...
    def _one_hot_encode(self, value: int | list[int]) -> _TArray: ...
    def _read_file(self, path: str) -> _TArray: ...


class BaseDataset(
    AnnotatedDataset[tuple[_TArray, _TTarget, DatumMetadata]],
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
        transforms: Transform[_TArray] | Sequence[Transform[_TArray]] | None = None,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        self._root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        transforms = transforms if transforms is not None else []
        self.transforms: Sequence[Transform[_TArray]] = transforms if isinstance(transforms, Sequence) else [transforms]
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
            id=self._unique_id(),
            index2label=self.index2label,
            split=self.image_set,
        )

        # Load the data
        self.path: Path = self._get_dataset_dir()
        self._filepaths, self._targets, self._datum_metadata = self._load_data()
        self.size: int = len(self._filepaths)

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{self.__class__.__name__} Dataset"
        sep = "-" * len(title)
        attrs = [f"{k.capitalize()}: {v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        return f"{title}\n{sep}{nt}{nt.join(attrs)}"

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

    def _transform(self, image: _TArray) -> _TArray:
        """Function to transform the image prior to returning based on parameters passed in."""
        for transform in self.transforms:
            image = transform(image)
        return image

    def __len__(self) -> int:
        return self.size


class BaseICDataset(
    BaseDataset[_TArray, _TArray, list[int], int],
    BaseDatasetMixin[_TArray],
    ImageClassificationDataset[_TArray],
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
        img = self._transform(img)

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}

        return img, score, _to_datum_metadata(index, img_metadata)


class BaseODDataset(
    BaseDataset[_TArray, ObjectDetectionTarget[_TArray], _TRawTarget, _TAnnotation],
    BaseDatasetMixin[_TArray],
    ObjectDetectionDataset[_TArray],
):
    """
    Base class for object detection datasets.
    """

    _bboxes_per_size: bool = False

    def __getitem__(self, index: int) -> tuple[_TArray, ObjectDetectionTarget[_TArray], DatumMetadata]:
        """
        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        tuple[TArray, ObjectDetectionTarget[TArray], DatumMetadata]
            Image, target, datum_metadata - target.boxes returns boxes in x0, y0, x1, y1 format
        """
        # Grab the bounding boxes and labels from the annotations
        annotation = cast(_TAnnotation, self._targets[index])
        boxes, labels, additional_metadata = self._read_annotations(annotation)
        # Get the image
        img = self._read_file(self._filepaths[index])
        img_size = img.shape
        img = self._transform(img)
        # Adjust labels if necessary
        if self._bboxes_per_size and boxes:
            boxes = boxes * np.array([[img_size[1], img_size[2], img_size[1], img_size[2]]])
        # Create the Object Detection Target
        target = ObjectDetectionTarget(self._as_array(boxes), self._as_array(labels), self._one_hot_encode(labels))

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}
        img_metadata = img_metadata | additional_metadata

        return img, target, _to_datum_metadata(index, img_metadata)

    @abstractmethod
    def _read_annotations(self, annotation: _TAnnotation) -> tuple[list[list[float]], list[int], dict[str, Any]]: ...
