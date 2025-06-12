from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Generic, TypedDict, TypeVar

from typing_extensions import NotRequired, Required

_T_co = TypeVar("_T_co", covariant=True)


class Dataset(Generic[_T_co]):
    """Abstract generic base class for PyTorch style Dataset"""

    def __getitem__(self, index: int) -> _T_co: ...
    def __add__(self, other: Dataset[_T_co]) -> Dataset[_T_co]: ...


class DatasetMetadata(TypedDict):
    id: Required[str]
    index2label: NotRequired[dict[int, str]]
    split: NotRequired[str]


class DatumMetadata(TypedDict, total=False):
    id: Required[str]


_TDatum = TypeVar("_TDatum")
_TArray = TypeVar("_TArray")


class AnnotatedDataset(Dataset[_TDatum]):
    metadata: DatasetMetadata

    def __len__(self) -> int: ...


class ImageClassificationDataset(
    AnnotatedDataset[tuple[_TArray, _TArray, DatumMetadata]]
): ...


@dataclass
class ObjectDetectionTarget(Generic[_TArray]):
    boxes: _TArray
    labels: _TArray
    scores: _TArray


class ObjectDetectionDataset(
    AnnotatedDataset[tuple[_TArray, ObjectDetectionTarget[_TArray], DatumMetadata]]
): ...
