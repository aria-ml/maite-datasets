"""
Common type protocols used for interoperability.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar, overload, runtime_checkable


@runtime_checkable
class Array(Protocol):
    """
    Protocol for interoperable array objects.

    Supports common array representations with popular libraries like
    PyTorch, Tensorflow and JAX, as well as NumPy arrays.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __array__(self) -> Any: ...
    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...


TBoxes = TypeVar("TBoxes", Array, Sequence)
TLabels = TypeVar("TLabels", Array, Sequence)
TScores = TypeVar("TScores", Array, Sequence)


class GenericObjectDetectionTarget(Generic[TBoxes, TLabels, TScores], Protocol):
    boxes: TBoxes
    labels: TLabels
    scores: TScores


@runtime_checkable
class HFDatasetInfo(Protocol):
    @property
    def dataset_name(self) -> str: ...


@runtime_checkable
class HFDataset(Protocol):
    @property
    def features(self) -> Mapping[str, Any]: ...

    @property
    def builder_name(self) -> str | None: ...

    @property
    def info(self) -> HFDatasetInfo: ...

    @overload
    def __getitem__(self, key: int | slice | Iterable[int]) -> dict[str, Any]: ...
    @overload
    def __getitem__(self, key: str) -> Sequence[int]: ...
    def __getitem__(self, key: str | int | slice | Iterable[int]) -> dict[str, Any] | Sequence[int]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class HFFeature(Protocol):
    @property
    def _type(self) -> str: ...


@runtime_checkable
class HFClassLabel(HFFeature, Protocol):
    @property
    def names(self) -> list[str]: ...

    @property
    def num_classes(self) -> int: ...


@runtime_checkable
class HFImage(HFFeature, Protocol):
    @property
    def decode(self) -> bool: ...


@runtime_checkable
class HFArray(HFFeature, Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> str: ...


@runtime_checkable
class HFList(HFFeature, Protocol):
    @property
    def feature(self) -> Any: ...
    @property
    def length(self) -> int: ...


@runtime_checkable
class HFValue(HFFeature, Protocol):
    @property
    def pa_type(self) -> Any: ...  # pyarrow type ... not documented
    @property
    def dtype(self) -> str: ...
