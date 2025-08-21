"""
Common type protocols used for interoperability.
"""

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


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
