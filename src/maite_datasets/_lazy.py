"""Lazy file-backed array proxy for deferred image decoding."""

from __future__ import annotations

__all__ = ["LazyArray", "pil_rgb_chw_load", "pil_rgb_chw_shape"]

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ArrayTransform = Callable[[NDArray[Any]], NDArray[Any]]


def pil_rgb_chw_load(path: Path | str) -> NDArray[np.uint8]:
    """Open ``path`` with PIL, force RGB, return CHW uint8 array."""
    return np.transpose(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), (2, 0, 1))


def pil_rgb_chw_shape(path: Path | str) -> tuple[int, ...]:
    """Read (3, H, W) from the PIL header without decoding pixels."""
    with Image.open(path) as im:
        w, h = im.size
    return (3, h, w)


class LazyArray:
    """File-backed array that decodes on first numpy access.

    Satisfies :class:`maite_datasets.protocols.Array`. ``shape`` resolves via
    ``shape_loader`` (cheap header read) without triggering pixel decode;
    ``__array__`` / ``__getitem__`` / ``__iter__`` materialize via ``loader``
    and apply ``pending`` transforms in order.
    """

    __slots__ = ("_path", "_loader", "_shape_loader", "_pending", "_array", "_shape")

    def __init__(
        self,
        path: str,
        loader: Callable[[str], NDArray[Any]],
        shape_loader: Callable[[str], tuple[int, ...]] | None = None,
        pending: Sequence[ArrayTransform] | None = None,
    ) -> None:
        self._path = path
        self._loader = loader
        self._shape_loader = shape_loader
        self._pending: list[ArrayTransform] = list(pending) if pending else []
        self._array: NDArray[Any] | None = None
        self._shape: tuple[int, ...] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._array is not None:
            return self._array.shape
        if self._shape is None:
            if self._shape_loader is not None and not self._pending:
                self._shape = self._shape_loader(self._path)
            else:
                self._shape = self._materialize().shape
        return self._shape

    def _materialize(self) -> NDArray[Any]:
        if self._array is None:
            arr = self._loader(self._path)
            for fn in self._pending:
                arr = fn(arr)
            self._array = arr
            self._pending = []
        return self._array

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        arr = self._materialize()
        needs_cast = dtype is not None and np.dtype(dtype) != arr.dtype
        if copy is False and needs_cast:
            raise ValueError(f"Unable to avoid copy while casting LazyArray from {arr.dtype} to {np.dtype(dtype)}")
        if needs_cast:
            return arr.astype(dtype, copy=True)
        if copy:
            return arr.copy()
        return arr

    def __getitem__(self, key: Any) -> Any:
        return self._materialize()[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._materialize())

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        state = "loaded" if self._array is not None else "lazy"
        return f"LazyArray(path={self._path!r}, shape={self.shape}, {state})"
