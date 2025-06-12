from __future__ import annotations

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from maite_datasets._base import BaseDatasetMixin


class BaseDatasetNumpyMixin(BaseDatasetMixin[NDArray[np.number[Any]]]):
    def _as_array(self, raw: list[Any]) -> NDArray[np.number[Any]]:
        return np.asarray(raw)

    def _one_hot_encode(self, value: int | list[int]) -> NDArray[np.number[Any]]:
        if isinstance(value, int):
            encoded = np.zeros(len(self.index2label))
            encoded[value] = 1
        else:
            encoded = np.zeros((len(value), len(self.index2label)))
            encoded[np.arange(len(value)), value] = 1
        return encoded

    def _read_file(self, path: str) -> NDArray[np.number[Any]]:
        return np.array(Image.open(path)).transpose(2, 0, 1)
