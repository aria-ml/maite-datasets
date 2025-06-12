from __future__ import annotations

__all__ = []

from typing import Any

import numpy as np
import torch
from PIL import Image

from maite_datasets._base import BaseDatasetMixin


class BaseDatasetTorchMixin(BaseDatasetMixin[torch.Tensor]):
    def _as_array(self, raw: list[Any]) -> torch.Tensor:
        return torch.as_tensor(raw)

    def _one_hot_encode(self, value: int | list[int]) -> torch.Tensor:
        if isinstance(value, int):
            encoded = torch.zeros(len(self.index2label))
            encoded[value] = 1
        else:
            encoded = torch.zeros((len(value), len(self.index2label)))
            encoded[torch.arange(len(value)), value] = 1
        return encoded

    def _read_file(self, path: str) -> torch.Tensor:
        return torch.as_tensor(np.array(Image.open(path)).transpose(2, 0, 1))
