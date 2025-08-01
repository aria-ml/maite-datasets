"""
Collate functions used with a PyTorch DataLoader to load data from MAITE compliant datasets.
"""

from __future__ import annotations

__all__ = []

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

from maite_datasets._protocols import ArrayLike

T_in = TypeVar("T_in")
T_tgt = TypeVar("T_tgt")
T_md = TypeVar("T_md")


def collate_as_list(
    batch_data_as_singles: Iterable[tuple[T_in, T_tgt, T_md]],
) -> tuple[Sequence[T_in], Sequence[T_tgt], Sequence[T_md]]:
    """
    A collate function that takes a batch of individual data points in the format
    (input, target, metadata) and returns three lists: the input batch, the target batch,
    and the metadata batch. This is useful for loading data with torch.utils.data.DataLoader
    when the target and metadata are not tensors.

    Parameters
    ----------
    batch_data_as_singles : An iterable of (input, target, metadata) tuples.

    Returns
    -------
    tuple[Sequence[T_in], Sequence[T_tgt], Sequence[T_md]]
        A tuple of three lists: the input batch, the target batch, and the metadata batch.
    """
    input_batch: list[T_in] = []
    target_batch: list[T_tgt] = []
    metadata_batch: list[T_md] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(input_datum)
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return input_batch, target_batch, metadata_batch


def collate_as_numpy(
    batch_data_as_singles: Iterable[tuple[ArrayLike, T_tgt, T_md]],
) -> tuple[NDArray[Any], Sequence[T_tgt], Sequence[T_md]]:
    """
    A collate function that takes a batch of individual data points in the format
    (input, target, metadata) and returns the batched input as a single NumPy array with two
    lists: the target batch, and the metadata batch. The inputs must be homogeneous arrays.

    Parameters
    ----------
    batch_data_as_singles : An iterable of (ArrayLike, target, metadata) tuples.

    Returns
    -------
    tuple[NDArray[Any], Sequence[T_tgt], Sequence[T_md]]
        A tuple of a NumPy array and two lists: the input batch, the target batch, and the metadata batch.
    """
    input_batch: list[NDArray[Any]] = []
    target_batch: list[T_tgt] = []
    metadata_batch: list[T_md] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(np.asarray(input_datum))
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return np.stack(input_batch) if input_batch else np.array([]), target_batch, metadata_batch


def collate_as_torch(
    batch_data_as_singles: Iterable[tuple[ArrayLike, T_tgt, T_md]],
) -> tuple[torch.Tensor, Sequence[T_tgt], Sequence[T_md]]:
    """
    A collate function that takes a batch of individual data points in the format
    (input, target, metadata) and returns the batched input as a single torch Tensor with two
    lists: the target batch, and the metadata batch. The inputs must be homogeneous arrays.

    Parameters
    ----------
    batch_data_as_singles : An iterable of (ArrayLike, target, metadata) tuples.

    Returns
    -------
    tuple[torch.Tensor, Sequence[T_tgt], Sequence[T_md]]
        A tuple of a torch Tensor and two lists: the input batch, the target batch, and the metadata batch.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install it to use this function.")

    input_batch: list[torch.Tensor] = []
    target_batch: list[T_tgt] = []
    metadata_batch: list[T_md] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(torch.as_tensor(input_datum))
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return torch.stack(input_batch) if input_batch else torch.tensor([]), target_batch, metadata_batch
