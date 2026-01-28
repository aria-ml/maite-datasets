from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from maite_datasets._base import (
    BaseDatasetNumpyMixin,
    BaseICDataset,
    DataLocation,
    NumpyArray,
    NumpyImageClassificationTransform,
)
from maite_datasets.image_classification._mnist_corruptions import ALL_CORRUPTIONS

MNISTClassStringMap = Literal["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TMNISTClassMap = TypeVar("TMNISTClassMap", MNISTClassStringMap, int, list[MNISTClassStringMap], list[int])
CorruptionStringMap = Literal[
    "identity",
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "gaussian_blur",
    "glass_blur",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "fog",
    "snow",
    "spatter",
    "contrast",
    "brightness",
    "saturate",
    "jpeg_compression",
    "pixelate",
    "elastic_transform",
    "quantize",
    "shear",
    "rotate",
    "scale",
    "translate",
    "line",
    "dotted_line",
    "zigzag",
    "inverse",
    "stripe",
    "canny_edges",
]


class MNIST(BaseICDataset[NumpyArray], BaseDatasetNumpyMixin):
    """`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ Dataset and `Corruptions <https://arxiv.org/abs/1906.02337>`_.

    There are 29 different styles of corruptions. This class downloads the original dataset and applies the
    corruptions if any corruptions are selected. Note that if corruption is "identity" or "None", the original
    dataset will be returned.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or the ``minst`` folder of the already downloaded data.
    image_set : "train", "test" or "base", default "train"
        If "base", returns all of the data to allow the user to create their own splits.
    corruption : "identity", "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "gaussian_blur", "glass_blur", "defocus_blur", "motion_blur", "zoom_blur", "fog", "snow", "spatter", "contrast", "brightness", "saturate", "jpeg_compression", "pixelate", "elastic_transform", "quantize", "shear", "rotate", "scale", "translate", "line", "dotted_line", "zigzag", "inverse", "stripe", "canny_edges", or None, default None
        Corruption to apply to the data.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "test" or "base"
        The selected image set from the dataset.
    index2label : dict[int, str]
        Dictionary which translates from class integers to the associated class strings.
    label2index : dict[str, int]
        Dictionary which translates from class strings to the associated class integers.
    metadata : DatasetMetadata
        Typed dictionary containing dataset metadata, such as `id` which returns the dataset class name.
    corruption : str or None
        Corruption applied to the data.
    transforms : Sequence[Transform]
        The transforms to be applied to the data.
    size : int
        The size of the dataset.
    
    Note
    ----
    Data License: `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_ for corruption dataset
    """  # noqa: E501

    _resources = [
        DataLocation(
            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            filename="mnist.npz",
            md5=False,
            checksum="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        ),
    ]

    index2label: dict[int, str] = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "test", "base"] = "train",
        corruption: CorruptionStringMap | None = None,
        transforms: NumpyImageClassificationTransform | Sequence[NumpyImageClassificationTransform] | None = None,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        self.corruption = corruption
        if self.corruption == "identity" and verbose:
            print("Identity is not a corrupted dataset but the original MNIST dataset.")
        if corruption not in list(ALL_CORRUPTIONS.keys()):
            raise ValueError(f"Provided corruption - {corruption} - is not an approved corruption.")
        self._resource_index = 0

        super().__init__(
            root,
            image_set,
            transforms,
            download,
            verbose,
        )

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels from the correct data format"""
        file_path = self.path / self._resource.filename
        self._loaded_data, labels = self._grab_data(file_path)

        if self.corruption is not None:
            self._loaded_data = self._load_corruption()

        index_strings = np.arange(self._loaded_data.shape[0]).astype(str).tolist()
        return index_strings, labels.tolist(), {}

    def _load_corruption(self) -> tuple[NumpyArray, NDArray[np.uintp]]:
        """Function to load in the file paths for the data and labels for the different corrupt data formats"""
        corruption = ALL_CORRUPTIONS[self.corruption]
        original = self._loaded_data.squeeze()
        data = corruption(original)
        data = data.astype(np.uint8)
        return np.expand_dims(data, axis=1)


    def _grab_data(self, path: Path) -> tuple[NumpyArray, NDArray[np.uintp]]:
        """Function to load in the data numpy array"""
        with np.load(path, allow_pickle=True) as data_array:
            if self.image_set == "base":
                data = np.concatenate([data_array["x_train"], data_array["x_test"]], axis=0)
                labels = np.concatenate([data_array["y_train"], data_array["y_test"]], axis=0).astype(np.uintp)
            else:
                data, labels = (
                    data_array[f"x_{self.image_set}"],
                    data_array[f"y_{self.image_set}"].astype(np.uintp),
                )
            data = np.expand_dims(data, axis=1)
        return data, labels

    def _read_file(self, path: str) -> NumpyArray:
        """
        Function to grab the correct image from the loaded data.
        Overwrite of the base `_read_file` because data is an all or nothing load.
        """
        index = int(path)
        return self._loaded_data[index]
