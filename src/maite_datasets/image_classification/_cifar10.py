from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from maite_datasets._base import BaseDatasetNumpyMixin, BaseICDataset, NumpyArray, NumpyImageClassificationTransform
from maite_datasets._fileio import ResourcePart, URLResource

CIFARClassStringMap = Literal[
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
TCIFARClassMap = TypeVar("TCIFARClassMap", CIFARClassStringMap, int, list[CIFARClassStringMap], list[int])


class CIFAR10(BaseICDataset[NumpyArray], BaseDatasetNumpyMixin):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset as NumPy arrays.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or the ``cifar10`` folder of the already downloaded data.
    image_set : "train", "test" or "base", default "train"
        If "base", returns all of the data to allow the user to create their own splits.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    verbose : bool, default False
        If True, outputs print statements.
    lazy : bool, default False
        Accepted for API consistency with the file-backed datasets. CIFAR10
        decodes its data fully into memory, so lazy access is a no-op here.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "test" or "base"
        The selected image set from the dataset.
    transforms : Sequence[Transform]
        The transforms to be applied to the data.
    size : int
        The size of the dataset.
    index2label : dict[int, str]
        Dictionary which translates from class integers to the associated class strings.
    label2index : dict[str, int]
        Dictionary which translates from class strings to the associated class integers.
    metadata : DatasetMetadata
        Typed dictionary containing dataset metadata, such as `id` which returns the dataset class name.
    """

    _resources = [
        ResourcePart(
            "cifar10",
            (
                URLResource(
                    url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                    filename="cifar-10-binary.tar.gz",
                    md5=True,
                    checksum="c32a1d4ab5d03f1284b67883e8d87530",
                ),
            ),
        ),
    ]

    index2label: dict[int, str] = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "test", "base"] = "train",
        transforms: NumpyImageClassificationTransform | Sequence[NumpyImageClassificationTransform] | None = None,
        download: bool = False,
        verbose: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            root,
            image_set,
            transforms,
            download,
            verbose,
            lazy,
        )

    def _load_bin_data(self, data_folder: list[Path]) -> tuple[list[str], list[int], dict[str, Any]]:
        images_list: list[NDArray[np.uint8]] = []
        labels_list: list[NDArray[np.uint8]] = []
        batch_list: list[NDArray[np.uint8]] = []
        # Process each batch file in order (data_batch_1..5, then test_batch),
        # accumulating rather than preallocating so any per-batch size works.
        for batch_file in data_folder:
            # Get batch parameters
            batch_type = "test" if "test" in batch_file.stem else "train"
            batch_num = 5 if batch_type == "test" else int(batch_file.stem.split("_")[-1]) - 1

            # Load data
            batch_images, batch_labels = self._unpack_batch_files(batch_file)

            # Stack data
            images_list.append(batch_images)
            labels_list.append(batch_labels)
            batch_list.append(np.full(batch_images.shape[0], batch_num, dtype=np.uint8))

        all_images = np.concatenate(images_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        batch_nums = np.concatenate(batch_list, axis=0)

        # Save data
        self._loaded_data = all_images
        np.savez(
            self.path / "cifar10",
            images=self._loaded_data,
            labels=all_labels,
            batches=batch_nums,
        )

        # Select data
        image_list = np.arange(all_labels.shape[0]).astype(str)
        if self.image_set == "train":
            return (
                image_list[np.nonzero(batch_nums != 5)[0]].tolist(),
                all_labels[batch_nums != 5].tolist(),
                {"batch_num": batch_nums[batch_nums != 5].tolist()},
            )
        if self.image_set == "test":
            return (
                image_list[np.nonzero(batch_nums == 5)[0]].tolist(),
                all_labels[batch_nums == 5].tolist(),
                {"batch_num": batch_nums[batch_nums == 5].tolist()},
            )
        return (
            image_list.tolist(),
            all_labels.tolist(),
            {"batch_num": batch_nums.tolist()},
        )

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels and retrieve metadata"""
        data_file = self.path / "cifar10.npz"
        if not data_file.exists():
            data_folder = sorted((self.path / "cifar-10-batches-bin").glob("*.bin"))
            if not data_folder:
                raise FileNotFoundError
            return self._load_bin_data(data_folder)

        # Load data
        data = np.load(data_file)
        self._loaded_data = data["images"]
        all_labels = data["labels"]
        batch_nums = data["batches"]

        # Select data
        image_list = np.arange(all_labels.shape[0]).astype(str)
        if self.image_set == "train":
            return (
                image_list[np.nonzero(batch_nums != 5)[0]].tolist(),
                all_labels[batch_nums != 5].tolist(),
                {"batch_num": batch_nums[batch_nums != 5].tolist()},
            )
        if self.image_set == "test":
            return (
                image_list[np.nonzero(batch_nums == 5)[0]].tolist(),
                all_labels[batch_nums == 5].tolist(),
                {"batch_num": batch_nums[batch_nums == 5].tolist()},
            )
        return (
            image_list.tolist(),
            all_labels.tolist(),
            {"batch_num": batch_nums.tolist()},
        )

    def _unpack_batch_files(self, file_path: Path) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        # Load pickle data with latin1 encoding
        with file_path.open("rb") as f:
            buffer = np.frombuffer(f.read(), dtype=np.uint8)
        # Each entry is 1 byte for label + 3072 bytes for image (3*32*32), and the
        # CIFAR format stores the image as three 1024-byte channel blocks (R, G, B),
        # which is exactly the (3, 32, 32) C×H×W layout once reshaped.
        entry_size = 1 + 3072
        num_entries = buffer.size // entry_size
        entries = buffer[: num_entries * entry_size].reshape(num_entries, entry_size)
        labels = entries[:, 0]
        images = entries[:, 1:].reshape(num_entries, 3, 32, 32)
        return images, labels

    def _read_file(self, path: str) -> NumpyArray:
        """
        Function to grab the correct image from the loaded data.
        Overwrite of the base `_read_file` because data is an all or nothing load.
        """
        index = int(path)
        return self._loaded_data[index]
