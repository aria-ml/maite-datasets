from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Literal

import yaml

from maite_datasets._base import DatasetMetadata, ReaderTransforms, _dataset_dir
from maite_datasets._fileio import ResourcePart, URLResource, _download_part, _print
from maite_datasets.object_detection._yolo import YOLODataset, YOLODatasetReader


class SkySeaLand(YOLODataset):
    """
    A satellite imagery dataset that focuses on identifying different types of transportation.

    The dataset is 262 MB and can be found on `kaggle <https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand/data>`_.
    Dataset includes images from 4 different locations around the world in a variety of settings.
    Ground truth labels are provided for the train, validation and test set.

    There are 1307 images: 1048 images in the train set, 132 images in the validation set and
    127 images in the test set.
    The dataset consists of 4 classes with a fairly even split across 19,103 bounding boxes.
    Images come in a variety of sizes.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``skysealand`` folder of the already downloaded data.
    image_set : "train", "val", "test" or "base", default "train"
        If "base", returns all of the data to allow the user to create their own splits.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    verbose : bool, default False
        If True, outputs print statements.
    lazy : bool, default False
        When True, the image element of each datum is returned as a
        :class:`LazyArray` that defers PIL decode until first numpy access.
        Useful for metadata-only iteration over large image folders.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "val", "test" or "base"
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

    Note
    ----
    Data License: `MIT <https://choosealicense.com/licenses/mit/>`_
    """

    _resources = [
        ResourcePart(
            "skysealand",
            (
                URLResource(
                    url="https://www.kaggle.com/api/v1/datasets/download/mdzahidhasanriad/skysealand?datasetVersionNumber=1",
                    filename="archive.zip",
                    md5=False,
                    checksum="aa391650d46dfacf7eee73036ca2321d5e90cef8a253d43a2936f3d089214e7b",
                ),
            ),
        ),
    ]

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val", "test", "base"] = "train",
        transforms: ReaderTransforms | None = None,
        download: bool = False,
        verbose: bool = False,
        lazy: bool = False,
    ) -> None:
        self._root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        self.image_set = image_set
        self._download = download
        self._verbose = verbose
        self.path: Path = self._get_dataset_dir()
        unique_id = f"{self.__class__.__name__}_{self.image_set}"

        # Load the data
        if download:
            self._resource: ResourcePart = self._resources[0]
            self._load_data()
        reader = YOLODatasetReader(self.path, dataset_id=unique_id, image_extensions=[".jpg"], image_set=self.image_set)

        super().__init__(reader, lazy, transforms)
        # ``image_set`` is not a DatasetMetadata key, so rebuild via ** rather than item assignment
        self.metadata: DatasetMetadata = DatasetMetadata(**{**self.metadata, "image_set": image_set})

    def _get_dataset_dir(self) -> Path:
        # Create a designated folder for this dataset (named after the class)
        return _dataset_dir(self._root, self.__class__.__name__)

    def _load_data(self) -> None:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        if (self.path / "data.yaml").exists() or (self.path / "classes.txt").exists():
            _print("Data already downloaded, skipping download.", self._verbose)
            return

        _print("Downloading files from kaggle.", self._verbose)

        _download_part(self._resource, self.path, self._root, self._download, self._verbose)

        with open(self.path / "data.yaml") as f:
            config = yaml.safe_load(f)

        config["train"] = config["train"].replace("../", "")
        config["val"] = config["val"].replace("../", "")
        config["test"] = config["test"].replace("../", "")

        with open(self.path / "data.yaml", "w") as f:
            yaml.safe_dump(config, f)
