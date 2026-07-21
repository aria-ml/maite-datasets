from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Literal

from maite_datasets._base import DataLocation, DatasetMetadata, ReaderTransforms, _dataset_dir
from maite_datasets._fileio import _ensure_exists, _hf_extract, _print, _remove_folder_nest
from maite_datasets._reader import DEFAULT_IMAGES_DIR
from maite_datasets.object_detection._yolo import YOLODataset, YOLODatasetReader


class DroneSwarm(YOLODataset):
    """
    A synthetic multi-drone dataset from Simuletic.

    The dataset is approximately 153 MB and can be found `here <https://huggingface.co/datasets/Simuletic/Military-Drone-Swarm-Saturation-Attack-Dataset>`_.
    Images are simulated with varying backgrounds, time of day and weather with varying range and number of drones.
    Ground truth labels are provided for the dataset.

    There are 117 images with only a single class - drone.
    Ground-truth bounding boxes are provided in yolo format - (xc, yc, w, h) using normalized coordinates (0-1).
    The images are 1024 x 1024.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``m3fd`` folder of the already downloaded data.
    image_set: "base"
        Only exists to maintain continuity with other datasets.
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
    image_set : "base"
        Only contains a "base" set.
    index2label : dict[int, str]
        Dictionary which translates from class integers to the associated class strings.
    label2index : dict[str, int]
        Dictionary which translates from class strings to the associated class integers.
    metadata : DatasetMetadata
        Typed dictionary containing dataset metadata, such as `id` which returns the dataset class name.
    transforms : Sequence[Transform]
        The transforms to be applied to the data.
    size : int
        The size of the dataset.

    Note
    ----
    Data License: `CC By 4.0 <https://choosealicense.com/licenses/cc-by-4.0/>`_
    """

    _repo_id: str = "Simuletic/Military-Drone-Swarm-Saturation-Attack-Dataset"
    _repo_type: Literal["dataset", "model"] = "dataset"
    _limit: list[str] | str | None = "Drone_Swarm_Dataset/*"

    _resources = [
        DataLocation(
            url="https://www.kaggle.com/api/v1/datasets/download/simuletic/military-drone-swarm-and-saturation-attack-dataset?datasetVersionNumber=1",
            filename="archive.zip",
            md5=False,
            checksum="26430a62d7de2cad1feb7ae4f2ab3b2aa76bb0f016ff047a23364b2352b14931",
            kaggle=True,
        ),
    ]

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["base"] = "base",
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
            self._resource: DataLocation = self._resources[0]
            self._load_data()
            # self._load_hf_data()
        reader = YOLODatasetReader(self.path, dataset_id=unique_id, image_extensions=[".png"], image_set="train")

        super().__init__(reader, lazy, transforms)
        # ``image_set`` is not a DatasetMetadata key, so rebuild via ** rather than item assignment
        self.metadata: DatasetMetadata = DatasetMetadata(**{**self.metadata, "image_set": image_set})

    def _get_dataset_dir(self) -> Path:
        # Create a designated folder for this dataset (named after the class)
        return _dataset_dir(self._root, self.__class__.__name__)

    def _load_hf_data(self) -> None:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        # The download arrives nested under Drone_Swarm_Dataset/ and is then flattened
        # into self.path. Once flattened, _hf_extract no longer sees the files at their
        # repo-relative paths, so re-check the flattened layout or every construction
        # re-downloads the whole dataset.
        if (self.path / DEFAULT_IMAGES_DIR).is_dir():
            _print("Data already downloaded, skipping download.", self._verbose)
            return

        _print("Downloading files from huggingface.", self._verbose)

        _hf_extract(repo_id=self._repo_id, repo_type=self._repo_type, local_dir=self.path, allow_patterns=self._limit)

        self._remove_nested_folder()

    def _remove_nested_folder(self) -> None:
        nested = self.path / "Drone_Swarm_Dataset"
        if nested.is_dir():
            _remove_folder_nest(nested, verbose=self._verbose)

    def _load_data(self) -> None:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        if (self.path / "data.yaml").exists() or (self.path / "classes.txt").exists():
            _print("Data already downloaded, skipping download.", self._verbose)
            return

        _print("Downloading files from kaggle.", self._verbose)

        _ensure_exists(*self._resource, self.path, self._root, self._download, self._verbose)

        self._remove_nested_folder()
