from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from maite_datasets._base import (
    BaseDatasetNumpyMixin,
    BaseICDataset,
    NumpyArray,
    NumpyImageClassificationTransform,
    _merge_datum_metadata,
)
from maite_datasets._fileio import HFResource, ResourcePart


class MilitaryVehicles(BaseICDataset[NumpyArray], BaseDatasetNumpyMixin):
    """
    A dataset that focuses on identifying different types of military vehicles.

    The dataset comes from the paper
    `Error Detection and Constraint Recovery in Hierarchical Multi-Label Classification
    without Prior Knowledge <https://dl.acm.org/doi/10.1145/3627673.3679918>`_ by Joshua Kricheli et al. (2024).

    The dataset is approximately 100 MB and can be found on `huggingface <https://huggingface.co/datasets/leibnitz-lab/military_vehicles>`_.
    Images are curated from other resources that show the vehicles in a variety of environments.
    Ground truth labels are provided for the train and test set.

    There are 9,444 images: 7,823 images in the train set and 1,621 images in the test set.
    The dataset has 24 classes from 6 overarching military vehicle categories.
    There is wide variation in image sizes.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``militaryvehicles`` folder of the already downloaded data.
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
        When True, the image element of each datum is returned as a
        :class:`LazyArray` that defers PIL decode until first numpy access.
        Useful for metadata-only iteration over large image folders.

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
    hierarchy : dict[str, Any]
        Dictionary form of the label hierarchy. Can be used to create a class ontology.

    Note
    ----
    Data License: `MIT <https://choosealicense.com/licenses/mit/>`_
    """

    _repo_id: str = "leibnitz-lab/military_vehicles"

    # Published only on huggingface. The class-level entry fetches the whole repo;
    # _load_data narrows it to the selected image set before anything downloads.
    _resources = [
        ResourcePart(
            "military_vehicles",
            (HFResource(repo_id=_repo_id),),
        ),
    ]

    index2label: dict[int, str] = {
        0: "2S19 MSTA",
        1: "30N6E",
        2: "BM-30",
        3: "BMD",
        4: "BMP-1",
        5: "BMP-2",
        6: "BMP-T15",
        7: "BRDM",
        8: "BTR-60",
        9: "BTR-70",
        10: "BTR-80",
        11: "D-30",
        12: "Iskander",
        13: "MT LB",
        14: "Pantsir-S1",
        15: "Rs-24",
        16: "T-14",
        17: "T-62",
        18: "T-64",
        19: "T-72",
        20: "T-80",
        21: "T-90",
        22: "TOS-1",
        23: "Tornado",
    }

    hierarchy: dict[str, Any] = {
        "vehicle": {
            "military": {
                "land vehicle": {
                    "Tank": ["T-14", "T-62", "T-64", "T-72", "T-80", "T-90"],
                    "BMP": ["BMP-1", "BMP-2", "BMP-T15"],
                    "BTR": ["BRDM", "BTR-60", "BTR-70", "BTR-80"],
                    "Self Propelled Artillery": ["2S19 MSTA", "BM-30", "D-30", "Tornado", "TOS-1"],
                    "Air Defense": ["30N6E", "Iskander", "Pantsir-S1", "Rs-24"],
                    "MT LB": ["MT LB"],
                    "BMD": ["BMD"],
                },
                "watercraft": None,
                "aircraft": None,
            }
        }
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
        super().__init__(root, image_set, transforms, download, verbose, lazy)

    def _load_data(self) -> tuple[list[str], Sequence[int], dict[str, Any]]:
        # Only the selected image set is worth fetching, and the pattern that expresses
        # that depends on self.image_set -- so the part is narrowed here rather than
        # declared statically on the class.
        image_sets = ["train", "test"] if self.image_set == "base" else [self.image_set]
        self._resource = ResourcePart(
            "military_vehicles",
            (HFResource(repo_id=self._repo_id, allow_patterns=[f"{img_set}_fine/*" for img_set in image_sets]),),
        )
        return super()._load_data()

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        filepaths: list[str] = []
        targets: list[int] = []
        datum_metadata: dict[str, list[Any]] = {}

        image_sets = ["train", "test"] if self.image_set == "base" else [self.image_set]
        for img_set in image_sets:
            annotations_path = self.path / f"{img_set}_fine/{img_set}_true_fine.npy"
            if not annotations_path.exists():
                raise FileNotFoundError
            annotations: NDArray = np.load(annotations_path)
            targets.extend(annotations.tolist())
            for group in self._label2index:
                data, file_data = self._load_group(img_set, group)
                filepaths.extend(data)
                _merge_datum_metadata(datum_metadata, file_data)

        return filepaths, targets, datum_metadata

    def _load_group(self, set_name: str, group_name: str) -> tuple[list[str], dict[str, Any]]:
        """Paths and per-datum metadata for one class folder within one image set."""
        group_dir = group_name.replace(" ", "_")
        base_dir = self.path / f"{set_name}_fine/{group_dir}"
        data_folder = sorted(base_dir.glob("*.jpg"))
        if not data_folder:
            raise FileNotFoundError

        file_data = {"image_id": [f"{group_dir}_{entry.stem}" for entry in data_folder]}
        return [str(entry) for entry in data_folder], file_data
