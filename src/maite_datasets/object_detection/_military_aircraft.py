from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from maite_datasets._base import (
    BaseDatasetNumpyMixin,
    BaseODDataset,
    NumpyArray,
    NumpyObjectDetectionTarget,
    NumpyObjectDetectionTransform,
)
from maite_datasets._fileio import ResourcePart, URLResource


class MilitaryAircraft(BaseODDataset[NumpyArray, NumpyObjectDetectionTarget, list[str], str], BaseDatasetNumpyMixin):
    """
    The object detection version of a dataset that focuses on identifying different types of military aircraft.

    The dataset is 2.88 GB and can be found on `huggingface <https://huggingface.co/datasets/Ahnuf/Military_Aircraft_Detection_Classification_Image_Dataset>`_.
    Dataset includes background-only images for the testing of different background suppression strategies.
    Ground truth labels are provided for the train, validation and test set.

    There are 26,668 images: 21,342 images in the train set (including 2,508 background-only images),
    2,641 images in the validation set (including 295 background-only images) and
    2,645 images in the test set (including 284 background-only images).
    The dataset has 88 aircraft classes from 4 overarching military aircraft categories.
    All images are 640 x 640 images.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``militaryvehicles`` folder of the already downloaded data.
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
    hierarchy : dict[str, Any]
        Dictionary form of the label hierarchy. Can be used to create a class ontology.

    Note
    ----
    Data License: `Apache 2.0 <https://choosealicense.com/licenses/apache-2.0/>`_
    """

    # One archive, two mirrors: Kaggle first, the huggingface copy as fallback.
    _resources = [
        ResourcePart(
            "aeroscan",
            (
                URLResource(
                    url="https://www.kaggle.com/api/v1/datasets/download/ahnuf05/aeroscan-military-aircraft-classification?datasetVersionNumber=1",
                    filename="archive.zip",
                    md5=False,
                    checksum="6f3b0bb890cda7004b04eb430d3ae2c571b9a407b9ac5e3c47b2921c3545686f",
                ),
                URLResource(
                    url="https://huggingface.co/datasets/Ahnuf/Military_Aircraft_Detection_Classification_Image_Dataset/resolve/main/dataset.zip?download=true",
                    filename="dataset.zip",
                    md5=False,
                    checksum="abce22bab42d8b0c544961a25469f4e0fc10cd08fd4fd0dc0aae1ff1673e8514",
                ),
            ),
        ),
    ]

    index2label: dict[int, str] = {
        0: "A10",
        1: "A400M",
        2: "AG600",
        3: "AH64",
        4: "AKINCI",
        5: "AV8B",
        6: "An124",
        7: "An22",
        8: "An225",
        9: "An72",
        10: "B1",
        11: "B2",
        12: "B52",
        13: "Be200",
        14: "C1",
        15: "C130",
        16: "C17",
        17: "C2",
        18: "C390",
        19: "C5",
        20: "CH47",
        21: "CH53",
        22: "CL415",
        23: "E2",
        24: "E7",
        25: "EF2000",
        26: "EMB314",
        27: "F117",
        28: "F14",
        29: "F15",
        30: "F16",
        31: "F18",
        32: "F2",
        33: "F22",
        34: "F35",
        35: "F4",
        36: "FCK1",
        37: "H6",
        38: "Il76",
        39: "J10",
        40: "J20",
        41: "J35",
        42: "J36",
        43: "JAS39",
        44: "JF17",
        45: "JH7",
        46: "KAAN",
        47: "KC135",
        48: "KF21",
        49: "KJ600",
        50: "Ka27",
        51: "Ka52",
        52: "MQ9",
        53: "Mi24",
        54: "Mi26",
        55: "Mi28",
        56: "Mi8",
        57: "Mig29",
        58: "Mig31",
        59: "Mirage2000",
        60: "P3",
        61: "RQ4",
        62: "Rafale",
        63: "SR71",
        64: "Su24",
        65: "Su25",
        66: "Su34",
        67: "Su47",
        68: "Su57",
        69: "TB001",
        70: "TB2",
        71: "Tejas",
        72: "Tornado",
        73: "Tu160",
        74: "Tu22M",
        75: "Tu95",
        76: "U2",
        77: "UH60",
        78: "US2",
        79: "V22",
        80: "Vulcan",
        81: "WZ7",
        82: "X32",
        83: "XB70",
        84: "Y20",
        85: "YF23",
        86: "Z10",
        87: "Z19",
    }

    hierarchy: dict[str, Any] = {
        "vehicle": {
            "military": {
                "land vehicle": None,
                "watercraft": None,
                "aircraft": {
                    "fixed_wing": {
                        "fighter": {
                            "air_superiority_interceptor": ["F14", "F15", "F4", "Mig29", "Mig31"],
                            "multirole_fighter": [
                                "EF2000",
                                "F16",
                                "F18",
                                "F2",
                                "FCK1",
                                "J10",
                                "JAS39",
                                "JF17",
                                "KF21",
                                "Mirage2000",
                                "Rafale",
                                "Tejas",
                                "Tornado",
                            ],
                            "stealth_fighter": ["F22", "F35", "J20", "J35", "J36", "KAAN", "Su57"],
                        },
                        "attack_ground_attack": ["A10", "AV8B", "EMB314", "F117", "JH7", "Su24", "Su25", "Su34"],
                        "bomber": ["B1", "B2", "B52", "H6", "Tu160", "Tu22M", "Tu95", "Vulcan"],
                        "transport": {
                            "strategic_transport": ["An124", "An22", "An225", "C17", "C5", "Il76", "Y20"],
                            "tactical_transport": ["A400M", "An72", "C1", "C130", "C2", "C390"],
                        },
                        "tanker": ["KC135"],
                        "airborne_early_warning": ["E2", "E7", "KJ600"],
                        "maritime_patrol": ["P3"],
                        "reconnaissance": ["SR71", "U2"],
                        "amphibious_seaplane": ["AG600", "Be200", "CL415", "US2"],
                        "experimental_prototype": ["Su47", "X32", "YF23", "XB70"],
                    },
                    "rotary_wing": {
                        "attack_helicopter": ["AH64", "Ka52", "Mi24", "Mi28", "Z10", "Z19"],
                        "transport_helicopter": ["CH47", "CH53", "Mi26"],
                        "utility_helicopter": ["UH60", "Mi8"],
                        "naval_helicopter": ["Ka27"],
                    },
                    "tiltrotor": ["V22"],
                    "unmanned": {
                        "combat_uav": ["AKINCI", "TB001", "TB2", "MQ9"],
                        "reconnaissance_uav": ["RQ4", "WZ7"],
                    },
                },
            }
        }
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val", "test", "base"] = "train",
        transforms: NumpyObjectDetectionTransform | Sequence[NumpyObjectDetectionTransform] | None = None,
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
        # Annotations are normalized YOLO coordinates; scale them to pixels on access.
        self._bboxes_per_size = True

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        """Function to load in the file paths for the data and labels"""
        folders = ["train", "val", "test"] if self.image_set == "base" else [self.image_set]
        data_folder: list[Path] = []
        for folder in folders:
            data_folder.extend(list((self.path / folder).glob("*.jpg")))
            data_folder.extend(list((self.path / folder).glob("*.jpeg")))
            data_folder.extend(list((self.path / folder).glob("*.png")))
        if not data_folder:
            raise FileNotFoundError

        data_folder = sorted(data_folder)
        data = [str(entry) for entry in data_folder]
        labels = [str(entry.with_suffix(".txt")) for entry in data_folder]

        return data, labels, {}

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        """Function for extracting the info for the label and boxes"""
        labels: list[int] = []
        boxes: list[list[float]] = []
        with open(annotation) as f:
            for line in f:
                out = line.split()
                # Skip blank/malformed lines, matching the YOLO reader's leniency.
                if len(out) != 5:
                    continue
                labels.append(int(out[0]))

                xcenter, ycenter, width, height = (float(value) for value in out[1:])
                x0 = xcenter - width / 2
                y0 = ycenter - height / 2
                boxes.append([x0, y0, x0 + width, y0 + height])

        return boxes, labels, {}
