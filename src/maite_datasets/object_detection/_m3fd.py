from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from defusedxml.ElementTree import parse

from maite_datasets._base import (
    BaseDatasetNumpyMixin,
    BaseODDataset,
    DataLocation,
    DatumMetadata,
    NumpyArray,
    NumpyObjectDetectionTarget,
    NumpyObjectDetectionTransform,
    ObjectDetectionTargetTuple,
    _merge_datum_metadata,
)
from maite_datasets._fileio import _hf_extract


def _bbox_coord(obj: Any, name: str, annotation: str) -> int:
    """Read one bndbox coordinate, failing loudly when the annotation omits it."""
    value = obj.findtext(f"bndbox/{name}")
    if value is None:
        raise ValueError(f"Missing bndbox/{name} in {annotation}")
    return int(value)


class M3FD(BaseODDataset[NumpyArray, NumpyObjectDetectionTarget, list[str], str], BaseDatasetNumpyMixin):
    """
    A computer vision dataset focused on fusing IR with RGB for detecting people and vehicles in natural images.

    The dataset comes from the paper
    `Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and
    Visible for Object Detection <https://ieeexplore.ieee.org/document/9879642>`_ by Jinyuan Liu et. al. (2022).

    The dataset is approximately 5.8 GB and can be found `here <https://github.com/dlut-dimt/TarDAL>`_
    or on `huggingface <https://huggingface.co/datasets/Frencis/M3FD_RGBT>`_.
    Images are collected with varying backgrounds, time of day and weather.
    Ground truth labels are provided for the train, validation and test set.

    There are 4200 ir-rgb paired images: 3200 images in the training set and 1000 images in the
    validation set.
    The dataset has six classes - people, car, bus, motorcycle, lamp and truck.
    Ground-truth bounding boxes are provided as absolute pixel coordinates - (x0, y0, x1, y1).
    The images are mostly 1024 x 768, with a few exceptions.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``m3fd`` folder of the already downloaded data.
    image_set: "train", "val", "operational" or "base", default "train"
        If "base", then the full dataset is selected (train and val).
        "operational" selects the separate M3FD_Fusion set instead of the detection set.
        Splitting only applies when downloading from huggingface; the direct-download
        archive is not split, so "train", "val" and "base" all yield the full set.
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
    image_set : "train", "val", "operational" or "base"
        The selected image set from the dataset.
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
    Data License: `GPL 3.0 <https://choosealicense.com/licenses/gpl-3.0/>`_
    """

    # Downloading from huggingface takes hours - avoid if possible
    _repo_id: str = "Frencis/M3FD_RGBT"
    _repo_type: Literal["dataset", "model"] = "dataset"
    _limit: list[str] | str | None

    # Actual google drive location for files but google limits downloads due to file sizes - back up only
    _resources = [
        DataLocation(
            url="https://drive.google.com/uc?export=download&id=1C8kkYkj1Xls6UtvJ4h6UajiPcvaQ7eeI",
            filename="M3FD_Detection.zip",
            md5=False,
            checksum="62780f67569cd35e631aba449a45781e4eff4f3e2a4bb962dd02551f110a3407",
        ),
        DataLocation(
            url="https://drive.google.com/uc?export=download&id=1pjdhjVTpOsj2qMBVIRpLOLA7UWIuHt0P",
            filename="M3FD_Fusion.zip",
            md5=False,
            checksum="ec33d031bbd26697b75061972786526cdd815ee8111586813427d155ec522dfc",
        ),
        # alternative download location
        DataLocation(
            url="https://www.kaggle.com/api/v1/datasets/download/nus1998/m3fd-dataset?datasetVersionNumber=1",
            filename="archive.zip",
            md5=True,
            checksum="6VVWpm+e2Zb12isdBahe1w==",
        ),
    ]

    index2label: dict[int, str] = {
        0: "people",
        1: "car",
        2: "bus",
        3: "motorcycle",
        4: "lamp",
        5: "truck",
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val", "operational", "base"] = "train",
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
            hf=False,
        )

    def _load_hf_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}

        splits = ["train", "val"] if self.image_set == "base" else [self.image_set]
        self._limit = [f"{split}/*" for split in splits]

        _hf_extract(repo_id=self._repo_id, repo_type=self._repo_type, local_dir=self.path, allow_patterns=self._limit)

        for split in splits:
            base_dir = self.path / split
            data_folder = sorted((base_dir / "images").glob("*.tiff"))
            if not data_folder:
                raise FileNotFoundError

            filepaths.extend(str(entry) for entry in data_folder)
            targets.extend(sorted(str(entry) for entry in (base_dir / "labels").glob("*.txt")))
            _merge_datum_metadata(datum_metadata, {"image_id": [entry.stem for entry in data_folder]})

        return filepaths, targets, datum_metadata

    def _load_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}

        # If base, load all resources
        if self.image_set == "operational":
            self.path: Path = self.path / "extra"
            self.path.mkdir(parents=True, exist_ok=True)
            self._resource = self._resources[1]
            resource_filepaths, resource_targets, _ = super()._load_data()
            filepaths.extend(resource_filepaths)
            targets.extend(resource_targets)

        else:
            self._resource = self._resources[0]
            resource_filepaths, resource_targets, resource_metadata = super()._load_data()
            filepaths.extend(resource_filepaths)
            targets.extend(resource_targets)
            datum_metadata.update(resource_metadata)

        return filepaths, targets, datum_metadata

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        rgb_dir = self.path / "Vis"
        label_dir = self.path / "Annotation"
        data_folder = sorted(rgb_dir.glob("*.png"))
        if not data_folder:
            raise FileNotFoundError

        file_data = {"image_id": [f"{entry.stem}" for entry in data_folder]}
        data = [str(entry) for entry in data_folder]
        annotations = sorted(str(entry) for entry in label_dir.glob("*.xml")) if label_dir.is_dir() else []

        return data, annotations, file_data

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        """Function for extracting the info for the label and boxes"""
        boxes: list[list[float]] = []
        text_labels: list[str] = []
        root = parse(annotation).getroot()
        if root is None:
            raise ValueError(f"Unable to parse {annotation}")
        additional_meta: dict[str, Any] = {
            "image_width": int(root.findtext("size/width", default="-1")),
            "image_height": int(root.findtext("size/height", default="-1")),
            "image_depth": int(root.findtext("size/depth", default="-1")),
        }
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").lower()
            text_labels.append(name)
            xmin = _bbox_coord(obj, "xmin", annotation)
            xmax = _bbox_coord(obj, "xmax", annotation)
            ymin = _bbox_coord(obj, "ymin", annotation)
            ymax = _bbox_coord(obj, "ymax", annotation)
            boxes.append([xmin, ymin, xmax, ymax])

        labels = [self._label2index[lbl] for lbl in text_labels]

        return boxes, labels, additional_meta

    def __getitem__(self, index: int) -> tuple[NumpyArray, NumpyObjectDetectionTarget, DatumMetadata]:
        """
        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        tuple[TArray, ObjectDetectionTarget, DatumMetadata]
            Image, target, datum_metadata - target.boxes returns boxes in x0, y0, x1, y1 format
        """
        # Grab the bounding boxes and labels from the annotations
        annotation = self._targets[index]
        boxes, labels, additional_metadata = self._read_annotations(annotation)
        # Stack the paired IR channel onto the RGB image. Both are read eagerly:
        # np.concatenate materializes its inputs, so lazy mode cannot defer here.
        rgb_img = np.asarray(self._get_image(self._filepaths[index]))
        ir_img = np.asarray(self._get_image(self._filepaths[index].replace("Vis", "Ir")))
        img = np.concatenate([rgb_img, ir_img[:1]])
        # Create the Object Detection Target
        # Cast target explicitly to ODTarget as namedtuple does not provide any typing metadata
        target = cast(
            NumpyObjectDetectionTarget,
            ObjectDetectionTargetTuple(self._as_array(boxes), self._as_array(labels), self._one_hot_encode(labels)),
        )

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}
        img_metadata = img_metadata | additional_metadata

        return self._transform((cast(NumpyArray, img), target, self._to_datum_metadata(index, img_metadata)))
