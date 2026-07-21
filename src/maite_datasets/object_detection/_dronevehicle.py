from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from defusedxml.ElementTree import parse

from maite_datasets._base import (
    BaseDatasetNumpyMixin,
    BaseODDataset,
    DataLocation,
    NumpyArray,
    NumpyObjectDetectionTarget,
    NumpyObjectDetectionTransform,
    _merge_datum_metadata,
)
from maite_datasets._fileio import _extract_archive, _hf_extract, _print


class DroneVehicle(BaseODDataset[NumpyArray, NumpyObjectDetectionTarget, list[str], str], BaseDatasetNumpyMixin):
    """
    A computer vision dataset focused on vehicle detection from RGB-Infrared drone images.

    The dataset comes from the paper
    `Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware
    Learning <https://ieeexplore.ieee.org/abstract/document/9759286>`_ by Yiming Sun et. al. (2022).

    The dataset is approximately 14 GB and can be found `here <https://github.com/VisDrone/DroneVehicle>`_
    or on `huggingface <https://huggingface.co/datasets/McCheng/DroneVehicle>`_.
    Images are collected with varying backgrounds, time of day and lighting conditions.
    Ground truth labels are provided for the train, validation and test set.

    There are 56,878 images (28,439 ir-rgb paired images): 17,990 image pairs in the train set,
    1,469 image pairs in the validation set, and 8,980 image pairs in the test set.
    The dataset has five classes - car, truck, bus, van, and freight car.
    Ground-truth bounding boxes are provided in yolo format - (xc, yc, w, h) using normalized coordinates (0-1).
    Because there are low-light conditions - there are separate annotations for both the rgb and ir versions
    of the image. The images are 712 x 840 with a 100 pixel white border on all sides (artifact from annotating);
    actual image size is 512 x 640, which will require cropping to get this size.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``m3fd`` folder of the already downloaded data.
    image_set: "train", "val", or "base", default "train"
        If "base", then the full dataset is selected (train, val and test).
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
    Data License: None provided
    """

    _repo_id: str = "McCheng/DroneVehicle"
    _repo_type: Literal["dataset", "model"] = "dataset"
    _limit: list[str] | str | None

    # Pulling directly from huggingface
    _resources = [
        DataLocation(
            url="https://huggingface.co/datasets/McCheng/DroneVehicle/resolve/main/train.zip?download=true",
            filename="train.zip",
            md5=False,
            checksum="d22eccae518728352b40bb758b383e64db2b1b38e3d8c5d14406724dc869614f",
        ),
        DataLocation(
            url="https://huggingface.co/datasets/McCheng/DroneVehicle/resolve/main/val.zip?download=true",
            filename="val.zip",
            md5=False,
            checksum="043b7944ebb8ce076c1e5cfd37c33de6a59a9f62cf47c0f028387f703d4f5250",
        ),
        DataLocation(
            url="https://huggingface.co/datasets/McCheng/DroneVehicle/resolve/main/test.zip?download=true",
            filename="test.zip",
            md5=False,
            checksum="94bf47e493e7a57fd5d900439f614f339e7b6c7181e4d781bcfebaeb963ffd8e",
        ),
    ]

    index2label: dict[int, str] = {
        0: "car",
        1: "truck",
        2: "bus",
        3: "van",
        4: "freight car",
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
            hf=False,  # Can switch to True, if desired
        )

    def _load_hf_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}

        self._limit = f"{self.image_set}.zip" if self.image_set != "base" else None

        _hf_extract(repo_id=self._repo_id, repo_type=self._repo_type, local_dir=self.path, allow_patterns=self._limit)

        # If base, load all resources
        if self.image_set == "base":
            _print("Extracting train.zip, val.zip, and test.zip ...", self._verbose)
            for file in ["train.zip", "val.zip", "test.zip"]:
                filepath = self.path / file
                filename = filepath.stem
                file_ext = filepath.suffix
                _extract_archive(file_ext, filepath, self.path, False, self._verbose)

                data, annotations, file_data = self._load_data_inner(filename)
                filepaths.extend(data)
                targets.extend(annotations)
                _merge_datum_metadata(datum_metadata, file_data)

        else:
            data, annotations, file_data = self._load_data_inner(self.image_set)
            filepaths.extend(data)
            targets.extend(annotations)
            _merge_datum_metadata(datum_metadata, file_data)

        return filepaths, targets, datum_metadata

    def _load_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}

        # If base, load all resources; otherwise grab only the desired data
        for resource in self._resources:
            if self.image_set != "base" and self.image_set not in resource.filename:
                continue
            self._resource = resource
            resource_filepaths, resource_targets, resource_metadata = super()._load_data()
            filepaths.extend(resource_filepaths)
            targets.extend(resource_targets)
            _merge_datum_metadata(datum_metadata, resource_metadata)

        return filepaths, targets, datum_metadata

    def _load_data_inner(self, resrc_name: str | None = None) -> tuple[list[str], list[str], dict[str, Any]]:
        resource_name = resrc_name if resrc_name is not None else self._resource.filename[:-4]
        base_dir = self.path / resource_name
        data_folder = sorted((base_dir / f"{resource_name}img").glob("*.jpg"))
        if not data_folder:
            raise FileNotFoundError

        file_data = {"image_id": [f"{resource_name}_{entry.name}" for entry in data_folder]}
        data = [str(entry) for entry in data_folder]
        annotations = sorted(str(entry) for entry in (base_dir / f"{resource_name}labelr").glob("*.xml"))

        return data, annotations, file_data

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        """Function for extracting the info for the label and boxes"""
        boxes: list[list[float]] = []
        text_labels: list[str] = []
        root = parse(annotation).getroot()
        if root is None:
            raise ValueError(f"Unable to parse {annotation}")
        additional_meta: dict[str, Any] = {
            "infrared_filename": root.findtext("filename", default=""),
            "image_width": int(root.findtext("size/width", default="-1")),
            "image_height": int(root.findtext("size/height", default="-1")),
            "image_depth": int(root.findtext("size/depth", default="-1")),
        }
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").replace("feright", "freight")
            text_labels.append(name)
            # Annotations are rotated quadrilaterals; take their axis-aligned extent.
            xs = [int(obj.findtext(f"polygon/x{corner}", default="0")) for corner in range(1, 5)]
            ys = [int(obj.findtext(f"polygon/y{corner}", default="0")) for corner in range(1, 5)]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

        labels = [self._label2index[lbl] for lbl in text_labels]

        rgb_annotation = annotation.replace("labelr/", "label/")
        root = parse(rgb_annotation).getroot()
        if root is not None:
            additional_meta["rgb_filename"] = root.findtext("filename", default="")
            additional_meta["image_depth"] += int(root.findtext("size/depth", default="0"))
        return boxes, labels, additional_meta
