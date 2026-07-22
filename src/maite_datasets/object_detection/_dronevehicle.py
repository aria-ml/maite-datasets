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
    DatumMetadata,
    NumpyArray,
    NumpyObjectDetectionTarget,
    NumpyObjectDetectionTransform,
    ObjectDetectionTargetTuple,
    _merge_datum_metadata,
)
from maite_datasets._fileio import ResourcePart, URLResource, _remove_folder_nest


class DroneVehicle(BaseODDataset[NumpyArray, NumpyObjectDetectionTarget, list[str], str], BaseDatasetNumpyMixin):
    """
    A computer vision dataset focused on vehicle detection from RGB-Infrared drone images.

    The dataset comes from the paper
    `Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware
    Learning <https://ieeexplore.ieee.org/abstract/document/9759286>`_ by Yiming Sun et. al. (2022).

    The dataset is approximately 14 GB and can be found `here <https://github.com/VisDrone/DroneVehicle>`_
    or on `kaggle <https://www.kaggle.com/datasets/brendanalvey/visdrone-dronevehicle>`_.
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

    _resources = [
        ResourcePart(
            "dronevehicle",
            (
                URLResource(
                    url="https://www.kaggle.com/api/v1/datasets/download/brendanalvey/visdrone-dronevehicle?datasetVersionNumber=2",
                    filename="archive.zip",
                    md5=False,
                    checksum="35271ccb2adfd7e34719ad59616fcecb398a798c0bcbc5cded6e30aa62835b78",
                ),
            ),
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
        )

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}

        # The Kaggle archive lands nested under VisDrone-DroneVehicle/; flattening is a
        # no-op once done (and for the huggingface layout, which is already flat), so it
        # runs unconditionally rather than sniffing which resource we came from.
        self._remove_nested_folder()

        for resource in ["train", "val", "test"]:
            if self.image_set != "base" and resource != self.image_set:
                continue
            base_dir = self.path / resource
            data_folder = sorted((base_dir / f"{resource}img").glob("*.jpg"))
            if not data_folder:
                raise FileNotFoundError

            filepaths.extend([str(entry) for entry in data_folder])
            targets.extend(sorted(str(entry) for entry in (base_dir / f"{resource}labelr").glob("*.xml")))
            file_data = {"image_id": [f"{resource}_{entry.name}" for entry in data_folder]}
            _merge_datum_metadata(datum_metadata, file_data)

        return filepaths, targets, datum_metadata

    def _remove_nested_folder(self) -> None:
        nested = self.path / "VisDrone-DroneVehicle"
        if nested.is_dir():
            _remove_folder_nest(nested, verbose=self._verbose)

    @staticmethod
    def _infrared_path(rgb_path: str) -> str:
        """Paired IR image for `rgb_path`: ``<split>img/x.jpg`` -> ``<split>imgr/x.jpg``.

        Rebuilt from the path components rather than by string replacement, so a root
        directory that itself contains a segment ending in ``img`` is left alone.
        """
        path = Path(rgb_path)
        return str(path.parent.with_name(f"{path.parent.name}r") / path.name)

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

        # Paired RGB annotation: <split>labelr/x.xml -> <split>label/x.xml, built from
        # path components so a root directory ending in "labelr" is left alone.
        ir_annotation = Path(annotation)
        rgb_annotation = ir_annotation.parent.with_name(ir_annotation.parent.name.removesuffix("r"))
        root = parse(str(rgb_annotation / ir_annotation.name)).getroot()
        if root is not None:
            additional_meta["rgb_filename"] = root.findtext("filename", default="")
            additional_meta["image_depth"] += int(root.findtext("size/depth", default="0"))
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
        ir_img = np.asarray(self._get_image(self._infrared_path(self._filepaths[index])))
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
