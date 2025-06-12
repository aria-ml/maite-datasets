from __future__ import annotations

__all__ = []

from torch import Tensor

from maite_datasets._base import BaseODDataset
from maite_datasets._types import ObjectDetectionTarget
from maite_datasets._mixin._torch import BaseDatasetTorchMixin
from maite_datasets.object_detection._voc import BaseVOCDataset


class VOCDetectionTorch(
    BaseVOCDataset[Tensor, ObjectDetectionTarget[Tensor]],
    BaseODDataset[Tensor, list[str], str],
    BaseDatasetTorchMixin,
):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset as PyTorch tensors.

    Parameters
    ----------
    root : str or pathlib.Path
        Because of the structure of the PASCAL VOC datasets, the root needs to be one of 4 folders.
        1) Directory containing the year of the **already downloaded** dataset (i.e. .../VOCdevkit/VOC2012 <-)
        2) Directory to the VOCdevkit folder of the **already downloaded** dataset (i.e. .../VOCdevkit <- /VOC2012)
        3) Directory to the folder one level up from the VOCdevkit folder,
        data **may** or **may not** be already downloaded (i.e. ... <- /VOCdevkit/VOC2012)
        4) Directory to where you would like the dataset to be downloaded
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007" or "2012". Note that the 2012 test set does not contain annotations.
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
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
    year : "2007", "2008", "2009", "2010", "2011" or "2012"
        The selected dataset year.
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
    Data License: `Flickr Terms of Use <http://www.flickr.com/terms.gne?legacy=1>`_
    """
