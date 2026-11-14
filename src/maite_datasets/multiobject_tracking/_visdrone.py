from __future__ import annotations

__all__ = []

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from maite.protocols import DatasetMetadata
from numpy.typing import NDArray

from maite_datasets._base import (
    MultiobjectTrackingTargetTuple,
    SingleFrameObjectTrackingTargetTuple,
    _dataset_dir,
)
from maite_datasets._fileio import HFResource, ResourcePart, _download_part, _print
from maite_datasets._lazy import LazyArray, chw_loaders
from maite_datasets._video import VideoFrameTuple


class _FrameFiles:
    """Re-iterable stream over one sequence's frame files, in frame order.

    The pixels of each frame are a :class:`LazyArray`, so walking the stream for its
    timings or frame sizes reads image headers and decodes nothing. Some sequences are 4K.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = paths

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[VideoFrameTuple]:
        for index, path in enumerate(self._paths):
            loader, shape_loader = chw_loaders(path)
            # LazyArray stands in for the NDArray the tuple declares, as it does for the image datasets.
            pixels = cast("NDArray[Any]", LazyArray(str(path), loader=loader, shape_loader=shape_loader))
            # ponytail: nominal timing, as frames_to_stream gives frames with no container timing --
            # the export publishes no frame rate, so time_s counts frames. Add an fps option if seconds matter.
            yield VideoFrameTuple(pixels=pixels, time_s=float(index), pts=index, frame_index=index)


class VisDroneMOT:
    """
    Drone footage with tracked vehicles and people, from the VisDrone2019-MOT benchmark.

    The dataset is approximately 1.7 GB and is read from the
    `Voxel51 export <https://huggingface.co/datasets/Voxel51/visdrone-mot>`_ of the benchmark. The export
    holds 7 sequences and 2,846 frames, which is the size of the benchmark's validation set, and
    publishes no other split. Its card does not name the split, so ``image_set`` follows the size.
    The official zips for every split are on the
    `VisDrone <https://github.com/VisDrone/VisDrone-Dataset>`_ repository.
    The benchmark is described in `Detection and Tracking Meet Drones Challenge
    <https://arxiv.org/abs/2001.06303>`_ by Pengfei Zhu et al. (2021).

    The sequences are drone footage of roads, intersections and a sporting event, by day, dusk and night.
    Every datum is one sequence: a lazy stream of its frames, the boxes, labels and track ids of every
    frame, and metadata for the sequence. Frames keep the size they are published at, which differs
    between sequences, from 1344 x 756 to 3840 x 2160.

    The 10 classes are the benchmark's object categories. Regions the benchmark marks as ignored,
    and its one unclassified ``others`` box, carry a confidence of 0 in the export and are not in the
    targets. Track ids are per sequence.

    The frames carry no timestamps and the export gives no frame rate, so ``frame_index`` and ``pts``
    count frames and ``time_s`` is the same count: the timing is nominal, not seconds. The
    ``scene_type``, ``time_of_day`` and ``pedestrian_density`` of each datum's metadata are labels the
    Voxel51 export adds, not part of the benchmark's own annotations.

    The download needs the ``hf`` extra. Without a huggingface token it fetches one file at a time.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``visdronemot`` folder of the already downloaded data.
    image_set : "val", default "val"
        The only image set the export publishes. Only exists to keep the signature of the other datasets.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "val"
        The selected image set from the dataset.
    index2label : dict[int, str]
        Dictionary which translates from class integers to the associated class strings.
    metadata : DatasetMetadata
        Typed dictionary containing dataset metadata, such as `id` which returns the dataset class name.

    Note
    ----
    Data License: `CC BY-NC-SA 3.0 <https://creativecommons.org/licenses/by-nc-sa/3.0/>`_. The
    VisDrone copyright is reserved by the AISKYEYE team at Tianjin University. The Voxel51 card's
    metadata names the license ``cc-by-sa-3.0``, but its text says non-commercial and share-alike,
    and this is the stricter reading.
    """

    _repo_id: str = "Voxel51/visdrone-mot"

    # What the loader reads: the annotations and the frames, not the card or FiftyOne's own bookkeeping.
    _resource = ResourcePart(
        "visdrone_mot",
        (HFResource(repo_id=_repo_id, allow_patterns=["samples.json", "data/*.jpg"]),),
    )

    # The benchmark's categories 1-10, shifted to start at 0. Category 0 (ignored regions)
    # and 11 (others) are what the export gives a confidence of 0.
    index2label: dict[int, str] = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor",
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["val"] = "val",
        *,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        if image_set != "val":
            raise ValueError(f"image_set must be 'val', the only split the export publishes, got {image_set!r}.")
        self.image_set = image_set
        self._root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        self._download = download
        self._verbose = verbose
        self._label2index = {label: index for index, label in self.index2label.items()}
        # ``split`` is not a declared DatasetMetadata key, so it goes in through ** as it does for the other datasets
        self.metadata: DatasetMetadata = DatasetMetadata(
            **{"id": f"{self.__class__.__name__}_{image_set}", "index2label": self.index2label, "split": image_set}
        )
        self.path: Path = _dataset_dir(self._root, self.__class__.__name__)

        _print(f"Determining if {self._resource.name} needs to be downloaded.", verbose)
        try:
            self._index_sequences()
            _print("No download needed, loaded data successfully.", verbose)
        except FileNotFoundError:
            # A file the annotations name but the disk lacks is an unfinished download, so it
            # gets the same recovery as no download at all.
            _download_part(self._resource, self.path, self._root, download, verbose)
            self._index_sequences()

    def _index_sequences(self) -> None:
        """Group the annotated frames into sequences and build each one's target and metadata."""
        with open(self.path / "samples.json") as f:
            samples = json.load(f)["samples"]
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            by_scene[sample["scene_id"]].append(sample)

        sequences = []
        for scene_id in sorted(by_scene):
            frames = sorted(by_scene[scene_id], key=lambda sample: sample["frame_number"])
            paths = [self.path / frame["filepath"] for frame in frames]
            first = frames[0]
            sequences.append(
                (
                    paths,
                    MultiobjectTrackingTargetTuple(frame_tracks=[self._frame_target(frame) for frame in frames]),
                    {
                        "id": scene_id,
                        "height": first["metadata"]["height"],
                        "width": first["metadata"]["width"],
                        "time_base": Fraction(1, 1),
                        # stat raises FileNotFoundError for a frame that was never fetched
                        "size": sum(path.stat().st_size for path in paths),
                        "scene_type": first["scene_type"]["label"],
                        "time_of_day": first["time_of_day"]["label"],
                        "pedestrian_density": first["pedestrian_density"]["label"],
                    },
                )
            )
        self._sequences = sequences

    def _frame_target(self, sample: dict[str, Any]) -> SingleFrameObjectTrackingTargetTuple:
        width, height = sample["metadata"]["width"], sample["metadata"]["height"]
        kept = [detection for detection in sample["detections"] if detection["confidence"] > 0]
        undeclared = {detection["label"] for detection in kept} - self._label2index.keys()
        if undeclared:
            raise ValueError(
                f"{self._repo_id} labels objects {sorted(undeclared)} that {self.__class__.__name__} does not declare."
            )
        # The export stores boxes as relative (x, y, w, h); the pixels they come back to are whole numbers.
        boxes = [
            [x * width, y * height, (x + w) * width, (y + h) * height]
            for x, y, w, h in (detection["bounding_box"] for detection in kept)
        ]
        return SingleFrameObjectTrackingTargetTuple(
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray([self._label2index[detection["label"]] for detection in kept], dtype=np.int64),
            scores=np.ones(len(kept), dtype=np.float32),
            track_ids=np.asarray([detection["index"] for detection in kept], dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(
        self, index: int, /
    ) -> tuple[Iterable[VideoFrameTuple], MultiobjectTrackingTargetTuple, dict[str, Any]]:
        paths, target, metadata = self._sequences[index]
        # A copy, so a caller editing one read's metadata does not edit the next read's.
        return _FrameFiles(paths), target, dict(metadata)
