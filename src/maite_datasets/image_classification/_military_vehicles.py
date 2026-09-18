from __future__ import annotations

__all__ = []

import json
from collections import Counter
from collections.abc import Sequence
from functools import partial
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
from maite_datasets._fileio import HFParquetResource, HFResource, ResourcePart, _hf_list_files


def _import_parquet() -> Any:
    """The parquet reader, or an ImportError explaining how to get it.

    Raising rather than degrading silently is deliberate: this module is reached
    through a mirror, so the exception is what hands the download back to the plain
    file-tree :class:`HFResource` listed behind it.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as e:
        raise ImportError(
            "pyarrow is required to restore MilitaryVehicles from its parquet conversion; "
            "install maite-datasets[hf] for the fast download path."
        ) from e
    return parquet


def _fine_tree_index(listing: Sequence[str], prefix: str) -> set[tuple[str, str]]:
    """``(class directory, filename)`` for every image the repo publishes under `prefix`."""
    entries: set[tuple[str, str]] = set()
    for path in listing:
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == prefix and parts[2].lower().endswith(".jpg"):
            entries.add((parts[1], parts[2]))
    return entries


def _coarse_mapping(
    index2label: dict[int, str],
    categories: dict[str, list[str]],
) -> tuple[dict[int, str], list[int]]:
    """The coarse labeling implied by `categories`, plus a fine-id to coarse-id lookup.

    Ids are assigned alphabetically, which is the order the source repo's own
    ``*_true_coarse.npy`` uses, so a coarse label here means the same integer it does
    upstream. ``BMD`` and ``MT LB`` are categories of one and so appear in both
    labelings under the same name.
    """
    names = sorted(categories)
    fine2category = {member: category for category, members in categories.items() for member in members}
    missing = set(index2label.values()) - set(fine2category)
    if missing:
        raise ValueError(f"hierarchy does not place {sorted(missing)} in a category.")
    return dict(enumerate(names)), [names.index(fine2category[index2label[i]]) for i in sorted(index2label)]


def _materialize_fine_tree(
    parquet_files: Sequence[Path],
    directory: Path,
    *,
    repo_id: str,
    index2label: dict[int, str],
) -> None:
    """Rebuild the ``{split}_fine`` image tree from the hub's parquet conversion.

    The conversion collapsed the repo's four top-level directories into one config, so
    every image appears twice -- once under its fine class and once under its coarse
    category -- and ``image.path`` is a bare filename that does not say which. The split
    is recovered by matching each row against the repo's own file listing: a row is fine
    exactly when ``{split}_fine/<label>/<path>`` is a file that exists upstream. That is
    decisive (measured: every row matches one side and no row matches both) and costs one
    listing request, where deriving it from the rows alone would mean hashing 165 MB of
    image bytes to pair the duplicates back up.
    """
    parquet = _import_parquet()
    listing = _hf_list_files(repo_id)

    for parquet_file in sorted(parquet_files):
        # ``default/<split>/0000.parquet`` -- the config is flat, the split is the parent.
        split = parquet_file.parent.name
        expected = _fine_tree_index(listing, f"{split}_fine")
        target = directory / f"{split}_fine"
        if not expected:
            raise RuntimeError(f"{repo_id} publishes no {split}_fine images; the parquet conversion cannot be split.")

        # Class directories from the listing become filesystem paths below, so they are
        # checked against what this class declares rather than trusted. That also catches
        # the repo gaining, losing or renaming a class out from under index2label.
        declared = {label.replace(" ", "_") for label in index2label.values()}
        undeclared = {class_dir for class_dir, _ in expected} - declared
        if undeclared:
            raise RuntimeError(f"{split}_fine holds classes {sorted(undeclared)} that {__name__} does not declare.")

        reader = parquet.ParquetFile(parquet_file)
        names = json.loads(reader.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]

        written: set[tuple[str, str]] = set()
        # Batched rather than one read_table: the train conversion is 136 MB of image
        # blobs, and nothing here needs more than a row at a time.
        for batch in reader.iter_batches():
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image, label in zip(images, labels):
                key = (names[label], image["path"])
                if key not in expected:
                    continue
                destination = target / key[0] / key[1]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(image["bytes"])
                written.add(key)

        if written != expected:
            raise RuntimeError(
                f"Restoring {split}_fine from parquet recovered {len(written)} of {len(expected)} images. "
                "The conversion no longer matches the repo it was generated from."
            )

        # ``{split}_true_fine.npy`` is not in the conversion -- it is not an image, so the
        # hub dropped it. Upstream's own copy is simply each class index repeated by that
        # class's file count (verified: non-decreasing, 24 distinct values), which is what
        # _load_data_inner's positional pairing against a class-ordered walk requires.
        counts = Counter(class_dir for class_dir, _ in written)
        targets = np.repeat(
            np.fromiter(index2label, dtype=np.int64),
            [counts[label.replace(" ", "_")] for label in index2label.values()],
        )
        np.save(target / f"{split}_true_fine.npy", targets, allow_pickle=False)


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
    The dataset has 24 fine classes grouped into 7 overarching military vehicle categories,
    and ``labels`` selects which of the two each datum is labeled with.
    There is wide variation in image sizes.

    The download uses the hub's parquet conversion of the repo, which is two files rather
    than the 9,466 the image tree holds, and falls back to the image tree if that branch
    is unavailable. Neither path needs a huggingface token.

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
    labels : "fine" or "coarse", default "fine"
        Which labeling each datum carries. "fine" is the 24 vehicle classes; "coarse" is
        the 7 categories of ``hierarchy`` that group them, numbered alphabetically. Both
        read the same images, so the two can be built over one download.

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
    labels : "fine" or "coarse"
        The labeling this dataset was built with. Read-only.

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
        *,
        labels: Literal["fine", "coarse"] = "fine",
        as_datamaite: bool = False,
    ) -> None:
        if labels not in ("fine", "coarse"):
            raise ValueError(f"labels must be 'fine' or 'coarse', got {labels!r}.")
        self._labels: Literal["fine", "coarse"] = labels

        # The images on disk are always the fine tree, whichever labeling is asked for,
        # so the fine mapping is kept separately: it is what names the class directories.
        self._fine_index2label = type(self).index2label
        self._fine_label2index = {v: k for k, v in self._fine_index2label.items()}
        coarse_index2label, self._fine_to_coarse = _coarse_mapping(
            self._fine_index2label, self.hierarchy["vehicle"]["military"]["land vehicle"]
        )
        if labels == "coarse":
            # Shadows the class attribute for this instance only, before the base class
            # reads it -- label2index, metadata and the one-hot width all follow from it.
            self.index2label = coarse_index2label

        super().__init__(root, image_set, transforms, download, verbose, lazy, as_datamaite=as_datamaite)

    @property
    def labels(self) -> Literal["fine", "coarse"]:
        """Which labeling ``index2label`` and each datum's target describe.

        Fixed at construction: a dataset that could change labeling underneath a wrapper
        would leave the wrapper holding a stale ``index2label`` and target width.
        """
        return self._labels

    def _load_data(self) -> tuple[list[str], Sequence[int], dict[str, Any]]:
        # Only the selected image set is worth fetching, and the pattern that expresses
        # that depends on self.image_set -- so the part is narrowed here rather than
        # declared statically on the class.
        #
        # Two mirrors of the same images. The hub's parquet conversion is a couple of
        # files where the file tree is thousands, which without a token (one worker, to
        # stay under the anonymous rate limit) is the difference between a minute and the
        # better part of an hour. It is an auto-generated branch though, so the tree it
        # was generated from stays behind it: anything the recovery cannot vouch for
        # raises, and _download_part falls through.
        image_sets = ["train", "test"] if self.image_set == "base" else [self.image_set]
        self._resource = ResourcePart(
            "military_vehicles",
            (
                HFParquetResource(
                    repo_id=self._repo_id,
                    materialize=partial(
                        _materialize_fine_tree,
                        repo_id=self._repo_id,
                        index2label=self._fine_index2label,
                    ),
                    allow_patterns=[f"default/{img_set}/*" for img_set in image_sets],
                ),
                HFResource(
                    repo_id=self._repo_id,
                    # The jpgs and the targets file, which is everything the loader reads.
                    # A bare ``{img_set}_fine/*`` also drags down 20 per-class
                    # binary_true.npy files and a zip that nothing here opens.
                    allow_patterns=[
                        pattern
                        for img_set in image_sets
                        for pattern in (f"{img_set}_fine/*/*.jpg", f"{img_set}_fine/{img_set}_true_fine.npy")
                    ],
                ),
            ),
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
            for group in self._fine_label2index:
                data, file_data = self._load_group(img_set, group)
                filepaths.extend(data)
                _merge_datum_metadata(datum_metadata, file_data)

        # Targets come from the .npy and filepaths from walking the tree, paired by
        # position -- so a tree that is short by one file relabels everything after the
        # gap rather than failing. FileNotFoundError is the type _load_data retries a
        # download on, which is the right recovery for a half-fetched tree.
        if len(filepaths) != len(targets):
            raise FileNotFoundError(
                f"{self.path} holds {len(filepaths)} images but {len(targets)} targets; the download is incomplete."
            )

        if self._labels == "coarse":
            targets = [self._fine_to_coarse[target] for target in targets]

        return filepaths, targets, datum_metadata

    def _load_group(self, set_name: str, group_name: str) -> tuple[list[str], dict[str, Any]]:
        """Paths and per-datum metadata for one class folder within one image set."""
        group_dir = group_name.replace(" ", "_")
        base_dir = self.path / f"{set_name}_fine/{group_dir}"
        data_folder = sorted(base_dir.glob("*.jpg"))
        if not data_folder:
            raise FileNotFoundError

        file_data = {"id": [f"{group_dir}_{entry.stem}" for entry in data_folder]}
        return [str(entry) for entry in data_folder], file_data
