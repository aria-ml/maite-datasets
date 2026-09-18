"""Tests for the MilitaryVehicles parquet fast path and its fine/coarse labelings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maite_datasets._fileio import HFParquetResource, HFResource
from maite_datasets.image_classification._military_vehicles import (
    MilitaryVehicles,
    _materialize_fine_tree,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def write_parquet(path: Path, label_names: list[str], rows: list[tuple[str, str, bytes]]) -> Path:
    """Write a parquet file shaped like the hub's own image-folder conversion.

    `rows` are ``(label_name, filename, blob)``; the label is stored as an index into
    `label_names`, whose spelling lives in the schema's ``huggingface`` metadata exactly
    as the hub writes it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    images = pa.array([{"bytes": blob, "path": name} for _, name, blob in rows], type=image_type)
    labels = pa.array([label_names.index(label) for label, _, _ in rows], type=pa.int64())
    metadata = {
        "info": {
            "features": {
                "image": {"_type": "Image"},
                "label": {"names": label_names, "_type": "ClassLabel"},
            }
        }
    }
    schema = pa.schema(
        [("image", image_type), ("label", pa.int64())],
        metadata={b"huggingface": json.dumps(metadata).encode()},
    )
    pq.write_table(pa.Table.from_arrays([images, labels], schema=schema), path)
    return path


# A miniature of the real repo: two fine classes collapsing into one coarse group, with
# the coarse rows first and zero-padded, exactly as the hub's conversion orders them.
INDEX2LABEL = {0: "Alpha One", 1: "Bravo"}
LABEL_NAMES = ["Alpha_One", "Bravo", "Group"]
FINE_ROWS = [("Alpha_One", "1.jpg", b"alpha-one"), ("Alpha_One", "2.jpg", b"alpha-two"), ("Bravo", "3.jpg", b"bravo")]
COARSE_ROWS = [
    ("Group", "0001.jpg", b"alpha-one"),
    ("Group", "0002.jpg", b"alpha-two"),
    ("Group", "0003.jpg", b"bravo"),
]
LISTING = [
    "README.md",
    "WEO_Data_Sheet.xlsx",
    *(
        f"{split}_{side}/{label}/{name}"
        for split in ("train", "test")
        for side, rows in (("coarse", COARSE_ROWS), ("fine", FINE_ROWS))
        for label, name, _ in rows
    ),
    # Non-image files the conversion drops but the tree carries; none should be matched.
    "train_coarse/train_true_coarse.npy",
    "train_fine/train_true_fine.npy",
    "train_fine/Alpha_One/binary_true.npy",
    "train_fine/zipped_binary_true.zip",
]


@pytest.fixture
def listing(monkeypatch):
    """Serve a canned main-branch listing, and report how often it was asked for."""
    calls = []

    def fake_list(repo_id, repo_type="dataset", revision=None):
        calls.append(repo_id)
        return LISTING

    monkeypatch.setattr("maite_datasets.image_classification._military_vehicles._hf_list_files", fake_list)
    return calls


@pytest.fixture
def parquet(tmp_path):
    return write_parquet(
        tmp_path / "_parquet" / "default" / "train" / "0000.parquet", LABEL_NAMES, COARSE_ROWS + FINE_ROWS
    )


class TestMaterializeFineTree:
    def test_writes_the_fine_side_and_leaves_the_coarse_side_out(self, listing, parquet, tmp_path):
        """Both labelings share one parquet; only the fine tree is what this dataset loads."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()

        _materialize_fine_tree([parquet], directory, repo_id="owner/name", index2label=INDEX2LABEL)

        assert (directory / "train_fine" / "Alpha_One" / "1.jpg").read_bytes() == b"alpha-one"
        assert (directory / "train_fine" / "Alpha_One" / "2.jpg").read_bytes() == b"alpha-two"
        assert (directory / "train_fine" / "Bravo" / "3.jpg").read_bytes() == b"bravo"
        assert not (directory / "train_coarse").exists()
        assert sorted(p.name for p in (directory / "train_fine").rglob("*.jpg")) == ["1.jpg", "2.jpg", "3.jpg"]

    def test_synthesizes_the_targets_file_in_class_order(self, listing, parquet, tmp_path):
        """The tree alone is not loadable: _load_data_inner pairs it with the .npy positionally."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()

        _materialize_fine_tree([parquet], directory, repo_id="owner/name", index2label=INDEX2LABEL)

        targets = np.load(directory / "train_fine" / "train_true_fine.npy")
        assert targets.tolist() == [0, 0, 1], "two Alpha One images then one Bravo, in index2label order"
        assert targets.dtype == np.int64

    def test_asks_the_hub_for_one_listing_however_many_splits(self, listing, tmp_path):
        """Request count is the whole point of this path; a per-split listing would erode it."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()
        files = [
            write_parquet(tmp_path / "_p" / "default" / split / "0000.parquet", LABEL_NAMES, COARSE_ROWS + FINE_ROWS)
            for split in ("train", "test")
        ]

        _materialize_fine_tree(files, directory, repo_id="owner/name", index2label=INDEX2LABEL)

        assert listing == ["owner/name"]

    def test_rejects_a_split_the_repo_does_not_publish(self, listing, tmp_path):
        """A conversion whose layout has drifted must read as a failed download, not an empty one."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()
        parquet = write_parquet(tmp_path / "_p" / "default" / "val" / "0000.parquet", LABEL_NAMES, FINE_ROWS)

        with pytest.raises(RuntimeError, match="no val_fine images"):
            _materialize_fine_tree([parquet], directory, repo_id="owner/name", index2label=INDEX2LABEL)

    def test_rejects_a_recovery_that_misses_images(self, listing, tmp_path):
        """Silently short trees are the failure mode worth spending a check on."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()
        parquet = write_parquet(
            tmp_path / "_p" / "default" / "train" / "0000.parquet", LABEL_NAMES, COARSE_ROWS + FINE_ROWS[:2]
        )

        with pytest.raises(RuntimeError, match="recovered 2 of 3"):
            _materialize_fine_tree([parquet], directory, repo_id="owner/name", index2label=INDEX2LABEL)

    def test_rejects_a_class_the_dataset_does_not_declare(self, monkeypatch, tmp_path):
        """Class dirs become filesystem paths, so they are checked against index2label, not trusted."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()
        monkeypatch.setattr(
            "maite_datasets.image_classification._military_vehicles._hf_list_files",
            lambda *a, **k: [*LISTING, "train_fine/Charlie/9.jpg"],
        )
        parquet = write_parquet(tmp_path / "_p" / "default" / "train" / "0000.parquet", LABEL_NAMES, FINE_ROWS)

        with pytest.raises(RuntimeError, match="Charlie"):
            _materialize_fine_tree([parquet], directory, repo_id="owner/name", index2label=INDEX2LABEL)

    def test_keeps_both_copies_of_byte_identical_images(self, listing, tmp_path):
        """Three train groups hold four rows because two source files share bytes; both are real images."""
        directory = tmp_path / "militaryvehicles"
        directory.mkdir()
        twins = [("Alpha_One", "1.jpg", b"same"), ("Alpha_One", "2.jpg", b"same"), ("Bravo", "3.jpg", b"same")]
        parquet = write_parquet(
            tmp_path / "_p" / "default" / "train" / "0000.parquet", LABEL_NAMES, COARSE_ROWS + twins
        )

        _materialize_fine_tree([parquet], directory, repo_id="owner/name", index2label=INDEX2LABEL)

        assert sorted(p.name for p in (directory / "train_fine").rglob("*.jpg")) == ["1.jpg", "2.jpg", "3.jpg"]


COARSE_CATEGORIES = ["Air Defense", "BMD", "BMP", "BTR", "MT LB", "Self Propelled Artillery", "Tank"]


@pytest.mark.optional
class TestCoarseLabels:
    @pytest.fixture(autouse=True)
    def _no_download(self, monkeypatch):
        monkeypatch.setattr("maite_datasets._fileio._hf_extract", lambda **kw: None)

    def test_coarse_exposes_the_seven_categories(self, military_vehicles_fake):
        """Ids are alphabetical, which is also how upstream's own *_true_coarse.npy encodes them."""
        dataset = MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        assert dataset.index2label == dict(enumerate(COARSE_CATEGORIES))
        assert dataset.label2index["Tank"] == 6

    def test_fine_stays_the_default(self, military_vehicles_fake):
        dataset = MilitaryVehicles(root=military_vehicles_fake)
        assert len(dataset.index2label) == 24
        assert dataset.labels == "fine"

    def test_the_class_attribute_still_reports_the_fine_classes(self, military_vehicles_fake):
        """A coarse instance must not rewrite the shared class-level mapping."""
        MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        assert len(MilitaryVehicles.index2label) == 24
        assert MilitaryVehicles.index2label[16] == "T-14"

    def test_targets_are_remapped_through_the_hierarchy(self, military_vehicles_fake):
        """The fixture holds one image per fine class, in index2label order."""
        dataset = MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        _, score, _ = dataset[16]  # fine class 16 is T-14, a Tank
        assert score.shape == (7,)
        assert score.argmax() == COARSE_CATEGORIES.index("Tank")

    def test_a_singleton_category_keeps_its_own_name(self, military_vehicles_fake):
        """BMD and MT LB are their own coarse categories, so they survive the collapse."""
        dataset = MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        _, score, _ = dataset[3]  # fine class 3 is BMD
        assert score.argmax() == COARSE_CATEGORIES.index("BMD")

    def test_length_and_files_are_unchanged_by_the_labeling(self, military_vehicles_fake):
        fine = MilitaryVehicles(root=military_vehicles_fake)
        coarse = MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        assert len(coarse) == len(fine)
        assert coarse._filepaths == fine._filepaths

    def test_metadata_carries_the_active_labeling(self, military_vehicles_fake):
        dataset = MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        assert dataset.metadata["index2label"] == dict(enumerate(COARSE_CATEGORIES))

    def test_labels_is_read_only(self, military_vehicles_fake):
        """Swapping the labeling under a wrapper would strand its cached index2label."""
        dataset = MilitaryVehicles(root=military_vehicles_fake, labels="coarse")
        with pytest.raises(AttributeError):
            dataset.labels = "fine"

    def test_an_unknown_labeling_is_rejected_at_construction(self, military_vehicles_fake):
        with pytest.raises(ValueError, match="labels must be"):
            MilitaryVehicles(root=military_vehicles_fake, labels="medium")

    def test_the_hierarchy_covers_every_fine_class_exactly_once(self):
        """The coarse mapping is derived from `hierarchy`, so drift there is a silent relabeling."""
        categories = MilitaryVehicles.hierarchy["vehicle"]["military"]["land vehicle"]
        covered = [name for members in categories.values() for name in members]
        assert sorted(covered) == sorted(MilitaryVehicles.index2label.values())
        assert len(covered) == len(set(covered))


@pytest.mark.optional
class TestTargetAlignment:
    def build_tree(self, root, targets):
        """One image per fine class, plus a targets file of the caller's choosing."""
        from PIL import Image

        split_dir = root / "militaryvehicles" / "train_fine"
        for i, group in enumerate(MilitaryVehicles.index2label.values()):
            group_dir = split_dir / group.replace(" ", "_")
            group_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8)).save(group_dir / f"{i:05}.jpg")
        np.save(split_dir / "train_true_fine.npy", np.asarray(targets))
        return root

    def test_a_short_tree_is_a_failed_download_not_a_mislabeled_dataset(self, monkeypatch, tmp_path):
        """Files and targets are paired positionally, so a gap silently relabels everything after it."""
        root = self.build_tree(tmp_path, np.arange(25))
        monkeypatch.setattr("maite_datasets._base._download_part", lambda *a, **kw: None)

        with pytest.raises(FileNotFoundError, match="24 images but 25 targets"):
            MilitaryVehicles(root=root, download=True)

    def test_a_matching_tree_loads(self, monkeypatch, tmp_path):
        root = self.build_tree(tmp_path, np.arange(24))
        monkeypatch.setattr("maite_datasets._base._download_part", lambda *a, **kw: None)

        assert len(MilitaryVehicles(root=root, download=True)) == 24


@pytest.mark.optional
class TestMirrors:
    @pytest.fixture
    def part(self, monkeypatch):
        """Build a dataset with nothing on disk and hand back the part it tried to fetch."""

        def build(root, image_set):
            captured = {}
            monkeypatch.setattr(
                "maite_datasets._base._download_part",
                lambda part, directory, dataset_root, download, verbose: captured.update(part=part),
            )
            with pytest.raises(FileNotFoundError):
                MilitaryVehicles(root=root, image_set=image_set, download=True)
            return captured["part"]

        return build

    def test_parquet_is_tried_before_the_file_tree(self, part, tmp_path):
        """Two requests against 1,622; the tree is still there when the conversion is not."""
        mirrors = part(tmp_path, "test").mirrors

        assert isinstance(mirrors[0], HFParquetResource)
        assert mirrors[0].revision == "refs/convert/parquet"
        assert mirrors[0].allow_patterns == ["default/test/*"]
        assert isinstance(mirrors[1], HFResource)

    def test_base_fetches_both_splits_from_either_mirror(self, part, tmp_path):
        mirrors = part(tmp_path, "base").mirrors

        assert mirrors[0].allow_patterns == ["default/train/*", "default/test/*"]
        assert mirrors[1].allow_patterns == [
            "train_fine/*/*.jpg",
            "train_fine/train_true_fine.npy",
            "test_fine/*/*.jpg",
            "test_fine/test_true_fine.npy",
        ]

    def test_the_tree_mirror_skips_files_the_loader_never_reads(self, part, tmp_path):
        """train_fine also holds 20 per-class binary_true.npy and a zip; 21 needless transfers."""
        patterns = part(tmp_path, "train").mirrors[1].allow_patterns

        assert patterns == ["train_fine/*/*.jpg", "train_fine/train_true_fine.npy"]

    def test_the_materializer_is_bound_to_this_datasets_fine_classes(self, part, tmp_path, monkeypatch):
        """The mirror carries the recovery with it, so _fileio needs no dataset knowledge."""
        seen = {}
        monkeypatch.setattr(
            "maite_datasets.image_classification._military_vehicles._materialize_fine_tree",
            lambda files, directory, *, repo_id, index2label: seen.update(repo=repo_id, labels=index2label),
        )
        part(tmp_path, "train").mirrors[0].materialize([], tmp_path)

        assert seen["repo"] == "leibnitz-lab/military_vehicles"
        assert seen["labels"] == MilitaryVehicles.index2label
