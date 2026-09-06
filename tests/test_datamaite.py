"""Tests for datamaite integration."""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from PIL import Image

from maite_datasets._datamaite import (
    check_datamaite_available,
    detect_format,
    format_for_task,
    get_dataset_task,
)
from maite_datasets._fileio import ResourcePart, URLResource
from maite_datasets.image_classification import MNIST
from maite_datasets.object_detection import MILCO, VOCDetection


def _mock_voc_resources(base: Path, year: str = "2012") -> list[ResourcePart]:
    tar_path = base / f"VOCtrainval-{year}.tar"
    hasher = hashlib.sha256()
    with open(tar_path, "rb") as f:
        while chunk := f.read(65535):
            hasher.update(chunk)
    return [
        ResourcePart(
            f"VOCtrainval-{year}",
            (
                URLResource(
                    url=f"https://data.brainchip.com/dataset-mirror/voc/VOCtrainval-{year}.tar",
                    filename=f"VOCtrainval-{year}.tar",
                    md5=False,
                    checksum=hasher.hexdigest(),
                ),
            ),
        ),
    ]


class TestDatamaiteAvailability:
    def test_check_datamaite_available_success(self):
        # Should not raise when datamaite is installed
        check_datamaite_available()

    def test_check_datamaite_available_missing(self, monkeypatch):
        with (
            patch.dict(sys.modules, {"datamaite": None}),
            pytest.raises(ImportError, match="requires datamaite to be installed"),
        ):
            check_datamaite_available()


class TestTaskAndFormatDetection:
    def test_get_dataset_task(self):
        assert get_dataset_task(MILCO) == "od"
        assert get_dataset_task(VOCDetection) == "od"
        assert get_dataset_task(MNIST) == "ic"

    def test_each_task_has_one_format(self):
        """One writable format per task: COCO refuses IC, and YOLO cannot record OD metadata."""
        assert format_for_task("od") == "coco"
        assert format_for_task("ic") == "yolo"


class TestFormatIsNotSelectable:
    def test_export_takes_no_format_argument(self, milco_fake, tmp_path):
        raw = MILCO(root=milco_fake)
        with pytest.raises(TypeError):
            raw.to_datamaite(dest=tmp_path / "out", output_format="yolo")

    def test_construction_takes_no_format_argument(self, milco_fake):
        with pytest.raises(TypeError):
            MILCO(root=milco_fake, as_datamaite=True, datamaite_format="yolo")

    def test_object_detection_always_writes_coco(self, milco_fake, tmp_path):
        dest = tmp_path / "out"
        MILCO(root=milco_fake).to_datamaite(dest=dest)
        assert detect_format(dest, "od") == "coco"

    def test_image_classification_always_writes_yolo(self, ship_fake, tmp_path):
        from maite_datasets.image_classification import Ships

        dest = tmp_path / "out"
        Ships(root=ship_fake).to_datamaite(dest=dest)
        assert detect_format(dest, "ic") == "yolo"

    def test_an_existing_yolo_export_is_still_readable(self, milco_fake, tmp_path):
        """Reading stays more permissive than writing: a YOLO directory already on disk loads."""
        import datamaite

        from maite_datasets._datamaite import _convert_od_dataset

        dest = tmp_path / "milco_datamaite"
        datamaite.write(_convert_od_dataset(MILCO(root=milco_fake), split="train"), dest, output_format="yolo")
        assert detect_format(dest, "od") == "yolo"

        loaded = MILCO(root=tmp_path, as_datamaite=True)
        assert len(loaded) == len(MILCO(root=milco_fake))


class TestDatamaiteODIntegration:
    def test_unreadable_export_dir_raises_value_error(self, tmp_path):
        """An export directory holding something other than a datamaite dataset is an error."""
        junk = tmp_path / "milco_datamaite"
        junk.mkdir()
        (junk / "notes.txt").write_text("not a dataset")
        with pytest.raises(ValueError, match="already exists but is not in a datamaite-compatible format"):
            MILCO(root=tmp_path, as_datamaite=True)

    def test_milco_not_found_raises_file_not_found(self, tmp_path):
        """If destination does not exist and download=False, raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            MILCO(root=tmp_path / "nonexistent", download=False, as_datamaite=True)

    def test_milco_to_datamaite_and_reload(self, milco_fake, tmp_path):
        """Test exporting an existing MILCO dataset to datamaite format and reloading it."""
        raw_milco = MILCO(root=milco_fake)
        dest_dir = tmp_path / "milco_datamaite"

        dm_milco = raw_milco.to_datamaite(dest=dest_dir)
        assert isinstance(dm_milco, ObjectDetectionDataset)
        assert len(dm_milco) == len(raw_milco)
        assert detect_format(dest_dir, "od") == "coco"

        # Reloading with as_datamaite=True should detect existing datamaite format and load it
        reloaded = MILCO(root=tmp_path, as_datamaite=True)
        assert isinstance(reloaded, ObjectDetectionDataset)
        assert len(reloaded) == len(raw_milco)

    def test_download_uses_tempdir_and_cleans_up(self, milco_fake, tmp_path, monkeypatch):
        """Test that download=True with as_datamaite=True uses a temp directory and cleans it up."""
        dest_root = tmp_path / "fresh"
        dest_dir = dest_root / "milco_datamaite"
        assert not dest_dir.exists()

        created_tmp_dirs: list[Path] = []

        # Stand in for the network: the data only materializes when download is allowed,
        # so the conversion has to take the download-into-a-tempdir path.
        raw_milco = MILCO(root=milco_fake)

        def fake_load_data(self):
            if not self._download:
                raise FileNotFoundError
            return (raw_milco._filepaths, raw_milco._targets, raw_milco._datum_metadata)

        monkeypatch.setattr(MILCO, "_load_data", fake_load_data)

        # Track temp dir
        orig_tempdir = tempfile.TemporaryDirectory

        class TrackingTempDir:
            def __init__(self, *args, **kwargs):
                self._td = orig_tempdir(*args, **kwargs)
                created_tmp_dirs.append(Path(self._td.name))

            def __enter__(self):
                return self._td.__enter__()

            def __exit__(self, *args):
                return self._td.__exit__(*args)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", TrackingTempDir)

        ds = MILCO(root=dest_root, download=True, as_datamaite=True)
        assert isinstance(ds, ObjectDetectionDataset)
        assert len(ds) == len(raw_milco)

        # Output was written to dest_dir
        assert dest_dir.exists()
        assert detect_format(dest_dir, "od") == "coco"

        # Temp directory was cleaned up and no longer exists
        assert len(created_tmp_dirs) == 1
        assert not created_tmp_dirs[0].exists()

    def test_voc_to_datamaite_and_reload(self, voc_fake, tmp_path, monkeypatch):
        """Test exporting VOCDetection to COCO datamaite format and reloading it."""
        monkeypatch.setattr(VOCDetection, "_resources", _mock_voc_resources(voc_fake))
        _ = VOCDetection(root=voc_fake)
        dir_path = voc_fake / "vocdataset" / "VOCdevkit" / "VOC2012"
        raw_voc = VOCDetection(root=dir_path, image_set="val")
        dest_dir = tmp_path / "vocdetection_datamaite"

        dm_voc = raw_voc.to_datamaite(dest=dest_dir)
        assert isinstance(dm_voc, ObjectDetectionDataset)
        assert len(dm_voc) == len(raw_voc)
        assert detect_format(dest_dir, "od") == "coco"

        # Reloading from root
        reloaded = VOCDetection(root=tmp_path, as_datamaite=True)
        assert isinstance(reloaded, ObjectDetectionDataset)
        assert len(reloaded) == len(raw_voc)


class TestDatamaiteICIntegration:
    def test_mnist_to_datamaite_and_reload(self, mnist_npy, tmp_path):
        """Test exporting in-memory array MNIST to YOLO ImageFolder datamaite format and reloading it."""
        raw_mnist = MNIST(root=str(mnist_npy), image_set="test")
        dest_dir = tmp_path / "mnist_datamaite"

        dm_mnist = raw_mnist.to_datamaite(dest=dest_dir)
        assert isinstance(dm_mnist, ImageClassificationDataset)
        assert len(dm_mnist) == len(raw_mnist)
        assert detect_format(dest_dir, "ic") == "yolo"

        # Reloading from root
        reloaded = MNIST(root=tmp_path, image_set="test", as_datamaite=True)
        assert isinstance(reloaded, ImageClassificationDataset)
        assert len(reloaded) == len(raw_mnist)


def _save_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    from PIL import Image

    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


@pytest.fixture
def droneswarm_fake(tmp_path):
    base = tmp_path / "droneswarm"
    (base / "images").mkdir(parents=True)
    (base / "labels").mkdir(parents=True)
    _save_image(base / "images" / "00000.png", size=(64, 48))
    (base / "labels" / "00000.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (base / "classes.txt").write_text("drone\n")
    return tmp_path


@pytest.fixture
def milco_operational_fake(tmp_path):
    """MILCO layout holding only the `operational` resources (2010, 2018)."""
    from PIL import Image

    for year in ("2010", "2018"):
        year_dir = tmp_path / "milco" / year
        year_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            Image.fromarray(np.full((10, 10, 3), 128, dtype=np.uint8)).save(year_dir / f"{i}_{year}.jpg")
            (year_dir / f"{i}_{year}.txt").write_text(f"0 {300 / 1024} {753 / 1024} {56 / 1024} {43 / 1024}")
    return tmp_path


class TestReaderBackedDatasets:
    def test_droneswarm_to_datamaite_roundtrip(self, droneswarm_fake, tmp_path):
        """Reader-backed datasets expose `classes`, not `index2label`; conversion must still work."""
        from maite_datasets.object_detection import DroneSwarm

        raw = DroneSwarm(root=droneswarm_fake)
        dm = raw.to_datamaite(dest=tmp_path / "out")
        assert isinstance(dm, ObjectDetectionDataset)
        assert len(dm) == len(raw)

    def test_droneswarm_export_writes_image_pixels(self, droneswarm_fake, tmp_path):
        """Reader-backed datasets have no `_filepaths`; the export must still carry pixels."""
        from maite_datasets.object_detection import DroneSwarm

        raw = DroneSwarm(root=droneswarm_fake)
        dest = tmp_path / "out"
        raw.to_datamaite(dest=dest)
        images = [p for p in dest.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        assert len(images) == len(raw)


class TestSplitHandling:
    def test_nonstandard_image_set_roundtrips(self, milco_operational_fake, tmp_path):
        """`operational` is a valid MILCO image_set but not a YOLO split directory."""
        raw = MILCO(root=milco_operational_fake, image_set="operational")
        assert len(raw) == 6
        with pytest.warns(UserWarning, match="exporting image_set 'operational' as 'train'"):
            dm = raw.to_datamaite(dest=tmp_path / "out")
        assert len(dm) == len(raw)

    def test_base_image_set_folds_without_warning(self, droneswarm_fake, tmp_path, recwarn):
        """`base` already means "no particular split", so folding it is not worth reporting."""
        from maite_datasets.object_detection import DroneSwarm

        raw = DroneSwarm(root=droneswarm_fake)
        assert raw.metadata["image_set"] == "base"
        raw.to_datamaite(dest=tmp_path / "out")
        assert [w for w in recwarn if "image_set" in str(w.message)] == []

    def test_operational_export_records_original_split_in_coco(self, milco_operational_fake, tmp_path):
        """COCO carries per-sample metadata, so the pre-fold image_set survives there."""
        raw = MILCO(root=milco_operational_fake, image_set="operational")
        with pytest.warns(UserWarning):
            dm = raw.to_datamaite(dest=tmp_path / "out")
        assert dm.samples[0].metadata["original_split"] == "operational"


class TestExportIdempotence:
    def test_repeated_export_replaces_instead_of_appending(self, milco_fake, tmp_path):
        raw = MILCO(root=milco_fake)
        dest = tmp_path / "out"
        first = raw.to_datamaite(dest=dest)
        second = raw.to_datamaite(dest=dest)
        assert len(second) == len(first) == len(raw)


class TestImageConversion:
    def test_float_image_preserves_intensity(self):
        from maite_datasets._datamaite import _array_to_png_bytes

        rng = np.random.default_rng(0)
        arr = rng.uniform(0.25, 1.0, size=(3, 8, 8)).astype(np.float32)
        decoded = np.array(Image.open(io.BytesIO(_array_to_png_bytes(arr))))
        assert decoded.max() > 0

    def test_two_channel_image_is_convertible(self):
        from maite_datasets._datamaite import _array_to_png_bytes

        arr = np.full((2, 8, 8), 200, dtype=np.uint8)
        with pytest.warns(UserWarning, match="2-channel image"):
            decoded = np.array(Image.open(io.BytesIO(_array_to_png_bytes(arr))))
        assert decoded.shape == (8, 8)
        assert decoded.max() == 200


class TestAsDatamaiteSignature:
    def test_as_datamaite_is_keyword_only(self, milco_fake):
        """Positional `as_datamaite` must not be silently ignored."""
        with pytest.raises(TypeError):
            MILCO(milco_fake, "train", None, False, False, False, True)


class TestDatasetNotMutated:
    def test_export_restores_lazy_flag(self, milco_fake, tmp_path):
        raw = MILCO(root=milco_fake)
        assert raw.lazy is False
        raw.to_datamaite(dest=tmp_path / "out")
        assert raw.lazy is False


class TestExistingRawData:
    def test_existing_raw_download_is_converted(self, milco_fake):
        """A root holding a normal (non-datamaite) download converts instead of failing."""
        ds = MILCO(root=milco_fake, as_datamaite=True)
        assert isinstance(ds, ObjectDetectionDataset)
        assert len(ds) == len(MILCO(root=milco_fake))


class TestTransformConsistency:
    def test_od_transform_keeps_pixels_and_dimensions_in_sync(self, milco_fake, tmp_path):
        """A geometry-changing transform must not leave annotations pointing at untransformed pixels."""
        raw = MILCO(root=milco_fake, transforms=lambda image: image[:, :5, :5])
        dest = tmp_path / "out"
        raw.to_datamaite(dest=dest)
        images = [p for p in dest.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        assert images
        for path in images:
            assert Image.open(path).size == (5, 5)

    def test_ic_transform_keeps_pixels_in_sync(self, ship_fake, tmp_path):
        """File-backed IC datasets must not drop transforms when exporting."""
        from maite_datasets.image_classification import Ships

        raw = Ships(root=ship_fake, transforms=lambda image: image[:, :5, :5])
        dest = tmp_path / "out"
        raw.to_datamaite(dest=dest)
        images = [p for p in dest.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        assert images
        assert Image.open(images[0]).size == (5, 5)


class TestNoWastedDecode:
    def test_file_backed_ic_export_does_not_decode(self, ship_fake, tmp_path, monkeypatch):
        """Without transforms, a file-backed IC export references files and never decodes."""
        from maite_datasets.image_classification import Ships

        raw = Ships(root=ship_fake)
        decodes = []
        original = Ships._read_file

        def counting_read_file(self, path):
            decodes.append(path)
            return original(self, path)

        monkeypatch.setattr(Ships, "_read_file", counting_read_file)
        raw.to_datamaite(dest=tmp_path / "out")
        assert decodes == []


class TestDatumMetadata:
    def test_scalar_datum_metadata_is_carried_into_samples(self, milco_fake, tmp_path):
        """Per-datum metadata is the input to downstream metadata analysis; it must survive export."""
        raw = MILCO(root=milco_fake)
        _, _, datum_meta = raw[0]
        assert datum_meta["year"] == "2015"

        dm = raw.to_datamaite(dest=tmp_path / "out")
        assert dm.samples[0].metadata["year"] == "2015"

    def test_non_scalar_datum_metadata_is_skipped(self, droneswarm_fake, tmp_path):
        """Reader datasets attach per-object annotation lists; those belong in detections, not metadata."""
        from maite_datasets.object_detection import DroneSwarm

        raw = DroneSwarm(root=droneswarm_fake)
        _, _, datum_meta = raw[0]
        assert "annotations" in datum_meta

        dm = raw.to_datamaite(dest=tmp_path / "out")
        assert "annotations" not in dm.samples[0].metadata
        assert dm.samples[0].metadata["num_annotations"] == 1
        assert dm.samples[0].metadata["label_file_exists"] is True

    def test_ic_datum_metadata_is_carried_into_samples(self, ship_fake):
        from maite_datasets._datamaite import _convert_ic_dataset
        from maite_datasets.image_classification import Ships

        raw = Ships(root=ship_fake)
        _, _, datum_meta = raw[0]
        assert datum_meta["scene_id"] == "abc"

        converted = _convert_ic_dataset(raw, split="train")
        assert converted.samples[0].metadata["scene_id"] == "abc"
        assert converted.samples[0].metadata["latitude"] == 25.0

    def test_yolo_export_cannot_keep_datum_metadata(self, ship_fake, tmp_path):
        """The YOLO wire format has nowhere to record it -- COCO is the format that keeps it."""
        from maite_datasets.image_classification import Ships

        dm = Ships(root=ship_fake).to_datamaite(dest=tmp_path / "out")
        assert "scene_id" not in dm.samples[0].metadata


class TestImageIds:
    def test_underscored_ids_do_not_become_numbers(self, milco_fake, tmp_path):
        """`int("1_2015")` is 12015 in Python -- a digit-separator artifact, not an id."""
        from maite_datasets._datamaite import _convert_od_dataset

        raw = MILCO(root=milco_fake)
        assert raw[1][2]["id"] == "1_2015"

        converted = _convert_od_dataset(raw, split="train")
        assert 12015 not in {s.image_id for s in converted.samples}
        assert len({s.image_id for s in converted.samples}) == len(raw)
        assert converted.samples[1].metadata["original_id"] == "1_2015"

    def test_numeric_ids_are_kept(self, tmp_path):
        """A dataset whose ids really are integers keeps them as COCO image ids."""
        from maite_datasets._datamaite import _convert_od_dataset

        class FakeNumericIdDataset:
            metadata = {"index2label": {0: "thing"}}
            transforms: list = []
            lazy = False

            def __len__(self):
                return 2

            def __getitem__(self, index):
                target = type("T", (), {"boxes": [], "labels": [], "scores": None})()
                return np.zeros((3, 4, 4), np.uint8), target, {"id": 48 if index == 0 else 97}

        converted = _convert_od_dataset(FakeNumericIdDataset(), split="val")
        assert [s.image_id for s in converted.samples] == [48, 97]


def _fake_od_dataset(datum_meta: dict, num_boxes: int = 2):
    """Minimal OD dataset standing in for one whose metadata holds per-object lists."""

    class FakeODDataset:
        metadata = {"index2label": {0: "thing"}}
        transforms: list = []
        lazy = False

        def __len__(self):
            return 1

        def __getitem__(self, index):
            target = type(
                "T",
                (),
                {"boxes": [[0, 0, 4, 4]] * num_boxes, "labels": [0] * num_boxes, "scores": None},
            )()
            return np.zeros((3, 8, 8), np.uint8), target, dict(datum_meta)

    return FakeODDataset()


class TestPerObjectMetadata:
    def test_aligned_lists_become_detection_attributes(self):
        """SeaDrone's object_id/object_size are per-object, so they belong on the detections."""
        from maite_datasets._datamaite import _convert_od_dataset

        ds = _fake_od_dataset({"id": 7, "object_id": [444, 555], "object_size": [1.5, 2.5]})
        converted = _convert_od_dataset(ds, split="val")

        sample = converted.samples[0]
        assert [d.attributes["object_id"] for d in sample.detections] == [444, 555]
        assert [d.attributes["object_size"] for d in sample.detections] == [1.5, 2.5]
        assert "object_id" not in sample.metadata

    def test_misaligned_lists_are_dropped(self):
        """A list that does not line up with the boxes cannot be attributed to any of them."""
        from maite_datasets._datamaite import _convert_od_dataset

        ds = _fake_od_dataset({"id": 7, "notes": ["a", "b", "c"]}, num_boxes=2)
        converted = _convert_od_dataset(ds, split="val")

        sample = converted.samples[0]
        assert "notes" not in sample.metadata
        assert all("notes" not in d.attributes for d in sample.detections)

    def test_detection_attributes_survive_coco(self, tmp_path):
        from maite_datasets._datamaite import _convert_od_dataset, load_datamaite_dataset, write_as_datamaite

        ds = _fake_od_dataset({"id": 7, "object_id": [444, 555]})
        dest = tmp_path / "out"
        write_as_datamaite(ds, dest, task="od", split="val")
        back = load_datamaite_dataset(dest, task="od", dataset_format="coco", split="val")
        assert [d.attributes["object_id"] for d in back.samples[0].detections] == [444, 555]
        assert _convert_od_dataset(ds, split="val").samples[0].detections[0].attributes["object_id"] == 444


class TestProvenanceMetadata:
    def test_preserved_integer_id_adds_no_extra_column(self):
        """When the COCO image id already is the dataset's id, `original_id` says nothing."""
        from maite_datasets._datamaite import _convert_od_dataset

        converted = _convert_od_dataset(_fake_od_dataset({"id": 48}), split="val")
        assert converted.samples[0].image_id == 48
        assert "original_id" not in converted.samples[0].metadata

    def test_unrepresentable_id_is_recorded(self, milco_fake, tmp_path):
        """A string id cannot be a COCO image id, so it has to survive in metadata."""
        raw = MILCO(root=milco_fake)
        dm = raw.to_datamaite(dest=tmp_path / "out")
        assert dm.samples[0].metadata["original_id"] == "0_2015"

    def test_unfolded_split_adds_no_extra_column(self, milco_fake, tmp_path):
        """`original_split` only says something when a fold happened; otherwise it is noise."""
        raw = MILCO(root=milco_fake, image_set="train")
        dm = raw.to_datamaite(dest=tmp_path / "out")
        assert "original_split" not in dm.samples[0].metadata

    def test_folded_split_is_recorded(self, milco_operational_fake, tmp_path):
        raw = MILCO(root=milco_operational_fake, image_set="operational")
        with pytest.warns(UserWarning):
            dm = raw.to_datamaite(dest=tmp_path / "out")
        assert dm.samples[0].metadata["original_split"] == "operational"
