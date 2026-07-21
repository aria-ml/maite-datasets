"""Tests for the recently added dataset loaders (M3FD, DroneVehicle, DroneSwarm, SkySeaLand, Military*)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PIL import Image

from maite_datasets.image_classification._military_aircraft import MilitaryAircraft as ICMilitaryAircraft
from maite_datasets.image_classification._military_vehicles import MilitaryVehicles
from maite_datasets.object_detection._droneswarm import DroneSwarm
from maite_datasets.object_detection._dronevehicle import DroneVehicle
from maite_datasets.object_detection._m3fd import M3FD
from maite_datasets.object_detection._military_aircraft import MilitaryAircraft as ODMilitaryAircraft
from maite_datasets.object_detection._skysealand import SkySeaLand

M3FD_ANNOTATION = """<annotation>
    <size><width>10</width><height>10</height><depth>3</depth></size>
    <object>
        <name>People</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>6</ymax></bndbox>
    </object>
    <object>
        <name>Car</name>
        <bndbox><xmin>3</xmin><ymin>3</ymin><xmax>8</xmax><ymax>9</ymax></bndbox>
    </object>
</annotation>"""

DRONEVEHICLE_IR_ANNOTATION = """<annotation>
    <filename>ir_00001.jpg</filename>
    <size><width>840</width><height>712</height><depth>1</depth></size>
    <object>
        <name>feright car</name>
        <polygon>
            <x1>10</x1><y1>20</y1>
            <x2>40</x2><y2>25</y2>
            <x3>38</x3><y3>60</y3>
            <x4>12</x4><y4>55</y4>
        </polygon>
    </object>
</annotation>"""

DRONEVEHICLE_RGB_ANNOTATION = """<annotation>
    <filename>rgb_00001.jpg</filename>
    <size><width>840</width><height>712</height><depth>3</depth></size>
</annotation>"""


def _save_image(path, size=(10, 10)):
    Image.fromarray(np.ones((size[1], size[0], 3), dtype=np.uint8)).save(path)


@pytest.fixture(scope="session")
def m3fd_fake(tmp_path_factory):
    """M3FD detection layout (Vis/Ir/Annotation) plus the "operational" extra/ set."""
    temp = tmp_path_factory.mktemp("data")
    for base, count in ((temp / "m3fd", 3), (temp / "m3fd" / "extra", 2)):
        for folder in ("Vis", "Ir", "Annotation"):
            (base / folder).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            _save_image(base / "Vis" / f"{i:05}.png")
            _save_image(base / "Ir" / f"{i:05}.png")
            (base / "Annotation" / f"{i:05}.xml").write_text(M3FD_ANNOTATION)
    yield temp


@pytest.fixture(scope="session")
def dronevehicle_fake(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    base = temp / "dronevehicle"
    for split, count in (("train", 3), ("val", 2), ("test", 1)):
        img_dir = base / split / f"{split}img"
        img_ir_dir = base / split / f"{split}imgr"
        ir_dir = base / split / f"{split}labelr"
        rgb_dir = base / split / f"{split}label"
        for folder in (img_dir, img_ir_dir, ir_dir, rgb_dir):
            folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            _save_image(img_dir / f"{i:05}.jpg")
            _save_image(img_ir_dir / f"{i:05}.jpg")
            (ir_dir / f"{i:05}.xml").write_text(DRONEVEHICLE_IR_ANNOTATION)
            (rgb_dir / f"{i:05}.xml").write_text(DRONEVEHICLE_RGB_ANNOTATION)
    yield temp


@pytest.fixture(scope="session")
def military_aircraft_fake(tmp_path_factory):
    """Shared layout for both MilitaryAircraft datasets - 640x640 jpgs beside YOLO txts."""
    temp = tmp_path_factory.mktemp("data")
    base = temp / "militaryaircraft"
    contents = {
        "train": ["0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.9 0.9\n", ""],
        "val": ["3 0.5 0.5 0.2 0.2\n"],
        "test": ["4 0.1 0.1 0.05 0.05\nbad line\n"],
    }
    for split, labels in contents.items():
        (base / split).mkdir(parents=True, exist_ok=True)
        for i, label in enumerate(labels):
            _save_image(base / split / f"{i:05}.jpg", size=(640, 640))
            (base / split / f"{i:05}.txt").write_text(label)
    yield temp


@pytest.fixture(scope="session")
def military_vehicles_fake(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    base = temp / "militaryvehicles"
    groups = list(MilitaryVehicles.index2label.values())
    for split in ("train", "test"):
        split_dir = base / f"{split}_fine"
        for i, group in enumerate(groups):
            group_dir = split_dir / group.replace(" ", "_")
            group_dir.mkdir(parents=True, exist_ok=True)
            _save_image(group_dir / f"{i:05}.jpg")
        np.save(split_dir / f"{split}_true_fine.npy", np.arange(len(groups)))
    yield temp


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
def skysealand_fake(tmp_path):
    base = tmp_path / "skysealand"
    (base / "images").mkdir(parents=True)
    (base / "labels").mkdir(parents=True)
    _save_image(base / "images" / "00000.jpg", size=(64, 48))
    (base / "labels" / "00000.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (base / "classes.txt").write_text("airplane\n")
    return tmp_path


@pytest.mark.optional
class TestM3FD:
    def test_m3fd_train(self, m3fd_fake):
        dataset = M3FD(root=m3fd_fake)
        assert len(dataset) == 3
        img, target, datum_meta = dataset[0]
        # RGB image with the paired IR channel stacked on
        assert img.shape == (4, 10, 10)
        assert np.array_equal(target.labels, [0, 1])
        assert np.array_equal(target.boxes[0], [1, 2, 5, 6])
        assert datum_meta["image_width"] == 10
        assert datum_meta["image_id"] == "00000"

    def test_m3fd_operational(self, m3fd_fake):
        dataset = M3FD(root=m3fd_fake, image_set="operational")
        assert dataset.path.name == "extra"
        assert len(dataset) == 2

    def test_m3fd_missing_bndbox_coordinate(self, m3fd_fake, tmp_path):
        dataset = M3FD(root=m3fd_fake)
        annotation = tmp_path / "bad.xml"
        annotation.write_text(M3FD_ANNOTATION.replace("<xmax>5</xmax>", ""))
        with pytest.raises(ValueError, match="Missing bndbox/xmax"):
            dataset._read_annotations(str(annotation))

    def test_m3fd_hf_load(self, m3fd_fake, monkeypatch, tmp_path):
        monkeypatch.setattr("maite_datasets.object_detection._m3fd._hf_extract", lambda **kwargs: None)
        dataset = M3FD(root=m3fd_fake)
        dataset.path = tmp_path / "hf"
        for split, count in (("train", 2), ("val", 1)):
            (dataset.path / split / "images").mkdir(parents=True)
            (dataset.path / split / "labels").mkdir(parents=True)
            for i in range(count):
                _save_image(dataset.path / split / "images" / f"{i:05}.tiff")
                (dataset.path / split / "labels" / f"{i:05}.txt").write_text("")

        dataset.image_set = "base"
        filepaths, targets, datum_metadata = dataset._load_hf_data()
        assert len(filepaths) == 3
        assert len(targets) == 3
        assert datum_metadata["image_id"] == ["00000", "00001", "00000"]

        dataset.image_set = "test"
        with pytest.raises(FileNotFoundError):
            dataset._load_hf_data()


@pytest.mark.optional
class TestDroneVehicle:
    def test_dronevehicle_train(self, dronevehicle_fake):
        dataset = DroneVehicle(root=dronevehicle_fake)
        assert len(dataset) == 3
        img, target, datum_meta = dataset[0]
        assert img.shape == (4, 10, 10)
        # "feright car" is corrected to the "freight car" class
        assert np.array_equal(target.labels, [4])
        # Rotated quadrilateral reduced to its axis-aligned extent
        assert np.array_equal(target.boxes[0], [10, 20, 40, 60])
        assert datum_meta["image_id"] == "train_00000.jpg"
        assert datum_meta["infrared_filename"] == "ir_00001.jpg"
        assert datum_meta["rgb_filename"] == "rgb_00001.jpg"
        # IR depth plus RGB depth
        assert datum_meta["image_depth"] == 4

    def test_dronevehicle_base(self, dronevehicle_fake):
        dataset = DroneVehicle(root=dronevehicle_fake, image_set="base")
        assert len(dataset) == 6
        assert len(dataset._datum_metadata["image_id"]) == 6

    @pytest.mark.parametrize("image_set", ["train", "base"])
    def test_dronevehicle_hf_load(self, dronevehicle_fake, monkeypatch, image_set):
        monkeypatch.setattr("maite_datasets.object_detection._dronevehicle._hf_extract", lambda **kwargs: None)
        monkeypatch.setattr("maite_datasets.object_detection._dronevehicle._extract_archive", lambda *args: None)
        dataset = DroneVehicle(root=dronevehicle_fake, image_set=image_set, verbose=True)
        filepaths, targets, datum_metadata = dataset._load_hf_data()
        expected = 6 if image_set == "base" else 3
        assert len(filepaths) == expected
        assert len(targets) == expected
        assert len(datum_metadata["image_id"]) == expected


@pytest.mark.optional
class TestMilitaryAircraftObjectDetection:
    def test_detection_train(self, military_aircraft_fake):
        dataset = ODMilitaryAircraft(root=military_aircraft_fake)
        assert len(dataset) == 2
        img, target, _ = dataset[0]
        assert img.shape == (3, 640, 640)
        assert np.array_equal(target.labels, [0, 1])
        # Normalized yolo boxes scaled to pixels: xc 0.5, w 0.1 -> x0 288, x1 352
        assert np.array_equal(target.boxes[0], [288, 288, 352, 352])
        # Second image has an empty annotation file
        _, empty_target, _ = dataset[1]
        assert empty_target.boxes.size == 0

    def test_detection_base(self, military_aircraft_fake):
        dataset = ODMilitaryAircraft(root=military_aircraft_fake, image_set="base")
        assert len(dataset) == 4

    def test_detection_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ODMilitaryAircraft(root=tmp_path, image_set="val")


@pytest.mark.optional
class TestMilitaryAircraftImageClassification:
    def test_classification_train(self, military_aircraft_fake):
        dataset = ICMilitaryAircraft(root=military_aircraft_fake)
        # Two annotated objects plus one background-only image
        assert len(dataset) == 3
        assert dataset._targets == [0, 1, 88]
        img, score, _ = dataset[0]
        # Small objects are widened to the minimum crop size
        assert img.shape == (3, 128, 128)
        assert score.shape == (89,)
        assert score[0] == 1
        # Background data point carries no crop, so the whole image is returned
        background_img, background_score, _ = dataset[2]
        assert background_img.shape == (3, 640, 640)
        assert background_score[88] == 1

    def test_classification_crop_is_clamped_and_shifted(self, military_aircraft_fake):
        dataset = ICMilitaryAircraft(root=military_aircraft_fake)
        # Object larger than the image is clamped to the image size
        assert dataset._get_image_crop(320, 320, 900, 900) == (0, 0, 640, 640)
        # Box hanging off the edge slides back in bounds at its original size
        assert dataset._get_image_crop(600, 600, 200, 200) == (440, 440, 640, 640)

    def test_classification_lazy(self, military_aircraft_fake):
        dataset = ICMilitaryAircraft(root=military_aircraft_fake, lazy=True)
        img, _, _ = dataset[0]
        assert np.asarray(img).shape == (3, 128, 128)

    def test_classification_base(self, military_aircraft_fake):
        dataset = ICMilitaryAircraft(root=military_aircraft_fake, image_set="base")
        assert len(dataset) == 5

    def test_classification_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ICMilitaryAircraft(root=tmp_path, image_set="val")


@pytest.mark.optional
class TestMilitaryVehicles:
    @pytest.fixture(autouse=True)
    def _no_download(self, monkeypatch):
        monkeypatch.setattr("maite_datasets.image_classification._military_vehicles._hf_extract", lambda **kw: None)

    def test_vehicles_train(self, military_vehicles_fake):
        dataset = MilitaryVehicles(root=military_vehicles_fake)
        assert len(dataset) == len(MilitaryVehicles.index2label)
        img, score, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert score.shape == (24,)
        assert score[0] == 1
        assert datum_meta["image_id"] == "2S19_MSTA_00000"

    def test_vehicles_base(self, military_vehicles_fake):
        dataset = MilitaryVehicles(root=military_vehicles_fake, image_set="base")
        assert len(dataset) == 2 * len(MilitaryVehicles.index2label)

    def test_vehicles_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MilitaryVehicles(root=tmp_path)


@pytest.mark.optional
class TestDroneSwarm:
    def test_droneswarm_local(self, droneswarm_fake):
        dataset = DroneSwarm(root=droneswarm_fake)
        assert len(dataset) == 1
        assert dataset.metadata["image_set"] == "base"
        assert dataset.metadata["index2label"] == {0: "drone"}
        img, target, _ = dataset[0]
        assert img.shape == (3, 48, 64)
        assert target.boxes.shape == (1, 4)

    def test_droneswarm_hf_load(self, droneswarm_fake, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("maite_datasets.object_detection._droneswarm._hf_extract", lambda **kwargs: None)
        dataset = DroneSwarm(root=droneswarm_fake, verbose=True)
        dataset.path = tmp_path
        datapath = tmp_path / "Drone_Swarm_Dataset"
        (datapath / "images").mkdir(parents=True)
        (datapath / "labels").mkdir(parents=True)
        for i in range(3):
            _save_image(datapath / "images" / f"{i:05}.tiff")
            (datapath / "labels" / f"{i:05}.txt").write_text("")

        dataset._load_hf_data()
        assert "Downloading files from huggingface." in capsys.readouterr().out

        dataset._load_hf_data()
        assert "Data already downloaded, skipping download." in capsys.readouterr().out

    def test_droneswarm_download(self, tmp_path, monkeypatch, capsys):
        def fake_extract(url, filename, md5, checksum, kaggle, local_dir, root, download, verbose):
            nested = local_dir / "Drone_Swarm_Dataset"
            (nested / "images").mkdir(parents=True)
            (nested / "labels").mkdir(parents=True)
            _save_image(nested / "images" / "00000.png", size=(64, 48))
            (nested / "labels" / "00000.txt").write_text("0 0.5 0.5 0.4 0.4\n")
            (nested / "classes.txt").write_text("drone\n")

        monkeypatch.setattr("maite_datasets.object_detection._droneswarm._ensure_exists", fake_extract)
        dataset = DroneSwarm(root=tmp_path, download=True, verbose=True)
        assert len(dataset) == 1
        assert "Downloading files from kaggle." in capsys.readouterr().out

    def test_droneswarm_skips_existing_download(self, droneswarm_fake, monkeypatch, capsys):
        def fail(**kwargs):
            raise AssertionError("should not re-download")

        monkeypatch.setattr("maite_datasets.object_detection._droneswarm._hf_extract", fail)
        DroneSwarm(root=droneswarm_fake, download=True, verbose=True)
        assert "Data already downloaded, skipping download." in capsys.readouterr().out


@pytest.mark.optional
class TestSkySeaLand:
    def test_skysealand_local(self, skysealand_fake):
        dataset = SkySeaLand(root=skysealand_fake)
        assert len(dataset) == 1
        assert dataset.metadata["image_set"] == "train"
        assert dataset.metadata["index2label"] == {0: "airplane"}
        img, target, _ = dataset[0]
        assert img.shape == (3, 48, 64)
        assert target.boxes.shape == (1, 4)

    def test_skysealand_download(self, tmp_path, monkeypatch, capsys):
        def fake_extract(url, filename, md5, checksum, kaggle, local_dir, root, download, verbose):
            nested = local_dir / "train"
            (nested / "images").mkdir(parents=True)
            (nested / "labels").mkdir(parents=True)
            _save_image(nested / "images" / "00000.jpg", size=(64, 48))
            (nested / "labels" / "00000.txt").write_text("0 0.5 0.5 0.4 0.4\n")
            config = {
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 4,
                "names": ["airplane", "boat", "car", "ship"],
            }
            with open(local_dir / "data.yaml", "w") as f:
                yaml.safe_dump(config, f)

        monkeypatch.setattr("maite_datasets.object_detection._skysealand._ensure_exists", fake_extract)
        dataset = SkySeaLand(root=tmp_path, download=True, verbose=True)
        assert len(dataset) == 1
        assert "Downloading files from kaggle." in capsys.readouterr().out

    def test_droneswarm_skips_existing_download(self, skysealand_fake, monkeypatch, capsys):
        def fail(*args):
            raise AssertionError("should not re-download")

        monkeypatch.setattr("maite_datasets.object_detection._skysealand._ensure_exists", fail)
        SkySeaLand(root=skysealand_fake, download=True, verbose=True)
        assert "Data already downloaded, skipping download." in capsys.readouterr().out


def test_multiobject_tracking_exports():
    import maite_datasets.multiobject_tracking as mot

    assert set(mot.__all__).issubset(dir(mot))
