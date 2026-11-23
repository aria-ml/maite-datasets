"""Tests for the recently added dataset loaders (M3FD, DroneVehicle, DroneSwarm, SkySeaLand, Military*)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PIL import Image, UnidentifiedImageError

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


def _polygon_object(name: str, x0: int, y0: int, x1: int, y1: int) -> str:
    return (
        f"<object><name>{name}</name><polygon><x1>{x0}</x1><y1>{y0}</y1><x2>{x1}</x2><y2>{y0}</y2>"
        f"<x3>{x1}</x3><y3>{y1}</y3><x4>{x0}</x4><y4>{y1}</y4></polygon></object>"
    )


# Every name the published train infrared labels use, in this order, each on a box of its own extent.
DRONEVEHICLE_TRAIN_NAMES = ["car", "feright car", "feright_car", "feright", "*"]
DRONEVEHICLE_TRAIN_IR_ANNOTATION = (
    "<annotation><filename>ir_00001.jpg</filename>"
    "<size><width>840</width><height>712</height><depth>1</depth></size>"
    + "".join(_polygon_object(name, i, i, i + 10, i + 20) for i, name in enumerate(DRONEVEHICLE_TRAIN_NAMES))
    + "</annotation>"
)

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
            ir_annotation = DRONEVEHICLE_TRAIN_IR_ANNOTATION if split == "train" else DRONEVEHICLE_IR_ANNOTATION
            (ir_dir / f"{i:05}.xml").write_text(ir_annotation)
            (rgb_dir / f"{i:05}.xml").write_text(DRONEVEHICLE_RGB_ANNOTATION)
    yield temp


@pytest.fixture
def dronevehicle_empty_files(tmp_path):
    """A train split of 4 datums, with the published archive's two read defects.

    Datum 1's RGB file and datum 2's IR file are 0 bytes, and datum 3's RGB file is a JPEG cut short, which PIL
    reports as "image file is truncated".
    """
    base = tmp_path / "dronevehicle" / "train"
    for suffix in ("img", "imgr", "label", "labelr"):
        (base / f"train{suffix}").mkdir(parents=True)
    for i in range(4):
        _save_image(base / "trainimg" / f"{i:05}.jpg")
        _save_image(base / "trainimgr" / f"{i:05}.jpg")
        (base / "trainlabelr" / f"{i:05}.xml").write_text(DRONEVEHICLE_IR_ANNOTATION)
        (base / "trainlabel" / f"{i:05}.xml").write_text(DRONEVEHICLE_RGB_ANNOTATION)
    (base / "trainimg" / "00001.jpg").write_bytes(b"")
    (base / "trainimgr" / "00002.jpg").write_bytes(b"")
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(noise).save(base / "trainimg" / "00003.jpg")
    data = (base / "trainimg" / "00003.jpg").read_bytes()
    (base / "trainimg" / "00003.jpg").write_bytes(data[: len(data) * 9 // 10])
    return tmp_path


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
        assert datum_meta["id"] == "00000"
        assert "image_id" not in datum_meta

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


@pytest.mark.optional
class TestDroneVehicle:
    def test_dronevehicle_train(self, dronevehicle_fake):
        dataset = DroneVehicle(root=dronevehicle_fake)
        assert len(dataset) == 3
        img, target, datum_meta = dataset[0]
        assert img.shape == (4, 10, 10)
        assert np.array_equal(target.labels, [0, 6, 7, 5, 4])
        # Rotated quadrilateral reduced to its axis-aligned extent
        assert np.array_equal(target.boxes[0], [0, 0, 10, 20])
        assert datum_meta["id"] == "train_00000.jpg"
        assert datum_meta["infrared_filename"] == "ir_00001.jpg"
        assert datum_meta["rgb_filename"] == "rgb_00001.jpg"
        # IR depth plus RGB depth
        assert datum_meta["image_depth"] == 4
        assert "dropped_points" not in datum_meta

    def test_class_list_is_the_published_names_as_written(self, dronevehicle_fake):
        published = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "*", 5: "feright", 6: "feright car", 7: "feright_car"}
        dataset = DroneVehicle(root=dronevehicle_fake)
        assert DroneVehicle.index2label == published
        assert dataset.label2index == {name: index for index, name in published.items()}
        assert dataset.metadata["index2label"] == published
        assert dataset[0][1].scores.shape == (5, 8)

    def test_each_box_carries_the_index_of_its_name_as_written(self, dronevehicle_fake):
        dataset = DroneVehicle(root=dronevehicle_fake)
        _, target, _ = dataset[0]
        assert target.labels.tolist() == [dataset.label2index[name] for name in DRONEVEHICLE_TRAIN_NAMES]
        assert target.labels.tolist() == [0, 6, 7, 5, 4]

    def test_every_split_shares_one_class_list(self, dronevehicle_fake):
        val_first = DroneVehicle(root=dronevehicle_fake, image_set="val")
        train_after = DroneVehicle(root=dronevehicle_fake)
        train_first = DroneVehicle(root=dronevehicle_fake)
        val_after = DroneVehicle(root=dronevehicle_fake, image_set="val")
        for val, train in ((val_first, train_after), (val_after, train_first)):
            assert val.index2label == train.index2label
            assert val.label2index == train.label2index
            assert val[0][1].labels.tolist() == [6]

    def test_unknown_name_raises_naming_the_file_and_the_name(self, dronevehicle_fake, tmp_path):
        dataset = DroneVehicle(root=dronevehicle_fake)
        annotation = tmp_path / "trainlabelr" / "00000.xml"
        annotation.parent.mkdir()
        (tmp_path / "trainlabel").mkdir()
        (tmp_path / "trainlabel" / "00000.xml").write_text(DRONEVEHICLE_RGB_ANNOTATION)
        annotation.write_text(DRONEVEHICLE_IR_ANNOTATION.replace("feright car", "freight car"))
        with pytest.raises(ValueError, match="'freight car'") as error:
            dataset._read_annotations(str(annotation))
        assert str(annotation) in str(error.value)

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("lazy", [False, True])
    def test_empty_or_truncated_image_file_reads_as_zero_size(self, dronevehicle_empty_files, lazy):
        dataset = DroneVehicle(root=dronevehicle_empty_files, lazy=lazy)
        assert len(dataset) == 4
        full_img, full_target, _ = dataset[0]
        assert np.asarray(full_img).shape == (4, 10, 10)
        for index in (1, 2, 3):
            img, target, _ = dataset[index]
            img = np.asarray(img)
            assert img.shape == (4, 0, 0)
            assert img.dtype == np.uint8
            assert np.array_equal(target.boxes, full_target.boxes)
            assert np.array_equal(target.labels, full_target.labels)

    def test_empty_rgb_file_with_a_missing_ir_file_raises(self, dronevehicle_empty_files):
        (dronevehicle_empty_files / "dronevehicle" / "train" / "trainimgr" / "00001.jpg").unlink()
        dataset = DroneVehicle(root=dronevehicle_empty_files)
        with pytest.raises(FileNotFoundError):
            dataset[1]

    def test_other_read_errors_still_raise(self, dronevehicle_empty_files):
        (dronevehicle_empty_files / "dronevehicle" / "train" / "trainimg" / "00000.jpg").write_bytes(b"not a jpeg")
        dataset = DroneVehicle(root=dronevehicle_empty_files)
        with pytest.raises(UnidentifiedImageError):
            dataset[0]

    def test_empty_image_goes_through_the_transforms(self, dronevehicle_empty_files):
        seen = []

        def record(image):
            seen.append(image.shape)
            return image

        dataset = DroneVehicle(root=dronevehicle_empty_files, transforms=record)
        dataset[0]
        dataset[1]
        assert seen == [(4, 10, 10), (4, 0, 0)]

    def test_dronevehicle_base(self, dronevehicle_fake):
        dataset = DroneVehicle(root=dronevehicle_fake, image_set="base")
        assert len(dataset) == 6
        assert len(dataset._datum_metadata["id"]) == 6

    def test_dronevehicle_root_containing_img_segment(self, tmp_path):
        """The IR/RGB siblings are derived per path component, not by string replacement.

        A root whose own path contains ``img``/``labelr`` used to be rewritten too,
        pointing the paired lookups at directories that do not exist.
        """
        root = tmp_path / "img" / "labelr"
        base = root / "dronevehicle" / "train"
        for suffix in ("img", "imgr", "label", "labelr"):
            (base / f"train{suffix}").mkdir(parents=True)
        _save_image(base / "trainimg" / "00000.jpg")
        _save_image(base / "trainimgr" / "00000.jpg")
        (base / "trainlabelr" / "00000.xml").write_text(DRONEVEHICLE_IR_ANNOTATION)
        (base / "trainlabel" / "00000.xml").write_text(DRONEVEHICLE_RGB_ANNOTATION)

        dataset = DroneVehicle(root=root)
        img, _, datum_meta = dataset[0]
        assert img.shape == (4, 10, 10)
        assert datum_meta["rgb_filename"] == "rgb_00001.jpg"

    @staticmethod
    def _annotation_file(tmp_path, text):
        annotation = tmp_path / "trainlabelr" / "00000.xml"
        annotation.parent.mkdir()
        (tmp_path / "trainlabel").mkdir()
        (tmp_path / "trainlabel" / "00000.xml").write_text(DRONEVEHICLE_RGB_ANNOTATION)
        annotation.write_text(text)
        return annotation

    def test_every_annotation_is_kept(self, dronevehicle_fake, tmp_path):
        """A bndbox is read as given, a point is a zero-area box, and a polygon whose corners coincide keeps them."""
        dataset = DroneVehicle(root=dronevehicle_fake)
        extra = """
    <object>
        <name>car</name>
        <bndbox><xmin>5</xmin><ymin>6</ymin><xmax>15</xmax><ymax>16</ymax></bndbox>
    </object>
    <object>
        <name>car</name>
        <point><x>7</x><y>8</y></point>
    </object>
    <object>
        <name>car</name>
        <polygon><x1>9</x1><y1>9</y1><x2>9</x2><y2>9</y2><x3>9</x3><y3>9</y3><x4>9</x4><y4>9</y4></polygon>
    </object>
</annotation>"""
        annotation = self._annotation_file(tmp_path, DRONEVEHICLE_IR_ANNOTATION.replace("</annotation>", extra))
        boxes, labels, meta = dataset._read_annotations(str(annotation))
        assert boxes == [[10, 20, 40, 60], [5, 6, 15, 16], [7, 8, 7, 8], [9, 9, 9, 9]]
        assert labels == [6, 0, 0, 0]
        assert "dropped_points" not in meta

    def test_point_without_a_coordinate_raises_naming_the_file(self, dronevehicle_fake, tmp_path):
        dataset = DroneVehicle(root=dronevehicle_fake)
        extra = "<object><name>car</name><point><x>7</x></point></object></annotation>"
        annotation = self._annotation_file(tmp_path, DRONEVEHICLE_IR_ANNOTATION.replace("</annotation>", extra))
        with pytest.raises(ValueError, match="point") as error:
            dataset._read_annotations(str(annotation))
        assert str(annotation) in str(error.value)

    def test_object_with_no_geometry_raises_naming_the_file(self, dronevehicle_fake, tmp_path):
        dataset = DroneVehicle(root=dronevehicle_fake)
        extra = "<object><name>car</name></object></annotation>"
        annotation = self._annotation_file(tmp_path, DRONEVEHICLE_IR_ANNOTATION.replace("</annotation>", extra))
        with pytest.raises(ValueError) as error:
            dataset._read_annotations(str(annotation))
        assert str(annotation) in str(error.value)


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
        monkeypatch.setattr("maite_datasets._fileio._hf_extract", lambda **kw: None)

    def test_vehicles_train(self, military_vehicles_fake):
        dataset = MilitaryVehicles(root=military_vehicles_fake)
        assert len(dataset) == len(MilitaryVehicles.index2label)
        img, score, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert score.shape == (24,)
        assert score[0] == 1
        assert datum_meta["id"] == "2S19_MSTA_00000"

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

    def test_droneswarm_download(self, tmp_path, monkeypatch, capsys):
        def fake_extract(part, local_dir, root, download, verbose):
            nested = local_dir / "Drone_Swarm_Dataset"
            (nested / "images").mkdir(parents=True)
            (nested / "labels").mkdir(parents=True)
            _save_image(nested / "images" / "00000.png", size=(64, 48))
            (nested / "labels" / "00000.txt").write_text("0 0.5 0.5 0.4 0.4\n")
            (nested / "classes.txt").write_text("drone\n")

        monkeypatch.setattr("maite_datasets.object_detection._droneswarm._download_part", fake_extract)
        dataset = DroneSwarm(root=tmp_path, download=True, verbose=True)
        assert len(dataset) == 1
        assert "Downloading files from kaggle." in capsys.readouterr().out

    def test_droneswarm_skips_existing_download(self, droneswarm_fake, monkeypatch, capsys):
        def fail(*args, **kwargs):
            raise AssertionError("should not re-download")

        monkeypatch.setattr("maite_datasets.object_detection._droneswarm._download_part", fail)
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
        def fake_extract(part, local_dir, root, download, verbose):
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

        monkeypatch.setattr("maite_datasets.object_detection._skysealand._download_part", fake_extract)
        dataset = SkySeaLand(root=tmp_path, download=True, verbose=True)
        assert len(dataset) == 1
        assert "Downloading files from kaggle." in capsys.readouterr().out

    def test_droneswarm_skips_existing_download(self, skysealand_fake, monkeypatch, capsys):
        def fail(*args):
            raise AssertionError("should not re-download")

        monkeypatch.setattr("maite_datasets.object_detection._skysealand._download_part", fail)
        SkySeaLand(root=skysealand_fake, download=True, verbose=True)
        assert "Data already downloaded, skipping download." in capsys.readouterr().out


def test_multiobject_tracking_exports():
    import maite_datasets.multiobject_tracking as mot

    assert set(mot.__all__).issubset(dir(mot))
