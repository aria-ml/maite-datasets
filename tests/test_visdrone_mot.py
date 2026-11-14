"""Tests for the VisDroneMOT dataset, read from a miniature of the Voxel51 hub export."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import maite_datasets._lazy as lazy
from maite_datasets import validate_dataset
from maite_datasets._lazy import LazyArray
from maite_datasets.multiobject_tracking import VisDroneMOT

# scene id -> (width, height, frame file names). The names mimic the hub, where each scene
# numbers its frames from 1 and a name another scene already took gets a "-2" suffix, so a
# frame's file can only be found through its recorded `filepath`, never by its number.
SCENES = {
    "uav_a": (32, 16, ["0000001.jpg", "0000002.jpg", "0000003.jpg"]),
    "uav_b": (40, 20, ["0000001-2.jpg", "0000002-2.jpg"]),
}


def _detection(label: str, box: list[float], index: int, confidence: float = 1.0) -> dict:
    return {"label": label, "bounding_box": box, "index": index, "confidence": confidence}


def _sample(scene: str, number: int, name: str, detections: list[dict]) -> dict:
    width, height, _ = SCENES[scene]
    return {
        "filepath": f"data/{name}",
        "scene_id": scene,
        "frame_number": number,
        "scene_type": {"label": "road"},
        "time_of_day": {"label": "dusk" if scene == "uav_a" else "night"},
        "pedestrian_density": {"label": "low"},
        "metadata": {"width": width, "height": height},
        "detections": detections,
    }


def write_tree(directory: Path, drop: str | None = None) -> None:
    """Write the miniature export. Frame `n` of every scene is filled with the value 10 * n."""
    (directory / "data").mkdir(parents=True, exist_ok=True)
    samples = []
    for scene, (width, height, names) in SCENES.items():
        for number, name in enumerate(names, start=1):
            detections = [
                # Relative xywh, as FiftyOne stores it: on 32 x 16 this is pixel (8, 8)-(24, 12).
                _detection("car", [0.25, 0.5, 0.5, 0.25], index=7),
                _detection("pedestrian", [0.0, 0.0, 0.25, 0.5], index=3),
                # An ignored region carries confidence 0 and is not an object.
                _detection("ignored_region", [0.5, 0.5, 0.5, 0.5], index=0, confidence=0.0),
            ]
            if number == 2:
                detections = []
            samples.append(_sample(scene, number, name, detections))
            if name != drop:
                frame = np.full((height, width, 3), 10 * number, dtype=np.uint8)
                Image.fromarray(frame).save(directory / "data" / name)
    # Out of order on purpose: the loader has to impose the scene and frame order itself.
    samples.reverse()
    (directory / "samples.json").write_text(json.dumps({"samples": samples}))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_tree(tmp_path / "visdronemot")
    return tmp_path


class TestVisDroneMOT:
    def test_scenes_are_sorted_and_each_is_a_datum(self, root):
        dataset = VisDroneMOT(root)
        assert len(dataset) == 2
        assert [dataset[i][2]["id"] for i in range(2)] == ["uav_a", "uav_b"]
        assert dataset.metadata["id"] == "VisDroneMOT_val"
        assert dataset.metadata["index2label"][3] == "car"

    def test_frames_follow_frame_number_and_resolve_through_filepath(self, root):
        stream, target, _ = VisDroneMOT(root)[1]
        frames = list(stream)
        assert len(stream) == len(frames) == len(target.frame_tracks) == 2
        assert [(f.frame_index, f.pts) for f in frames] == [(0, 0), (1, 1)]
        # nominal timing: no frame rate is published, so time_s counts frames
        assert [f.time_s for f in frames] == [0.0, 1.0]
        assert [int(np.asarray(f.pixels)[0, 0, 0]) for f in frames] == [10, 20]

    def test_pixels_are_chw_uint8(self, root):
        (frame,) = list(VisDroneMOT(root)[0][0])[:1]
        pixels = np.asarray(frame.pixels)
        assert pixels.shape == (3, 16, 32)
        assert pixels.dtype == np.uint8

    def test_stream_can_be_walked_again(self, root):
        stream = VisDroneMOT(root)[0][0]
        assert len(list(stream)) == len(list(stream)) == 3

    def test_walking_a_stream_decodes_no_pixels(self, root, monkeypatch):
        decoded = []
        real = lazy.pil_rgb_chw_load
        monkeypatch.setattr(lazy, "pil_rgb_chw_load", lambda path: decoded.append(path) or real(path))
        frames = list(VisDroneMOT(root)[0][0])
        assert [f.pixels.shape for f in frames] == [(3, 16, 32)] * 3
        assert all(isinstance(f.pixels, LazyArray) for f in frames)
        assert decoded == []
        np.asarray(frames[0].pixels)
        assert len(decoded) == 1

    def test_targets_are_pixel_xyxy_with_labels_and_track_ids(self, root):
        _, target, _ = VisDroneMOT(root)[0]
        first = target.frame_tracks[0]
        np.testing.assert_array_equal(first.boxes, [[8, 8, 24, 12], [0, 0, 8, 8]])
        assert first.boxes.dtype == np.float32
        np.testing.assert_array_equal(first.labels, [3, 0])
        np.testing.assert_array_equal(first.track_ids, [7, 3])
        np.testing.assert_array_equal(first.scores, [1.0, 1.0])

    def test_ignored_regions_are_dropped_and_empty_frames_keep_their_shape(self, root):
        _, target, _ = VisDroneMOT(root)[0]
        empty = target.frame_tracks[1]
        assert empty.boxes.shape == (0, 4)
        assert len(empty.labels) == len(empty.scores) == len(empty.track_ids) == 0

    def test_a_box_is_dropped_by_its_confidence_not_by_its_label(self, root):
        path = root / "visdronemot" / "samples.json"
        data = json.loads(path.read_text())
        for sample in data["samples"]:
            for detection in sample["detections"]:
                if detection["label"] == "car":
                    detection["confidence"] = 0.0
        path.write_text(json.dumps(data))
        _, target, _ = VisDroneMOT(root)[0]
        np.testing.assert_array_equal(target.frame_tracks[0].labels, [0])

    def test_boxes_scale_with_each_scenes_own_size(self, root):
        _, target, _ = VisDroneMOT(root)[1]
        np.testing.assert_array_equal(target.frame_tracks[0].boxes[0], [10, 10, 30, 15])

    def test_datum_metadata(self, root):
        _, _, metadata = VisDroneMOT(root)[0]
        assert (metadata["width"], metadata["height"]) == (32, 16)
        assert metadata["time_base"] == Fraction(1, 1)
        files = (root / "visdronemot" / "data").glob("000000[123].jpg")
        assert metadata["size"] == sum(p.stat().st_size for p in files)
        assert (metadata["scene_type"], metadata["time_of_day"], metadata["pedestrian_density"]) == (
            "road",
            "dusk",
            "low",
        )

    def test_datum_metadata_is_not_shared_between_reads(self, root):
        dataset = VisDroneMOT(root)
        dataset[0][2]["id"] = "changed"
        assert dataset[0][2]["id"] == "uav_a"

    def test_validates_as_multiobject_tracking(self, root):
        validate_dataset(VisDroneMOT(root), "mot")

    def test_unknown_image_set_is_refused(self, root):
        with pytest.raises(ValueError, match="image_set"):
            VisDroneMOT(root, image_set="train")  # type: ignore[arg-type]

    def test_an_undeclared_label_is_refused_not_dropped(self, root):
        path = root / "visdronemot" / "samples.json"
        data = json.loads(path.read_text())
        data["samples"][0]["detections"] = [_detection("boat", [0.1, 0.1, 0.1, 0.1], index=1)]
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="boat"):
            VisDroneMOT(root)


class TestVisDroneMOTDownload:
    def test_missing_data_without_download_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VisDroneMOT(tmp_path)

    def test_download_fetches_only_what_the_loader_reads(self, tmp_path, monkeypatch):
        parts = []

        def fake_download(part, local_dir, root, download, verbose):
            parts.append(part)
            write_tree(local_dir)

        monkeypatch.setattr("maite_datasets.multiobject_tracking._visdrone._download_part", fake_download)
        assert len(VisDroneMOT(tmp_path, download=True)) == 2
        ((mirror,),) = [part.mirrors for part in parts]
        assert mirror.repo_id == "Voxel51/visdrone-mot"
        assert set(mirror.allow_patterns) == {"samples.json", "data/*.jpg"}

    def test_existing_data_is_not_downloaded_again(self, root, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("should not re-download")

        monkeypatch.setattr("maite_datasets.multiobject_tracking._visdrone._download_part", fail)
        VisDroneMOT(root, download=True)

    def test_a_missing_frame_means_an_incomplete_download_and_is_fetched_again(self, tmp_path, monkeypatch):
        write_tree(tmp_path / "visdronemot", drop="0000002-2.jpg")
        calls = []

        def fake_download(part, local_dir, root, download, verbose):
            calls.append(part)
            write_tree(local_dir)

        monkeypatch.setattr("maite_datasets.multiobject_tracking._visdrone._download_part", fake_download)
        assert len(VisDroneMOT(tmp_path, download=True)) == 2
        assert len(calls) == 1

    def test_a_missing_frame_without_download_raises(self, tmp_path):
        write_tree(tmp_path / "visdronemot", drop="0000002-2.jpg")
        with pytest.raises(FileNotFoundError):
            VisDroneMOT(tmp_path)
