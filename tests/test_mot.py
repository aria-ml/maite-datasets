"""Unit tests for multi-object tracking foundation tooling."""

import numpy as np
import pytest

from maite_datasets._base import MultiobjectTrackingTargetTuple, SingleFrameObjectTrackingTargetTuple
from maite_datasets._builder import _validate_mot_data, to_multiobject_tracking_dataset
from maite_datasets._validate import (
    ValidationMessages,
    _detect_target_type,
    _validate_datum_target_mot,
    _validate_datum_video,
    validate_dataset,
)
from maite_datasets._video import VideoFrameTuple, frames_to_stream


@pytest.fixture(scope="module")
def videos() -> list[list[np.ndarray]]:
    # 4 videos, i+1 frames each, raw (C, H, W) arrays.
    return [[np.random.random((3, 16, 16)) for _ in range(i + 1)] for i in range(4)]


@pytest.fixture(scope="module")
def labels() -> list[list[list[int]]]:
    # Per video, per frame, one detection per frame index.
    return [[[0 for _ in range(f + 1)] for f in range(i + 1)] for i in range(4)]


@pytest.fixture(scope="module")
def bboxes() -> list[list[list[tuple[float, float, float, float]]]]:
    return [[[(0.0, 0.0, 4.0, 4.0) for _ in range(f + 1)] for f in range(i + 1)] for i in range(4)]


@pytest.fixture(scope="module")
def track_ids() -> list[list[list[int]]]:
    return [[list(range(f + 1)) for f in range(i + 1)] for i in range(4)]


@pytest.fixture(scope="module")
def metadata() -> list[dict[str, int]]:
    return [{"foo": i} for i in range(4)]


@pytest.fixture(scope="module")
def classes() -> list[str]:
    return ["car"]


class TestVideoFrame:
    def test_frames_to_stream_synthesizes_timing(self):
        frames = frames_to_stream([np.zeros((3, 4, 4)) for _ in range(3)])
        assert len(frames) == 3
        assert [f.frame_index for f in frames] == [0, 1, 2]
        assert [f.pts for f in frames] == [0, 1, 2]
        assert [f.time_s for f in frames] == [0.0, 1.0, 2.0]
        assert frames[0].pixels.shape == (3, 4, 4)

    def test_video_frame_tuple_attribute_access(self):
        frame = VideoFrameTuple(pixels=np.zeros((3, 2, 2)), time_s=0.5, pts=1, frame_index=0)
        assert frame.pixels.shape == (3, 2, 2)
        assert frame.time_s == 0.5
        assert frame.pts == 1
        assert frame.frame_index == 0


class TestMOTValidateData:
    def test_empty_videos_raises(self):
        with pytest.raises(ValueError, match="non-empty sequence"):
            _validate_mot_data([], [], [], [], None)

    def test_mismatched_top_level_length(self, videos, labels, bboxes, track_ids):
        with pytest.raises(ValueError, match="Number of labels"):
            _validate_mot_data(videos, labels[:-1], bboxes, track_ids, None)

    def test_mismatched_metadata_length(self, videos, labels, bboxes, track_ids):
        with pytest.raises(ValueError, match="Number of metadata"):
            _validate_mot_data(videos, labels, bboxes, track_ids, [{"a": 1}])

    def test_mismatched_frame_count(self, videos, labels, bboxes, track_ids):
        bad_labels = [v[:-1] if i == 1 and len(v) > 1 else v for i, v in enumerate(labels)]
        with pytest.raises(ValueError, match="one entry per frame"):
            _validate_mot_data(videos, bad_labels, bboxes, track_ids, None)

    def test_mismatched_detection_count(self, videos, labels, bboxes, track_ids):
        bad_track_ids = [[[*frame, 99] for frame in video] for video in track_ids]
        with pytest.raises(ValueError, match="equal length"):
            _validate_mot_data(videos, labels, bboxes, bad_track_ids, None)


class TestMOTValidateDatum:
    @pytest.fixture
    def good_target(self):
        frame = SingleFrameObjectTrackingTargetTuple(
            boxes=np.zeros((2, 4), np.float32),
            labels=np.array([0, 1]),
            scores=np.ones(2, np.float32),
            track_ids=np.array([0, 1]),
        )
        return MultiobjectTrackingTargetTuple(frame_tracks=[frame])

    def test_detect_target_type_mot(self, good_target):
        assert _detect_target_type(good_target) == "mot"

    def test_validate_video_ok(self):
        assert _validate_datum_video(frames_to_stream([np.zeros((3, 4, 4))])) == []

    def test_validate_video_single_array_rejected(self):
        assert ValidationMessages.DATUM_VIDEO_TYPE in _validate_datum_video(np.zeros((3, 4, 4)))

    def test_validate_video_bad_pixels(self):
        assert ValidationMessages.DATUM_VIDEO_FORMAT in _validate_datum_video([np.zeros((4, 4))])

    def test_validate_video_empty_ok(self):
        assert _validate_datum_video([]) == []

    def test_validate_target_mot_ok(self, good_target):
        assert _validate_datum_target_mot(good_target) == []

    def test_validate_target_mot_no_frame_tracks(self):
        assert ValidationMessages.DATUM_TARGET_MOT_TYPE in _validate_datum_target_mot(object())

    def test_validate_target_mot_bad_boxes(self):
        frame = SingleFrameObjectTrackingTargetTuple(
            np.zeros((1, 3), np.float32), np.array([0]), np.ones(1), np.array([0])
        )
        target = MultiobjectTrackingTargetTuple(frame_tracks=[frame])
        assert ValidationMessages.DATUM_TARGET_MOT_BOXES_TYPE in _validate_datum_target_mot(target)

    def test_validate_target_mot_bad_track_ids(self):
        frame = SingleFrameObjectTrackingTargetTuple(
            np.zeros((1, 4), np.float32), np.array([0]), np.ones(1), np.array([0.5])
        )
        target = MultiobjectTrackingTargetTuple(frame_tracks=[frame])
        assert ValidationMessages.DATUM_TARGET_MOT_TRACK_IDS_TYPE in _validate_datum_target_mot(target)


class TestMOTBuilder:
    def test_build_and_index(self, videos, labels, bboxes, track_ids, classes):
        ds = to_multiobject_tracking_dataset(videos, labels, bboxes, track_ids, None, classes)
        assert len(ds) == 4
        stream, target, md = ds[1]
        assert len(list(stream)) == 2
        assert len(target.frame_tracks) == 2
        assert target.frame_tracks[1].boxes.shape == (2, 4)
        assert list(target.frame_tracks[1].track_ids) == [0, 1]
        assert "id" in md

    def test_build_no_classes(self, videos, labels, bboxes, track_ids):
        ds = to_multiobject_tracking_dataset(videos, labels, bboxes, track_ids, None, None)
        assert len(ds) == 4

    def test_build_with_metadata(self, videos, labels, bboxes, track_ids, metadata, classes):
        ds = to_multiobject_tracking_dataset(videos, labels, bboxes, track_ids, metadata, classes)
        assert ds[2][2]["foo"] == 2

    def test_build_with_name(self, videos, labels, bboxes, track_ids, classes):
        name = "Test_MOT_Dataset"
        ds = to_multiobject_tracking_dataset(videos, labels, bboxes, track_ids, None, classes, name)
        assert name in ds.__class__.__name__

    def test_build_passthrough_video_frames(self, labels, bboxes, track_ids, classes):
        videos = [frames_to_stream([np.zeros((3, 8, 8)) for _ in range(i + 1)]) for i in range(4)]
        ds = to_multiobject_tracking_dataset(videos, labels, bboxes, track_ids, None, classes)
        stream, _, _ = ds[0]
        assert all(hasattr(f, "pixels") for f in stream)

    def test_built_dataset_validates(self, videos, labels, bboxes, track_ids, classes):
        ds = to_multiobject_tracking_dataset(videos, labels, bboxes, track_ids, None, classes)
        validate_dataset(ds, "mot")
        validate_dataset(ds, "auto")


class TestPyAVVideoStream:
    @pytest.fixture
    def clip(self, tmp_path):
        av = pytest.importorskip("av")
        path = str(tmp_path / "clip.mp4")
        with av.open(path, mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv420p"
            for i in range(5):
                frame = av.VideoFrame.from_ndarray(np.full((16, 32, 3), i * 40, dtype=np.uint8), format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
        return path

    def test_lazy_stream_decodes_chw_and_is_reiterable(self, clip):
        from maite_datasets._video import PyAVVideoStream

        stream = PyAVVideoStream(clip)
        frames = list(stream)
        assert len(frames) == 5
        assert frames[0].pixels.shape == (3, 16, 32)
        assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]
        assert len(list(stream)) == 5  # re-iterating decodes from the start again

    def test_decode_video_returns_frames_and_metadata(self, clip):
        from maite_datasets._video import decode_video

        frames, metadata = decode_video(clip)
        assert len(frames) == 5
        assert {"height", "width", "time_base", "size"} <= set(metadata)
        assert metadata["height"] == 16 and metadata["width"] == 32
        assert metadata["size"] > 0
