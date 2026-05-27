"""Lazy-loading regression tests for COCO/YOLO/HuggingFace datasets."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from maite_datasets._lazy import LazyArray
from maite_datasets.object_detection._coco import COCODataset, COCODatasetReader
from maite_datasets.object_detection._yolo import YOLODataset, YOLODatasetReader


@pytest.fixture
def coco_dataset_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    coco_data = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 64, "height": 48}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 15], "area": 300},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    Image.new("RGB", (64, 48), color="red").save(images_dir / "image1.jpg")
    (tmp_path / "annotations.json").write_text(json.dumps(coco_data))
    (tmp_path / "classes.txt").write_text("person\n")
    return tmp_path


@pytest.fixture
def yolo_dataset_dir(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (64, 48), color="blue").save(images_dir / "image1.jpg")
    (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (tmp_path / "classes.txt").write_text("person\n")
    return tmp_path


class TestCOCOLazy:
    def test_eager_default(self, coco_dataset_dir):
        ds = COCODatasetReader(coco_dataset_dir).create_dataset()
        img, _, _ = ds[0]
        assert isinstance(img, np.ndarray)

    def test_lazy_returns_lazyarray(self, coco_dataset_dir):
        ds = COCODataset(COCODatasetReader(coco_dataset_dir), lazy=True)
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)

    def test_lazy_shape_no_decode(self, coco_dataset_dir, monkeypatch):
        from maite_datasets.object_detection import _coco

        decoded = {"n": 0}
        real_loader = _coco.pil_rgb_chw_load

        def spy_loader(path):
            decoded["n"] += 1
            return real_loader(path)

        monkeypatch.setattr(_coco, "pil_rgb_chw_load", spy_loader)
        ds = COCODataset(COCODatasetReader(coco_dataset_dir), lazy=True)
        img, target, _ = ds[0]
        assert target.boxes.shape == (1, 4)
        # img.shape uses cheap header read, not full decode.
        assert img.shape == (3, 48, 64)
        assert decoded["n"] == 0, "lazy mode must not invoke the pixel loader during __getitem__/shape"

    def test_lazy_materialize_matches_eager(self, coco_dataset_dir):
        eager = COCODataset(COCODatasetReader(coco_dataset_dir))
        eager_img, _, _ = eager[0]
        lazy_ds = COCODataset(COCODatasetReader(coco_dataset_dir), lazy=True)
        lazy_img, _, _ = lazy_ds[0]
        np.testing.assert_array_equal(np.asarray(lazy_img), eager_img)


class TestYOLOLazy:
    def test_eager_default(self, yolo_dataset_dir):
        ds = YOLODatasetReader(yolo_dataset_dir).create_dataset()
        img, _, _ = ds[0]
        assert isinstance(img, np.ndarray)

    def test_lazy_returns_lazyarray(self, yolo_dataset_dir):
        ds = YOLODataset(YOLODatasetReader(yolo_dataset_dir), lazy=True)
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)

    def test_lazy_bbox_scaling_uses_lazy_shape(self, yolo_dataset_dir):
        """YOLO normalized coords are scaled by image dimensions — must work without full decode."""
        ds = YOLODataset(YOLODatasetReader(yolo_dataset_dir), lazy=True)
        eager = YOLODataset(YOLODatasetReader(yolo_dataset_dir))
        _, lazy_tgt, _ = ds[0]
        _, eager_tgt, _ = eager[0]
        np.testing.assert_array_equal(lazy_tgt.boxes, eager_tgt.boxes)

    def test_lazy_materialize_matches_eager(self, yolo_dataset_dir):
        eager = YOLODataset(YOLODatasetReader(yolo_dataset_dir))
        eager_img, _, _ = eager[0]
        lazy_ds = YOLODataset(YOLODatasetReader(yolo_dataset_dir), lazy=True)
        lazy_img, _, _ = lazy_ds[0]
        np.testing.assert_array_equal(np.asarray(lazy_img), eager_img)


class TestHFLazy:
    """HF dataset uses an arrow-backed source; lazy defers the per-item access."""

    def _make_hf_ic(self, lazy: bool):
        from maite_datasets.adapters._huggingface import HFImageClassificationDataset
        from maite_datasets.protocols import HFClassLabel, HFImage, HFValue

        access_count = {"n": 0}

        class FakeFeatures(dict):
            def __init__(self):
                super().__init__(
                    {
                        "image": type("F", (HFImage,), {"_type": "Image", "decode": True})(),
                        "label": type(
                            "L",
                            (HFClassLabel,),
                            {"_type": "ClassLabel", "names": ["a", "b"], "num_classes": 2},
                        )(),
                        "id": type("V", (HFValue,), {"_type": "Value", "pa_type": None, "dtype": "int64"})(),
                    }
                )

        class FakeSource:
            features = FakeFeatures()
            info = type("I", (), {"dataset_name": "fake", "__dict__": {"dataset_name": "fake"}})()

            def __init__(self, image):
                self._image = image

            def __getitem__(self, key):
                access_count["n"] += 1
                return {"image": self._image, "label": 0, "id": key if isinstance(key, int) else 0}

            def __len__(self):
                return 4

        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        source = FakeSource(arr)
        with patch("maite_datasets.adapters._huggingface.isinstance", return_value=True):
            ds = HFImageClassificationDataset(source, "image", "label", lazy=lazy)
        return ds, access_count

    def test_lazy_kwarg_accepted(self):
        ds, _ = self._make_hf_ic(lazy=True)
        assert ds.lazy is True

    def test_lazy_getitem_returns_lazyarray(self):
        ds, _ = self._make_hf_ic(lazy=True)
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)

    def test_lazy_getitem_defers_image_access(self):
        ds, access = self._make_hf_ic(lazy=True)
        access["n"] = 0
        img, _label, _md = ds[0]
        # source[index] is accessed for label/metadata, but image fetch must be deferred.
        accesses_before = access["n"]
        np.asarray(img)  # force materialize
        assert access["n"] > accesses_before

    def test_lazy_materialize_matches_eager(self):
        lazy_ds, _ = self._make_hf_ic(lazy=True)
        eager_ds, _ = self._make_hf_ic(lazy=False)
        lazy_img, _, _ = lazy_ds[0]
        eager_img, _, _ = eager_ds[0]
        np.testing.assert_array_equal(np.asarray(lazy_img), eager_img)

    def _make_hf_od(self, lazy: bool):
        """Mirror tests/test_huggingface.py setup for HFObjectDetectionDataset."""
        from unittest.mock import MagicMock

        from maite_datasets.adapters._huggingface import HFObjectDetectionDataset

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.info.dataset_name = "test_od"

        mock_bbox_value = MagicMock(spec=["dtype"])
        type(mock_bbox_value).__name__ = "HFValue"
        mock_bbox_value.dtype = "float32"

        mock_bbox_feature = MagicMock(spec=["length", "feature"])
        type(mock_bbox_feature).__name__ = "HFList"
        mock_bbox_feature.length = 4
        mock_bbox_feature.feature = mock_bbox_value

        mock_label_feature = MagicMock(spec=["names", "num_classes"])
        type(mock_label_feature).__name__ = "HFClassLabel"
        mock_label_feature.names = ["person", "car"]
        mock_label_feature.num_classes = 2

        mock_label_container = MagicMock(spec=["feature"])
        type(mock_label_container).__name__ = "HFList"
        mock_label_container.feature = mock_label_feature

        mock_objects_feature = MagicMock(spec=["feature"])
        type(mock_objects_feature).__name__ = "HFList"
        mock_objects_feature.feature = {"bbox": mock_bbox_feature, "category_id": mock_label_container}

        mock_image_feature = MagicMock(spec=["decode"])
        type(mock_image_feature).__name__ = "HFImage"
        mock_dataset.features = {"image": mock_image_feature, "objects": mock_objects_feature}

        mock_image = np.random.rand(8, 8, 3).astype(np.uint8)
        mock_dataset.__getitem__ = MagicMock(
            return_value={
                "image": mock_image,
                "objects": {"bbox": [[0, 0, 4, 4]], "category_id": [0]},
            }
        )

        def isinstance_side_effect(obj, types):
            if hasattr(obj, "__class__"):
                class_name = obj.__class__.__name__
                type_names = [getattr(t, "__name__", str(t)) for t in (types if isinstance(types, tuple) else [types])]
                return class_name in type_names
            return False

        with patch("maite_datasets.adapters._huggingface.isinstance", side_effect=isinstance_side_effect):
            return HFObjectDetectionDataset(mock_dataset, "image", "objects", "bbox", "category_id", "xyxy", lazy=lazy)

    def test_od_lazy_kwarg_accepted(self):
        ds = self._make_hf_od(lazy=True)
        assert ds.lazy is True

    def test_od_lazy_getitem_works(self):
        """HF OD must accept lazy=True and produce correct boxes even when shape access materializes."""
        ds = self._make_hf_od(lazy=True)
        img, target, _ = ds[0]
        # Boxes come back regardless of whether image was materialized for shape access.
        assert len(target.boxes) == 1
        # Image is either a LazyArray (if shape access skipped materialization) or ndarray.
        assert isinstance(img, (LazyArray, np.ndarray))
