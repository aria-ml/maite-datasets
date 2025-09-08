"""
Unit tests for maite_datasets.adapters._huggingface module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from maite_datasets.adapters._huggingface import (
    HFDatasetInfo,
    HFImageClassificationDataset,
    HFImageClassificationDatasetInfo,
    HFObjectDetectionDataset,
    HFObjectDetectionDatasetInfo,
    find_od_keys,
    from_huggingface,
    get_dataset_info,
    is_bbox,
    is_label,
)


class TestHFImageClassificationDataset:
    def setup_method(self):
        self.mock_dataset = Mock()
        self.mock_dataset.__len__ = Mock(return_value=2)
        self.mock_dataset.info.dataset_name = "test_dataset"

        # Mock features with proper class setup
        mock_image_feature = Mock(spec=["decode"])
        type(mock_image_feature).__name__ = "HFImage"

        mock_label_feature = Mock(spec=["num_classes", "names"])
        type(mock_label_feature).__name__ = "HFClassLabel"
        mock_label_feature.num_classes = 3
        mock_label_feature.names = ["cat", "dog", "bird"]

        mock_value_feature = Mock(spec=["dtype", "pa_type"])
        type(mock_value_feature).__name__ = "HFValue"

        self.mock_dataset.features = {
            "image": mock_image_feature,
            "label": mock_label_feature,
            "metadata_field": mock_value_feature,
        }

    @patch("maite_datasets.adapters._huggingface.isinstance")
    def test_init_success(self, mock_isinstance):
        def isinstance_side_effect(obj, types):
            if hasattr(obj, "__class__"):
                class_name = obj.__class__.__name__
                if "HFImage" in str(types) or "HFArray" in str(types):
                    return class_name in ["HFImage", "HFArray"]
                if "HFClassLabel" in str(types):
                    return class_name == "HFClassLabel"
                if "HFValue" in str(types):
                    return class_name == "HFValue"
            return False

        mock_isinstance.side_effect = isinstance_side_effect

        dataset = HFImageClassificationDataset(self.mock_dataset, "image", "label")

        assert dataset._image_key == "image"
        assert dataset._label_key == "label"
        assert dataset._num_classes == 3
        assert dataset.metadata["id"] == "test_dataset"
        assert "index2label" in dataset.metadata

    @patch("maite_datasets.adapters._huggingface.isinstance")
    def test_init_missing_image_key(self, mock_isinstance):
        mock_isinstance.return_value = False

        with pytest.raises(ValueError, match="Image key 'nonexistent' not found"):
            HFImageClassificationDataset(self.mock_dataset, "nonexistent", "label")

    @patch("maite_datasets.adapters._huggingface.isinstance")
    def test_getitem_success(self, mock_isinstance):
        def isinstance_side_effect(obj, types):
            if hasattr(obj, "__class__"):
                class_name = obj.__class__.__name__
                if "HFImage" in str(types) or "HFArray" in str(types):
                    return class_name in ["HFImage", "HFArray"]
                if "HFClassLabel" in str(types):
                    return class_name == "HFClassLabel"
                if "HFValue" in str(types):
                    return class_name == "HFValue"
            return False

        mock_isinstance.side_effect = isinstance_side_effect

        # Mock dataset item
        mock_image = np.random.rand(32, 32, 3).astype(np.uint8)
        self.mock_dataset.__getitem__ = Mock(
            return_value={"image": mock_image, "label": 1, "metadata_field": "test_value"}
        )

        dataset = HFImageClassificationDataset(self.mock_dataset, "image", "label")
        image, label, metadata = dataset[0]

        assert image.shape == (3, 32, 32)  # CHW format
        assert label.shape == (3,)
        assert np.sum(label) == 1.0  # One-hot encoded
        assert metadata["id"] == 0
        assert metadata["metadata_field"] == "test_value"

    def test_getitem_index_out_of_range(self):
        with patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance:
            mock_isinstance.return_value = True
            dataset = HFImageClassificationDataset(self.mock_dataset, "image", "label")

            with pytest.raises(IndexError):
                dataset[10]


class TestHFObjectDetectionDataset:
    def setup_method(self):
        self.mock_dataset = Mock()
        self.mock_dataset.__len__ = Mock(return_value=2)
        self.mock_dataset.info.dataset_name = "test_od"

        # Mock nested features for objects with proper class setup
        mock_bbox_value = Mock(spec=["dtype"])
        type(mock_bbox_value).__name__ = "HFValue"
        mock_bbox_value.dtype = "float32"

        mock_bbox_feature = Mock(spec=["length", "feature"])
        type(mock_bbox_feature).__name__ = "HFList"
        mock_bbox_feature.length = 4
        mock_bbox_feature.feature = mock_bbox_value

        # Create the actual label feature (HFClassLabel)
        mock_label_feature = Mock(spec=["names", "num_classes"])
        type(mock_label_feature).__name__ = "HFClassLabel"
        mock_label_feature.names = ["person", "car"]
        mock_label_feature.num_classes = 2

        # Create the label container that wraps the HFClassLabel
        mock_label_container = Mock(spec=["feature"])
        type(mock_label_container).__name__ = "HFList"
        mock_label_container.feature = mock_label_feature

        mock_objects_feature = Mock(spec=["feature"])
        type(mock_objects_feature).__name__ = "HFList"
        mock_objects_feature.feature = {"bbox": mock_bbox_feature, "category_id": mock_label_container}

        mock_image_feature = Mock(spec=["decode"])
        type(mock_image_feature).__name__ = "HFImage"

        self.mock_dataset.features = {"image": mock_image_feature, "objects": mock_objects_feature}

    @patch("maite_datasets.adapters._huggingface.isinstance")
    def test_init_success(self, mock_isinstance):
        def isinstance_side_effect(obj, types):
            if hasattr(obj, "__class__"):
                class_name = obj.__class__.__name__
                type_names = [getattr(t, "__name__", str(t)) for t in (types if isinstance(types, tuple) else [types])]
                return class_name in type_names
            return False

        mock_isinstance.side_effect = isinstance_side_effect

        dataset = HFObjectDetectionDataset(self.mock_dataset, "image", "objects", "bbox", "category_id")

        assert dataset._image_key == "image"
        assert dataset._objects_key == "objects"
        assert dataset._bbox_key == "bbox"
        assert dataset._label_key == "category_id"

    @patch("maite_datasets.adapters._huggingface.isinstance")
    def test_getitem_success(self, mock_isinstance):
        def isinstance_side_effect(obj, types):
            if hasattr(obj, "__class__"):
                class_name = obj.__class__.__name__
                type_names = [getattr(t, "__name__", str(t)) for t in (types if isinstance(types, tuple) else [types])]
                return class_name in type_names
            return False

        mock_isinstance.side_effect = isinstance_side_effect

        # Mock dataset item
        mock_image = np.random.rand(32, 32, 3).astype(np.uint8)
        self.mock_dataset.__getitem__ = Mock(
            return_value={
                "image": mock_image,
                "objects": {"bbox": [[0, 0, 10, 10], [5, 5, 15, 15]], "category_id": [0, 1]},
            }
        )

        dataset = HFObjectDetectionDataset(self.mock_dataset, "image", "objects", "bbox", "category_id")

        image, target, metadata = dataset[0]

        assert image.shape == (3, 32, 32)
        assert len(target.boxes) == 2
        assert len(target.labels) == 2
        assert metadata["id"] == 0


class TestUtilityFunctions:
    def test_is_bbox_true(self):
        mock_value = Mock(spec=["dtype"])
        type(mock_value).__name__ = "HFValue"
        mock_value.dtype = "float32"

        mock_feature = Mock(spec=["length", "feature"])
        type(mock_feature).__name__ = "HFList"
        mock_feature.length = 4
        mock_feature.feature = mock_value

        with patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, types: type(obj).__name__ in [
                t.__name__ for t in (types if isinstance(types, tuple) else [types])
            ]
            assert is_bbox(mock_feature)

    def test_is_bbox_false(self):
        mock_feature = Mock(spec=["dtype"])
        type(mock_feature).__name__ = "HFValue"

        with patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance:
            mock_isinstance.return_value = False
            assert not is_bbox(mock_feature)

    def test_is_label_true(self):
        mock_feature = Mock(spec=["names", "num_classes"])
        type(mock_feature).__name__ = "HFClassLabel"

        with patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, types: type(obj).__name__ in [
                t.__name__ for t in (types if isinstance(types, tuple) else [types])
            ]
            assert is_label(mock_feature)

    def test_find_od_keys_success(self):
        mock_bbox = Mock(spec=["length", "feature"])
        mock_label = Mock(spec=["names", "num_classes"])

        mock_feature = Mock(spec=["feature"])
        type(mock_feature).__name__ = "HFList"
        mock_feature.feature = {"bbox": mock_bbox, "category_id": mock_label}

        def mock_is_bbox(feature):
            # Only return True for the actual bbox feature
            return feature is mock_bbox

        def mock_is_label(feature):
            # Only return True for the actual label feature
            return feature is mock_label

        with (
            patch("maite_datasets.adapters._huggingface.is_bbox", side_effect=mock_is_bbox),
            patch("maite_datasets.adapters._huggingface.is_label", side_effect=mock_is_label),
            patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance,
        ):
            mock_isinstance.side_effect = lambda obj, types: type(obj).__name__ in [
                t.__name__ for t in (types if isinstance(types, tuple) else [types])
            ]

            bbox_key, label_key = find_od_keys(mock_feature)
            assert bbox_key == "bbox"
            assert label_key == "category_id"


class TestGetDatasetInfo:
    def test_image_classification_detection(self):
        mock_dataset = Mock()

        mock_image_feature = Mock(spec=["decode"])
        type(mock_image_feature).__name__ = "HFImage"

        mock_label_feature = Mock(spec=["names", "num_classes"])
        type(mock_label_feature).__name__ = "HFClassLabel"

        mock_dataset.features = {"image": mock_image_feature, "label": mock_label_feature}

        with patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, types: type(obj).__name__ in [
                t.__name__ for t in (types if isinstance(types, tuple) else [types])
            ]

            info = get_dataset_info(mock_dataset)
            assert isinstance(info, HFImageClassificationDatasetInfo)
            assert info.image_key == "image"
            assert info.label_key == "label"

    def test_object_detection_detection(self):
        mock_dataset = Mock()

        mock_image_feature = Mock(spec=["decode"])
        type(mock_image_feature).__name__ = "HFImage"

        mock_objects_feature = Mock(spec=["feature"])

        mock_dataset.features = {"image": mock_image_feature, "objects": mock_objects_feature}

        with (
            patch("maite_datasets.adapters._huggingface.isinstance") as mock_isinstance,
            patch("maite_datasets.adapters._huggingface.find_od_keys", return_value=("bbox", "label")),
        ):
            mock_isinstance.side_effect = lambda obj, types: type(obj).__name__ in [
                t.__name__ for t in (types if isinstance(types, tuple) else [types])
            ]

            info = get_dataset_info(mock_dataset)
            assert isinstance(info, HFObjectDetectionDatasetInfo)
            assert info.image_key == "image"
            assert info.objects_key == "objects"

    def test_no_image_key_error(self):
        mock_dataset = Mock()

        mock_text_feature = Mock(spec=["dtype"])
        type(mock_text_feature).__name__ = "HFValue"

        mock_dataset.features = {"text": mock_text_feature}

        with (
            patch("maite_datasets.adapters._huggingface.isinstance", return_value=False),
            pytest.raises(ValueError, match="No image key found"),
        ):
            get_dataset_info(mock_dataset)


class TestFromHuggingface:
    def setup_method(self):
        self.mock_dataset = Mock()

    @patch("maite_datasets.adapters._huggingface.get_dataset_info")
    @patch("maite_datasets.adapters._huggingface.HFImageClassificationDataset")
    def test_from_huggingface_image_classification_auto(self, mock_ic_dataset, mock_get_info):
        mock_info = HFImageClassificationDatasetInfo("image", "label")
        mock_get_info.return_value = mock_info
        mock_ic_dataset.return_value = Mock()

        from_huggingface(self.mock_dataset, "auto")

        mock_ic_dataset.assert_called_once_with(self.mock_dataset, "image", "label")

    @patch("maite_datasets.adapters._huggingface.get_dataset_info")
    @patch("maite_datasets.adapters._huggingface.HFObjectDetectionDataset")
    def test_from_huggingface_object_detection_explicit(self, mock_od_dataset, mock_get_info):
        mock_info = HFObjectDetectionDatasetInfo("image", "objects", "bbox", "label")
        mock_get_info.return_value = mock_info
        mock_od_dataset.return_value = Mock()

        from_huggingface(self.mock_dataset, "object_detection")

        mock_od_dataset.assert_called_once_with(self.mock_dataset, "image", "objects", "bbox", "label")

    @patch("maite_datasets.adapters._huggingface.get_dataset_info")
    def test_from_huggingface_task_mismatch_error(self, mock_get_info):
        mock_info = HFImageClassificationDatasetInfo("image", "label")
        mock_get_info.return_value = mock_info

        with pytest.raises(ValueError, match="Task mismatch"):
            from_huggingface(self.mock_dataset, "object_detection")

    @patch("maite_datasets.adapters._huggingface.get_dataset_info")
    def test_from_huggingface_auto_detection_failure(self, mock_get_info):
        mock_info = HFDatasetInfo("image")  # Base info, no task-specific info
        mock_get_info.return_value = mock_info
        self.mock_dataset.features = {"image": Mock(), "text": Mock()}

        with pytest.raises(ValueError, match="Could not automatically determine task"):
            from_huggingface(self.mock_dataset, "auto")


@pytest.fixture
def sample_image_classification_dataset():
    """Create a mock HF dataset for image classification testing."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=3)
    mock_dataset.info.dataset_name = "test_ic"

    # Create realistic mock features with proper class names
    mock_image_feature = Mock(spec=["decode"])
    type(mock_image_feature).__name__ = "HFImage"

    mock_label_feature = Mock(spec=["num_classes", "names"])
    type(mock_label_feature).__name__ = "HFClassLabel"
    mock_label_feature.num_classes = 2
    mock_label_feature.names = ["cat", "dog"]

    mock_dataset.features = {"pixel_values": mock_image_feature, "labels": mock_label_feature}

    return mock_dataset


@pytest.fixture
def sample_object_detection_dataset():
    """Create a mock HF dataset for object detection testing."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=2)
    mock_dataset.info.dataset_name = "test_od"

    # Create realistic mock nested features with proper class names
    mock_bbox_value = Mock(spec=["dtype"])
    type(mock_bbox_value).__name__ = "HFValue"
    mock_bbox_value.dtype = "float32"

    mock_bbox_list = Mock(spec=["length", "feature"])
    type(mock_bbox_list).__name__ = "HFList"
    mock_bbox_list.length = 4
    mock_bbox_list.feature = mock_bbox_value

    # Create the actual label feature (HFClassLabel)
    mock_label_feature = Mock(spec=["names", "num_classes"])
    type(mock_label_feature).__name__ = "HFClassLabel"
    mock_label_feature.names = ["person", "car"]
    mock_label_feature.num_classes = 2

    # Create the label container that wraps the HFClassLabel
    mock_label_container = Mock(spec=["feature"])
    type(mock_label_container).__name__ = "HFList"
    mock_label_container.feature = mock_label_feature

    mock_objects_feature = Mock(spec=["feature"])
    type(mock_objects_feature).__name__ = "HFList"
    mock_objects_feature.feature = {"bbox": mock_bbox_list, "category_id": mock_label_container}

    mock_image_feature = Mock(spec=["decode"])
    type(mock_image_feature).__name__ = "HFImage"

    mock_dataset.features = {"image": mock_image_feature, "objects": mock_objects_feature}

    return mock_dataset
