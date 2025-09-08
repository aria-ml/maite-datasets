"""Unit tests for TorchvisionWrapper class."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

from maite_datasets._base import ObjectDetectionTargetTuple
from maite_datasets.wrappers._torch import TorchvisionWrapper


class MockArray:
    """Mock array-like object for testing."""

    def __init__(self, data):
        self.data = np.array(data)

    def __array__(self):
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __getitem__(self, key: Any, /) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class MockObjectDetectionTarget:
    """Mock ObjectDetectionTarget for testing."""

    def __init__(self, boxes, labels, scores):
        self.boxes = np.array(boxes)
        self.labels = np.array(labels)
        self.scores = np.array(scores)


class MockDataset:
    """Mock dataset for testing TorchvisionWrapper."""

    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {"id": "test_dataset", "index2label": {0: "class0", 1: "class1"}}
        self.custom_attr = "test_value"

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@pytest.fixture
def ic_numpy_image():
    """Image classification numpy image (C, H, W)."""
    return np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)


@pytest.fixture
def ic_target():
    """Image classification target."""
    return np.array([0.0, 1.0, 0.0])


@pytest.fixture
def od_target():
    """Object detection target."""
    return MockObjectDetectionTarget(boxes=[[10, 20, 30, 40], [50, 60, 70, 80]], labels=[0, 1], scores=[0.9, 0.8])


@pytest.fixture
def datum_metadata():
    """Sample datum metadata."""
    return {"id": "test_image_001"}


@pytest.fixture
def ic_dataset(ic_numpy_image, ic_target, datum_metadata):
    """Image classification dataset."""
    data = [(ic_numpy_image, ic_target, datum_metadata)]
    return MockDataset(data)


@pytest.fixture
def od_dataset(ic_numpy_image, od_target, datum_metadata):
    """Object detection dataset."""
    data = [(ic_numpy_image, od_target, datum_metadata)]
    return MockDataset(data)


class TestTorchWrapper:
    """Test TorchvisionWrapper functionality."""

    def test_init_without_transforms(self, ic_dataset):
        """Test initialization without transforms."""
        wrapper = TorchvisionWrapper(ic_dataset)

        assert wrapper._dataset is ic_dataset
        assert wrapper.transforms is None
        assert wrapper.metadata["id"] == "TorchvisionWrapper(test_dataset)"
        assert wrapper.metadata["index2label"] == {0: "class0", 1: "class1"}

    def test_init_with_transforms(self, ic_dataset):
        """Test initialization with transforms."""
        transform_fn = Mock()
        wrapper = TorchvisionWrapper(ic_dataset, transforms=transform_fn)

        assert wrapper.transforms is transform_fn

    def test_init_missing_index2label(self, ic_numpy_image, ic_target, datum_metadata):
        """Test initialization when dataset metadata missing index2label."""
        dataset = MockDataset([(ic_numpy_image, ic_target, datum_metadata)], {"id": "test"})
        wrapper = TorchvisionWrapper(dataset)

        assert wrapper.metadata["index2label"] == {}

    def test_getattr_forwards_to_dataset(self, ic_dataset):
        """Test attribute forwarding to wrapped dataset."""
        wrapper = TorchvisionWrapper(ic_dataset)

        assert wrapper.custom_attr == "test_value"

    def test_getattr_raises_for_nonexistent(self, ic_dataset):
        """Test AttributeError for non-existent attributes."""
        wrapper = TorchvisionWrapper(ic_dataset)

        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attr

    def test_dir_includes_dataset_attrs(self, ic_dataset):
        """Test __dir__ includes wrapped dataset attributes."""
        wrapper = TorchvisionWrapper(ic_dataset)
        attrs = dir(wrapper)

        assert "custom_attr" in attrs
        assert "_dataset" in attrs
        assert "transforms" in attrs

    def test_len(self, ic_dataset):
        """Test __len__ forwards to dataset."""
        wrapper = TorchvisionWrapper(ic_dataset)

        assert len(wrapper) == len(ic_dataset)

    def test_getitem_image_classification(self, ic_dataset):
        """Test __getitem__ for image classification."""
        wrapper = TorchvisionWrapper(ic_dataset)
        torch_image, torch_target, metadata = wrapper[0]

        # Check image conversion
        assert isinstance(torch_image, Image)
        assert torch_image.dtype == torch.uint8
        assert torch_image.shape == (3, 32, 32)  # CHW format

        # Check target conversion
        assert isinstance(torch_target, torch.Tensor)
        assert torch_target.dtype == torch.float32
        np.testing.assert_array_equal(torch_target.numpy(), [0.0, 1.0, 0.0])

        # Check metadata passthrough
        assert metadata == {"id": "test_image_001"}

    def test_getitem_object_detection(self, od_dataset):
        """Test __getitem__ for object detection."""
        wrapper = TorchvisionWrapper(od_dataset)
        torch_image, torch_target, metadata = wrapper[0]

        # Check image conversion
        assert isinstance(torch_image, Image)
        assert torch_image.dtype == torch.uint8

        # Check target conversion
        assert isinstance(torch_target, ObjectDetectionTargetTuple)
        assert isinstance(torch_target.boxes, BoundingBoxes)
        assert torch_target.boxes.format == BoundingBoxFormat.XYXY
        assert torch_target.boxes.canvas_size == (32, 32)
        assert torch_target.labels.dtype == torch.int64
        assert torch_target.scores.dtype == torch.float32

        # Check tensor values
        expected_boxes = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        torch.testing.assert_close(torch_target.boxes.data, expected_boxes, check_dtype=False)
        torch.testing.assert_close(torch_target.labels, torch.tensor([0, 1]), check_dtype=False)
        torch.testing.assert_close(torch_target.scores, torch.tensor([0.9, 0.8]), check_dtype=False)

    def test_getitem_with_transforms_ic(self, ic_dataset):
        """Test __getitem__ with transforms for image classification."""

        def mock_transform(datum):
            image, target, metadata = datum
            return image * 0.5, target * 2, metadata

        wrapper = TorchvisionWrapper(ic_dataset, transforms=mock_transform)
        torch_image, torch_target, metadata = wrapper[0]

        # Check transform was applied
        assert torch_image.max() <= ic_dataset[0][0].max() * 0.5
        np.testing.assert_array_equal(torch_target.numpy(), [0.0, 2.0, 0.0])

    def test_getitem_with_transforms_od(self, od_dataset):
        """Test __getitem__ with transforms for object detection."""

        def mock_transform(datum):
            image, target, metadata = datum
            return image, ObjectDetectionTargetTuple(target.boxes * 2, target.labels, target.scores), metadata

        wrapper = TorchvisionWrapper(od_dataset, transforms=mock_transform)
        torch_image, torch_target, metadata = wrapper[0]

        # Check transform was applied to boxes
        expected_boxes = torch.tensor([[20, 40, 60, 80], [100, 120, 140, 160]])
        torch.testing.assert_close(torch_target.boxes.data, expected_boxes, check_dtype=False)

    def test_getitem_float_image_no_normalization(self, ic_target, datum_metadata):
        """Test __getitem__ with float image doesn't get normalized."""
        float_image = np.random.rand(3, 32, 32).astype(np.float32)
        dataset = MockDataset([(float_image, ic_target, datum_metadata)])
        wrapper = TorchvisionWrapper(dataset)

        torch_image, _, _ = wrapper[0]
        # Float images should not be divided by 255
        assert torch_image.max() > 0.5

    def test_getitem_non_numpy_array_target(self, ic_numpy_image, datum_metadata):
        """Test __getitem__ with array-like target."""
        array_target = MockArray([0.0, 1.0, 0.0])
        dataset = MockDataset([(ic_numpy_image, array_target, datum_metadata)])
        wrapper = TorchvisionWrapper(dataset)

        torch_image, torch_target, metadata = wrapper[0]
        assert isinstance(torch_target, torch.Tensor)
        np.testing.assert_array_equal(torch_target.numpy(), [0.0, 1.0, 0.0])

    def test_getitem_unsupported_target_type(self, ic_numpy_image, datum_metadata):
        """Test __getitem__ raises TypeError for unsupported target."""
        unsupported_target = "invalid_target"
        dataset = MockDataset([(ic_numpy_image, unsupported_target, datum_metadata)])
        wrapper = TorchvisionWrapper(dataset)

        with pytest.raises(TypeError, match="Unsupported target type"):
            _ = wrapper[0]

    def test_str_representation(self, ic_dataset):
        """Test string representation."""
        wrapper = TorchvisionWrapper(ic_dataset)
        str_repr = str(wrapper)

        assert "Torchvision Wrapped Mock Dataset" in str_repr
        assert "Transforms: None" in str_repr
        assert str(ic_dataset) in str_repr

    def test_str_representation_avoids_double_torch(self):
        """Test string representation doesn't double 'Torch' prefix."""

        class TorchvisionMockDataset(MockDataset):
            pass

        dataset = TorchvisionMockDataset([])
        wrapper = TorchvisionWrapper(dataset)
        str_repr = str(wrapper)

        # Should not have "Torchvision Wrapped"
        assert "Torchvision Wrapped" not in str_repr
