"""
Unit tests for dataset readers (COCO and YOLO formats).
"""

import json

import numpy as np
import pytest
from PIL import Image

# Import the modules to test
from maite_datasets._reader import BaseDatasetReader, create_dataset_reader
from maite_datasets.object_detection._coco import COCODatasetReader
from maite_datasets.object_detection._yolo import YOLODatasetReader


@pytest.fixture
def sample_coco_data():
    """Sample COCO annotation data."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 50, 200, 150],  # x, y, w, h
                "area": 30000,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [300, 200, 100, 100],
                "area": 10000,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 25, 150, 200],
                "area": 30000,
            },
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
        ],
    }


@pytest.fixture
def sample_yolo_labels():
    """Sample YOLO label data."""
    return {
        "image1.txt": "0 0.5 0.3 0.2 0.4\n1 0.7 0.8 0.1 0.2\n",
        "image2.txt": "0 0.3 0.5 0.3 0.6\n",
    }


@pytest.fixture
def sample_classes():
    """Sample class names."""
    return ["person", "car", "bicycle"]


@pytest.fixture
def temp_coco_dataset(tmp_path, sample_coco_data, sample_classes):
    """Create temporary COCO dataset structure."""
    # Create directories
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create dummy images
    for img_info in sample_coco_data["images"]:
        img_path = images_dir / img_info["file_name"]
        dummy_img = Image.new("RGB", (img_info["width"], img_info["height"]), color="red")
        dummy_img.save(img_path)

    # Create annotations.json
    annotations_path = tmp_path / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(sample_coco_data, f)

    # Create classes.txt
    classes_path = tmp_path / "classes.txt"
    with open(classes_path, "w") as f:
        for class_name in sample_classes:
            f.write(f"{class_name}\n")

    return tmp_path


@pytest.fixture
def temp_yolo_dataset(tmp_path, sample_yolo_labels, sample_classes):
    """Create temporary YOLO dataset structure."""
    # Create directories
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create dummy images and labels
    for label_file, content in sample_yolo_labels.items():
        # Create image
        img_name = label_file.replace(".txt", ".jpg")
        img_path = images_dir / img_name
        dummy_img = Image.new("RGB", (640, 480), color="blue")
        dummy_img.save(img_path)

        # Create label file
        label_path = labels_dir / label_file
        with open(label_path, "w") as f:
            f.write(content)

    # Create classes.txt
    classes_path = tmp_path / "classes.txt"
    with open(classes_path, "w") as f:
        for class_name in sample_classes:
            f.write(f"{class_name}\n")

    return tmp_path


class TestBaseDatasetReader:
    """Test the BaseDatasetReader abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseDatasetReader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDatasetReader("/some/path")

    def test_subclass_inheritance(self, temp_coco_dataset):
        """Test that concrete classes properly inherit from BaseDatasetReader."""
        reader = COCODatasetReader(temp_coco_dataset)
        assert isinstance(reader, BaseDatasetReader)

        # Test common interface
        assert hasattr(reader, "dataset_path")
        assert hasattr(reader, "dataset_id")
        assert hasattr(reader, "index2label")
        assert hasattr(reader, "create_dataset")
        assert hasattr(reader, "validate_structure")

        # Test that get_dataset returns expected type
        dataset = reader.create_dataset()
        assert hasattr(dataset, "metadata")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")

        # Test validation method
        result = reader.validate_structure()
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "issues" in result
        assert "stats" in result


class TestCOCODatasetReader:
    """Test the COCODatasetReader class."""

    def test_initialization_success(self, temp_coco_dataset):
        """Test successful initialization."""
        reader = COCODatasetReader(temp_coco_dataset)

        assert reader.dataset_path == temp_coco_dataset
        assert reader.dataset_id == temp_coco_dataset.name
        assert len(reader.index2label) == 3  # From classes.txt
        assert reader.index2label[0] == "person"

    def test_initialization_missing_path(self):
        """Test initialization with missing dataset path."""
        with pytest.raises(FileNotFoundError, match="Dataset path not found"):
            COCODatasetReader("/nonexistent/path")

    def test_initialization_missing_annotations(self, tmp_path):
        """Test initialization with missing annotations file."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            COCODatasetReader(tmp_path)

    def test_initialization_missing_images(self, tmp_path, sample_coco_data):
        """Test initialization with missing images directory."""
        annotations_path = tmp_path / "annotations.json"
        with open(annotations_path, "w") as f:
            json.dump(sample_coco_data, f)

        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            COCODatasetReader(tmp_path)

    def test_get_dataset(self, temp_coco_dataset):
        """Test dataset creation."""
        reader = COCODatasetReader(temp_coco_dataset)
        dataset = reader.create_dataset()

        assert len(dataset) == 2
        assert dataset.metadata["id"] == temp_coco_dataset.name
        assert "person" in dataset.metadata["index2label"].values()

    def test_dataset_getitem(self, temp_coco_dataset):
        """Test dataset item retrieval."""
        reader = COCODatasetReader(temp_coco_dataset)
        dataset = reader.create_dataset()

        image, target, metadata = dataset[0]

        # Check image shape (CHW format)
        assert image.shape == (3, 480, 640)

        # Check target
        assert len(target.boxes) == 2  # Two annotations for first image
        assert len(target.labels) == 2
        assert len(target.scores) == 2

        # Check bbox conversion (COCO to x1,y1,x2,y2)
        expected_boxes = np.array([[100, 50, 300, 200], [300, 200, 400, 300]])
        np.testing.assert_array_equal(target.boxes, expected_boxes)

        # Check comprehensive metadata
        assert metadata["id"].startswith(temp_coco_dataset.name)
        assert metadata["coco_image_id"] == 1
        assert metadata["file_name"] == "image1.jpg"
        assert metadata["width"] == 640
        assert metadata["height"] == 480
        assert metadata["num_annotations"] == 2
        assert len(metadata["annotations"]) == 2

        # Check annotation metadata structure
        ann_meta = metadata["annotations"][0]
        assert "annotation_id" in ann_meta
        assert "category_id" in ann_meta
        assert "area" in ann_meta
        assert "iscrowd" in ann_meta

    def test_dataset_empty_annotations(self, temp_coco_dataset, sample_coco_data):
        """Test dataset with image having no annotations."""
        # Modify data to have image with no annotations
        sample_coco_data["images"].append({"id": 3, "file_name": "image3.jpg", "width": 320, "height": 240})
        sample_coco_data["annotations"] = [ann for ann in sample_coco_data["annotations"] if ann["image_id"] != 2]

        # Recreate annotations file
        annotations_path = temp_coco_dataset / "annotations.json"
        with open(annotations_path, "w") as f:
            json.dump(sample_coco_data, f)

        # Add image3.jpg
        img_path = temp_coco_dataset / "images" / "image3.jpg"
        dummy_img = Image.new("RGB", (320, 240), color="green")
        dummy_img.save(img_path)

        reader = COCODatasetReader(temp_coco_dataset)
        dataset = reader.create_dataset()

        # Get item with no annotations (image2)
        image, target, metadata = dataset[1]

        assert target.boxes.shape == (0, 4)
        assert target.labels.shape == (0,)
        assert target.scores.shape == (0,)


class TestYOLODatasetReader:
    """Test the YOLODatasetReader class."""

    def test_initialization_success(self, temp_yolo_dataset):
        """Test successful initialization."""
        reader = YOLODatasetReader(temp_yolo_dataset)

        assert reader.dataset_path == temp_yolo_dataset
        assert reader.dataset_id == temp_yolo_dataset.name
        assert len(reader.index2label) == 3
        assert len(reader._image_files) == 2

    def test_initialization_missing_path(self):
        """Test initialization with missing dataset path."""
        with pytest.raises(FileNotFoundError, match="Dataset path not found"):
            YOLODatasetReader("/nonexistent/path")

    def test_initialization_missing_labels(self, tmp_path):
        """Test initialization with missing labels directory."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        classes_path = tmp_path / "classes.txt"
        classes_path.write_text("person\ncar\n")

        with pytest.raises(FileNotFoundError, match="Labels directory not found"):
            YOLODatasetReader(tmp_path)

    def test_initialization_missing_classes(self, tmp_path):
        """Test initialization with missing classes file."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Classes file not found"):
            YOLODatasetReader(tmp_path)

    def test_initialization_no_images(self, tmp_path):
        """Test initialization with no image files."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        classes_path = tmp_path / "classes.txt"
        classes_path.write_text("person\ncar\n")

        with pytest.raises(ValueError, match="No image files found"):
            YOLODatasetReader(tmp_path)

    def test_get_dataset(self, temp_yolo_dataset):
        """Test dataset creation."""
        reader = YOLODatasetReader(temp_yolo_dataset)
        dataset = reader.create_dataset()

        assert len(dataset) == 2
        assert dataset.metadata["id"] == temp_yolo_dataset.name
        assert "person" in dataset.metadata["index2label"].values()

    def test_dataset_getitem(self, temp_yolo_dataset):
        """Test dataset item retrieval."""
        reader = YOLODatasetReader(temp_yolo_dataset)
        dataset = reader.create_dataset()

        image, target, metadata = dataset[0]

        # Check image shape (CHW format)
        assert image.shape == (3, 480, 640)

        # Check target
        assert len(target.boxes) == 2  # Two annotations in image1.txt
        assert len(target.labels) == 2
        assert len(target.scores) == 2

        # Check coordinate conversion (normalized to absolute)
        # First annotation: 0 0.5 0.3 0.2 0.4 -> center_x=0.5*640, center_y=0.3*480, w=0.2*640, h=0.4*480
        # -> center=(320, 144), w=128, h=192 -> bbox=[256, 48, 384, 240]
        expected_x1 = (0.5 - 0.2 / 2) * 640  # 256
        expected_y1 = (0.3 - 0.4 / 2) * 480  # 48
        expected_x2 = (0.5 + 0.2 / 2) * 640  # 384
        expected_y2 = (0.3 + 0.4 / 2) * 480  # 240

        np.testing.assert_array_almost_equal(target.boxes[0], [expected_x1, expected_y1, expected_x2, expected_y2])

        # Check comprehensive metadata
        assert metadata["id"].startswith(temp_yolo_dataset.name)
        assert metadata["file_name"] == "image1.jpg"
        assert metadata["width"] == 640
        assert metadata["height"] == 480
        assert metadata["label_file_exists"] is True
        assert metadata["label_file"] == "image1.txt"
        assert metadata["num_annotations"] == 2
        assert len(metadata["annotations"]) == 2

        # Check annotation metadata structure
        ann_meta = metadata["annotations"][0]
        assert "line_number" in ann_meta
        assert "class_id" in ann_meta
        assert "yolo_center_x" in ann_meta
        assert "yolo_center_y" in ann_meta
        assert "yolo_width" in ann_meta
        assert "yolo_height" in ann_meta
        assert "absolute_bbox" in ann_meta

    def test_dataset_missing_label_file(self, temp_yolo_dataset):
        """Test dataset with missing label file."""
        # Remove one label file
        label_file = temp_yolo_dataset / "labels" / "image2.txt"
        label_file.unlink()

        reader = YOLODatasetReader(temp_yolo_dataset)
        dataset = reader.create_dataset()

        # Get item with missing label file (should have empty annotations)
        image, target, metadata = dataset[1]  # image2

        assert target.boxes.shape == (0, 4)
        assert target.labels.shape == (0,)
        assert target.scores.shape == (0,)


class TestCreateDatasetReader:
    """Test the create_dataset_reader factory function."""

    def test_auto_detect_coco(self, temp_coco_dataset):
        """Test auto-detection of COCO format."""
        reader = create_dataset_reader(temp_coco_dataset)
        assert isinstance(reader, BaseDatasetReader)
        assert isinstance(reader, COCODatasetReader)

    def test_auto_detect_yolo(self, temp_yolo_dataset):
        """Test auto-detection of YOLO format."""
        reader = create_dataset_reader(temp_yolo_dataset)
        assert isinstance(reader, BaseDatasetReader)
        assert isinstance(reader, YOLODatasetReader)

    def test_explicit_format_hint(self, temp_coco_dataset):
        """Test explicit format hint."""
        reader = create_dataset_reader(temp_coco_dataset, format_hint="coco")
        assert isinstance(reader, BaseDatasetReader)
        assert isinstance(reader, COCODatasetReader)

    def test_invalid_format_hint(self, temp_coco_dataset):
        """Test invalid format hint."""
        with pytest.raises(ValueError, match="Unsupported format hint"):
            create_dataset_reader(temp_coco_dataset, format_hint="invalid")

    def test_ambiguous_format(self, tmp_path, sample_coco_data, sample_classes):
        """Test ambiguous format detection."""
        # Create dataset with both COCO and YOLO indicators
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create annotations.json (COCO indicator)
        annotations_path = tmp_path / "annotations.json"
        with open(annotations_path, "w") as f:
            json.dump(sample_coco_data, f)

        with pytest.raises(ValueError, match="Ambiguous format"):
            create_dataset_reader(tmp_path)

    def test_undetectable_format(self, tmp_path):
        """Test undetectable format."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        with pytest.raises(ValueError, match="Cannot detect dataset format"):
            create_dataset_reader(tmp_path)


class TestCOCODatasetValidation:
    """Test COCO dataset validation methods."""

    def test_validate_structure_success(self, temp_coco_dataset):
        """Test successful COCO dataset validation."""
        reader = COCODatasetReader(temp_coco_dataset)
        result = reader.validate_structure()

        assert result["is_valid"] is True
        assert len(result["issues"]) == 0
        assert result["stats"]["num_images"] == 2
        assert result["stats"]["num_images"] == 2
        assert result["stats"]["num_annotations"] == 3
        assert result["stats"]["num_categories"] == 2

    def test_validate_structure_missing_annotation(self, tmp_path):
        """Test COCO validation with missing annotations file."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create dummy image
        dummy_img = Image.new("RGB", (100, 100), color="red")
        dummy_img.save(images_dir / "test.jpg")

        # This should fail during reader initialization
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            COCODatasetReader(tmp_path)

    def test_validate_structure_invalid_json(self, tmp_path):
        """Test COCO validation with invalid JSON."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create dummy image
        dummy_img = Image.new("RGB", (100, 100), color="red")
        dummy_img.save(images_dir / "test.jpg")

        # Create invalid JSON
        annotation_path = tmp_path / "annotations.json"
        annotation_path.write_text("invalid json content")

        # This should fail during reader initialization
        with pytest.raises(json.JSONDecodeError):
            COCODatasetReader(tmp_path)


class TestYOLODatasetValidation:
    """Test YOLO dataset validation methods."""

    def test_validate_structure_success(self, temp_yolo_dataset):
        """Test successful YOLO dataset validation."""
        reader = YOLODatasetReader(temp_yolo_dataset)
        result = reader.validate_structure()

        assert result["is_valid"] is True
        assert len(result["issues"]) == 0
        assert result["stats"]["num_images"] == 2
        assert result["stats"]["num_label_files"] == 2
        assert result["stats"]["num_classes"] == 3

    def test_validate_structure_missing_labels(self, tmp_path):
        """Test YOLO validation with missing labels directory."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create dummy image
        dummy_img = Image.new("RGB", (100, 100), color="red")
        dummy_img.save(images_dir / "test.jpg")

        # Create classes file
        classes_path = tmp_path / "classes.txt"
        classes_path.write_text("person\ncar\n")

        # This should fail during reader initialization
        with pytest.raises(FileNotFoundError, match="Labels directory not found"):
            YOLODatasetReader(tmp_path)

    def test_validate_structure_missing_classes(self, tmp_path):
        """Test YOLO validation with missing classes file."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create dummy image
        dummy_img = Image.new("RGB", (100, 100), color="red")
        dummy_img.save(images_dir / "test.jpg")

        # This should fail during reader initialization
        with pytest.raises(FileNotFoundError, match="Classes file not found"):
            YOLODatasetReader(tmp_path)

    def test_validate_yolo_label_format_invalid(self, temp_yolo_dataset):
        """Test YOLO label format validation with invalid format."""
        # Modify one of the label files to have invalid format
        label_file = temp_yolo_dataset / "labels" / "image1.txt"
        label_file.write_text("0 0.5 0.3\n")  # Missing width and height

        reader = YOLODatasetReader(temp_yolo_dataset)
        result = reader.validate_structure()

        assert result["is_valid"] is False
        assert any("expected 5 values" in issue for issue in result["issues"])

    def test_validate_yolo_label_format_out_of_range(self, temp_yolo_dataset):
        """Test YOLO label format validation with out-of-range coordinates."""
        # Modify one of the label files to have out-of-range coordinates
        label_file = temp_yolo_dataset / "labels" / "image1.txt"
        label_file.write_text("0 1.5 0.3 0.2 0.4\n")  # center_x > 1.0

        reader = YOLODatasetReader(temp_yolo_dataset)
        result = reader.validate_structure()

        assert result["is_valid"] is False
        assert any("out of range" in issue for issue in result["issues"])


@pytest.mark.parametrize(
    "format_type,expected_reader",
    [
        ("coco", COCODatasetReader),
        ("yolo", YOLODatasetReader),
    ],
)
def test_format_consistency(format_type, expected_reader, tmp_path):
    """Test that format detection and reader creation are consistent."""
    # This test ensures that our format detection logic matches reader expectations
    # It's a bit meta but helps catch inconsistencies

    if format_type == "coco":
        # Create minimal COCO structure
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (tmp_path / "annotations.json").write_text('{"images":[],"annotations":[],"categories":[]}')
    else:  # yolo
        # Create minimal YOLO structure
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        (tmp_path / "classes.txt").write_text("class1\n")

        # Create at least one dummy image file (required for YOLO reader)
        dummy_img = Image.new("RGB", (100, 100), color="red")
        dummy_img.save(images_dir / "test.jpg")

    # Auto-detection should work
    reader = create_dataset_reader(tmp_path)
    assert isinstance(reader, BaseDatasetReader)
    assert isinstance(reader, expected_reader)

    # Explicit format should work
    reader = create_dataset_reader(tmp_path, format_hint=format_type)
    assert isinstance(reader, BaseDatasetReader)
    assert isinstance(reader, expected_reader)

    # Validation should pass
    result = reader.validate_structure()
    # Note: might not be fully valid due to minimal structure, but should not crash
    assert "is_valid" in result
    assert "issues" in result
    assert "stats" in result
