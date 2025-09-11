import pytest

from maite_datasets._bbox import (
    BoundingBoxFormat,
    _check_if_normalized,
    _is_plausible_cxcywh,
    _is_plausible_xywh,
    _is_plausible_xyxy,
    _scale_box_if_normalized,
    convert_to_xyxy,
    detect_bbox_format,
)

image_shapes = [(3, 100, 200), (3, 150, 150)]
h1, w1 = 100, 200
h2, w2 = 150, 150


class TestBoundingBox:
    def test_xyxy(self):
        bboxes_xyxy = [[(10, 20, 50, 80), (100, 50, 150, 90)], [(5, 5, 25, 25)]]
        detected = detect_bbox_format(bboxes_xyxy, image_shapes)
        assert detected == BoundingBoxFormat.XYXY

    def test_xywh(self):
        bboxes_xywh = [
            [(10, 20, 40, 60), (100, 50, 50, 40)],  # 10+40=50, 20+60=80
            [(5, 5, 20, 20)],
        ]
        detected = detect_bbox_format(bboxes_xywh, image_shapes)
        assert detected == BoundingBoxFormat.XYWH

    def test_cxcywh(self):
        bboxes_cxcywh = [
            [(30, 50, 40, 60), (125, 70, 50, 40)],  # (30-20=10, 30+20=50), (50-30=20, 50+30=80)
            [(15, 15, 20, 20)],
        ]
        detected = detect_bbox_format(bboxes_cxcywh, image_shapes)
        assert detected == BoundingBoxFormat.CXCYWH

    def test_normalized_xyxy(self):
        bboxes_norm_xyxy = [
            [(10 / w1, 20 / h1, 50 / w1, 80 / h1), (100 / w1, 50 / h1, 150 / w1, 90 / h1)],
            [(5 / w2, 5 / h2, 25 / w2, 25 / h2)],
        ]
        detected = detect_bbox_format(bboxes_norm_xyxy, image_shapes)
        assert detected == BoundingBoxFormat.NORMALIZED_XYXY

    def test_normalized_xywh(self):
        bboxes_norm_xywh = [
            [(10 / w1, 20 / h1, 40 / w1, 60 / h1), (100 / w1, 50 / h1, 50 / w1, 40 / h1)],
            [(5 / w2, 5 / h2, 20 / w2, 20 / h2)],
        ]
        detected = detect_bbox_format(bboxes_norm_xywh, image_shapes)
        assert detected == BoundingBoxFormat.NORMALIZED_XYWH

    def test_normalized_cxcywh(self):
        bboxes_norm_cxcywh = [
            [(30 / w1, 50 / h1, 40 / w1, 60 / h1), (125 / w1, 70 / h1, 50 / w1, 40 / h1)],
            [(15 / w2, 15 / h2, 20 / w2, 20 / h2)],
        ]
        detected = detect_bbox_format(bboxes_norm_cxcywh, image_shapes)
        assert detected == BoundingBoxFormat.NORMALIZED_CXCYWH

    def test_invalid_oob(self):
        bboxes_invalid = [
            [(10, 20, 50, 80), (180, 50, 210, 90)],  # x2=210 > w1=200
            [(5, 5, 25, 25)],
        ]
        detected = detect_bbox_format(bboxes_invalid, image_shapes)
        assert detected == BoundingBoxFormat.UNKNOWN

    def test_ambiguous_resolved(self):
        bboxes_ambiguous_then_resolved = [
            [(1, 1, 2, 2)],  # This box is valid as XYWH and CXCYWH
            [(70, 70, 10, 10)],  # This box is only valid as CXCYWH (center is 70,70) not XYWH (would go to 80,80)
        ]
        image_shapes = [(3, 10, 10), (3, 75, 75)]
        detected = detect_bbox_format(bboxes_ambiguous_then_resolved, image_shapes)
        assert detected == BoundingBoxFormat.CXCYWH

    def test_ambiguous_unresolved(self):
        bboxes_ambiguous_not_resolved = [
            [(1, 1, 2, 2)],  # This box is valid as XYWH and CXCYWH
            [(70, 70, 10, 10)],  # This box is also valid as XYWH and CXCYWH
        ]
        image_shapes = [(3, 10, 10), (3, 80, 80)]
        detected = detect_bbox_format(bboxes_ambiguous_not_resolved, image_shapes)
        assert detected == BoundingBoxFormat.UNKNOWN


class TestBoundingBoxFormat:
    """Test BoundingBoxFormat enum methods."""

    def test_is_xyxy(self):
        assert BoundingBoxFormat.XYXY.is_xyxy()
        assert BoundingBoxFormat.NORMALIZED_XYXY.is_xyxy()
        assert not BoundingBoxFormat.XYWH.is_xyxy()
        assert not BoundingBoxFormat.CXCYWH.is_xyxy()

    def test_is_xywh(self):
        assert BoundingBoxFormat.XYWH.is_xywh()
        assert BoundingBoxFormat.NORMALIZED_XYWH.is_xywh()
        assert not BoundingBoxFormat.XYXY.is_xywh()
        assert not BoundingBoxFormat.CXCYWH.is_xywh()

    def test_is_cxcywh(self):
        assert BoundingBoxFormat.CXCYWH.is_cxcywh()
        assert BoundingBoxFormat.NORMALIZED_CXCYWH.is_cxcywh()
        assert not BoundingBoxFormat.XYXY.is_cxcywh()
        assert not BoundingBoxFormat.XYWH.is_cxcywh()

    def test_is_normalized(self):
        assert BoundingBoxFormat.NORMALIZED_XYXY.is_normalized()
        assert BoundingBoxFormat.NORMALIZED_XYWH.is_normalized()
        assert BoundingBoxFormat.NORMALIZED_CXCYWH.is_normalized()
        assert not BoundingBoxFormat.XYXY.is_normalized()
        assert not BoundingBoxFormat.XYWH.is_normalized()
        assert not BoundingBoxFormat.CXCYWH.is_normalized()

    def test_to_normalized(self):
        assert BoundingBoxFormat.XYXY.to_normalized(True) == BoundingBoxFormat.NORMALIZED_XYXY
        assert BoundingBoxFormat.XYXY.to_normalized(False) == BoundingBoxFormat.XYXY
        assert BoundingBoxFormat.XYWH.to_normalized(True) == BoundingBoxFormat.NORMALIZED_XYWH
        assert BoundingBoxFormat.CXCYWH.to_normalized(True) == BoundingBoxFormat.NORMALIZED_CXCYWH


class TestConvertToXYXY:
    """Test BoundingBox class functionality."""

    @pytest.fixture
    def image_shape(self):
        return (3, 100, 200)  # CHW format

    def test_xyxy_format_init(self, image_shape):
        bbox = convert_to_xyxy(10, 20, 50, 80, bbox_format=BoundingBoxFormat.XYXY, image_shape=image_shape)
        assert bbox == (10, 20, 50, 80)

    def test_xywh_format_init(self, image_shape):
        bbox = convert_to_xyxy(10, 20, 40, 60, bbox_format=BoundingBoxFormat.XYWH, image_shape=image_shape)
        assert bbox == (10, 20, 50, 80)

    def test_cxcywh_format_init(self, image_shape):
        bbox = convert_to_xyxy(30, 50, 40, 60, bbox_format=BoundingBoxFormat.CXCYWH, image_shape=image_shape)
        assert bbox == (10, 20, 50, 80)

    def test_normalized_xyxy_format(self, image_shape):
        # Normalized coordinates: 0.05, 0.2, 0.25, 0.8 for 200x100 image
        bbox = convert_to_xyxy(
            0.05, 0.2, 0.25, 0.8, bbox_format=BoundingBoxFormat.NORMALIZED_XYXY, image_shape=image_shape
        )
        assert bbox == (10, 20, 50, 80)


class TestPlausibilityCheckers:
    """Test bounding box format plausibility checkers."""

    def test_is_plausible_xyxy(self):
        # Valid XYXY
        assert _is_plausible_xyxy((10, 20, 50, 80), 100, 200)
        # Invalid: x1 > x2
        assert not _is_plausible_xyxy((50, 20, 10, 80), 100, 200)
        # Invalid: outside bounds
        assert not _is_plausible_xyxy((10, 20, 250, 80), 100, 200)

    def test_is_plausible_xywh(self):
        # Valid XYWH
        assert _is_plausible_xywh((10, 20, 40, 60), 100, 200)
        # Invalid: negative width
        assert not _is_plausible_xywh((10, 20, -40, 60), 100, 200)
        # Invalid: outside bounds
        assert not _is_plausible_xywh((10, 20, 200, 60), 100, 200)

    def test_is_plausible_cxcywh(self):
        # Valid CXCYWH
        assert _is_plausible_cxcywh((30, 50, 40, 60), 100, 200)
        # Invalid: negative width
        assert not _is_plausible_cxcywh((30, 50, -40, 60), 100, 200)
        # Invalid: center too close to edge
        assert not _is_plausible_cxcywh((10, 50, 40, 60), 100, 200)  # Would extend outside


class TestHelperFunctions:
    """Test helper functions for format detection."""

    def test_check_if_normalized_true(self):
        all_bboxes = [[(0.1, 0.2, 0.5, 0.8), (0.3, 0.4, 0.7, 0.9)], [(0.2, 0.1, 0.6, 0.7)]]
        assert _check_if_normalized(all_bboxes) is True

    def test_check_if_normalized_false(self):
        all_bboxes = [[(10, 20, 50, 80), (30, 40, 70, 90)], [(20, 10, 60, 70)]]
        assert _check_if_normalized(all_bboxes) is False

    def test_check_if_normalized_mixed(self):
        all_bboxes = [
            [(0.1, 0.2, 0.5, 0.8), (30, 40, 70, 90)],  # One normalized, one not
        ]
        assert _check_if_normalized(all_bboxes) is False

    def test_scale_box_if_normalized(self):
        box = (0.1, 0.2, 0.5, 0.8)
        scaled = _scale_box_if_normalized(box, True, 100, 200)
        expected = (20, 20, 100, 80)  # (0.1*200, 0.2*100, 0.5*200, 0.8*100)
        assert scaled == expected

        # Not normalized - should return original
        unscaled = _scale_box_if_normalized(box, False, 100, 200)
        assert unscaled == box


class TestFormatDetection:
    """Test the main format detection function."""

    def test_detect_xyxy_format(self):
        # XYXY format boxes
        all_bboxes = [[(10, 20, 50, 80), (100, 120, 150, 180)], [(20, 30, 60, 90)]]
        image_shapes = [(3, 200, 300), (3, 200, 300)]

        format_detected = detect_bbox_format(all_bboxes, image_shapes)
        assert format_detected == BoundingBoxFormat.XYXY

    def test_detect_xywh_format(self):
        # XYWH format boxes
        all_bboxes = [
            [(10, 20, 40, 60), (100, 120, 50, 60)],  # x, y, width, height
            [(20, 30, 40, 60)],
        ]
        image_shapes = [(3, 200, 300), (3, 200, 300)]

        format_detected = detect_bbox_format(all_bboxes, image_shapes)
        assert format_detected == BoundingBoxFormat.XYWH

    def test_detect_normalized_format(self):
        # Normalized XYXY format
        all_bboxes = [[(0.1, 0.2, 0.5, 0.8), (0.3, 0.4, 0.7, 0.9)], [(0.2, 0.3, 0.6, 0.9)]]
        image_shapes = [(3, 200, 300), (3, 200, 300)]

        format_detected = detect_bbox_format(all_bboxes, image_shapes)
        assert format_detected == BoundingBoxFormat.NORMALIZED_XYXY

    def test_detect_empty_input(self):
        assert detect_bbox_format([], []) == BoundingBoxFormat.UNKNOWN
        assert detect_bbox_format([[], []], [(3, 100, 100)] * 2) == BoundingBoxFormat.UNKNOWN


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_format_detection_with_ambiguous_data(self):
        """Test when multiple formats are equally plausible."""
        # Data that could be either XYXY or CXCYWH
        all_bboxes = [
            [(50, 50, 100, 100)],  # Could be XYXY (50,50 to 100,100) or CXCYWH (center 50,50, size 100x100)
        ]
        image_shapes = [(3, 200, 200)]

        # This should still return a format (implementation dependent)
        result = detect_bbox_format(all_bboxes, image_shapes)
        assert result in [BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH, BoundingBoxFormat.UNKNOWN]

    def test_very_large_dataset_sampling(self):
        """Test that format detection samples appropriately for large datasets."""
        # This would be tested with a mock to verify sampling behavior
        pass

    def test_bbox_conversion_edge_coordinates(self):
        """Test bounding box conversion with edge coordinates."""
        # Test with boxes at image boundaries
        bbox = convert_to_xyxy(0, 0, 200, 100, bbox_format=BoundingBoxFormat.XYXY, image_shape=(3, 100, 200))
        assert bbox == (0, 0, 200, 100)


# Parameterized tests for different formats
@pytest.mark.parametrize(
    "input_format,input_coords,expected_xyxy",
    [
        (BoundingBoxFormat.XYXY, (10, 20, 50, 80), (10, 20, 50, 80)),
        (BoundingBoxFormat.XYWH, (10, 20, 40, 60), (10, 20, 50, 80)),
        (BoundingBoxFormat.CXCYWH, (30, 50, 40, 60), (10, 20, 50, 80)),
        (BoundingBoxFormat.NORMALIZED_XYXY, (0.05, 0.2, 0.25, 0.8), (10, 20, 50, 80)),
        (BoundingBoxFormat.NORMALIZED_XYWH, (0.05, 0.2, 0.2, 0.6), (10, 20, 50, 80)),
        (BoundingBoxFormat.NORMALIZED_CXCYWH, (0.15, 0.5, 0.2, 0.6), (10, 20, 50, 80)),
    ],
)
def test_format_conversions(input_format, input_coords, expected_xyxy):
    """Test all format conversions produce correct XYXY output."""
    image_shape = (3, 100, 200)  # H=100, W=200
    bbox = convert_to_xyxy(*input_coords, bbox_format=input_format, image_shape=image_shape)
    assert bbox == pytest.approx(expected_xyxy, abs=1e-10)
