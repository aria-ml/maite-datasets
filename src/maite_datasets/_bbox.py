from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable


class BoundingBoxFormat(Enum):
    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"
    NORMALIZED_XYXY = "NORMALIZED_XYXY"
    NORMALIZED_XYWH = "NORMALIZED_XYWH"
    NORMALIZED_CXCYWH = "NORMALIZED_CXCYWH"  # YOLO
    UNKNOWN = "UNKNOWN"

    def is_xyxy(self) -> bool:
        return self in (BoundingBoxFormat.XYXY, BoundingBoxFormat.NORMALIZED_XYXY)

    def is_xywh(self) -> bool:
        return self in (BoundingBoxFormat.XYWH, BoundingBoxFormat.NORMALIZED_XYWH)

    def is_cxcywh(self) -> bool:
        return self in (BoundingBoxFormat.CXCYWH, BoundingBoxFormat.NORMALIZED_CXCYWH)

    def is_normalized(self) -> bool:
        return self in (
            BoundingBoxFormat.NORMALIZED_XYXY,
            BoundingBoxFormat.NORMALIZED_XYWH,
            BoundingBoxFormat.NORMALIZED_CXCYWH,
        )

    def to_normalized(self, should_normalize: bool = True) -> BoundingBoxFormat:
        if self.is_xyxy():
            return BoundingBoxFormat.NORMALIZED_XYXY if should_normalize else BoundingBoxFormat.XYXY
        if self.is_xywh():
            return BoundingBoxFormat.NORMALIZED_XYWH if should_normalize else BoundingBoxFormat.XYWH
        if self.is_cxcywh():
            return BoundingBoxFormat.NORMALIZED_CXCYWH if should_normalize else BoundingBoxFormat.CXCYWH
        return BoundingBoxFormat.UNKNOWN


def convert_to_xyxy(
    v1: float, v2: float, v3: float, v4: float, bbox_format: BoundingBoxFormat, image_shape: tuple[int, ...]
) -> tuple[float, float, float, float]:
    h, w = image_shape[-2], image_shape[-1]

    if bbox_format in (
        BoundingBoxFormat.NORMALIZED_XYXY,
        BoundingBoxFormat.NORMALIZED_XYWH,
        BoundingBoxFormat.NORMALIZED_CXCYWH,
    ):
        v1 = v1 * w
        v2 = v2 * h
        v3 = v3 * w
        v4 = v4 * h

    if bbox_format in (BoundingBoxFormat.XYXY, BoundingBoxFormat.NORMALIZED_XYXY):
        if v1 > v3 or v2 > v4:
            warnings.warn(f"Invalid bounding box coordinates: {(v1, v2, v3, v4)} - swapping invalid coordinates.")
        x0 = min(v1, v3)
        y0 = min(v2, v4)
        x1 = max(v1, v3)
        y1 = max(v2, v4)
    elif bbox_format in (BoundingBoxFormat.XYWH, BoundingBoxFormat.NORMALIZED_XYWH):
        x0, y0 = v1, v2
        x1, y1 = v1 + v3, v2 + v4
    elif bbox_format in (BoundingBoxFormat.CXCYWH, BoundingBoxFormat.NORMALIZED_CXCYWH):
        center_x, center_y, w, h = v1, v2, v3, v4
        x0 = center_x - w / 2
        y0 = center_y - h / 2
        x1 = center_x + w / 2
        y1 = center_y + h / 2
    else:
        raise ValueError(f"Unsupported bounding box format: {bbox_format}")

    return x0, y0, x1, y1


def _is_plausible_xyxy(box: tuple[float, ...], h: int, w: int) -> bool:
    """
    Check if a box is plausible in XYXY format.

    Parameters
    ----------
    box : tuple[float, ...]
        Bounding box coordinates.
    h : int
        Image height.
    w : int
        Image width.

    Returns
    -------
    bool
        True if the box is plausible in XYXY format.
    """
    x1, y1, x2, y2 = box
    # Check for basic validity: coordinates are ordered correctly and within image bounds.
    # We use a small tolerance for floating point comparisons.
    return 0 <= x1 < x2 and 0 <= y1 < y2 and x2 <= w + 1e-4 and y2 <= h + 1e-4


def _is_plausible_xywh(box: tuple[float, ...], h: int, w: int) -> bool:
    """
    Check if a box is plausible in XYWH format.

    Parameters
    ----------
    box : tuple[float, ...]
        Bounding box coordinates.
    h : int
        Image height.
    w : int
        Image width.

    Returns
    -------
    bool
        True if the box is plausible in XYWH format.
    """
    x, y, box_w, box_h = box
    # Width and height must be positive, and the box must be within image bounds.
    return box_w > 0 and box_h > 0 and x >= 0 and y >= 0 and (x + box_w) <= w + 1e-4 and (y + box_h) <= h + 1e-4


def _is_plausible_cxcywh(box: tuple[float, ...], h: int, w: int) -> bool:
    """
    Check if a box is plausible in CXCYWH format.

    Parameters
    ----------
    box : tuple[float, ...]
        Bounding box coordinates.
    h : int
        Image height.
    w : int
        Image width.

    Returns
    -------
    bool
        True if the box is plausible in CXCYWH format.
    """
    cx, cy, box_w, box_h = box
    # Width and height must be positive.
    if box_w <= 0 or box_h <= 0:
        return False
    # Calculate corner coordinates and check if they are within image bounds.
    x1 = cx - box_w / 2
    y1 = cy - box_h / 2
    x2 = cx + box_w / 2
    y2 = cy + box_h / 2
    return x1 >= 0 and y1 >= 0 and x2 <= w + 1e-4 and y2 <= h + 1e-4


def _check_if_normalized(all_bboxes: list[list[tuple[float, float, float, float]]]) -> bool:
    """
    Check if bounding box coordinates are normalized (all values <= 1.0).

    Parameters
    ----------
    all_bboxes : list[list[tuple[float, float, float, float]]]
        List of lists of bounding boxes for each image.

    Returns
    -------
    bool
        True if coordinates appear to be normalized.
    """
    max_coords_to_check = 100
    coords_checked = 0

    for bbox_list in all_bboxes:
        for box in bbox_list:
            for coord in box:
                if coord > 1.0:
                    return False
                coords_checked += 1
                if coords_checked > max_coords_to_check:
                    return True
    return True


def _scale_box_if_normalized(
    box: tuple[float, float, float, float], is_normalized: bool, h: int, w: int
) -> tuple[float, float, float, float]:
    """
    Scale bounding box coordinates to image dimensions if normalized.

    Parameters
    ----------
    box : tuple[float, float, float, float]
        Bounding box coordinates.
    is_normalized : bool
        Whether the coordinates are normalized.
    h : int
        Image height.
    w : int
        Image width.

    Returns
    -------
    tuple[float, float, float, float]
        Scaled bounding box coordinates.
    """
    return (box[0] * w, box[1] * h, box[2] * w, box[3] * h) if is_normalized else box


def _get_format_checkers() -> dict[BoundingBoxFormat, Callable]:
    """
    Get mapping of bounding box formats to their plausibility checker functions.

    Returns
    -------
    dict[BoundingBoxFormat, Callable]
        Dictionary mapping formats to checker functions.
    """
    return {
        BoundingBoxFormat.XYXY: _is_plausible_xyxy,
        BoundingBoxFormat.XYWH: _is_plausible_xywh,
        BoundingBoxFormat.CXCYWH: _is_plausible_cxcywh,
    }


def _filter_implausible_formats(
    possible_formats: set[BoundingBoxFormat],
    box: tuple[float, float, float, float],
    h: int,
    w: int,
    format_checkers: dict[BoundingBoxFormat, Callable],
) -> set[BoundingBoxFormat]:
    """
    Filter out implausible formats for a given bounding box.

    Parameters
    ----------
    possible_formats : set[BoundingBoxFormat]
        Set of currently possible formats.
    box : tuple[float, float, float, float]
        Bounding box coordinates to check.
    h : int
        Image height.
    w : int
        Image width.
    format_checkers : dict[BoundingBoxFormat, Callable]
        Dictionary mapping formats to checker functions.

    Returns
    -------
    set[BoundingBoxFormat]
        Filtered set of possible formats.
    """
    return {fmt for fmt in possible_formats if format_checkers[fmt](box, h, w)}


def _handle_detection_result(possible_formats: set[BoundingBoxFormat], is_normalized: bool) -> BoundingBoxFormat:
    """
    Handle the final result of format detection.

    Parameters
    ----------
    possible_formats : set[BoundingBoxFormat]
        Set of remaining possible formats.
    is_normalized : bool
        Whether coordinates are normalized.

    Returns
    -------
    BoundingBoxFormat
        The detected format or UNKNOWN if detection failed.
    """
    if len(possible_formats) == 1:
        final_format = possible_formats.pop()
        return final_format.to_normalized(is_normalized)

    if len(possible_formats) > 1:
        print(f"Ambiguous result. Remaining possible formats: {possible_formats}")

    return BoundingBoxFormat.UNKNOWN


def detect_bbox_format(
    all_bboxes: list[list[tuple[float, float, float, float]]], image_shapes: list[tuple[int, ...]]
) -> BoundingBoxFormat:
    """
    Detect the bounding box format by checking for plausibility against image dimensions.

    Parameters
    ----------
    all_bboxes : list[list[tuple[float, float, float, float]]]
        A list of lists of bounding boxes. Each inner list corresponds to an image.
    images : list[np.ndarray]
        A list of images as numpy arrays in CHW (Channels, Height, Width) format.

    Returns
    -------
    BoundingBoxFormat
        An enum indicating the detected format (e.g., "XYXY", "NORMALIZED_XYWH"),
        or "UNKNOWN" if detection is inconclusive.
    """
    # Early exit for empty inputs
    if not all_bboxes or not any(all_bboxes) or not image_shapes:
        return BoundingBoxFormat.UNKNOWN

    # Check if coordinates are normalized
    is_normalized = _check_if_normalized(all_bboxes)

    # Initialize possible formats and format checkers
    possible_formats = {BoundingBoxFormat.XYXY, BoundingBoxFormat.XYWH, BoundingBoxFormat.CXCYWH}
    format_checkers = _get_format_checkers()

    # Process each image and its bounding boxes
    for i, (bboxes, image_shape) in enumerate(zip(all_bboxes, image_shapes)):
        if not bboxes:
            continue

        h, w = image_shape[-2], image_shape[-1]

        for box in bboxes:
            # Early exit if only one format remains
            if len(possible_formats) <= 1:
                break

            # Scale box coordinates if normalized
            box_to_check = _scale_box_if_normalized(box, is_normalized, h, w)

            # Filter out implausible formats
            possible_formats = _filter_implausible_formats(possible_formats, box_to_check, h, w, format_checkers)

        # Early exit if we've narrowed it down
        if len(possible_formats) <= 1:
            break

    return _handle_detection_result(possible_formats, is_normalized)
