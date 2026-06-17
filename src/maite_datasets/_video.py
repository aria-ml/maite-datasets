"""Video input abstractions for the multi-object tracking task.

The MAITE multi-object tracking protocol models an input as a ``VideoStream`` --
an ``Iterable`` of ``VideoFrame`` objects, where each frame exposes ``pixels``
(a ``(C, H, W)`` array), ``time_s``, ``pts``, and ``frame_index``.

This module provides:

* :class:`VideoFrameTuple` -- a lightweight, attribute-accessible frame that
  satisfies :class:`maite.protocols.multiobject_tracking.VideoFrame`.
* :func:`frames_to_stream` -- wrap already-decoded ``(C, H, W)`` frame arrays
  into a list of :class:`VideoFrameTuple` (the in-memory path).
* :class:`PyAVVideoStream` -- a lazy, PyAV-backed stream that decodes frames on
  iteration without holding the whole video in memory (the file-backed path).
  Requires the optional ``av`` dependency.
"""

from __future__ import annotations

__all__ = ["VideoFrameTuple", "frames_to_stream", "PyAVVideoStream", "decode_video"]

from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


# A NamedTuple gives read-only attribute access matching the VideoFrame protocol's
# property surface, mirroring the ObjectDetectionTargetTuple pattern in _base.py.
# Class-based (rather than the functional ``namedtuple``) so the exported fields
# carry type annotations.
class VideoFrameTuple(NamedTuple):
    pixels: NDArray[Any]
    time_s: float
    pts: int
    frame_index: int


def frames_to_stream(frames: Sequence[Any]) -> list[VideoFrameTuple]:
    """Wrap decoded ``(C, H, W)`` frame arrays into a list of :class:`VideoFrameTuple`.

    Timing fields are synthesized for in-memory frames that carry no container
    timing: ``frame_index`` and ``pts`` are set to the positional index and
    ``time_s`` to ``float(index)``. Use :class:`PyAVVideoStream` when real
    presentation timestamps are required.

    Parameters
    ----------
    frames : Sequence[ArrayLike]
        Decoded frames, each a ``(C, H, W)`` array.

    Returns
    -------
    list[VideoFrameTuple]
    """
    return [VideoFrameTuple(pixels=frame, time_s=float(i), pts=i, frame_index=i) for i, frame in enumerate(frames)]


def _av_frame_to_tuple(frame: Any, frame_index: int, time_base: Any) -> VideoFrameTuple:
    """Convert a decoded PyAV frame to a :class:`VideoFrameTuple`.

    PyAV yields HWC RGB; the pixels are transposed to the ``(C, H, W)`` the
    protocol expects, and ``pts``/``time_s`` are derived from the stream ``time_base``.
    """
    pixels: NDArray[Any] = np.transpose(frame.to_ndarray(format="rgb24"), (2, 0, 1))
    pts = 0 if frame.pts is None else int(frame.pts)
    time_s = 0.0 if frame.pts is None or time_base is None else float(frame.pts * time_base)
    return VideoFrameTuple(pixels=pixels, time_s=time_s, pts=pts, frame_index=frame_index)


class PyAVVideoStream:
    """Lazy, PyAV-backed :class:`VideoStream` that decodes frames on iteration.

    Each iteration opens the container, yields :class:`VideoFrameTuple` objects
    with real ``pts`` and ``time_s`` derived from the stream ``time_base``, and
    closes the container when exhausted. The stream is re-iterable: every
    ``iter()`` decodes from the start, so the full video is never held in memory.

    Requires the optional ``av`` dependency (``pip install maite-datasets[av]``).

    Parameters
    ----------
    path : str or Path
        Path to the video file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)

    def __iter__(self) -> Iterator[VideoFrameTuple]:
        try:
            import av
        except ImportError:
            raise ImportError(
                "PyAV is not installed. Install it with `pip install maite-datasets[av]` to decode video files."
            )

        with av.open(self._path) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base
            for frame_index, frame in enumerate(container.decode(stream)):
                yield _av_frame_to_tuple(frame, frame_index, time_base)


def decode_video(path: str | Path) -> tuple[list[VideoFrameTuple], dict[str, Any]]:
    """Eagerly decode a video file via PyAV into frames plus container metadata.

    Use :class:`PyAVVideoStream` instead when frames should be decoded lazily.

    Parameters
    ----------
    path : str or Path
        Path to the video file.

    Returns
    -------
    tuple[list[VideoFrameTuple], dict[str, Any]]
        Decoded frames and a metadata dict carrying ``height``, ``width``,
        ``time_base`` (:class:`fractions.Fraction`), and ``size`` (bytes) --
        the fields of the MAITE multi-object tracking ``DatumMetadata``.
    """
    try:
        import av
    except ImportError:
        raise ImportError(
            "PyAV is not installed. Install it with `pip install maite-datasets[av]` to decode video files."
        )

    frames: list[VideoFrameTuple] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        height = int(stream.height)
        width = int(stream.width)
        for frame_index, frame in enumerate(container.decode(stream)):
            frames.append(_av_frame_to_tuple(frame, frame_index, time_base))

    metadata = {
        "height": height,
        "width": width,
        "time_base": Fraction(time_base) if time_base is not None else Fraction(0, 1),
        "size": Path(path).stat().st_size,
    }
    return frames, metadata
