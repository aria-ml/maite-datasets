"""Module for MAITE compliant Multi-Object Tracking datasets."""

from maite_datasets._base import MultiObjectTrackingTargetTuple, SingleFrameObjectTrackingTargetTuple
from maite_datasets._builder import to_multi_object_tracking_dataset
from maite_datasets._video import PyAVVideoStream, VideoFrameTuple, decode_video, frames_to_stream

__all__ = [
    "MultiObjectTrackingTargetTuple",
    "PyAVVideoStream",
    "SingleFrameObjectTrackingTargetTuple",
    "VideoFrameTuple",
    "decode_video",
    "frames_to_stream",
    "to_multi_object_tracking_dataset",
]
