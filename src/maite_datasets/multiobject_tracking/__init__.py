"""Module for MAITE compliant Multi-Object Tracking datasets."""

from maite_datasets._base import MultiobjectTrackingTargetTuple, SingleFrameObjectTrackingTargetTuple
from maite_datasets._builder import to_multiobject_tracking_dataset
from maite_datasets._video import PyAVVideoStream, VideoFrameTuple, decode_video, frames_to_stream
from maite_datasets.multiobject_tracking._visdrone import VisDroneMOT

__all__ = [
    "MultiobjectTrackingTargetTuple",
    "PyAVVideoStream",
    "SingleFrameObjectTrackingTargetTuple",
    "VideoFrameTuple",
    "VisDroneMOT",
    "decode_video",
    "frames_to_stream",
    "to_multiobject_tracking_dataset",
]
