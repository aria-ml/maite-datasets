"""Module for MAITE compliant Object Detection datasets."""

from maite_datasets.object_detection._antiuav import AntiUAVDetection
from maite_datasets.object_detection._coco import COCODatasetReader
from maite_datasets.object_detection._droneswarm import DroneSwarm
from maite_datasets.object_detection._dronevehicle import DroneVehicle
from maite_datasets.object_detection._m3fd import M3FD
from maite_datasets.object_detection._milco import MILCO
from maite_datasets.object_detection._military_aircraft import MilitaryAircraft
from maite_datasets.object_detection._seadrone import SeaDrone
from maite_datasets.object_detection._skysealand import SkySeaLand
from maite_datasets.object_detection._voc import VOCDetection
from maite_datasets.object_detection._yolo import YOLODatasetReader

__all__ = [
    "AntiUAVDetection",
    "DroneSwarm",
    "DroneVehicle",
    "M3FD",
    "MILCO",
    "MilitaryAircraft",
    "SeaDrone",
    "SkySeaLand",
    "VOCDetection",
    "COCODatasetReader",
    "YOLODatasetReader",
]
