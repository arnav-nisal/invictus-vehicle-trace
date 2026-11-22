"""Vehicle Identification & Tracking Pipeline."""

from .pipeline import VehicleTrackingPipeline
from .config import ROAD_VEHICLE_CLASSES, COCO_TO_CATEGORY

__all__ = ["VehicleTrackingPipeline", "ROAD_VEHICLE_CLASSES", "COCO_TO_CATEGORY"]
