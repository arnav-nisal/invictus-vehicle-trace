"""Configuration and class mappings for vehicle tracking."""

import torch
from pathlib import Path

# Target COCO road vehicle classes
# 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
ROAD_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

ROAD_VEHICLE_CLASS_IDS = list(ROAD_VEHICLE_CLASSES.keys())

# Category mapping for high-level grouping
COCO_TO_CATEGORY = {
    "bicycle": "two-wheeler",
    "motorcycle": "two-wheeler",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}

# Default paths
DEFAULT_MODEL_PATH = "yolo26s.pt"
DEFAULT_TRACKER_CONFIG = str(Path(__file__).parent / "botsort_reid.yaml")
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TRACKS_DIR = "outputs/tracks"
DEFAULT_SUMMARY_PATH = "outputs/tracks_summary.json"

# Pipeline defaults tuned for CCTV vehicle tracking
DEFAULT_MIN_FRAMES = 15          # At 30 FPS, 15 frames = 0.5s to filter transient flickers
DEFAULT_CONF_THRESHOLD = 0.40    # Raised to eliminate low-confidence background clutter
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_MIN_BOX_AREA = 1200      # Minimum area in pixels to ignore tiny distant background noise
DEFAULT_MIN_BOX_DIM = 28         # Minimum width and height in pixels

# GPU Acceleration
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
