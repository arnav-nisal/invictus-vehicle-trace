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
DEFAULT_GRID_PATH = "outputs/tracks_grid.jpg"

# Pipeline defaults tuned for CCTV vehicle tracking
DEFAULT_IMGSZ = 1280             # Full-resolution inference (1280px) for sharp distant/moving two-wheelers
DEFAULT_CONF_THRESHOLD = 0.20    # Low detector threshold to retain motion-blurred vehicles
DEFAULT_IOU_THRESHOLD = 0.70

# Post-tracking filtering defaults (applied AFTER tracking completes)
DEFAULT_MIN_FRAMES = 15          # At 30 FPS, >= 15 frames (~0.5s) to filter transient flickers
DEFAULT_MIN_BOX_AREA = 1200      # Minimum bounding-box area in pixels for kept tracks
DEFAULT_MIN_BOX_DIM = 28         # Minimum bounding-box width/height in pixels

# Tracklet stitching defaults for reconnecting occluded vehicle tracks
DEFAULT_MAX_STITCH_GAP_SEC = 2.5       # Max occlusion gap to stitch (seconds)
DEFAULT_MAX_STITCH_SPEED_PX_S = 450.0   # Max plausible vehicle velocity across CCTV (pixels/sec)
DEFAULT_MIN_HIST_CORREL = 0.65         # Min HSV color histogram correlation
DEFAULT_MAX_LAB_DIST = 35.0            # Max Lab color space Euclidean distance (allows lighting variation while rejecting different vehicles)
DEFAULT_MAX_SIZE_RATIO = 2.1           # Max bounding-box scale ratio for stitching (up to 2.1x for perspective growth)
DEFAULT_MAX_LATERAL_ERR_PX = 65.0      # Max lateral cross-lane error from trajectory projection (pixels)
DEFAULT_MAX_CONSECUTIVE_AREA_JUMP = 2.2 # Max area jump/shrink ratio between consecutive frames before splitting

# GPU Acceleration
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
