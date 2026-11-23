# Changelog — Invictus Vehicle Trace (Stage 1)

Chronological record of repository setup, pipeline iterations, engineering bugfixes, and multi-clip verifications for Hack-a-Sol 5.0 (AIML-01 track).

---

## [2026-11-22] — Project Initialization & Environment Setup

### 2026-11-22 22:30:00 IST — Repository Initialization
- Initialized Git repository for Team Invictus (`invictus-vehicle-trace`).
- Created virtual environment `.venv` with core dependencies: Python 3.12, Ultralytics YOLO, OpenCV, NumPy, and PyYAML.
- Created `.gitignore` to prevent tracking of large video assets, virtual environments, and intermediate files.

### 2026-11-22 23:45:00 IST — Initial Vehicle Tracking Baseline
- Implemented first end-to-end tracking pipeline in `cv_pipeline/`:
  - `config.py`: Road vehicle class mappings (`ROAD_VEHICLE_CLASSES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}`).
  - `botsort_reid.yaml`: Initial BoT-SORT configuration.
  - `crop_selector.py`: Laplcian sharpness-scoring best-crop selector.
  - `pipeline.py`: Video capture, YOLO tracking, frame-by-frame annotation, and JSON summary export.
- Executed on `data/clips/activa_signal.mp4` on CPU.
- Identified baseline limitations: CPU processing was slow ($\sim 2.8$ FPS) and target Activa split into multiple IDs when blocked by cross-traffic.

---

## [2026-11-23] — Pipeline Optimization, Stitching Architecture & Multi-Clip Hardening

### 2026-11-23 03:15:00 IST — NVIDIA GPU Acceleration & CUDA Integration
- Reinstalled PyTorch and Torchvision inside `.venv` with CUDA 12.4 support (`torch==2.6.0+cu124`).
- Verified hardware detection: **NVIDIA GeForce RTX 3050 Laptop GPU** (`device=0`).
- Switched pipeline default to `device=0`, achieving $15\times$ speedup ($\sim 50$ FPS).

### 2026-11-23 03:45:00 IST — High-Resolution Inference & Clutter Filtering
- Increased YOLO inference resolution from default 640px to `imgsz=1280`.
- Lowered detector confidence input into tracker to `conf=0.20` to retain motion-blurred two-wheelers.
- Implemented post-tracking spatiotemporal clutter filter (`min_frames=15`, `min_box_area=1200px`, `min_box_dim=28px`) to prune transient background detections.
- Tuned BoT-SORT configuration for static CCTV: disabled Camera Motion Compensation (`gmc_method: none`) and extended memory window (`track_buffer: 90`).

### 2026-11-23 04:15:00 IST — Workspace Skills Installation
- Installed 7 specialized engineering skills into `.agent/skills/`:
  - `data-scientist`
  - `machine-learning-ops-ml-pipeline`
  - `systematic-debugging`
  - `writing-plans`
  - `pdf`
  - `backend-architect`
  - `antigravity-design-expert`
- Added `.agent/` to `.gitignore`.

### 2026-11-23 04:45:00 IST — Tracklet Stitcher Introduction (`stitcher.py`)
- Designed and built [cv_pipeline/stitcher.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/stitcher.py):
  - Spatiotemporal association across occlusion gaps up to 2.5s (`DEFAULT_MAX_STITCH_GAP_SEC = 2.5`).
  - Physical velocity constraint ($\le 450\text{ px/s}$).
  - Two-level appearance matching using 2D HSV normalized color histograms and Lab color Euclidean distance.
  - Disjoint-set union (DSU) graph clustering to consolidate multi-segment vehicle lifecycles into a single Master ID.

### 2026-11-23 05:15:00 IST — Test Clip Acquisition & Multi-Scenario Benchmarking
- Added [scripts/download_clips.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/scripts/download_clips.py) to fetch real-world traffic clips:
  - `intersection_traffic.mp4` (dense multi-lane intersection).
  - `highway_traffic.mp4` (fast multi-lane highway with overhead gantry).
  - `full_intersection.mp4` (full stress test video).
- Benchmarked pipeline across clips, uncovering edge cases: identity jumping on adjacent vehicles in dense lanes and overhead bar track splitting on highway.

### 2026-11-23 05:45:00 IST — Pre-Stitch Jump Splitter & Neighbor-Jump Protection
- Implemented [cv_pipeline/splitter.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/splitter.py):
  - Frame-by-frame anomaly detection before stitching.
  - Strict splitting if detected class switches between `two-wheeler` and `larger-vehicle`.
  - Tightened BoT-SORT matching threshold from 0.85 down to `match_thresh = 0.72`.
- Implemented automated contact-sheet grid generator ([cv_pipeline/grid_generator.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/grid_generator.py)) to export `tracks_grid.jpg` for visual QA.

### 2026-11-23 06:15:00 IST — Perspective Scale Growth Tuning (Activa Zero-Regression)
- Relaxed post-tracking stitcher bounding-box scale ratio limit to $2.1\times$ (`DEFAULT_MAX_SIZE_RATIO = 2.1`) for same-category vehicles.
- Verified on `data/clips/activa_signal.mp4`:
  - Moving white Activa consolidated into single **Track 27** across entire crossing (`00:00:02.100` to `00:00:08.733`, merged from `[27, 38, 51]`).
  - Stationary white Activa maintained **Track 4** across all 300 frames.

### 2026-11-23 06:45:00 IST — Overhead Bar Resilience & Trajectory Projection
- Deep root-cause debugging on `highway_traffic.mp4` (Track pairs 39/66, 61/93, 49/95, 55/90, 63/56):
  - In `splitter.py`: Added 1-frame dip/recovery resilience so momentary bounding-box occlusions under thin overhead bars do not split tracks.
  - In `stitcher.py`: Introduced `extract_central_body` ($[0.18, 0.82] \cdot H, [0.12, 0.88] \cdot W$) to eliminate roof bar and shadow distortion.
  - Implemented forward travel velocity vector projection and cross-lane lateral error enforcement (`lateral_err <= 65.0px`).
  - Added footprint centroid shift allowance ($\ge 60.0$px) to prevent bounding-box boundary expansion from tripping velocity ceilings.
- Verified on `highway_traffic.mp4`: All target cars stitched into single IDs with zero cross-lane swaps.
- Verified zero regression on `activa_signal.mp4`.

### 2026-11-23 07:00:00 IST — Clean Benchmark on simple_traffic_30s.mp4
- Prepared 1080p 30 FPS test clip `data/clips/simple_traffic_30s.mp4` (900 frames) from 4K source.
- Executed existing pipeline as-is with zero logic changes:
  - 144 clean vehicles tracked across 4 categories.
  - 57 merge events consolidated into 34 continuous multi-segment tracks.
  - Clean video, summary JSON, and contact sheet generated in `outputs/simple_traffic/`.

### 2026-11-23 07:15:00 IST — Living Documentation Suite & Git Preparation
- Created living documentation:
  - `challanges.md`: Full catalog of engineering obstacles.
  - `solutions.md`: Detailed engineering solutions, formulas, and verification metrics.
  - `changes.md`: Chronological changelog with timestamps.
- Updated `README.md` with architecture diagrams, usage guides, and complete Hack-a-Sol 5.0 AI and open-source compliance disclosures.
- Verified `.gitignore` and prepared staged Git tree for manual commit and push.

### 2026-11-23 20:30:00 IST — Stage 2: Vehicle Selection & Full-Track Crop Extraction Architecture
- Dropped contact-sheet grid (`tracks_grid.jpg`) generation in favour of full-track chronological crop extraction and frame-by-frame history.
- Upgraded `cv_pipeline/pipeline.py` to export `outputs/track_history.json`, storing frame-by-frame bounding boxes, timestamps, confidences, and areas for every kept vehicle.
- Created `cv_pipeline/crop_extractor.py`:
  - `extract_vehicle_crops`: Pulls every video frame in which a chosen vehicle was active, applies a $5\%$ safety margin padding around bounding boxes to prevent bumper plate clipping, and writes chronological crops `frame_XXXXXX_crop.jpg`.
  - `crop_manifest.json`: Generates a structured JSON metadata manifest containing crop indices, original/padded bounding boxes, dimensions, confidences, and timestamps.
  - `interactive_vehicle_selection`: Displays detected vehicle summary table, annotated video path, and prompts user for target ID (or accepts `--target-id`).
- Upgraded `scripts/run_tracking.py`:
  - Added flags: `--target-id`, `--interactive`, `--reuse-tracking`, and `--padding`.
  - Added instantaneous `--reuse-tracking` mode loading cached tracking and history data in $<50$ms.
- Verified on `data/clips/activa_signal.mp4` with Target ID 27 (crossing white Activa):
  - Extracted all 98 consecutive crops into `outputs/activa_signal/track_027/vehicle_crops/`.
  - Saved manifest `outputs/activa_signal/track_027/crop_manifest.json`.
