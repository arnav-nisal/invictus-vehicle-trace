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

### 2026-11-23 21:20:00 IST — Stage 2: Lossless Dual-Folder Crops, Adaptive Lanczos Upscaling & High-Res Source Mapping
- Upgraded `cv_pipeline/crop_extractor.py` and `scripts/run_tracking.py`:
  - **Lossless Dual-Folder Output (.png)**: Replaced lossy `.jpg` with `.png`. Split crops into `vehicle_crops_raw/` (untouched pixels with $8\%$ safety margin padding) and `vehicle_crops_enhanced/` (upscaled & clarity-enhanced).
  - **Adaptive Lanczos4 Upscaling**: For raw crops whose longer side is $<480\text{px}$, adaptively upscaled using `cv2.INTER_LANCZOS4` to reach $480\text{px}$ (capped at $4.0\times$ maximum scale factor). Crops $\ge 480\text{px}$ preserved at $1.0\times$.
  - **Controlled Blur & Clarity Enhancement**:
    - Edge-preserving bilateral filter (`d=5, sigmaColor=30, sigmaSpace=30`) cleans video compression noise without smearing text edges.
    - Luminance-only CLAHE (`clipLimit=1.8, tileGridSize=(4, 4)` on LAB $L$-channel) enhances shadowed plates without shifting vehicle paint color.
    - Controlled unsharp mask (`1.35 * img - 0.35 * blurred` with $\sigma=1.0$) sharpens character edges without halo ringing.
  - **High-Resolution Source Mapping (`--crop-video`)**: Implemented normalized coordinate mapping `(x/W_track, y/H_track)` and timestamp alignment so tracking runs on fast 1080p proxies while crops are extracted from original 4K/high-res source footage.
  - **Quality & Resolution Metrics**: Upgraded `crop_manifest.json` with `raw_width`, `raw_height`, `scale_factor`, `enhanced_width`, `enhanced_height`, `sharpness_score` (Laplacian variance), and `quality_flag` ("usable" vs "too_small").
- Verified on `data/clips/activa_signal.mp4` (Track 27):
  - 98 lossless `.png` crops extracted into `vehicle_crops_raw/` and `vehicle_crops_enhanced/`.
  - 95 / 98 usable frames, average sharpness 3159.0, upscaled up to $4.0\times$ (reaching up to $326\times 480$).
- Verified on `data/clips/simple_traffic_30s.mp4` with `--crop-video data/clips/simple_traffic.mp4` (Track 2):
  - Successfully mapped 1080p tracking bounding boxes onto 4K UHD ($3840\times 2160$) source video.
  - Extracted 202 lossless crops directly from 4K source video into `outputs/simple_traffic_30s/track_002/`.
  - 202 / 202 usable frames, adaptive scale factors up to $3.66\times$ reaching $480\times 410$.

### 2026-11-23 21:45:00 IST — Stage 2: Three-Tier Crop Architecture (Raw, Unenhanced, Enhanced)
- Upgraded `cv_pipeline/crop_extractor.py` to output three lossless `.png` directories inside `outputs/<clip_name>/track_<ID>/`:
  - `vehicle_crops_raw/`: Untouched original video crop at native resolution ($8\%$ safety padding).
  - `vehicle_crops_unenhanced/`: Enlarged using adaptive Lanczos4 upscaling (minimum longer side $480\text{px}$, capped at $4.0\times$) with zero post-processing applied.
  - `vehicle_crops_enhanced/`: Enlarged with exact matching dimensions, plus bilateral denoise + luminance CLAHE + unsharp masking.
- Upgraded `crop_manifest.json` schema:
  - Added `unenhanced_crops_dir` in the manifest header.
  - Added `unenhanced_path` and `unenhanced_crop_path` for every frame entry alongside `raw_path` and `enhanced_path`.
- Verified on `data/clips/activa_signal.mp4` (Track 27):
  - Generated all 3 folders cleanly with 98 lossless `.png` images each (294 total images).
  - Confirmed $100\%$ dimension parity between `vehicle_crops_unenhanced/` and `vehicle_crops_enhanced/` across all frames (e.g., Frame 63: $184\times 236$, Frame 214: $248\times 480$, Frame 238: $326\times 480$).

### 2026-11-23 23:25:00 IST — Stage 2.2: Dual-Detector License Plate Cropping & Dynamic Readability Selection
- Designed and built [cv_pipeline/plate_detector.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/plate_detector.py) and integrated into [scripts/run_tracking.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/scripts/run_tracking.py):
  - **Dual Open-Source Detectors**:
    - Detector A: Custom fine-tuned Indian plate model (`models/yolo26s_plate_best.pt`, trained on `kedarsai/indian-license-plates-with-labels`, Ultralytics YOLO26s architecture, run on CUDA `device=0`, `conf=0.15`).
    - Detector B: Global MIT-licensed ONNX model (`yolo-v9-s-608-license-plate-end2end` loaded via `open-image-models`, `conf_thresh=0.18`).
  - **Spatial Cross-Verification**:
    - Calculates IoU overlap for each frame: if $\text{IoU} \ge 0.35$, computes confidence-weighted average bounding box tagged `DUAL_VERIFIED` ($1.15\times$ readability bonus).
    - If only one detector fires, preserves detection tagged `SINGLE_CUSTOM_YOLO26` or `SINGLE_GLOBAL_YOLOV9`.
  - **Smart 12% Padding & Paired Lossless Cropping**:
    - Expands winning plate bounding box by $12\%$ horizontally and $12\%$ vertically (clamped to boundaries) to ensure edge characters and two-line Indian scooter/auto plates are never clipped.
    - Slices matching coordinates from both `vehicle_crops_unenhanced/` and `vehicle_crops_enhanced/` to produce lossless `.png` pairs with $100\%$ dimension parity.
    - Saves all plate crops to `plate_crops_all/unenhanced/` and `plate_crops_all/enhanced/`.
  - **Dynamic Readability Gate (Top ~60 Paired Plates)**:
    - Calculates native (un-upscaled) dimensions: $\text{native\_w} = \text{w} / \text{scale\_factor}$, $\text{native\_h} = \text{h} / \text{scale\_factor}$.
    - Hard-rejects micro-plates: native width $< 18\text{px}$, native height $< 8\text{px}$, or cropped width $< 40\text{px}$.
    - Calculates composite readability score based on unenhanced Laplacian variance, native plate area, contrast ($\sigma$), detector confidence, and cross-verification status.
    - Keeps all candidate plates scoring $\ge 45\%$ of peak score, capped at 60 evenly spaced frames.
    - Saves prioritized plates to `plate_crops_selected/unenhanced/` and `plate_crops_selected/enhanced/`.
  - **Audit Manifest**: Exports comprehensive `plate_manifest.json` with per-detector statistics, readability scores, and paths.
- Verified on 2 distinct test tracks:
  - **Track 27 (Scooter)** from `data/clips/activa_signal.mp4`:
    - 98 vehicle crops processed.
    - Custom YOLO26s: 67 frames | Global YOLOv9-s: 57 frames | Dual Verified: 51 frames ($\text{IoU} \ge 0.35$).
    - Total detected: 72 | Rejected micro-plates: 3 | Readable: 69.
    - Peak Readability Score: 2290.6 | Dynamic threshold: 1030.8.
    - Selected for OCR: 18 top paired crops saved in `plate_crops_selected/`.
  - **Track 2 (Car)** from `data/clips/simple_traffic_30s.mp4`:
    - 202 vehicle crops processed (from 4K source video `data/clips/simple_traffic.mp4`).
    - Custom YOLO26s: 84 frames | Global YOLOv9-s: 175 frames | Dual Verified: 53 frames ($\text{IoU} \ge 0.35$).
    - Total detected: 179 | Rejected micro-plates: 0 | Readable: 179.
    - Peak Readability Score: 2830.3 | Dynamic threshold: 1273.7.
    - Selected for OCR: 28 top paired crops saved in `plate_crops_selected/`.

---

## [2026-11-24] — Stage 2.2 Hardening: Border-Truncation Guard, Conflict Accounting, p90 Anchor & Pure Lanczos Enlargement

### 2026-11-24 00:40:00 IST — Plate Detector Hardening & Verification Suite (`cv_pipeline/plate_detector.py`)
- **1. Border-Truncation Guard (Fixes Cut-Off Exit Frames)**:
  - Added spatial boundary check rejecting detections whose unpadded plate box comes within $\le 2$ native pixels of vehicle crop edges or within $\le 6$ pixels of the original video frame's outer boundary.
  - Implemented exit aspect-ratio drop check: computes track median plate aspect ratio ($\text{median\_ar} = \text{median}(w/h)$); if a plate is within $5\%$ of any video/crop boundary and its aspect ratio drops below $0.75 \times \text{median\_ar}$, it is rejected.
  - Tagged rejected boundary plates with `rejection_reason: "border_truncated"` in `plate_manifest.json` and excluded them from `plate_crops_selected/`.
- **2. Balanced Conflict Accounting & Plate Geometry Filter**:
  - Implemented `is_valid_plate_geometry`: filters obvious non-plate candidate boxes with aspect ratio outside $[0.85, 6.5]$ or box area $> 45\%$ of crop area before conflict resolution.
  - Added explicit conflict resolution statuses: `CONFLICT_RESOLVED_CUSTOM` and `CONFLICT_RESOLVED_GLOBAL` for non-overlapping dual detections ($\text{IoU} < 0.35$).
  - Guaranteed $100\%$ exact mathematical balance in manifest and summary table:
    - $\text{custom\_yolo26\_detections} = \text{dual\_verified} + \text{single\_custom} + \text{conflict\_resolved\_custom} + \text{conflict\_resolved\_global}$
    - $\text{global\_yolov9\_detections} = \text{dual\_verified} + \text{single\_global} + \text{conflict\_resolved\_custom} + \text{conflict\_resolved\_global}$
  - Captured 5-frame diagnostic for Track 2 conflict frames demonstrating that Custom YOLO26 predicted spurious full-body regions ($>45\%$ crop area) that are cleanly caught and resolved.
- **3. Outlier-Sensitive Dynamic Readability Gate (p90 Anchor)**:
  - Replaced single-frame max anchor with the 90th percentile ($p90$) of passing readability scores: $\text{threshold} = 0.45 \times p90\_score$.
  - Increased cross-verification bonus from $1.15\times$ to $1.25\times$ for `DUAL_VERIFIED` frames over single-detector frames.
  - Prevents single glare/noise outliers from filtering out dozens of clear, readable plate crops.
- **4. Pure Unenhanced Plate Enlargement (Lanczos4)**:
  - Calculated native plate dimensions and Laplacian sharpness variance on untouched raw unenhanced crops before enlargement.
  - Slices small plate crops cleanly using `cv2.INTER_LANCZOS4` (preserving aspect ratio) to guarantee minimum width of $320\text{px}$ or minimum height of $120\text{px}$ (capped at $4.0\times$ zoom; already-large plates kept at $1.0\times$).
  - Applied zero extra image enhancements or filters during resize (no extra CLAHE, bilateral filter, or unsharp mask).
  - Maintained $100\%$ dimension parity between `plate_crops_all/unenhanced/` and `plate_crops_all/enhanced/`.
  - Added `plate_crop_raw_size`, `plate_upscale_factor`, and `plate_saved_size` to `plate_manifest.json`.
- **5. Re-Run & Verification Results**:
  - **Track 27 (`activa_signal.mp4`)**:
    - Processed 98 crops | Detected: 63 frames (Custom: 58, Global: 55).
    - Dual Verified: 50 | Single Custom: 8 | Single Global: 5 | Conflict: 0.
    - $100\%$ Math Balance: $50 + 8 + 0 + 0 = 58$ (Custom) and $50 + 5 + 0 + 0 = 55$ (Global).
    - Border-Truncated Rejected: 3 frames (Frames 255, 256, 260 cleanly dropped).
    - Last Selected Frame: **Frame 254** ($102\times 53$ raw enlarged $2.26\times$ to $231\times 120$), with plate 100% fully intact.
    - p90 Readability Score: 1774.6 | Threshold: 798.5 | Selected for OCR: 26 frames.
  - **Track 2 (`simple_traffic_30s.mp4`)**:
    - Processed 202 crops | Detected: 184 frames (Custom: 76, Global: 177).
    - Dual Verified: 54 | Single Custom: 7 | Single Global: 108 | Conflict Custom: 7 | Conflict Global: 8.
    - $100\%$ Math Balance: $54 + 7 + 7 + 8 = 76$ (Custom) and $54 + 108 + 7 + 8 = 177$ (Global).
    - Border-Truncated Rejected: 8 frames (Frame 187 clipped by bottom video edge cleanly rejected).
    - Last Selected Frame: **Frame 186** ($130\times 37$ raw enlarged $2.46\times$ to $320\times 91$), with plate 100% fully intact.
    - p90 Readability Score: 1923.6 | Threshold: 865.6 | Selected for OCR: 52 frames.

---

## [2026-11-24] — Stage 2.3: Dual-Pipeline Cross-Checked OCR & 3-Bucket Slot Arbitration

### 2026-11-24 01:50:00 IST — Stage 2.3 OCR Engine Implementation (`cv_pipeline/ocr_engine.py`, `scripts/run_ocr.py`)
- **1. Dual Local Open-Source OCR Engine Integration**:
  - Implemented local dual OCR architecture using **RapidOCR** (PP-OCRv6 DBNet detection + PP-OCRv5 English text recognition via ONNX Runtime) and **PARSeq** (`baudm/parseq`, PyTorch CUDA 12.4 on NVIDIA RTX 3050).
  - Explicitly evaluated and rejected `fast-plate-ocr` due to its hardcoded 9-character output sequence ceiling, which breaks standard 10-character Indian license plates (`SS DD LL DDDD`).
  - Implemented empty-detection recognition fallback and two-line Indian plate splitting (plates with aspect ratio $< 2.2$ split into top $0\text{--}58\%$ and bottom $42\text{--}100\%$ strips).
- **2. Dual Parallel Complementary Streams**:
  - **Pipeline A (Unenhanced)**: RapidOCR primary + PARSeq cross-check on natural crops (`plate_crops_selected/unenhanced/`).
  - **Pipeline B (Enhanced)**: PARSeq primary + RapidOCR cross-check on contrast-controlled crops (`plate_crops_selected/enhanced/`).
- **3. Anti-Hallucination 3-Bucket Slot Arbitration**:
  - **LOCKED**: Both pipelines agree on character candidate with effective combined confidence $\ge 0.65$ across $\ge 2$ supporting frames (or 1 frame with $\ge 0.85$); marked with `vlm_action: "TRIPLE_CHECK"`.
  - **CONTRADICTORY**: Pipeline disagreement or confidence in $[0.38, 0.65)$; marked with `vlm_action: "TIE_BREAK"`.
  - **UNREADABLE**: Combined confidence $< 0.38$ or frame coverage $< 15\%$; emits `'?'` and strictly forbids VLM guessing (`vlm_action: "FORBIDDEN_KEEP_UNKNOWN"`).
- **4. Calibrated Multi-Frame Confidence Bugfix**:
  - Fixed confidence accounting bug where vote share alone normalized unanimous blurry single-frame reads into false $1.0$ confidence.
  - Tracked both weighted vote share (`vote_share`) and weighted mean character confidence (`mean_char_conf`) per slot, computing $\text{effective\_conf} = \text{vote\_share} \times \text{mean\_char\_conf}$.
  - Confirmed via unit tests that unanimous mid-confidence reads ($\sim 0.50$) land in `CONTRADICTORY` and unanimous low-confidence reads ($\sim 0.25$) land in `UNREADABLE ('?')`.
- **5. Stage 2.4 VLM Evidence Pack Packaging**:
  - Automatically isolates and packages the top 4 highest-weight distinct frame plate crops with verified on-disk paths into `vlm_evidence_pack` inside `ocr_result.json`.
- **6. Verification Results**:
  - **Unit Test Suite (`tests/test_ocr_arbitration.py`)**: 40/40 tests passing ($100\%$).
  - **Track 27 (`activa_signal.mp4`, 26 selected frames)**:
    - Recovered plate: **`UP15BE5580`** (Uttar Pradesh, 10-char template).
    - State code valid: **True** (`UP`).
    - Overall confidence: **$0.6989$**.
    - Slots: **7 LOCKED**, **3 CONTRADICTORY**, **0 UNREADABLE**.
    - Top raw frame read: Frame 253 (`UP15BE5580`, weight $1.3856$).
    - Runtime: **$13.1\text{s}$** on RTX 3050 GPU.
  - **Track 2 (`simple_traffic_30s.mp4`, 52 selected frames)**:
    - Recovered plate: **`????BG718?`** (10-char template, core series and sequence identified).
    - Overall confidence: **$0.4274$**.
    - Slots: **0 LOCKED**, **5 CONTRADICTORY**, **5 UNREADABLE**.
    - Top raw frame read: Frame 184 (`KH01BG7182`, weight $1.1894$) / Frame 182 (`MH01BG7184`, weight $1.1888$).
    - Runtime: **$17.5\text{s}$** on RTX 3050 GPU.

---

### 2026-11-24 03:00:00 IST — Stage 2.4: Blind Multi-Tier VLM Triple-Check & Tie-Breaker (`cv_pipeline/vlm_verifier.py`, `scripts/run_vlm.py`)
- **1. 3-Tier Zero-Paid-API OpenAI-Compatible Fallback Architecture**:
  - Implemented 3-tier fallback chain via standard `openai` Python SDK:
    - **Tier 1 (Groq)**: `qwen/qwen3.8-27b`, low reasoning effort, hidden reasoning format, strict JSON response format, `max_tokens=600` for output token rate limit protection.
    - **Tier 2 (NVIDIA Build NIM)**: `nvidia/nemotron-3.5-super-vl-120b-a12b`, instruct mode (`enable_thinking: False`), zero-temperature JSON extraction, `max_tokens=600`.
    - **Tier 3 (Local Ollama)**: `qwen3-vl:4b-instruct` / `qwen3-vl:2b-instruct` for offline air-gapped environments.
  - Robust fallback tracing: each tier's attempt, latency, status (`SUCCESS`, `FAILED`, `SKIPPED`), and error details recorded in `vlm_result.json`.
  - Graceful degradation: if all tiers fail or API keys are missing, falls back cleanly to Stage 2.3 OCR arbitration (`vlm_status: "FALLBACK_TO_OCR"`).
- **2. 2x2 Labeled Evidence Grid Synthesis**:
  - `build_evidence_grids`: Takes top 4 distinct frames from `vlm_evidence_pack`.
  - Synthesizes two 2x2 labeled stitched composite grids (`vlm_evidence_unenhanced_grid.png` and `vlm_evidence_enhanced_grid.png` at 640x240px) with per-cell headers displaying Frame Index, Readability Score, and Weight.
  - Sends exactly 2 images in the multimodal payload, preventing rate limits and token bloat.
- **3. Blind Verification Protocol (Zero OCR Text Leakage)**:
  - System and user prompts strictly withhold Stage 2.3 OCR string to eliminate confirmation bias.
  - VLM evaluates layout (`single_line` vs `two_line`) and extracts slot-by-slot characters, confidences, and legibility booleans.
- **4. Slot-by-Slot 3-Bucket Reconciliation**:
  - **LOCKED Slots**: VLM acts as triple-check (`TRIPLE_CHECK_CONFIRMED`). Preserves OCR unless VLM shows $\ge 0.90$ certainty visual disagreement (`VLM_OVERRIDE_LOCKED`).
  - **CONTRADICTORY Slots**: VLM breaks tie between Pipe A and Pipe B (`TIE_BREAK_PIPE_A` / `TIE_BREAK_PIPE_B`), or selects independent candidate on $\ge 0.70$ confidence (`VLM_INDEPENDENT_CHOICE`).
  - **UNREADABLE Slots**: Anti-hallucination guard (`FORBIDDEN_KEEP_UNKNOWN`) preserves `'?'` unless VLM detects crisp character with $\ge 0.85$ confidence (`VLM_RESOLVED_UNREADABLE`).
- **5. Standalone CLI & Pipeline Integration**:
  - Created standalone CLI `scripts/run_vlm.py` with tabular visualization of slot reconciliation.
  - Integrated into `scripts/run_tracking.py` with `--skip-vlm` and `--vlm-provider auto|groq|nvidia|ollama`.
  - Fixed `argparse` literal `%` interpolation bug in `scripts/run_tracking.py`.
- **6. Verification Results**:
  - **Unit Tests (`tests/test_vlm_verifier.py`)**: 6/6 tests passing. Total test suite: 46/46 tests passing ($100\%$).
  - **Track 27 (`activa_signal/track_027`)**:
    - Provider: Groq (`qwen/qwen3.8-27b`) in 3.61s.
    - Layout: `two_line`.
    - Stage 2.3 OCR: `UP15BE5580` $\to$ VLM: `UP15BE5580` $\to$ **Reconciled: `UP15BE5580`** (Valid State Code: `UP` = Uttar Pradesh).
    - Actions: 6 `TRIPLE_CHECK_CONFIRMED`, 3 `TIE_BREAK_PIPE_A`, 1 `LOCKED_VLM_DISPUTED`.
  - **Track 2 (`simple_traffic_30s/track_002`)**:
    - Layout: `single_line`.
    - Stage 2.3 OCR: `????BG718?` (5 unreadable, 5 contradictory).
    - **Reconciled: `????BG718?`** (Valid State Code: False).
    - Actions: 5 `FORBIDDEN_KEEP_UNKNOWN`, 5 `TIE_BREAK_PIPE_A`.

---

### 2026-11-24 03:25:00 IST — Hardening VLM Anti-Hallucination & Override Guards
- **1. Hard-Block `UNREADABLE` Slots (`FORBIDDEN_KEEP_UNKNOWN`)**:
  - Removed `VLM_RESOLVED_UNREADABLE` completely.
  - Any slot marked `UNREADABLE` in Stage 2.3 strictly stays `'?'` in `final_plate`, regardless of VLM self-reported confidence.
  - Preserves VLM's blind guess in `vlm_blind_raw_char` for audit visibility without polluting reconstructed registration.
- **2. Restrict `CONTRADICTORY` Tie-Break to OCR Candidates Only**:
  - Removed `VLM_INDEPENDENT_CHOICE`.
  - On `CONTRADICTORY` slots, the VLM may only confirm Pipe A's candidate or Pipe B's candidate (`TIE_BREAK_PIPE_A` / `TIE_BREAK_PIPE_B`).
  - If the VLM outputs `'?'` or a character not present in OCR candidates, the slot is demoted to `'?'` with action `VLM_ABSTAINED`.
- **3. Protect `LOCKED` Slots from Single-VLM Overrides**:
  - Removed `VLM_OVERRIDE_LOCKED`.
  - If the VLM disagrees with a dual-OCR `LOCKED` slot, the OCR character is preserved and tagged `LOCKED_VLM_DISPUTED` with a $-0.05$ confidence penalty.
- **4. Instant Multi-Tier Failover (`max_retries=0`)**:
  - Configured `max_retries=0` on OpenAI client instances for Groq and NVIDIA NIM to avoid long internal SDK retry sleeps on rate limits, enabling immediate failover across tiers.
- **5. Verification Outcome**:
  - **Track 2 (`simple_traffic_30s/track_002`)**: Reconciled plate correctly outputs **`????BG718?`** with zero hallucinated state code or characters (5 `FORBIDDEN_KEEP_UNKNOWN`, 5 `TIE_BREAK_PIPE_A`).
  - **Track 27 (`activa_signal/track_027`)**: Reconciled plate held firm as **`UP15BE5580`** (6 `TRIPLE_CHECK_CONFIRMED`, 3 `TIE_BREAK_PIPE_A`, 1 `LOCKED_VLM_DISPUTED`).
  - **Unit Tests**: 46/46 unit tests passing ($100\%$).

---

### 2026-11-24 04:15:00 IST — Stage 2.4 Upgrades: Pre-Reconciliation VLM Disambiguation, Per-Slot Ghost Stack & Top-3 Beam Search Predictions
- **1. Pre-Reconciliation VLM Disambiguation (`cv_pipeline/vlm_verifier.py`)**:
  - Wired `_apply_typed_swap` to run against blind VLM slot predictions based on slot typing (`D` vs `L`) before reconciliation.
  - Automatically normalizes common optical ambiguities (`S->5`, `O->0`, `I->1`, `B->8`, `Z->2`, `G->6`, `A->4`, `T->7` on digit slots; `5->S`, `0->O`, `8->B`, etc. on letter slots).
  - Preserves exact raw reading in `vlm_blind_raw_char`, normalized char in `vlm_disambiguated_char`, applies a $0.92\times$ confidence penalty, and logs `vlm_disambiguated: "<raw>-><fixed>"` in `reasoning_notes`.
  - Cleanly converts false optical disputes (such as Track 27 Slot 6 `S` vs `5`) into `TRIPLE_CHECK_CONFIRMED`.
- **2. Per-Slot UI Tile Tone & Deduplicated Ghost Stack**:
  - Assigned visual styling tokens in `reconciled_slots` for frontend rendering:
    - `ui_tile_tone = "green"`: `TRIPLE_CHECK_CONFIRMED`.
    - `ui_tile_tone = "black"`: `TIE_BREAK_PIPE_A`, `TIE_BREAK_PIPE_B`, `LOCKED_VLM_DISPUTED`.
    - `ui_tile_tone = "unknown"`: Unresolved slots (`final_char == '?'`, i.e., `FORBIDDEN_KEEP_UNKNOWN` / `VLM_ABSTAINED`).
  - Computed per-candidate fused soft score:
    $$\text{soft\_score}(c) = 0.45 \cdot \text{conf}_{\text{vlm}}(c) + 0.30 \cdot \text{conf}_{\text{pipe\_a}}(c) + 0.25 \cdot \text{conf}_{\text{pipe\_b}}(c)$$
  - Deduplicated `ghost_stack` (max 3 entries, ordered descending by `soft_score`, pruning candidates $< 0.12$).
  - For green and black tiles, excludes `final_char` and tags remaining alternatives as `kind: "conflict_alt"`. For unknown tiles, includes candidates tagged as `kind: "ghost_fill"`.
- **3. Top-3 Ghost Predictions via Lightweight Beam Search**:
  - Implemented `compute_ghost_predictions` with beam width 8 to compute advisory plate hypotheses for operator hover paths:
    - Green slots: strictly anchored to `final_char`.
    - Black slots: branches across `final_char` and top conflict alternatives.
    - Unknown slots: branches across top ghost-fill candidates.
  - Scoring incorporates average slot soft score plus $+0.08$ bonus if prefix matches an official Indian RTO state/UT code (`validate_state_code`).
  - Emits top 3 ranked plates with metadata (`rank`, `plate`, `soft_score`, `state_code_valid`, `ghost_slot_indices`).
  - Purely advisory: `final_plate` remains strictly zero-hallucination ground truth.
- **4. Output Persistence & CLI Updates**:
  - Writes both `vlm_result.json` and `final_plate_result.json` in the track output directory.
  - Formatted terminal table in `scripts/run_vlm.py` with Tone, VLM Raw, Disambig, Reconciliation Action, and Ghost Stack columns, followed by the Ghost Predictions ranking block.
- **5. Comprehensive Unit Testing & Benchmarking**:
  - Added 3 dedicated unit tests in `tests/test_vlm_verifier.py` (`test_vlm_pre_disambiguation`, `test_ghost_stack_tagging_and_tone`, `test_beam_search_ghost_predictions`).
  - Test suite: **49/49 tests passing ($100\%$)** in 5.7s.
  - Benchmarked on `outputs/activa_signal/track_027` (Reconciled: `UP15BE5580` / Ghost 1: `UP15BE5580` at $0.779$) and `outputs/simple_traffic_30s/track_002` (Reconciled: `????B?718?` / Ghost 1: `HR01BC7184` at $0.645$).

---

### 2026-11-24 04:50:00 IST — Top-Level `final_plate` Key, Ghost Stack Pruning Floor (`0.05`), & Interactive Full-Stack Tester (`scripts/test_full_pipeline.py`)
- **1. Standardize Top-Level Plate Key (`cv_pipeline/vlm_verifier.py`)**:
  - Explicitly added `"final_plate"` alongside `"reconciled_plate"` across both `SUCCESS` and `FALLBACK_TO_OCR` returns.
  - Preserved `"reconciled_plate"` for backward compatibility across all existing callers and tests.
- **2. Lower `ghost_stack` Pruning Floor to `0.05`**:
  - Reduced soft-score cutoff from `0.12` to `0.05` so single-pipeline OCR candidates (weighted by $0.25-0.30$, e.g. $\sim 0.06-0.09$) survive in `ghost_stack`.
  - On `simple_traffic_30s` Track 2, confirmed slot 0 retains `'K'` ($0.069$) alongside `'M'` ($0.474$), and slot 9 retains `'2'` ($0.091$) alongside `'4'` ($0.406$).
  - Updated `tests/test_vlm_verifier.py` to assert `"final_plate"` is non-null and that low-score single-pipeline OCR candidates survive.
- **3. Interactive Full-Stack Tester (`scripts/test_full_pipeline.py`)**:
  - Built an interactive CLI tool testing Stage 1 (BoT-SORT vehicle tracking) and full Stage 2 (2.1 vehicle crop extraction $\to$ 2.2 dual-detector plate localization & readability scoring $\to$ 2.3 RapidOCR + PARSeq dual-pipeline OCR $\to$ 2.4 blind multi-tier VLM triple-check).
  - Isolates all runs into `test_outputs/<clip_stem>/` without touching `outputs/` (added `test_outputs/` to `.gitignore`).
  - Added robust fallback path resolution in `cv_pipeline/ocr_engine.py` and `cv_pipeline/vlm_verifier.py` to seamlessly handle arbitrary output directory nesting.
  - Includes interactive video selector scanning `data/Final` (10 videos), Stage 1 tracking caching/reuse, compact track selection table, option `[v]` to launch annotated video, full Stage 2 execution, terminal forensic report with Top-3 ghost predictions, and automatic OS image viewer launch of the clearest unenhanced plate crop.
- **4. Verification & PyTorch Device Fix**:
  - Fixed `torch.cuda.get_device_name` in `cv_pipeline/pipeline.py` to handle string `"0"` via integer index conversion.
  - End-to-end verification run on `activa_signal.mp4` Track 27 inside `test_outputs/activa_signal/track_027` succeeded with full forensic table and clearest crop image generated.
  - Full test suite: **77/77 tests passing ($100\%$)**.


