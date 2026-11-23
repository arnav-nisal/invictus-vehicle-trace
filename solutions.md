# Engineering Solutions — Stage 1: Vehicle Tracking & Traceability

This document describes the engineering solutions, mathematical models, algorithms, and empirical verification results implemented to resolve the challenges documented in [challanges.md](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/challanges.md).

---

## 1. Hardware-Accelerated GPU Inference Architecture
- **Solution**: Reconfigured the Python virtual environment (`.venv`) with PyTorch (`torch 2.6.0+cu124` / `torchvision 0.21.0+cu124`) with CUDA 12.4 support enabled on the host's dedicated **NVIDIA GeForce RTX 3050 Laptop GPU** (`device=0`).
- **Pipeline Implementation**: Integrated device auto-detection in [cv_pipeline/config.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/config.py) (`DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"`) and verified in [cv_pipeline/pipeline.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/pipeline.py).
- **Verification Outcome**:
  - Processing speed increased from **2.8 FPS on CPU** to **45–60 FPS on GPU** ($\sim 15\times$ speedup).
  - A 30-second 1080p clip (900 frames) now processes end-to-end (including ReID tracking, splitting, stitching, video annotation, crop generation, and contact-sheet export) in **under 20 seconds**.

---

## 2. Full-Resolution Inference with Decoupled Post-Tracking Filtering
- **Solution**: Decoupled the sensitivity needed for tracking moving objects from the selectivity needed for the final forensic report.
  1. *High-Resolution Detector*: Set YOLO inference resolution to `imgsz=1280` on the GPU. This quadruples the pixel density for distant objects compared to default 640px.
  2. *Sensitive Tracking Threshold*: Lowered detector confidence input to `conf=0.20`, allowing motion-blurred scooters and partially occluded vehicles to remain attached to their active tracks.
  3. *Decoupled Post-Tracking Filter*: Applied spatial and temporal thresholds strictly **after** tracking completes:
     - `DEFAULT_MIN_FRAMES = 15` (requires vehicle persistence for $\ge 0.5\text{s}$ at 30 FPS).
     - `DEFAULT_MIN_BOX_AREA = 1200\text{ px}^2` and `DEFAULT_MIN_BOX_DIM = 28\text{ px}`.
- **Verification Outcome**:
  - Distant roadside noise, pedestrian walkway flickers, and background clutter are completely eliminated from the output summary.
  - Fast-moving, motion-blurred two-wheelers maintain unbroken track attachments across entire intersections.

---

## 3. Fixed-CCTV Tracker Tuning
- **Solution**: Customized the BoT-SORT tracker configuration in [cv_pipeline/botsort_reid.yaml](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/botsort_reid.yaml) specifically for stationary surveillance cameras:
  - `gmc_method: none`: Disabled Camera Motion Compensation. Since CCTV cameras are static, GMC estimation introduces artificial ego-motion noise that corrupts track state vectors.
  - `track_buffer: 90`: Expanded memory buffer to 90 frames (3.0 seconds at 30 FPS) to maintain inactive tracks during short pauses.
  - `match_thresh: 0.72`: Tightened association tolerance from default 0.85 down to 0.72 to prevent tracks from latching onto neighboring vehicles in adjacent lanes.
  - `proximity_thresh: 0.5` and `appearance_thresh: 0.25`: Balanced geometric spatial IoU with native visual appearance features.

---

## 4. Pre-Stitch Sudden Jump Splitter
- **Solution**: Implemented [cv_pipeline/splitter.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/splitter.py) to scan raw tracks frame-by-frame before stitching and split any identity-jumping tracks.
  1. *Coarse Category Switch Check*:
     - Broad vehicle groups: `two-wheeler` (`motorcycle`, `bicycle`) vs. `larger-vehicle` (`car`, `truck`, `bus`).
     - If a track switches between these categories at frame $i$, it is **immediately split** into a separate tracklet.
  2. *Overhead Bar Resilience & Sustained Anomaly Check*:
     - If consecutive frame area ratio $\frac{\max(A_{i-1}, A_i)}{\min(A_{i-1}, A_i)} > 2.2\times$:
       - If extreme jump ($>3.5\times$), split immediately.
       - If moderate ($2.2\times < \text{ratio} \le 3.5\times$), check recovery at frame $i+1$. A momentary 1-frame dip/spike that immediately recovers back to baseline is recognized as a thin overhead bar/pole occlusion and is **preserved without splitting**.
       - Sustained multi-frame scale changes ($\ge 2$ frames) are split.

---

## 5. Spatiotemporal & Central-Body Trajectory-Projected Tracklet Stitching
- **Solution**: Implemented [cv_pipeline/stitcher.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/stitcher.py) to re-associate broken track segments belonging to the same physical vehicle using physical motion constraints and appearance cues.

### A. Central Body Extraction
- Function `extract_central_body(crop, y_margin=(0.18, 0.82), x_margin=(0.12, 0.88))` crops the inner $64\%$ vertical and $76\%$ horizontal region of the vehicle image.
- Completely ignores roof-level gantry bars, windshield glare, and ground shadows, isolating pure vehicle body paint.
- Features are extracted and compared at two levels: boundary crops (crop at end of track A vs. crop at start of track B) and best sharpest crops across each track.

### B. Trajectory Velocity Vector Projection & Lateral Lane Alignment
- For moving vehicle segment A, computes the travel velocity vector $\vec{v}_A = (v_x, v_y)$ over recent trajectory frames.
- Predicts vehicle position at segment B's start time:
  $$\vec{x}_{\text{pred}} = \vec{x}_{\text{last}} + \vec{v}_A \cdot \Delta f$$
- Computes unit velocity $\hat{u} = \frac{\vec{v}_A}{\|\vec{v}_A\|}$ and normal vector $\hat{n} = (-\hat{u}_y, \hat{u}_x)$.
- Measures cross-lane lateral error:
  $$\text{lateral\_err} = |(\vec{x}_B - \vec{x}_{\text{pred}}) \cdot \hat{n}|$$
- Strictly enforces $\text{lateral\_err} \le 65.0\text{ px}$ (`DEFAULT_MAX_LATERAL_ERR_PX = 65.0`). Parallel cross-lane merges are strictly rejected.

### C. Footprint Centroid Allowance
- Accommodates bounding box boundary expansions across occlusion boundaries:
  $$\text{max\_allowed\_dist} = \max\left(60.0, \max(w_A, h_A) \times 0.85, v_{\max} \cdot \Delta t\right)$$
- Prevents 1-frame centroid shifts from tripping instantaneous velocity ceilings.

### D. Perspective Scale Growth Allowance
- Allows bounding-box scale ratio up to $2.1\times$ (`DEFAULT_MAX_SIZE_RATIO = 2.1`) for same-category vehicles, accommodating camera approach without allowing cross-category mix-ups ($>2.7\times$).

### E. Composite Compatibility Score
$$\text{Score} = 100 \cdot \rho_{\text{correl}} - 0.5 \cdot d_{\text{Lab}} - 0.25 \cdot \text{lateral\_err} - 0.15 \cdot \text{total\_proj\_err} - 15.0 \cdot \Delta t_{\text{gap}}$$
Prioritizes immediate temporal continuation and tight lane alignment over coincidental distant color matches.

---

## 6. Lossless Dual-Folder Crop Extraction & Adaptive Clarity Enhancement
- **Solution**: Built [cv_pipeline/crop_extractor.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/crop_extractor.py) to extract every frame a selected vehicle appears in, outputting to two dedicated lossless (`.png`) directories:
  1. `vehicle_crops_raw/`: Untouched original pixels with an $8\%$ safety margin padding formula:
     $$\Delta x = \text{width} \times 0.08, \quad \Delta y = \text{height} \times 0.08$$
     $$x_1^{\text{padded}} = \max(0, \lfloor x_1 - \Delta x \rfloor), \quad y_1^{\text{padded}} = \max(0, \lfloor y_1 - \Delta y \rfloor)$$
     $$x_2^{\text{padded}} = \min(W, \lceil x_2 + \Delta x \rceil), \quad y_2^{\text{padded}} = \min(H, \lceil y_2 + \Delta y \rceil)$$
  2. `vehicle_crops_enhanced/`: Upscaled and clarity-enhanced crops ready for human inspection and automated license plate detection:
     - **Adaptive Lanczos4 Upscaling**: If $\max(w, h) < 480\text{px}$, upscales using `cv2.INTER_LANCZOS4` so the longer side reaches $480\text{px}$ (scale factor capped at $4.0\times$). Crops $\ge 480\text{px}$ remain at $1.0\times$.
     - **Compression Noise Mitigation**: Gentle bilateral filter ($d=5, \sigma_{\text{color}}=30, \sigma_{\text{space}}=30$) cleans macroblocking artifacts without blurring letter strokes.
     - **Luminance-Only CLAHE**: Converts BGR $\to$ LAB, applies CLAHE ($\text{clipLimit}=1.8, \text{grid}=(4,4)$) exclusively on the $L$-channel, then converts back to BGR. Illuminates shadowed bumper plates without altering vehicle paint color.
     - **Controlled Unsharp Mask**: Subtracts Gaussian blur ($\sigma=1.0$) as $1.35 \cdot I - 0.35 \cdot I_{\text{blur}}$ to accentuate plate characters without halo ringing.

---

## 7. High-Resolution Source Mapping (`--crop-video`)
- **Solution**: Implemented coordinate normalization and temporal mapping allowing the tracking engine to operate on high-speed 1080p proxies (e.g. `simple_traffic_30s.mp4`) while extracting pristine crops from the original high-resolution footage (e.g. 4K UHD $3840\times 2160$ `simple_traffic.mp4`):
  $$x_{\text{crop}} = \frac{x_{\text{track}}}{W_{\text{track}}} \times W_{\text{crop}}, \quad y_{\text{crop}} = \frac{y_{\text{track}}}{H_{\text{track}}} \times H_{\text{crop}}$$
  $$f_{\text{crop}} = \text{round}(t_{\text{sec}} \times \text{FPS}_{\text{crop}})$$
- **Benefits**: Eliminates the heavy GPU overhead of running full tracking at 4K resolution while preserving 4K pixel fidelity for downstream license plate character OCR.

---

## 8. Decoupled Frame-by-Frame Track History Serialization (`track_history.json`)
- **Solution**: Refactored [cv_pipeline/pipeline.py](file:///c:/Users/arnav/Documents/invictus-vehicle-trace/cv_pipeline/pipeline.py) to serialize all active observations for every kept vehicle into `outputs/track_history.json`.
- **Schema**:
  Each entry maps master `track_id` to its chronological array of frame observations:
  - `frame_idx`: Exact zero-indexed video frame number.
  - `timestamp_sec` & `timestamp`: Elapsed video time in seconds and formatted string `HH:MM:SS.mmm`.
  - `bbox`: `[x1, y1, x2, y2]` in original video coordinates.
  - `confidence`: YOLO detection confidence score.
  - `area`: Bounding box pixel area ($w \times h$).
- **Instantaneous `--reuse-tracking` Mode**:
  Enables the CLI runner (`scripts/run_tracking.py --reuse-tracking --target-id <ID>`) to load pre-computed tracking in $<50\text{ms}$, decoupling forensic crop extraction and vehicle selection from GPU tracking re-runs.

---

## 9. Downstream Plate Manifest Generation (`crop_manifest.json`)
- **Solution**: Automatically writes a structured metadata manifest `crop_manifest.json` inside each vehicle's track directory:
  - Header: `track_id`, `class_name`, `category`, `start_frame`, `end_frame`, `start_timestamp`, `end_timestamp`, `total_crops_saved`, `usable_crops_count`, `avg_sharpness_score`, `video_source`, `crop_video_source`, and `padding_percent`.
  - Itemized `crops` array: Lists every crop's index, original bbox, padded bbox, `raw_width`, `raw_height`, `scale_factor`, `enhanced_width`, `enhanced_height`, `sharpness_score`, `quality_flag`, and absolute lossless file paths (`raw_crop_path` and `enhanced_crop_path`).
- **Benefits**: Provides an automated, well-typed input contract for Stage 2 license plate detection and OCR pipelines without requiring video re-parsing.

---

## 10. Dual-Detector License Plate Cropping & Spatial Cross-Verification (`cv_pipeline/plate_detector.py`)
- **Solution**: Implemented an ensemble plate detection system running on top of vehicle crops:
  - **Detector A (Custom Indian YOLO26s)**: Fine-tuned on Indian HSRP and two-wheeler registration datasets (`models/yolo26s_plate_best.pt`, `conf=0.15`, CUDA `device=0`).
  - **Detector B (Global YOLOv9-s ONNX)**: MIT-licensed ONNX model (`yolo-v9-s-608-license-plate-end2end`, `conf=0.18`) loaded via `open-image-models`.
  - **Spatial IoU Cross-Verification**: Computes intersection-over-union between both detections. For $\text{IoU} \ge 0.35$, merges bounding boxes via confidence-weighted averaging and tags as `DUAL_VERIFIED` with a $1.25\times$ readability bonus.
  - **Smart 12% Padding**: Expands winning box by $12\%$ horizontally and vertically (clamped to boundaries) to ensure edge characters and two-line scooter plates are never clipped.
  - **Lossless Slicing**: Slices exact matching coordinates from both `vehicle_crops_unenhanced/` and `vehicle_crops_enhanced/` into `plate_crops_all/`.

---

## 11. Border-Truncation Guard & Exit Aspect-Ratio Drop Protection
- **Solution**: Evaluates unpadded plate boxes against both crop and video boundaries before scoring:
  - **Crop Edge Contact**: Rejects detections whose unpadded box comes within $\le 2$ native pixels of any vehicle crop boundary ($\text{dist} \le 2 \times \text{scale\_factor}$ in enhanced coordinates).
  - **Video Outer Boundary Contact**: Rejects detections whose mapped coordinates in video frame space touch within $\le 6\text{px}$ of the video frame's outer boundary.
  - **Exit Aspect-Ratio Drop**: Computes track median plate aspect ratio ($\text{median\_ar} = \text{median}(w/h)$). If a plate is within $5\%$ of a boundary and its aspect ratio drops below $0.75 \times \text{median\_ar}$, it is rejected as an exit-truncated plate.
  - **Tagging**: Dropped boundary plates are tagged `rejection_reason: "border_truncated"` in `plate_manifest.json` and excluded from `plate_crops_selected/`.

---

## 12. Plate Geometry Filtering & Balanced Conflict Accounting
- **Solution**: 
  - **Pre-Resolution Geometry Filter**: Implemented `is_valid_plate_geometry` requiring $0.85 \le \text{AR} \le 6.5$ and $\text{box\_area} \le 0.45 \times \text{crop\_area}$, rejecting spurious full-body hallucinations.
  - **Explicit Conflict Statuses**: Dual detections with $\text{IoU} < 0.35$ are explicitly designated as `CONFLICT_RESOLVED_CUSTOM` (Custom confidence higher) or `CONFLICT_RESOLVED_GLOBAL` (Global confidence higher).
  - **100% Balanced Math**: Ensures exact identity:
    $$\text{custom\_detections} = \text{dual\_verified} + \text{single\_custom} + \text{conflict\_custom} + \text{conflict\_global}$$
    $$\text{global\_detections} = \text{dual\_verified} + \text{single\_global} + \text{conflict\_custom} + \text{conflict\_global}$$

---

## 13. Outlier-Sensitive Dynamic Readability Gate (p90 Anchor)
- **Solution**: Replaced single peak score anchor with the 90th percentile ($p90$) of passing readability scores:
  $$\text{threshold} = 0.45 \times p90\_score$$
  $$\text{readability\_score} = (\sqrt{\text{sharpness}})^{0.8} \cdot (\sqrt{\text{native\_area}})^{0.8} \cdot (\text{contrast})^{0.4} \cdot \text{conf} \cdot \text{dual\_bonus}$$
  where $\text{dual\_bonus} = 1.25$ for `DUAL_VERIFIED` and $1.0$ otherwise.
- **Benefits**: Retains 26 and 52 clean, readable frames (capped at 60) without single glare/noise frames discarding dozens of valid candidate crops.

---

## 14. Pure Lanczos4 Plate Enlargement
- **Solution**:
  - Computes native plate dimensions and Laplacian sharpness variance on the untouched raw unenhanced slice before resize.
  - Slices small plate crops cleanly using `cv2.INTER_LANCZOS4` (preserving aspect ratio) so every saved plate has a minimum width of $320\text{px}$ or minimum height of $120\text{px}$ (capped at $4.0\times$ zoom; already-large plates kept at $1.0\times$).
  - Zero additional filters or CLAHE applied. Exact matching dimensions maintained between `unenhanced/` and `enhanced/` directories.
  - Records `plate_crop_raw_size`, `plate_upscale_factor`, and `plate_saved_size` in `plate_manifest.json`.

---

## 15. Verification Summary Across All Test Clips

| Test Clip | Resolution & FPS | Track Count | Key Target Behaviors Verified | Status |
| :--- | :---: | :---: | :--- | :---: |
| **`activa_signal.mp4`** | 1080p @ 30 FPS | 29 vehicles | Moving white Activa maintains continuous single ID **Track 27** across crossing (`[27, 38, 51]`); stationary Activa maintains **Track 4** for all 300 frames. | **PASSED** |
| **`highway_traffic.mp4`** | 1080p @ 30 FPS | 88 vehicles | Track pairs `39-66`, `49-95`, `55-90`, `56-87`, `61-91-93`, and `63-96-98` cleanly stitched under overhead bar. Cross-lane swaps (`55-98`, `63-56`, `96-106`) fully prevented. | **PASSED** |
| **`simple_traffic_30s.mp4`** | 1080p @ 30 FPS | 144 vehicles | 57 merge events successfully consolidated across 34 multi-segment tracks; zero ID jumping across cars, trucks, buses, and motorcycles. | **PASSED** |
| **`activa_signal.mp4` (Stage 2 Lossless Crops)** | 1080p @ 30 FPS | Track 27 | 98 lossless `.png` crops extracted into `vehicle_crops_raw/` & `vehicle_crops_enhanced/`. 95/98 usable, avg sharpness 3159.0, adaptively upscaled to $326\times 480$. | **PASSED** |
| **`simple_traffic_30s.mp4` (4K Source Mapping)** | 1080p tracking $\to$ 4K crop | Track 2 | Mapped 1080p tracking onto 4K ($3840\times 2160$) `simple_traffic.mp4`. 202 lossless crops extracted, 202/202 usable, scaled up to $480\times 410$. | **PASSED** |
| **`activa_signal.mp4` (Stage 2.2 Plate Selection)** | 1080p @ 30 FPS | Track 27 | 63 plates detected, 3 border-truncated rejected (Frames 255, 256, 260). Last selected Frame 254 fully intact ($231\times 120\text{px}$). 26 top paired crops selected. 100% math balanced ($58=58, 55=55$). | **PASSED** |
| **`simple_traffic_30s.mp4` (Stage 2.2 Plate Selection)** | 1080p tracking $\to$ 4K crop | Track 2 | 184 plates detected, 8 border-truncated rejected (Frame 187 excluded). Last selected Frame 186 fully intact ($320\times 91\text{px}$). 52 top paired crops selected. 100% math balanced ($76=76, 177=177$). | **PASSED** |
