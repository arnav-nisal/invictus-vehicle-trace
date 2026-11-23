# Engineering Challenges — Stage 1: Vehicle Tracking & Traceability

This document records the technical challenges encountered during the development of Stage 1 (vehicle detection, continuous tracking, identity preservation, and post-tracking tracklet stitching) for the Invictus Vehicle Trace system.

---

## 1. Slow CPU-Only Processing Latency
- **Symptom**: Initial execution on CPU ran at approximately 2.8–3.2 FPS. Processing a 10-second clip (`activa_signal.mp4`, 300 frames) took nearly 2 minutes, and testing a 30-second clip took over 4 minutes.
- **Root Cause**: The virtual environment was initially configured with CPU-only PyTorch wheels. Executing YOLO object detection and BoT-SORT tracking at high resolution saturated CPU cores and prevented rapid experimentation.
- **Impact**: Iteration cycles for tuning tracker hyper-parameters and verifying fixes were bottlenecked by compute latency.

---

## 2. Detection Trade-Off: Background Clutter vs. Motion-Blurred Two-Wheelers
- **Symptom**: Distant, stationary background vehicles and pedestrian walkways generated dozens of short, flickering tracklets. When detection confidence was raised (e.g. `conf=0.45` to `0.55`) to eliminate this noise, fast-moving two-wheelers (such as the white Activa crossing the intersection) were dropped mid-crossing due to motion blur and perspective changes.
- **Root Cause**: Fixed CCTV cameras have a deep depth of field. Far-away cars occupy very few pixels ($<20\text{px}$) and trigger low-confidence detections, whereas foreground two-wheelers accelerating across camera view experience significant motion blur and rapid aspect-ratio shifts.
- **Impact**: Either the system flooded the output with background clutter, or it lost track of target vehicles midway through their crossing.

---

## 3. Track Splitting from Intermittent Traffic Occlusions
- **Symptom**: The white Activa crossing the intersection was tracked as ID 33 from `00:00:03.00` to `00:00:05.07`, disappeared for $\sim 1.3$ seconds behind crossing cross-traffic, and was assigned a completely new ID (ID 42) upon reappearing from `00:00:06.33` to `00:00:08.77`.
- **Root Cause**: Online trackers (like standard BoT-SORT and ByteTrack) rely on Kalman filter position projections over a short memory window (`track_buffer`). When a vehicle is completely occluded by a larger vehicle for more than 1–2 seconds, the Kalman filter's predicted bounding box drifts significantly. When the vehicle emerges, the spatial IoU with the stale prediction is 0, causing the tracker to declare a new tracklet.
- **Impact**: Broken vehicle lifecycles, fragmented audit summaries, and lost vehicle continuity across urban intersections.

---

## 4. Perspective Scale Growth During Camera Approach
- **Symptom**: Moving vehicles approaching the CCTV camera naturally grow in size. For instance, the moving Activa's bounding-box area grew from $\sim 3,600\text{px}^2$ to $\sim 14,000\text{px}^2$ (a $\sim 1.9\times$ linear scale growth) as it crossed from the far lane toward the camera. Early tracklet association filters with strict scale ratio thresholds ($< 1.6\times$) rejected legitimate segment associations.
- **Root Cause**: Static scale ratio checks designed to prevent cars from merging with motorcycles did not account for perspective scaling in fixed CCTV angles, where vehicles moving along the optical axis grow by nearly $2\times$ across the camera frame.
- **Impact**: Valid vehicle tracks were prevented from reconnecting after brief occlusions, resulting in split identities.

---

## 5. Identity-Jumping onto Neighboring Vehicles in Dense Lanes
- **Symptom**: On multi-lane clips (`highway_traffic.mp4` and `intersection_traffic.mp4`), tracks occasionally jumped onto an adjacent vehicle passing close by (e.g., Track 15 jumping from a motorcycle to a neighboring car, or Track 28 hopping onto an adjacent scooter).
- **Root Cause**: Overly permissive tracking association thresholds (`match_thresh = 0.85`) combined with Kalman filter velocity extrapolation caused bounding boxes of vehicles in adjacent lanes to overlap in the assignment cost matrix. Additionally, when appearance ReID features for dark vehicles were similar, the tracker latch-on probability increased.
- **Impact**: Contaminated track histories where a single ID represented two entirely different physical vehicles in adjacent lanes.

---

## 6. Overhead Highway Gantry Bars Breaking Car Tracks
- **Symptom**: On `highway_traffic.mp4`, vehicles driving away from the camera under an overhead horizontal gantry bar ($y \approx 250\text{--}300$) repeatedly split into separate tracks (e.g., Track pairs 39/66, 61/93, 49/95, 55/90).
- **Root Causes**:
  1. *Bounding-box Clipping*: As a car passed under the thin overhead pole, the detector temporarily clipped the bounding box to the bottom half of the car for 1 frame, causing a sudden momentary area dip.
  2. *Centroid Shift Velocity Spikes*: When the car emerged 1 frame later ($0.033\text{s}$), the bounding box expanded back to the full car, shifting the centroid by $\sim 36\text{px}$. Dividing $36\text{px}$ by $0.05\text{s}$ effective gap produced an apparent instantaneous velocity of $>700\text{px/s}$, tripping velocity ceilings.
  3. *Overhead Bar Color Pollution*: The dark horizontal bar crossed the vehicle's roof and windshield, distorting full-crop color histograms and Lab distance calculations.
  4. *Cross-Lane Associations*: In parallel multi-lane traffic, tracks that split under the bar sometimes merged with cars in adjacent parallel lanes because both cars shared dark colors and similar forward speeds.
- **Impact**: Broken vehicle traces and cross-lane identity mix-ups at highway monitoring gantries.

---

## 7. Bumper-Edge License Plate Clipping in Standard Crops
- **Symptom**: Standard object detection bounding boxes tightly wrap the visual contour of a vehicle. For two-wheelers (scooters/motorcycles) and compact cars, the license plate is mounted at the extreme bottom edge of the rear fender or front bumper.
- **Root Cause**: Tight YOLO bounding boxes frequently clip or truncate the bottom 5–10 pixels of the bumper. In downstream plate detection and OCR, this edge clipping cuts off the state prefix or registration digits (e.g. cutting off the bottom of `MH` or numeric digits).
- **Impact**: Unreadable plates or OCR hallucination due to truncated characters at crop boundaries.

---

## 8. Random-Seek Latency in Video Crop Extraction
- **Symptom**: When extracting 100+ frame crops for a single tracked vehicle using standard `cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)`, OpenCV takes up to 45–60 seconds on Windows.
- **Root Cause**: MP4 video compression relies on GOPs (Group of Pictures) with inter-frame P/B compression and sparse I-frames (keyframes). Calling `cap.set()` forces the video decoder to repeatedly seek backward to the preceding keyframe and sequentially decode all intermediate frames up to the requested index.
- **Impact**: Severe interactive lag when an investigator selects a vehicle ID and waits for cropped images to be generated.

---

## 9. Monolithic Pipeline Coupling vs. Forensic Re-Analysis
- **Symptom**: Initially, selecting a different vehicle or extracting its full-track frame history required re-running the entire YOLO object detection and BoT-SORT tracking pipeline over the video.
- **Root Cause**: Frame-by-frame vehicle trajectories and bounding boxes were stored only in memory during the tracking pass and discarded after exporting the single-frame best crops and contact sheet.
- **Impact**: Inability to quickly inspect multiple vehicles or iteratively test downstream license plate detection without enduring redundant GPU computation.

---

## 10. Frame-Exit and Video Outer Boundary Plate Truncation
- **Symptom**: In both `activa_signal.mp4` (Track 27) and `simple_traffic_30s.mp4` (Track 2), the final selected plate crop had a cut-off plate (e.g. Frame 255 in Activa cut off on the right, and Frame 187 in Simple Traffic cut off at the bottom).
- **Root Cause**: As vehicles exit the video frame, they are at their closest perspective point to the camera, achieving maximum pixel dimensions and high sharpness scores. Standard readability gates that only evaluate size and Laplacian variance favored these exit frames despite the license plate being sliced by the video frame outer border or the vehicle crop boundary.
- **Impact**: Downstream OCR models received truncated, partial license plates as the highest-priority crops, inducing character recognition hallucination.

---

## 11. Dual-Detector Conflict Accounting Gaps & Spurious Box Hallucinations
- **Symptom**: Audit funnel counts showed an accounting discrepancy where custom detector counts did not match the sum of dual and single detections. In Track 2, 27 dual detections with non-overlapping boxes ($\text{IoU} < 0.35$) were silently counted as single global detections.
- **Root Cause**: When both detectors fired but had $\text{IoU} < 0.35$, the detector with higher confidence was designated the winner and its status was labeled `SINGLE_GLOBAL` or `SINGLE_CUSTOM`, masking the fact that both detectors had detected objects. Furthermore, Custom YOLO26 occasionally hallucinated large vehicle body regions ($>50\%$ of the vehicle crop) as plates.
- **Impact**: Obscured conflicting detections in audit manifests and allowed spurious body-sized bounding boxes into conflict resolution.

---

## 12. Glare & Noise Outlier Sensitivity in Dynamic Readability Gating
- **Symptom**: In multiple vehicle tracks, a single frame with camera sensor glare or high-frequency edge noise achieved an artificially extreme readability score (e.g. $>2800$). Because the dynamic gate used a threshold of $0.45 \times \text{peak\_score}$, dozens of clear, readable plate crops (scoring 800–1200) were rejected.
- **Root Cause**: A single-frame maximum anchor ($\text{peak\_score}$) is vulnerable to statistical outliers and sensor artifacts.
- **Impact**: Premature discarding of viable plate images, starving downstream multi-frame OCR consensus algorithms of valuable diverse frames.

---

## 13. Sub-Resolution Plate Dimensions for Manual Inspection and OCR
- **Symptom**: Raw sliced plate crops on 1080p surveillance video ranged between $70\text{--}130\text{px}$ wide and $20\text{--}50\text{px}$ high. At these dimensions, characters were difficult for investigators to inspect visually, and OCR engines designed for standard document resolutions suffered degraded text segmentation.
- **Root Cause**: Vehicles at mid-distance occupy only a fraction of the full camera frame, resulting in small native plate bounding boxes.
- **Impact**: Poor investigator ergonomics during manual verification and sub-optimal OCR tokenization.
