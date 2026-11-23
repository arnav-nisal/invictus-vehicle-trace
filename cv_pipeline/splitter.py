"""Track splitter module for detecting and splitting abrupt class or bounding-box size jumps.

Prevents identity-jumping where a track mistakenly latches onto an adjacent vehicle
passing close by (e.g., motorcycle jumping onto a car, or truck jumping onto a motorcycle).
"""

import logging
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from .config import COCO_TO_CATEGORY, DEFAULT_MAX_CONSECUTIVE_AREA_JUMP

logger = logging.getLogger(__name__)

TWO_WHEELER_CLASSES = {"motorcycle", "bicycle"}
LARGER_VEHICLE_CLASSES = {"car", "truck", "bus"}


def get_vehicle_type(label: str) -> str:
    """Classify vehicle into broad coarse types for jump detection."""
    label_lower = label.lower()
    if label_lower in TWO_WHEELER_CLASSES:
        return "two-wheeler"
    if label_lower in LARGER_VEHICLE_CLASSES:
        return "larger-vehicle"
    return "other"


def split_tracks_on_sudden_jumps(
    tracks_meta: Dict[int, dict],
    frame_observations: Dict[int, List[dict]],
    crop_selector: Any,
    video_path: str,
    fps: float = 30.0,
    max_area_jump_ratio: float = DEFAULT_MAX_CONSECUTIVE_AREA_JUMP,
) -> Tuple[Dict[int, dict], Dict[int, List[dict]], List[dict]]:
    """Scan each track frame-by-frame and split it if the vehicle class switches between

    a two-wheeler and a larger vehicle OR if the bounding-box size abruptly jumps/shrinks > 2.2x.

    Returns:
        (updated_tracks_meta, updated_frame_observations, split_audit_log)
    """
    logger.info("Scanning raw tracks for sudden class or bounding-box size jumps...")

    # Group all frame observations by track ID, ordered by frame_idx
    obs_by_track: Dict[int, List[Tuple[int, dict]]] = {}
    for f_idx in sorted(frame_observations.keys()):
        for det in frame_observations[f_idx]:
            tid = det["id"]
            obs_by_track.setdefault(tid, []).append((f_idx, det))

    next_new_id = (max(tracks_meta.keys()) + 1) if tracks_meta else 1000
    split_audit = []
    tracks_to_split: Dict[int, List[List[Tuple[int, dict]]]] = {}

    for tid, obs_list in obs_by_track.items():
        if len(obs_list) < 2:
            continue

        segments: List[List[Tuple[int, dict]]] = []
        current_segment: List[Tuple[int, dict]] = [obs_list[0]]

        for i in range(1, len(obs_list)):
            prev_f, prev_det = obs_list[i - 1]
            curr_f, curr_det = obs_list[i]

            prev_type = get_vehicle_type(prev_det.get("label", ""))
            curr_type = get_vehicle_type(curr_det.get("label", ""))

            # Condition 1: Class switch between two-wheeler and larger vehicle (car/truck/bus)
            class_switch = (
                (prev_type != curr_type)
                and (prev_type in ("two-wheeler", "larger-vehicle"))
                and (curr_type in ("two-wheeler", "larger-vehicle"))
            )

            # Condition 2: Bounding-box size abruptly jumps or shrinks
            prev_area = max(1.0, float(prev_det.get("area", 1.0)))
            curr_area = max(1.0, float(curr_det.get("area", 1.0)))
            area_ratio = max(prev_area, curr_area) / min(prev_area, curr_area)
            size_jump = False

            if area_ratio > max_area_jump_ratio:
                if area_ratio > 3.5:
                    # Extreme jump (> 3.5x): split immediately (e.g. hopped onto another vehicle)
                    size_jump = True
                elif i + 1 < len(obs_list):
                    # Check if this is a momentary 1-frame dip/recovery (e.g. thin overhead bar/pole)
                    next_f, next_det = obs_list[i + 1]
                    next_area = max(1.0, float(next_det.get("area", 1.0)))
                    recovery_ratio = max(prev_area, next_area) / min(prev_area, next_area)
                    if recovery_ratio <= max_area_jump_ratio:
                        # Momentary 1-frame dip/recovery under overhead bar: do not split
                        size_jump = False
                    else:
                        # Sustained multi-frame scale anomaly: split
                        size_jump = True
                else:
                    # Trailing boundary frame: do not split single frame unless extreme
                    size_jump = False

            if class_switch or size_jump:
                reason = []
                if class_switch:
                    reason.append(f"class switch '{prev_det['label']}' -> '{curr_det['label']}'")
                if size_jump:
                    reason.append(f"size jump {area_ratio:.2f}x ({prev_area:.0f}px -> {curr_area:.0f}px)")

                logger.info(
                    f"SPLIT DETECTED on Track {tid} at frame {curr_f} ({curr_f / fps:.2f}s): "
                    + ", ".join(reason)
                )

                split_audit.append(
                    {
                        "original_tid": int(tid),
                        "split_frame": int(curr_f),
                        "timestamp_sec": round(curr_f / fps, 2),
                        "reasons": reason,
                        "prev_class": prev_det.get("label", ""),
                        "curr_class": curr_det.get("label", ""),
                        "area_ratio": round(area_ratio, 2),
                    }
                )

                # End current segment and start a new one
                segments.append(current_segment)
                current_segment = [obs_list[i]]
            else:
                current_segment.append(obs_list[i])

        if segments:
            segments.append(current_segment)
            tracks_to_split[tid] = segments

    if not tracks_to_split:
        logger.info("No abrupt class or size jumps detected; all tracks consistent.")
        return tracks_meta, frame_observations, []

    # Open video capture once to populate best crop candidates for split segments if needed
    cap = cv2.VideoCapture(str(video_path))
    cap_opened = cap.isOpened()

    updated_meta = dict(tracks_meta)

    for orig_tid, seg_list in tracks_to_split.items():
        # First segment retains the original ID
        # Subsequent segments receive a new unique ID
        for s_idx, seg in enumerate(seg_list):
            seg_tid = orig_tid if s_idx == 0 else next_new_id
            if s_idx > 0:
                next_new_id += 1

            # Update detection references in frame_observations
            for f_idx, det in seg:
                det["id"] = seg_tid

            # Determine dominant class and category for this segment
            class_counts = Counter(d.get("label", "vehicle") for _, d in seg)
            dominant_class = class_counts.most_common(1)[0][0]
            category = COCO_TO_CATEGORY.get(dominant_class, "vehicle")

            start_f = seg[0][0]
            end_f = seg[-1][0]
            start_ts = start_f / fps
            end_ts = end_f / fps
            frame_cnt = len(seg)

            confs = [float(d.get("conf", 0.5)) for _, d in seg]
            areas = [float(d.get("area", 100.0)) for _, d in seg]
            dims = [
                max(float(d["bbox"][2] - d["bbox"][0]), float(d["bbox"][3] - d["bbox"][1]))
                for _, d in seg
            ]

            first_bbox = [float(x) for x in seg[0][1]["bbox"]]
            last_bbox = [float(x) for x in seg[-1][1]["bbox"]]

            # Trajectory centroids
            trajectory = [
                (
                    f,
                    (d["bbox"][0] + d["bbox"][2]) / 2.0,
                    (d["bbox"][1] + d["bbox"][3]) / 2.0,
                )
                for f, d in seg
            ]

            # Find the best frame with highest confidence / area for crop extraction
            best_f, best_det = max(seg, key=lambda x: (x[1].get("conf", 0.0), x[1].get("area", 0.0)))
            first_crop = None
            last_crop = None

            if cap_opened:
                # Extract first crop
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                ret, f_img = cap.read()
                if ret and f_img is not None:
                    x1, y1, x2, y2 = [int(round(c)) for c in first_bbox]
                    x1 = max(0, min(f_img.shape[1] - 1, x1))
                    y1 = max(0, min(f_img.shape[0] - 1, y1))
                    x2 = max(0, min(f_img.shape[1], x2))
                    y2 = max(0, min(f_img.shape[0], y2))
                    if x2 > x1 and y2 > y1:
                        first_crop = f_img[y1:y2, x1:x2].copy()

                # Extract last crop
                cap.set(cv2.CAP_PROP_POS_FRAMES, end_f)
                ret, f_img = cap.read()
                if ret and f_img is not None:
                    x1, y1, x2, y2 = [int(round(c)) for c in last_bbox]
                    x1 = max(0, min(f_img.shape[1] - 1, x1))
                    y1 = max(0, min(f_img.shape[0] - 1, y1))
                    x2 = max(0, min(f_img.shape[1], x2))
                    y2 = max(0, min(f_img.shape[0], y2))
                    if x2 > x1 and y2 > y1:
                        last_crop = f_img[y1:y2, x1:x2].copy()

                # Extract best crop candidate for crop_selector
                cap.set(cv2.CAP_PROP_POS_FRAMES, best_f)
                ret, f_img = cap.read()
                if ret and f_img is not None:
                    crop_selector.update(
                        track_id=int(seg_tid),
                        frame=f_img,
                        bbox=best_det["bbox"],
                        confidence=float(best_det["conf"]),
                        frame_idx=best_f,
                        timestamp_sec=best_f / fps,
                    )

            seg_meta = {
                "id": int(seg_tid),
                "class_name": dominant_class,
                "category": category,
                "start_frame": start_f,
                "end_frame": end_f,
                "start_timestamp_sec": start_ts,
                "end_timestamp_sec": end_ts,
                "frame_count": frame_cnt,
                "max_conf": max(confs),
                "conf_sum": sum(confs),
                "max_area": max(areas),
                "max_dim": max(dims),
                "first_bbox": first_bbox,
                "last_bbox": last_bbox,
                "first_crop": first_crop,
                "last_crop": last_crop,
                "trajectory": trajectory,
                "original_ids": [int(seg_tid)],
            }
            updated_meta[seg_tid] = seg_meta

    if cap_opened:
        cap.release()

    logger.info(
        f"Track splitting complete: {len(split_audit)} split event(s) executed across {len(tracks_to_split)} tracks."
    )
    return updated_meta, frame_observations, split_audit
