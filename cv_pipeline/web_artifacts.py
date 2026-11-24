"""Web Artifacts Generator for Invictus Vehicle Trace Frontend.

Generates:
1. boxes.json: O(1) frame-indexed lookup of active track bounding boxes for HTML5 Canvas overlays.
2. thumbnails/track_<ID:03d>.jpg: High-quality representative crop per track extracted
   in a single efficient sequential video pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

logger = logging.getLogger("cv_pipeline.web_artifacts")


def parse_resolution(res_val: Union[str, List[int], Tuple[int, int], None]) -> Tuple[int, int]:
    """Parse resolution into (width, height)."""
    if isinstance(res_val, (list, tuple)) and len(res_val) >= 2:
        return int(res_val[0]), int(res_val[1])
    if isinstance(res_val, str) and "x" in res_val.lower():
        parts = res_val.lower().split("x")
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            pass
    return 1920, 1080


def build_boxes_json(
    track_history_path: Union[str, Path],
    tracks_summary_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Generate O(1) frame-indexed boxes.json for web video canvas annotation."""
    hist_path = Path(track_history_path)
    summ_path = Path(tracks_summary_path)
    out_path = Path(output_path)

    if not hist_path.is_file() or not summ_path.is_file():
        raise FileNotFoundError(f"Missing history or summary file: {hist_path}, {summ_path}")

    with open(hist_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    with open(summ_path, "r", encoding="utf-8") as f:
        summ_data = json.load(f)

    # Filter to only clean tracks verified in tracks_summary
    clean_tracks = summ_data.get("tracks", [])
    valid_track_ids = {int(t["id"]) for t in clean_tracks if "id" in t}

    fps = float(summ_data.get("video_fps") or hist_data.get("video_fps") or 30.0)
    total_frames = int(summ_data.get("total_frames") or hist_data.get("total_frames") or 0)
    res_raw = summ_data.get("video_resolution") or hist_data.get("video_resolution")
    width, height = parse_resolution(res_raw)

    hist_tracks = hist_data.get("tracks", {})
    if isinstance(hist_tracks, list):
        tracks_dict = {t.get("id"): t for t in hist_tracks if "id" in t}
    else:
        tracks_dict = hist_tracks

    frames_lookup: Dict[str, List[Dict[str, Any]]] = {}

    for tid_key, track_info in tracks_dict.items():
        try:
            tid = int(track_info.get("id", tid_key))
        except (ValueError, TypeError):
            continue

        if tid not in valid_track_ids:
            continue

        observations = track_info.get("observations") or track_info.get("frame_observations") or []
        for obs in observations:
            f_idx = str(obs.get("frame_idx", 0))
            bbox = obs.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            rounded_bbox = [round(float(c), 1) for c in bbox[:4]]
            if f_idx not in frames_lookup:
                frames_lookup[f_idx] = []

            frames_lookup[f_idx].append({
                "track_id": tid,
                "bbox": rounded_bbox,
            })

    boxes_result = {
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames": frames_lookup,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(boxes_result, f, indent=None, separators=(",", ":"))

    logger.info(f"Generated boxes.json with {len(frames_lookup)} annotated frames -> {out_path}")
    return boxes_result


def generate_track_thumbnails(
    video_path: Union[str, Path],
    track_history_path: Union[str, Path],
    tracks_summary_path: Union[str, Path],
    thumbnails_dir: Union[str, Path],
    max_width: int = 320,
) -> Dict[int, Path]:
    """Extract 1 representative sharp thumbnail per track in a single sequential video pass."""
    v_path = Path(video_path)
    hist_path = Path(track_history_path)
    summ_path = Path(tracks_summary_path)
    thumb_dir = Path(thumbnails_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    with open(summ_path, "r", encoding="utf-8") as f:
        summ_data = json.load(f)
    with open(hist_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    clean_tracks = summ_data.get("tracks", [])
    valid_track_ids = {int(t["id"]): t for t in clean_tracks if "id" in t}

    hist_tracks = hist_data.get("tracks", {})
    if isinstance(hist_tracks, list):
        tracks_dict = {t.get("id"): t for t in hist_tracks if "id" in t}
    else:
        tracks_dict = hist_tracks

    # Determine best frame and bbox for each valid track
    # map frame_idx -> list of (track_id, bbox)
    frame_targets: Dict[int, List[Tuple[int, List[float]]]] = {}
    track_selected_frame: Dict[int, int] = {}

    for tid, s_track in valid_track_ids.items():
        t_hist = tracks_dict.get(tid) or tracks_dict.get(str(tid)) or {}
        observations = t_hist.get("observations") or t_hist.get("frame_observations") or []
        if not observations:
            continue

        best_f = s_track.get("best_frame")
        chosen_obs = None

        if best_f is not None:
            for obs in observations:
                if obs.get("frame_idx") == best_f:
                    chosen_obs = obs
                    break

        if not chosen_obs:
            # Pick non-interpolated observation with largest area or middle frame
            non_interp = [o for o in observations if not o.get("interpolated", False)]
            candidates = non_interp if non_interp else observations
            # Pick candidate closest to median frame index
            mid_idx = len(candidates) // 2
            chosen_obs = candidates[mid_idx]

        f_idx = int(chosen_obs["frame_idx"])
        bbox = chosen_obs["bbox"]
        track_selected_frame[tid] = f_idx
        frame_targets.setdefault(f_idx, []).append((tid, bbox))

    extracted_thumbnails: Dict[int, Path] = {}
    if not frame_targets:
        return extracted_thumbnails

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {v_path}")

    sorted_frames = sorted(frame_targets.keys())
    curr_frame = 0
    target_idx = 0

    while cap.isOpened() and target_idx < len(sorted_frames):
        target_f = sorted_frames[target_idx]

        if target_f - curr_frame > 5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            curr_frame = target_f
        else:
            while curr_frame < target_f:
                cap.grab()
                curr_frame += 1

        ret, frame = cap.read()
        if not ret:
            break
        curr_frame += 1

        H, W = frame.shape[:2]
        for tid, bbox in frame_targets[target_f]:
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            # Add ~8% margin
            px1 = max(0, int(x1 - 0.08 * bw))
            py1 = max(0, int(y1 - 0.08 * bh))
            px2 = min(W, int(x2 + 0.08 * bw))
            py2 = min(H, int(y2 + 0.08 * bh))

            crop = frame[py1:py2, px1:px2]
            if crop.size > 0:
                ch, cw = crop.shape[:2]
                if cw > max_width:
                    ch = int(ch * (float(max_width) / cw))
                    cw = max_width
                    crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)

                thumb_path = thumb_dir / f"track_{tid:03d}.jpg"
                cv2.imwrite(str(thumb_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extracted_thumbnails[tid] = thumb_path

        target_idx += 1

    cap.release()
    logger.info(f"Extracted {len(extracted_thumbnails)} track thumbnails into {thumb_dir}")
    return extracted_thumbnails


def build_web_artifacts(
    clip_stem: str,
    output_dir: Union[str, Path],
    video_path: Union[str, Path],
) -> Dict[str, Any]:
    """Orchestrate building boxes.json and thumbnail images for a processed video clip."""
    out_d = Path(output_dir)
    hist_p = out_d / "track_history.json"
    summ_p = out_d / "tracks_summary.json"
    boxes_p = out_d / "boxes.json"
    thumbs_d = out_d / "thumbnails"

    boxes = build_boxes_json(hist_p, summ_p, boxes_p)
    thumbs = generate_track_thumbnails(video_path, hist_p, summ_p, thumbs_d)

    return {
        "boxes_path": boxes_p,
        "boxes_url": f"/outputs/{clip_stem}/boxes.json",
        "thumbnails_dir": thumbs_d,
        "thumbnails_count": len(thumbs),
        "total_frames": boxes.get("total_frames", 0),
    }
