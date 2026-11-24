"""Vehicle tracking pipeline orchestrating detection, BoT-SORT tracking, visual annotation, crop selection, tracklet stitching, and contact-sheet generation."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Set

import cv2
import torch
from ultralytics import YOLO

from .annotator import VehicleAnnotator
from .config import (
    ROAD_VEHICLE_CLASS_IDS,
    ROAD_VEHICLE_CLASSES,
    COCO_TO_CATEGORY,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRACKER_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_IMGSZ,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_BOX_AREA,
    DEFAULT_MIN_BOX_DIM,
    DEFAULT_DEVICE,
    DEFAULT_MAX_STITCH_GAP_SEC,
    DEFAULT_MAX_STITCH_SPEED_PX_S,
    DEFAULT_MIN_HIST_CORREL,
    DEFAULT_MAX_LAB_DIST,
    DEFAULT_MAX_SIZE_RATIO,
    DEFAULT_MAX_LATERAL_ERR_PX,
    DEFAULT_MAX_CONSECUTIVE_AREA_JUMP,
)
from .crop_selector import BestCropSelector
from .stitcher import TrackletStitcher
from .splitter import split_tracks_on_sudden_jumps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def make_json_serializable(obj: Any) -> Any:
    """Recursively converts numpy and non-standard types to pure Python JSON-serializable types."""
    import numpy as np
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(x) for x in obj]
    return obj


def interpolate_track_gaps(
    tracks: Union[Dict[int, List[dict]], Dict[str, Any], List[dict]],
    max_gap_frames: int = 6,
    kept_track_ids: Optional[Union[Set[int], List[int]]] = None,
) -> Any:
    """Linearly interpolate bounding boxes across small temporal gaps (1 < gap <= max_gap_frames).

    Marks real detections with 'interpolated': False and synthesized frames with 'interpolated': True.
    Does not interpolate across gaps > max_gap_frames.
    Supports:
      1. frame_observations format: Dict[frame_idx -> List[detection_dict]]
      2. track_history['tracks'] format: Dict[track_id -> {'observations': List[dict]}]
      3. List of observation dicts: List[dict with 'frame_idx' and 'bbox']
    """
    if isinstance(tracks, list):
        if len(tracks) < 2:
            for d in tracks:
                d.setdefault("interpolated", False)
            return tracks

        sorted_obs = sorted(tracks, key=lambda x: x["frame_idx"])
        for d in sorted_obs:
            d.setdefault("interpolated", False)

        interpolated_list = []
        for i in range(len(sorted_obs) - 1):
            obs_a = sorted_obs[i]
            obs_b = sorted_obs[i + 1]
            interpolated_list.append(obs_a)

            f_a = obs_a["frame_idx"]
            f_b = obs_b["frame_idx"]
            gap = f_b - f_a

            if 1 < gap <= max_gap_frames:
                box_a = obs_a["bbox"]
                box_b = obs_b["bbox"]
                conf_a = float(obs_a.get("confidence", obs_a.get("conf", 0.0)))
                conf_b = float(obs_b.get("confidence", obs_b.get("conf", 0.0)))

                for f in range(f_a + 1, f_b):
                    alpha = (f - f_a) / float(gap)
                    interp_box = [
                        round(float(box_a[k] + alpha * (box_b[k] - box_a[k])), 2)
                        for k in range(4)
                    ]
                    w = max(0.0, interp_box[2] - interp_box[0])
                    h = max(0.0, interp_box[3] - interp_box[1])
                    interp_obs = {
                        "frame_idx": int(f),
                        "bbox": interp_box,
                        "confidence": round(float(conf_a + alpha * (conf_b - conf_a)), 3),
                        "conf": round(float(conf_a + alpha * (conf_b - conf_a)), 3),
                        "area": round(float(w * h), 1),
                        "interpolated": True,
                    }
                    if "id" in obs_a:
                        interp_obs["id"] = obs_a["id"]
                    if "label" in obs_a:
                        interp_obs["label"] = obs_a["label"]
                    if "timestamp_sec" in obs_a and "timestamp_sec" in obs_b:
                        ts_a = float(obs_a["timestamp_sec"])
                        ts_b = float(obs_b["timestamp_sec"])
                        interp_obs["timestamp_sec"] = round(float(ts_a + alpha * (ts_b - ts_a)), 3)
                    if "timestamp" in obs_a and "timestamp_sec" in interp_obs:
                        interp_obs["timestamp"] = format_timestamp(interp_obs["timestamp_sec"])
                    interpolated_list.append(interp_obs)

        interpolated_list.append(sorted_obs[-1])
        return interpolated_list

    if isinstance(tracks, dict) and any(isinstance(v, dict) and "observations" in v for v in tracks.values()):
        for tid, tdata in tracks.items():
            if kept_track_ids is not None and int(tid) not in kept_track_ids:
                continue
            if "observations" in tdata and isinstance(tdata["observations"], list):
                tdata["observations"] = interpolate_track_gaps(tdata["observations"], max_gap_frames)
                tdata["frame_count"] = len(tdata["observations"])
        return tracks

    if isinstance(tracks, dict):
        frame_obs = tracks
        for f_idx, dets in frame_obs.items():
            for d in dets:
                d.setdefault("interpolated", False)

        by_track: Dict[int, List[dict]] = {}
        for f_idx in sorted(frame_obs.keys()):
            for det in frame_obs[f_idx]:
                tid = det["id"]
                if kept_track_ids is not None and tid not in kept_track_ids:
                    continue
                d_copy = dict(det)
                d_copy["frame_idx"] = f_idx
                by_track.setdefault(tid, []).append(d_copy)

        for tid, obs_list in by_track.items():
            if len(obs_list) < 2:
                continue
            for i in range(len(obs_list) - 1):
                obs_a = obs_list[i]
                obs_b = obs_list[i + 1]
                f_a = obs_a["frame_idx"]
                f_b = obs_b["frame_idx"]
                gap = f_b - f_a

                if 1 < gap <= max_gap_frames:
                    box_a = obs_a["bbox"]
                    box_b = obs_b["bbox"]
                    conf_a = float(obs_a.get("conf", obs_a.get("confidence", 0.0)))
                    conf_b = float(obs_b.get("conf", obs_b.get("confidence", 0.0)))
                    label = obs_a.get("label", obs_b.get("label", "vehicle"))

                    for f in range(f_a + 1, f_b):
                        alpha = (f - f_a) / float(gap)
                        interp_box = [
                            round(float(box_a[k] + alpha * (box_b[k] - box_a[k])), 2)
                            for k in range(4)
                        ]
                        w = max(0.0, interp_box[2] - interp_box[0])
                        h = max(0.0, interp_box[3] - interp_box[1])
                        interp_det = {
                            "id": int(tid),
                            "label": label,
                            "conf": round(float(conf_a + alpha * (conf_b - conf_a)), 3),
                            "confidence": round(float(conf_a + alpha * (conf_b - conf_a)), 3),
                            "bbox": interp_box,
                            "area": round(float(w * h), 1),
                            "interpolated": True,
                        }
                        frame_obs.setdefault(f, []).append(interp_det)
        return frame_obs


class VehicleTrackingPipeline:
    """End-to-end pipeline for vehicle detection, tracking with ReID, annotation, and export."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        imgsz: int = DEFAULT_IMGSZ,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        min_frames: int = DEFAULT_MIN_FRAMES,
        min_box_area: int = DEFAULT_MIN_BOX_AREA,
        min_box_dim: int = DEFAULT_MIN_BOX_DIM,
        device: Union[int, str] = DEFAULT_DEVICE,
        max_stitch_gap_sec: float = DEFAULT_MAX_STITCH_GAP_SEC,
        max_stitch_speed_px_s: float = DEFAULT_MAX_STITCH_SPEED_PX_S,
        min_hist_correlation: float = DEFAULT_MIN_HIST_CORREL,
        max_lab_distance: float = DEFAULT_MAX_LAB_DIST,
        max_size_ratio: float = DEFAULT_MAX_SIZE_RATIO,
        max_lateral_err_px: float = DEFAULT_MAX_LATERAL_ERR_PX,
        max_area_jump_ratio: float = DEFAULT_MAX_CONSECUTIVE_AREA_JUMP,
        stride: Optional[int] = None,
        half: Optional[bool] = None,
        write_video: bool = True,
    ):
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.output_dir = Path(output_dir)
        self.tracks_dir = self.output_dir / "tracks"
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.min_frames = min_frames
        self.min_box_area = min_box_area
        self.min_box_dim = min_box_dim
        self.device = device
        self.max_area_jump_ratio = max_area_jump_ratio
        self.max_lateral_err_px = max_lateral_err_px
        self.stride = stride
        if half is None:
            is_cuda = torch.cuda.is_available() and (str(self.device) in ("0", "cuda") or isinstance(self.device, int))
            self.half = bool(is_cuda)
        else:
            self.half = bool(half)
        self.write_video = write_video

        self.stitcher = TrackletStitcher(
            max_gap_seconds=max_stitch_gap_sec,
            max_speed_px_s=max_stitch_speed_px_s,
            min_hist_correlation=min_hist_correlation,
            max_lab_distance=max_lab_distance,
            max_size_ratio=max_size_ratio,
            max_lateral_err_px=max_lateral_err_px,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

        # Check and log device status
        if torch.cuda.is_available() and (str(self.device) in ("0", "cuda") or isinstance(self.device, int)):
            dev_idx = int(self.device) if str(self.device).isdigit() else self.device
            self.device_name = torch.cuda.get_device_name(dev_idx)
            logger.info(f"NVIDIA GPU Acceleration enabled: {self.device_name} (device={self.device})")
        else:
            self.device_name = "CPU"
            logger.info(f"Running on device: {self.device} ({self.device_name})")

        logger.info(f"Loading YOLO model from '{self.model_path}' at imgsz={self.imgsz}...")
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded. Using tracker config: '{self.tracker_config}'")
        logger.info(
            f"Pipeline config: detector_conf={self.conf_threshold}, imgsz={self.imgsz}, "
            f"post-filter: min_frames={self.min_frames} (>=0.5s), min_box_area={self.min_box_area}px, "
            f"stitcher: max_gap={max_stitch_gap_sec}s, max_speed={max_stitch_speed_px_s}px/s"
        )

    def run(
        self,
        video_path: str,
        output_video_name: str = "activa_signal_tracked.mp4",
        write_video: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute tracking pipeline on the specified video file."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        effective_stride = self.stride if self.stride is not None else (2 if fps >= 24.0 else 1)
        do_write_video = self.write_video if write_video is None else write_video

        logger.info(
            f"Processing video: {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames, {duration:.2f}s)"
        )
        logger.info(
            f"Pass 1 Config: imgsz={self.imgsz}, vid_stride={effective_stride}, half={self.half}, write_video={do_write_video}"
        )

        crop_selector = BestCropSelector(padding_ratio=0.06, min_size=8)
        annotator = VehicleAnnotator(font_scale=0.6, box_thickness=2)

        # Track history and frame observations
        tracks_meta: Dict[int, dict] = {}
        frame_observations: Dict[int, List[dict]] = {}

        # PASS 1: Stream tracking with GPU FP16 acceleration and optimal resolution
        logger.info(
            f"Pass 1: Running BoT-SORT tracking at imgsz={self.imgsz} (vid_stride={effective_stride}, half={self.half}) on {self.device_name}..."
        )
        results_generator = self.model.track(
            source=str(video_path),
            tracker=self.tracker_config,
            classes=ROAD_VEHICLE_CLASS_IDS,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            vid_stride=effective_stride,
            show=False,
            stream=True,
            verbose=False,
            persist=True,
        )

        for step_idx, result in enumerate(results_generator):
            frame = result.orig_img
            orig_frame_idx = step_idx * effective_stride
            timestamp_sec = orig_frame_idx / fps if fps > 0 else 0.0
            current_frame_dets = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for bbox, tid, cid, conf in zip(boxes_xyxy, track_ids, class_ids, confidences):
                    box_w = bbox[2] - bbox[0]
                    box_h = bbox[3] - bbox[1]
                    area = box_w * box_h
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0

                    class_name = self.model.names.get(cid, ROAD_VEHICLE_CLASSES.get(cid, "vehicle"))
                    category = COCO_TO_CATEGORY.get(class_name, "vehicle")

                    # Extract raw vehicle crop for boundary visual matching
                    x1, y1, x2, y2 = [int(round(c)) for c in bbox]
                    x1 = max(0, min(frame.shape[1] - 1, x1))
                    y1 = max(0, min(frame.shape[0] - 1, y1))
                    x2 = max(0, min(frame.shape[1], x2))
                    y2 = max(0, min(frame.shape[0], y2))
                    raw_crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

                    if tid not in tracks_meta:
                        tracks_meta[tid] = {
                            "id": int(tid),
                            "class_name": class_name,
                            "category": category,
                            "start_frame": orig_frame_idx,
                            "end_frame": orig_frame_idx,
                            "start_timestamp_sec": timestamp_sec,
                            "end_timestamp_sec": timestamp_sec,
                            "frame_count": 1,
                            "max_conf": float(conf),
                            "conf_sum": float(conf),
                            "max_area": float(area),
                            "max_dim": max(float(box_w), float(box_h)),
                            "first_bbox": bbox.tolist(),
                            "last_bbox": bbox.tolist(),
                            "first_crop": raw_crop,
                            "last_crop": raw_crop,
                            "head_crops": [raw_crop] if raw_crop is not None else [],
                            "tail_crops": [raw_crop] if raw_crop is not None else [],
                            "trajectory": [(orig_frame_idx, cx, cy)],
                        }
                    else:
                        tracks_meta[tid]["end_frame"] = orig_frame_idx
                        tracks_meta[tid]["end_timestamp_sec"] = timestamp_sec
                        tracks_meta[tid]["frame_count"] += 1
                        tracks_meta[tid]["max_conf"] = max(tracks_meta[tid]["max_conf"], float(conf))
                        tracks_meta[tid]["conf_sum"] += float(conf)
                        tracks_meta[tid]["max_area"] = max(tracks_meta[tid]["max_area"], float(area))
                        tracks_meta[tid]["max_dim"] = max(tracks_meta[tid]["max_dim"], float(box_w), float(box_h))
                        tracks_meta[tid]["last_bbox"] = bbox.tolist()
                        if raw_crop is not None:
                            tracks_meta[tid]["last_crop"] = raw_crop
                            if len(tracks_meta[tid]["head_crops"]) < 6:
                                tracks_meta[tid]["head_crops"].append(raw_crop)
                            tracks_meta[tid]["tail_crops"].append(raw_crop)
                            if len(tracks_meta[tid]["tail_crops"]) > 6:
                                tracks_meta[tid]["tail_crops"].pop(0)
                        tracks_meta[tid]["trajectory"].append((orig_frame_idx, cx, cy))

                    crop_selector.update(
                        track_id=int(tid),
                        frame=frame,
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        frame_idx=orig_frame_idx,
                        timestamp_sec=timestamp_sec,
                    )

                    current_frame_dets.append(
                        {
                            "id": int(tid),
                            "label": class_name,
                            "conf": float(conf),
                            "bbox": bbox.tolist(),
                            "area": float(area),
                            "interpolated": False,
                        }
                    )

            frame_observations[orig_frame_idx] = current_frame_dets

            if (orig_frame_idx % (50 * effective_stride) == 0) or orig_frame_idx >= total_frames - effective_stride:
                logger.info(
                    f"Tracking Frame {orig_frame_idx:3d}/{total_frames} | Raw Tracks: {len(tracks_meta):2d}"
                )

        # PRE-STITCH TRACK SPLITTING: Split tracks with abrupt class switches or size jumps (> 2.2x)
        tracks_meta, frame_observations, split_audit = split_tracks_on_sudden_jumps(
            tracks_meta=tracks_meta,
            frame_observations=frame_observations,
            crop_selector=crop_selector,
            video_path=str(video_path),
            fps=fps,
            max_area_jump_ratio=self.max_area_jump_ratio,
        )

        # POST-TRACKING TRACKLET STITCHING: Reconnect occluded vehicle tracks across brief gaps
        logger.info("Running post-tracking tracklet stitching on raw tracks...")
        tracks_meta, frame_observations, stitched_pairs = self.stitcher.stitch(
            tracks_meta=tracks_meta,
            crop_selector=crop_selector,
            frame_observations=frame_observations,
            fps=fps,
        )
        logger.info(f"Stitching finished: {len(stitched_pairs)} track merges executed.")

        # POST-TRACKING FILTERING: Exclude tiny distant clutter and brief transient flickers
        logger.info(f"Applying post-tracking filters (min_frames >= {self.min_frames}, max_area >= {self.min_box_area}px)...")
        effective_min_frames = max(5, math.ceil(self.min_frames / effective_stride)) if effective_stride > 1 else self.min_frames
        kept_track_ids = set()
        for tid, meta in tracks_meta.items():
            duration_obs = meta["end_timestamp_sec"] - meta["start_timestamp_sec"]
            if meta["frame_count"] < effective_min_frames and duration_obs < 0.5:
                continue
            if meta["max_area"] < self.min_box_area or meta["max_dim"] < self.min_box_dim:
                continue
            kept_track_ids.add(tid)

        logger.info(f"Post-filtering: {len(kept_track_ids)} / {len(tracks_meta)} tracks kept.")

        # SMOOTH TEMPORAL BOX INTERPOLATION: linearly bridge small gaps (<= max_gap_frames) across strided frames
        max_gap_frames = max(6, effective_stride * 3)
        frame_observations = interpolate_track_gaps(
            tracks=frame_observations,
            max_gap_frames=max_gap_frames,
            kept_track_ids=kept_track_ids,
        )

        # PASS 2: Render output video with ONLY kept tracks for pristine visual verification (optional)
        output_video_path = None
        if do_write_video:
            output_video_path = self.output_dir / output_video_name
            logger.info(f"Pass 2: Writing annotated preview video (max 1280px) to {output_video_path}...")
            cap = cv2.VideoCapture(str(video_path))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            scale = min(1.0, 1280.0 / width) if width > 1280 else 1.0
            out_w = int(round(width * scale))
            out_h = int(round(height * scale))
            out_w -= (out_w % 2)
            out_h -= (out_h % 2)

            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (out_w, out_h))

            for f_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Filter frame detections to kept tracks only
                valid_dets = [d for d in frame_observations.get(f_idx, []) if d["id"] in kept_track_ids]

                if scale < 1.0:
                    frame_to_draw = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    scaled_dets = []
                    for d in valid_dets:
                        d_copy = dict(d)
                        d_copy["bbox"] = [c * scale for c in d["bbox"]]
                        scaled_dets.append(d_copy)
                    dets_to_draw = scaled_dets
                else:
                    frame_to_draw = frame
                    dets_to_draw = valid_dets

                annotated_frame = annotator.annotate_frame(
                    frame=frame_to_draw,
                    tracks=dets_to_draw,
                    frame_idx=f_idx,
                    total_frames=total_frames,
                    fps=fps,
                    total_unique_ids=len(kept_track_ids),
                )
                writer.write(annotated_frame)

            cap.release()
            writer.release()
            logger.info(f"Pass 2 complete. Video saved: {output_video_path}")

            # Create alias for activa_signal if needed
            alias_video_path = self.output_dir / "annotated_activa_signal.mp4"
            if "activa_signal" in video_path.stem and output_video_path != alias_video_path:
                import shutil
                shutil.copyfile(str(output_video_path), str(alias_video_path))
        else:
            logger.info("Pass 2: Skipped annotated video rendering (write_video=False).")

        # Clean existing crops directory
        for f in self.tracks_dir.glob("*.jpg"):
            f.unlink()

        # Build final qualified track records and save best crops
        qualified_tracks = []
        for tid in sorted(kept_track_ids):
            meta = tracks_meta[tid]
            best_crop = crop_selector.get_best(tid)
            saved_image_rel_path = None
            sharpness_val = None
            best_frame_idx = None
            crop_img = None

            if best_crop is not None and best_crop.crop_image is not None and best_crop.crop_image.size > 0:
                img_filename = f"track_{tid:03d}_{meta['class_name']}.jpg"
                img_path = self.tracks_dir / img_filename
                cv2.imwrite(str(img_path), best_crop.crop_image)
                saved_image_rel_path = f"{self.output_dir.as_posix()}/tracks/{img_filename}"
                sharpness_val = best_crop.sharpness
                best_frame_idx = best_crop.frame_idx
                crop_img = best_crop.crop_image

            avg_conf = round(meta["conf_sum"] / meta["frame_count"], 3)
            duration_sec = round(meta["end_timestamp_sec"] - meta["start_timestamp_sec"] + (1.0 / fps), 3)

            track_entry = {
                "id": meta["id"],
                "class_name": meta["class_name"],
                "category": meta["category"],
                "start_frame": meta["start_frame"],
                "end_frame": meta["end_frame"],
                "start_timestamp": format_timestamp(meta["start_timestamp_sec"]),
                "end_timestamp": format_timestamp(meta["end_timestamp_sec"]),
                "start_time_seconds": round(meta["start_timestamp_sec"], 3),
                "end_time_seconds": round(meta["end_timestamp_sec"], 3),
                "duration_seconds": duration_sec,
                "frame_count": meta["frame_count"],
                "avg_confidence": avg_conf,
                "max_confidence": round(meta["max_conf"], 3),
                "max_box_area": round(meta["max_area"], 1),
                "best_frame": best_frame_idx,
                "sharpness_score": sharpness_val,
                "saved_image_path": saved_image_rel_path,
                "crop_image": crop_img,
                "original_ids": meta.get("original_ids", [meta["id"]]),
            }
            qualified_tracks.append(track_entry)

        # Save frame-by-frame track history for all kept vehicles
        track_history_data: Dict[str, Any] = {
            "video_source": str(video_path),
            "video_resolution": f"{width}x{height}",
            "video_fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 3),
            "total_tracked_vehicles": len(kept_track_ids),
            "tracks": {},
        }

        for tid in sorted(kept_track_ids):
            meta = tracks_meta[tid]
            track_history_data["tracks"][str(tid)] = {
                "id": int(tid),
                "class_name": meta["class_name"],
                "category": meta["category"],
                "start_frame": int(meta["start_frame"]),
                "end_frame": int(meta["end_frame"]),
                "start_timestamp": format_timestamp(meta["start_timestamp_sec"]),
                "end_timestamp": format_timestamp(meta["end_timestamp_sec"]),
                "duration_seconds": round(meta["end_timestamp_sec"] - meta["start_timestamp_sec"] + (1.0 / fps), 3),
                "frame_count": int(meta["frame_count"]),
                "original_ids": [int(x) for x in meta.get("original_ids", [tid])],
                "observations": [],
            }

        for f_idx in sorted(frame_observations.keys()):
            t_sec = f_idx / fps
            for det in frame_observations[f_idx]:
                tid = det["id"]
                if tid in kept_track_ids:
                    obs_entry = {
                        "frame_idx": int(f_idx),
                        "timestamp_sec": round(t_sec, 3),
                        "timestamp": format_timestamp(t_sec),
                        "bbox": [round(float(c), 2) for c in det["bbox"]],
                        "confidence": round(float(det.get("conf", 0.0)), 3),
                        "area": round(float(det.get("area", 0.0)), 1),
                        "interpolated": bool(det.get("interpolated", False)),
                    }
                    track_history_data["tracks"][str(tid)]["observations"].append(obs_entry)

        for tid in sorted(kept_track_ids):
            track_history_data["tracks"][str(tid)]["frame_count"] = len(
                track_history_data["tracks"][str(tid)]["observations"]
            )

        history_file = self.output_dir / "track_history.json"
        clean_history = make_json_serializable(track_history_data)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(clean_history, f, indent=2)
        logger.info(f"Saved frame-by-frame track history to '{history_file}'.")

        # Clean non-serializable fields before dumping to JSON
        json_tracks = []
        for t in qualified_tracks:
            entry = dict(t)
            entry.pop("crop_image", None)
            json_tracks.append(entry)

        summary_data = {
            "video_source": str(video_path),
            "video_resolution": f"{width}x{height}",
            "video_fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 3),
            "inference_imgsz": self.imgsz,
            "inference_conf_threshold": self.conf_threshold,
            "device": str(self.device),
            "device_name": self.device_name,
            "post_filters": {
                "min_frames": self.min_frames,
                "min_box_area": self.min_box_area,
                "min_box_dim": self.min_box_dim,
            },
            "stitching_config": {
                "max_gap_seconds": self.stitcher.max_gap_seconds,
                "max_speed_px_s": self.stitcher.max_speed_px_s,
                "max_size_ratio": self.stitcher.max_size_ratio,
                "min_hist_correlation": self.stitcher.min_hist_correlation,
                "max_lab_distance": self.stitcher.max_lab_distance,
            },
            "split_jumps_count": len(split_audit),
            "split_jumps": split_audit,
            "stitched_merges_count": len(stitched_pairs),
            "stitched_merges": stitched_pairs,
            "total_raw_tracks": len(tracks_meta),
            "tracked_vehicles_count": len(qualified_tracks),
            "output_video_path": f"{self.output_dir.as_posix()}/{output_video_name}" if do_write_video else None,
            "track_history_path": f"{self.output_dir.as_posix()}/track_history.json",
            "tracks": json_tracks,
        }

        summary_file = self.output_dir / "tracks_summary.json"
        clean_summary = make_json_serializable(summary_data)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(clean_summary, f, indent=2)

        logger.info(f"Saved tracks summary to '{summary_file}' ({len(qualified_tracks)} clean vehicles recorded).")
        return clean_summary

    process_video = run

