"""Vehicle tracking pipeline orchestrating detection, BoT-SORT ReID tracking, visual annotation, crop selection, and summary generation."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

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
    DEFAULT_MIN_FRAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MIN_BOX_AREA,
    DEFAULT_MIN_BOX_DIM,
    DEFAULT_DEVICE,
)
from .crop_selector import BestCropSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


class VehicleTrackingPipeline:
    """End-to-end pipeline for vehicle detection, tracking with ReID, annotation, and export."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        min_frames: int = DEFAULT_MIN_FRAMES,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        min_box_area: int = DEFAULT_MIN_BOX_AREA,
        min_box_dim: int = DEFAULT_MIN_BOX_DIM,
        device: Union[int, str] = DEFAULT_DEVICE,
    ):
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.output_dir = Path(output_dir)
        self.tracks_dir = self.output_dir / "tracks"
        self.min_frames = min_frames
        self.conf_threshold = conf_threshold
        self.min_box_area = min_box_area
        self.min_box_dim = min_box_dim
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

        # Check and log device status
        if torch.cuda.is_available() and str(self.device) in ("0", "cuda") or isinstance(self.device, int):
            self.device_name = torch.cuda.get_device_name(self.device)
            logger.info(f"NVIDIA GPU Acceleration enabled: {self.device_name} (device={self.device})")
        else:
            self.device_name = "CPU"
            logger.info(f"Running on device: {self.device} ({self.device_name})")

        logger.info(f"Loading YOLO model from '{self.model_path}'...")
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded. Using tracker config: '{self.tracker_config}'")
        logger.info(
            f"Noise filtering active: conf_threshold={self.conf_threshold}, "
            f"min_box_area={self.min_box_area}px, min_box_dim={self.min_box_dim}px, "
            f"min_frames={self.min_frames} (under 0.5s ignored)"
        )

    def run(self, video_path: str, output_video_name: str = "activa_signal_tracked.mp4") -> Dict[str, Any]:
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

        output_video_path = self.output_dir / output_video_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        logger.info(
            f"Processing video: {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames, {duration:.2f}s)"
        )
        logger.info(f"Output video destination: {output_video_path}")

        crop_selector = BestCropSelector(padding_ratio=0.06, min_size=8)
        annotator = VehicleAnnotator(font_scale=0.6, box_thickness=2)

        # Track history store
        tracks_meta: Dict[int, dict] = {}
        frame_idx = 0

        # Stream tracking with GPU acceleration and BoT-SORT native ReID
        results_generator = self.model.track(
            source=str(video_path),
            tracker=self.tracker_config,
            classes=ROAD_VEHICLE_CLASS_IDS,
            conf=self.conf_threshold,
            device=self.device,
            show=False,
            stream=True,
            verbose=False,
            persist=True,
        )

        for result in results_generator:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = result.orig_img

            timestamp_sec = frame_idx / fps if fps > 0 else 0.0
            active_tracks = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for bbox, tid, cid, conf in zip(boxes_xyxy, track_ids, class_ids, confidences):
                    box_w = bbox[2] - bbox[0]
                    box_h = bbox[3] - bbox[1]
                    area = box_w * box_h

                    # Filter out tiny distant background noise
                    if area < self.min_box_area or box_w < self.min_box_dim or box_h < self.min_box_dim:
                        continue

                    class_name = self.model.names.get(cid, ROAD_VEHICLE_CLASSES.get(cid, "vehicle"))
                    category = COCO_TO_CATEGORY.get(class_name, "vehicle")

                    if tid not in tracks_meta:
                        tracks_meta[tid] = {
                            "id": int(tid),
                            "class_name": class_name,
                            "category": category,
                            "start_frame": frame_idx,
                            "end_frame": frame_idx,
                            "start_timestamp_sec": timestamp_sec,
                            "end_timestamp_sec": timestamp_sec,
                            "frame_count": 1,
                            "max_conf": float(conf),
                            "conf_sum": float(conf),
                            "max_area": float(area),
                        }
                    else:
                        tracks_meta[tid]["end_frame"] = frame_idx
                        tracks_meta[tid]["end_timestamp_sec"] = timestamp_sec
                        tracks_meta[tid]["frame_count"] += 1
                        tracks_meta[tid]["max_conf"] = max(tracks_meta[tid]["max_conf"], float(conf))
                        tracks_meta[tid]["conf_sum"] += float(conf)
                        tracks_meta[tid]["max_area"] = max(tracks_meta[tid]["max_area"], float(area))

                    crop_selector.update(
                        track_id=int(tid),
                        frame=frame,
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                    )

                    active_tracks.append(
                        {
                            "id": int(tid),
                            "label": class_name,
                            "conf": float(conf),
                            "bbox": bbox.tolist(),
                        }
                    )

            # Draw visual annotations
            annotated_frame = annotator.annotate_frame(
                frame=frame,
                tracks=active_tracks,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                total_unique_ids=len(tracks_meta),
            )

            writer.write(annotated_frame)
            frame_idx += 1

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                logger.info(
                    f"Frame {frame_idx:3d}/{total_frames} | Active: {len(active_tracks):2d} | Total IDs: {len(tracks_meta):2d}"
                )

        cap.release()
        writer.release()
        logger.info("Video processing complete. Exporting cropped photos and tracks summary...")

        # Create annotated_activa_signal.mp4 alias if not already named that
        alias_video_path = self.output_dir / "annotated_activa_signal.mp4"
        if output_video_path != alias_video_path:
            import shutil
            shutil.copyfile(str(output_video_path), str(alias_video_path))

        # Clean existing tracks directory
        for f in self.tracks_dir.glob("*.jpg"):
            f.unlink()

        # Filter tracks with sufficient presence (ignore very brief detections under half a second)
        qualified_tracks = []
        for tid, meta in sorted(tracks_meta.items()):
            if meta["frame_count"] < self.min_frames:
                continue

            best_crop = crop_selector.get_best(tid)
            saved_image_rel_path = None
            sharpness_val = None
            best_frame_idx = None

            if best_crop is not None and best_crop.crop_image is not None and best_crop.crop_image.size > 0:
                img_filename = f"track_{tid:03d}_{meta['class_name']}.jpg"
                img_path = self.tracks_dir / img_filename
                cv2.imwrite(str(img_path), best_crop.crop_image)
                saved_image_rel_path = f"outputs/tracks/{img_filename}"
                sharpness_val = best_crop.sharpness
                best_frame_idx = best_crop.frame_idx

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
            }
            qualified_tracks.append(track_entry)

        summary_data = {
            "video_source": str(video_path),
            "video_resolution": f"{width}x{height}",
            "video_fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 3),
            "device": str(self.device),
            "device_name": self.device_name,
            "min_frames_threshold": self.min_frames,
            "confidence_threshold": self.conf_threshold,
            "min_box_area_threshold": self.min_box_area,
            "total_raw_tracks": len(tracks_meta),
            "tracked_vehicles_count": len(qualified_tracks),
            "output_video_path": f"outputs/{output_video_name}",
            "tracks": qualified_tracks,
        }

        summary_file = self.output_dir / "tracks_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Saved tracks summary to '{summary_file}' ({len(qualified_tracks)} clean vehicles recorded).")
        return summary_data
