#!/usr/bin/env python3
"""CLI runner for the vehicle tracking pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so cv_pipeline is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_pipeline.pipeline import VehicleTrackingPipeline
from cv_pipeline.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRACKER_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MIN_FRAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MIN_BOX_AREA,
    DEFAULT_DEVICE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invictus Vehicle Identification & Tracking Pipeline with BoT-SORT + Appearance ReID."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/clips/activa_signal.mp4",
        help="Path to input video file (default: data/clips/activa_signal.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Ultralytics model weights path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER_CONFIG,
        help=f"BoT-SORT tracker YAML config (default: {DEFAULT_TRACKER_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="activa_signal_tracked.mp4",
        help="Name of annotated output video (default: activa_signal_tracked.mp4)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help=f"Minimum frames for a vehicle to be saved to summary (default: {DEFAULT_MIN_FRAMES}, ~0.5s at 30fps)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Detection confidence threshold (default: {DEFAULT_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_BOX_AREA,
        help=f"Minimum bounding box area in pixels to ignore distant clutter (default: {DEFAULT_MIN_BOX_AREA})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Inference device: 0, cuda, or cpu (default: {DEFAULT_DEVICE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("  INVICTUS VEHICLE TRACKING PIPELINE (STAGE 1)")
    print("=" * 80)
    print(f" Input Video:        {args.video}")
    print(f" Model:              {args.model}")
    print(f" Tracker Config:     {args.tracker}")
    print(f" Inference Device:   {args.device}")
    print(f" Output Directory:   {args.output_dir}")
    print(f" Output Video Name:  {args.output_video}")
    print(f" Min Frames Filter:  {args.min_frames} (~{args.min_frames/30.0:.2f}s)")
    print(f" Confidence Thresh:  {args.conf}")
    print(f" Min Box Area:       {args.min_area} px")
    print("=" * 80)

    pipeline = VehicleTrackingPipeline(
        model_path=args.model,
        tracker_config=args.tracker,
        output_dir=args.output_dir,
        min_frames=args.min_frames,
        conf_threshold=args.conf,
        min_box_area=args.min_area,
        device=args.device,
    )

    summary = pipeline.run(video_path=args.video, output_video_name=args.output_video)

    tracks = summary["tracks"]
    print("\n" + "=" * 80)
    print(f"  TRACKING RESULTS: {len(tracks)} CLEAN VEHICLES DETECTED (>= {args.min_frames} frames)")
    print("=" * 80)
    print(f"{'ID':>4} | {'Class':<12} | {'Category':<12} | {'Frames':>6} | {'Duration':>8} | {'Start Time':<12} | {'End Time':<12} | {'Saved Image':<28}")
    print("-" * 115)

    for trk in tracks:
        image_name = Path(trk["saved_image_path"]).name if trk["saved_image_path"] else "None"
        highlight = " <-- WHITE SCOOTER (ACTIVA)" if trk["id"] == 3 else ""
        print(
            f"{trk['id']:>4} | {trk['class_name']:<12} | {trk['category']:<12} | {trk['frame_count']:>6} | "
            f"{trk['duration_seconds']:>7.2f}s | {trk['start_timestamp']:<12} | {trk['end_timestamp']:<12} | {image_name:<28}{highlight}"
        )

    print("-" * 115)
    print(f"Inference Device: {summary.get('device_name', summary.get('device'))}")
    print(f"Annotated Video:  {summary['output_video_path']}")
    print(f"Summary JSON:     {args.output_dir}/tracks_summary.json")
    print(f"Cropped Images:   {args.output_dir}/tracks/ ({len(tracks)} images saved)")
    print("=" * 80)


if __name__ == "__main__":
    main()
