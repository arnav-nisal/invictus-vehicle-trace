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
    DEFAULT_IMGSZ,
    DEFAULT_MIN_FRAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MIN_BOX_AREA,
    DEFAULT_DEVICE,
    DEFAULT_MAX_STITCH_GAP_SEC,
    DEFAULT_MAX_STITCH_SPEED_PX_S,
    DEFAULT_MIN_HIST_CORREL,
    DEFAULT_MAX_LAB_DIST,
    DEFAULT_MAX_SIZE_RATIO,
    DEFAULT_MAX_LATERAL_ERR_PX,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invictus Vehicle Identification & Tracking Pipeline with BoT-SORT + Tracklet Stitcher on NVIDIA GPU."
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
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Inference image size (default: {DEFAULT_IMGSZ})",
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
        help=f"Detection confidence threshold into tracker (default: {DEFAULT_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_BOX_AREA,
        help=f"Minimum bounding box area in pixels for post-filter (default: {DEFAULT_MIN_BOX_AREA})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Inference device: 0, cuda, or cpu (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_STITCH_GAP_SEC,
        help=f"Maximum temporal gap in seconds for tracklet stitching (default: {DEFAULT_MAX_STITCH_GAP_SEC}s)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_MAX_STITCH_SPEED_PX_S,
        help=f"Maximum vehicle velocity in px/s for tracklet stitching (default: {DEFAULT_MAX_STITCH_SPEED_PX_S}px/s)",
    )
    parser.add_argument(
        "--max-size-ratio",
        type=float,
        default=DEFAULT_MAX_SIZE_RATIO,
        help=f"Maximum bounding-box scale ratio for tracklet stitching (default: {DEFAULT_MAX_SIZE_RATIO})",
    )
    parser.add_argument(
        "--max-lateral-err",
        type=float,
        default=DEFAULT_MAX_LATERAL_ERR_PX,
        help=f"Maximum lateral cross-lane error in px for tracklet stitching (default: {DEFAULT_MAX_LATERAL_ERR_PX}px)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_stem = Path(args.video).stem
    if args.output_video == "activa_signal_tracked.mp4" and video_stem != "activa_signal":
        output_video_name = f"{video_stem}_tracked.mp4"
    else:
        output_video_name = args.output_video

    print("=" * 80)
    print("  INVICTUS VEHICLE TRACKING PIPELINE (STAGE 1 — FULL RES + TRACKLET STITCHING)")
    print("=" * 80)
    print(f" Input Video:        {args.video}")
    print(f" Model:              {args.model}")
    print(f" Tracker Config:     {args.tracker}")
    print(f" Inference Device:   {args.device}")
    print(f" Inference Img Size: {args.imgsz} px")
    print(f" Detector Conf:      {args.conf}")
    print(f" Post-Filter Min F:  {args.min_frames} frames (~{args.min_frames/30.0:.2f}s)")
    print(f" Post-Filter Min A:  {args.min_area} px")
    print(f" Stitch Max Gap:     {args.max_gap}s")
    print(f" Stitch Max Speed:   {args.max_speed} px/s")
    print(f" Stitch Size Ratio:  {args.max_size_ratio}x")
    print(f" Stitch Max LatErr:  {args.max_lateral_err} px")
    print(f" Output Directory:   {args.output_dir}")
    print(f" Output Video Name:  {output_video_name}")
    print("=" * 80)

    pipeline = VehicleTrackingPipeline(
        model_path=args.model,
        tracker_config=args.tracker,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        min_frames=args.min_frames,
        conf_threshold=args.conf,
        min_box_area=args.min_area,
        device=args.device,
        max_stitch_gap_sec=args.max_gap,
        max_stitch_speed_px_s=args.max_speed,
        max_size_ratio=args.max_size_ratio,
        max_lateral_err_px=args.max_lateral_err,
    )

    summary = pipeline.run(video_path=args.video, output_video_name=output_video_name)

    # Report split abrupt jumps
    splits = summary.get("split_jumps", [])
    if splits:
        print("\n" + "=" * 80)
        print(f"  PRE-STITCH SPLIT REPORT: {len(splits)} ABRUPT JUMP(S) SPLIT")
        print("=" * 80)
        for s in splits:
            reasons_str = "; ".join(s.get("reasons", []))
            print(f"  -> Track {s['original_tid']} split at frame {s['split_frame']} ({s['timestamp_sec']}s): {reasons_str}")
        print("=" * 80)

    # Report stitched tracklets
    merges = summary.get("stitched_merges", [])
    if merges:
        print("\n" + "=" * 80)
        print(f"  TRACKLET STITCHING REPORT: {len(merges)} BROKEN TRACK(S) RECONNECTED")
        print("=" * 80)
        for m in merges:
            print(
                f"  -> Track {m['tid_a']} + Track {m['tid_b']} | "
                f"Gap: {m['gap_sec']}s ({int(round(m['gap_sec']*30))} frames), "
                f"Dist: {m['dist_px']}px, Speed: {m['speed_px_s']}px/s, "
                f"HSV Correl: {m['hist_correl']}, Lab Dist: {m['lab_dist']}"
            )
        print("=" * 80)

    tracks = summary["tracks"]
    print("\n" + "=" * 80)
    print(f"  TRACKING RESULTS: {len(tracks)} CLEAN VEHICLES DETECTED (>= {args.min_frames} frames)")
    print("=" * 80)
    print(f"{'ID':>4} | {'Class':<12} | {'Category':<12} | {'Frames':>6} | {'Duration':>8} | {'Start Time':<12} | {'End Time':<12} | {'Saved Image':<28}")
    print("-" * 115)

    for trk in tracks:
        image_name = Path(trk["saved_image_path"]).name if trk.get("saved_image_path") else "None"
        orig_ids = trk.get("original_ids", [trk["id"]])
        highlight = ""
        orig_str = f" [merged from IDs {orig_ids}]" if len(orig_ids) > 1 else ""

        if "activa_signal" in Path(args.video).stem:
            if trk["id"] == 25 or 38 in orig_ids or 42 in orig_ids:
                highlight = f" <-- WHITE ACTIVA (CROSSING INTERSECTION){orig_str}"
            elif trk["id"] == 4 and trk["category"] == "two-wheeler":
                highlight = " <-- WHITE ACTIVA (WAITING AT SIGNAL)"
        elif len(orig_ids) > 1:
            highlight = f" <-- STITCHED TRACKLET{orig_str}"

        print(
            f"{trk['id']:>4} | {trk['class_name']:<12} | {trk['category']:<12} | {trk['frame_count']:>6} | "
            f"{trk['duration_seconds']:>7.2f}s | {trk['start_timestamp']:<12} | {trk['end_timestamp']:<12} | {image_name:<28}{highlight}"
        )

    print("-" * 115)
    print(f"Inference Device: {summary.get('device_name', summary.get('device'))}")
    print(f"Annotated Video:  {summary['output_video_path']}")
    print(f"Contact Sheet:    {summary['contact_sheet_grid_path']}")
    print(f"Summary JSON:     {args.output_dir}/tracks_summary.json")
    print(f"Cropped Images:   {args.output_dir}/tracks/ ({len(tracks)} images saved)")
    print("=" * 80)


if __name__ == "__main__":
    main()
