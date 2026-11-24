#!/usr/bin/env python3
"""CLI runner for the vehicle tracking pipeline."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so cv_pipeline is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_pipeline.pipeline import VehicleTrackingPipeline
from cv_pipeline.crop_extractor import interactive_vehicle_selection, extract_vehicle_crops
from cv_pipeline.plate_detector import run_dual_plate_detection
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
        "--stride",
        type=int,
        default=None,
        help="Frame stride for tracking (default: None for smart auto, 2 if FPS>=24, 1 if low FPS)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        default=False,
        help="Skip Pass 2 annotated video rendering for maximum speed",
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
    parser.add_argument(
        "--target-id",
        type=int,
        default=None,
        help="Target Vehicle Tracking ID to extract full-track vehicle crops for (e.g. 27)",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt interactively to select a Vehicle Tracking ID after tracking (default: True, use --no-interactive to disable)",
    )
    parser.add_argument(
        "--reuse-tracking",
        action="store_true",
        default=False,
        help="Reuse existing tracks_summary.json and track_history.json if present in output-dir without re-running tracking",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Safety margin padding percentage around bounding box for vehicle crops (default: 0.08 = 8%%)",
    )
    parser.add_argument(
        "--crop-video",
        type=str,
        default=None,
        help="Optional path to higher-resolution original video for source-mapped crop extraction (e.g. 4K original source)",
    )
    parser.add_argument(
        "--detect-plates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Stage 2.2 Dual-Detector license plate cropping & dynamic readability selection (default: True)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        default=False,
        help="Skip Stage 2.3 OCR (useful when only re-running plate detection, default: False)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        default=False,
        help="Skip Stage 2.4 Blind VLM verification (default: False)",
    )
    parser.add_argument(
        "--vlm-provider",
        type=str,
        default="auto",
        choices=["auto", "groq", "nvidia", "ollama"],
        help="VLM provider for Stage 2.4 (default: auto)",
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

    # Check for reuse-tracking
    clip_specific_summary = Path(args.output_dir) / video_stem / "tracks_summary.json"
    clip_specific_history = Path(args.output_dir) / video_stem / "track_history.json"
    if clip_specific_summary.exists() and clip_specific_history.exists():
        summary_path = clip_specific_summary
        history_path = clip_specific_history
        args.output_dir = str(Path(args.output_dir) / video_stem)
    else:
        summary_path = Path(args.output_dir) / "tracks_summary.json"
        history_path = Path(args.output_dir) / "track_history.json"
    
    if args.reuse_tracking and summary_path.exists() and history_path.exists():
        print(f"\n[INFO] --reuse-tracking flag detected. Loading existing tracking data from {args.output_dir}...")
        with open(summary_path, "r") as f:
            summary = json.load(f)
        with open(history_path, "r") as f:
            track_history = json.load(f)
        print(f"[INFO] Loaded {len(summary.get('tracks', []))} tracks and {len(track_history)} history vehicle traces successfully.")
    else:
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
            stride=args.stride,
            write_video=not args.no_video,
        )

        summary = pipeline.run(
            video_path=args.video,
            output_video_name=output_video_name,
            write_video=not args.no_video,
        )
        
        # Load track history produced by pipeline
        if history_path.exists():
            with open(history_path, "r") as f:
                track_history = json.load(f)
        else:
            track_history = {}

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
    if summary.get('track_history_path'):
        print(f"Track History:    {summary['track_history_path']}")
    print(f"Summary JSON:     {args.output_dir}/tracks_summary.json")
    print(f"Cropped Images:   {args.output_dir}/tracks/ ({len(tracks)} images saved)")
    print("=" * 80)

    # Stage 2: Interactive or Target Vehicle Selection & Full-Track Cropping
    if (args.target_id is not None or args.interactive) and track_history:
        vehicle_manifest = interactive_vehicle_selection(
            summary_data=summary,
            track_history=track_history,
            video_path=args.video,
            output_dir=args.output_dir,
            target_id=args.target_id,
            padding_pct=args.padding,
            crop_video_path=args.crop_video,
        )

        # Stage 2.2: Dual-Detector License Plate Cropping & Dynamic Readability Selection
        if args.detect_plates and vehicle_manifest is not None and vehicle_manifest.get("crops"):
            raw_dir_str = vehicle_manifest.get("raw_crops_dir", "")
            if raw_dir_str:
                crop_manifest_path = Path(raw_dir_str).parent / "crop_manifest.json"
                if crop_manifest_path.exists():
                    run_dual_plate_detection(
                        crop_manifest_path=crop_manifest_path,
                        device=args.device,
                    )

            # Stage 2.3: Dual-Pipeline OCR & 3-Bucket Arbitration
            track_dir = Path(raw_dir_str).parent if raw_dir_str else None
            if not args.skip_ocr:
                plate_manifest_path = track_dir / "plate_manifest.json" if track_dir else None
                if plate_manifest_path and plate_manifest_path.exists():
                    print("\n" + "=" * 80)
                    print("  STAGE 2.3 — DUAL-PIPELINE OCR & SLOT ARBITRATION")
                    print("=" * 80)
                    try:
                        from cv_pipeline.ocr_engine import run_ocr_track
                        import json as _json
                        ocr_result = run_ocr_track(str(plate_manifest_path), device=args.device)
                        ocr_out = track_dir / "ocr_result.json"
                        with open(ocr_out, "w") as _f:
                            _json.dump(ocr_result, _f, indent=2)
                        print(f"  Recovered plate : {ocr_result['recovered_plate']}")
                        print(f"  Overall conf    : {ocr_result['overall_confidence']:.4f}")
                        print(f"  State valid     : {ocr_result['state_code_valid']}")
                        ss = ocr_result["slot_summary"]
                        print(f"  Slots  LOCKED={ss['locked']}  CONTRADICTORY={ss['contradictory']}  UNREADABLE={ss['unreadable']}")
                        print(f"  Runtime         : {ocr_result['runtime_seconds']:.1f}s")
                        print(f"  OCR result      : {ocr_out}")
                    except Exception as _exc:
                        print(f"[ERROR] Stage 2.3 OCR failed: {_exc}")
                        import traceback; traceback.print_exc()
            else:
                print("[INFO] --skip-ocr: Stage 2.3 skipped.")

            # Stage 2.4: Blind Multi-Tier VLM Triple-Check & Tie-Breaker
            if not args.skip_vlm:
                ocr_out = track_dir / "ocr_result.json" if track_dir else None
                if ocr_out and ocr_out.exists():
                    print("\n" + "=" * 80)
                    print("  STAGE 2.4 — BLIND MULTI-TIER VLM VERIFICATION & ARBITRATION")
                    print("=" * 80)
                    try:
                        from cv_pipeline.vlm_verifier import verify_track_vlm
                        vlm_result = verify_track_vlm(track_dir=track_dir, provider=args.vlm_provider)
                        print(f"  VLM Status      : {vlm_result.get('vlm_status')}")
                        print(f"  Provider Used   : {vlm_result.get('provider_used')} ({vlm_result.get('model_used')})")
                        print(f"  Stage 2.3 Plate : {vlm_result.get('stage_2_3_plate')}")
                        print(f"  VLM Raw Plate   : {vlm_result.get('vlm_raw_plate')}")
                        print(f"  FINAL RECONCILED: {vlm_result.get('reconciled_plate')} (Valid State: {vlm_result.get('state_code_valid')})")
                        print(f"  Runtime         : {vlm_result.get('runtime_seconds'):.2f}s")
                    except Exception as _exc:
                        print(f"[ERROR] Stage 2.4 VLM failed: {_exc}")
                        import traceback; traceback.print_exc()
            else:
                print("[INFO] --skip-vlm: Stage 2.4 skipped.")


if __name__ == "__main__":
    main()
