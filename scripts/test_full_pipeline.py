#!/usr/bin/env python3
"""Interactive End-to-End Test Suite for Vehicle Traceability (Stage 1 + Stage 2).

Tests full Stage 1 (BoT-SORT vehicle tracking) and Stage 2 (2.1 vehicle cropping,
2.2 dual-detector plate localization & readability scoring, 2.3 dual-pipeline OCR
with 3-bucket arbitration, and 2.4 blind multi-tier VLM triple-check).

All test artifacts are saved into test_outputs/<clip_stem>/ without touching outputs/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from cv_pipeline.crop_extractor import extract_vehicle_crops
from cv_pipeline.ocr_engine import run_ocr_track
from cv_pipeline.pipeline import VehicleTrackingPipeline
from cv_pipeline.plate_detector import run_dual_plate_detection
from cv_pipeline.vlm_verifier import verify_track_vlm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_full_pipeline")


def open_file_in_os(path: Union[str, Path]) -> bool:
    """Open a file (video/image) in the OS default viewer across platforms."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            print(f"[WARN] File to open does not exist: {p}")
            return False
        if sys.platform.startswith("win"):
            os.startfile(str(p))
            return True
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(p)])
            return True
        else:
            subprocess.Popen(["xdg-open", str(p)])
            return True
    except Exception as e:
        print(f"[WARN] Failed to open file in OS viewer: {e}")
        return False


def format_size(size_bytes: int) -> str:
    """Format file size into human-readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def find_test_videos() -> List[Path]:
    """Scan data/Final (case-insensitive fallback) for video files."""
    candidates: List[Path] = []
    data_dir = PROJECT_ROOT / "data"

    for d_name in ("Final", "final", "FINAL"):
        d = data_dir / d_name
        if d.is_dir():
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
                candidates.extend(d.glob(ext))
            if candidates:
                break

    # Fallback to data/clips if data/Final has no videos
    if not candidates:
        clips_d = data_dir / "clips"
        if clips_d.is_dir():
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
                candidates.extend(clips_d.glob(ext))

    return sorted(list(set(candidates)), key=lambda p: p.name.lower())


def pick_video_interactively(videos: List[Path]) -> Optional[Path]:
    """Display numbered list of videos and prompt user to select one."""
    if not videos:
        print("[ERROR] No video files found in data/Final or data/clips.")
        return None

    print("\n" + "=" * 80)
    print("  STEP 1 — SELECT TEST VIDEO FROM data/Final")
    print("=" * 80)
    for i, v in enumerate(videos, start=1):
        sz = format_size(v.stat().st_size)
        print(f"  [{i:>2}] {v.name:<45} ({sz:>8})")
    print("=" * 80)

    while True:
        try:
            choice = input("Enter video number (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None

        if choice.lower() in ("q", "quit", "exit"):
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(videos):
                return videos[idx - 1]
        print(f"Invalid selection. Please enter a number between 1 and {len(videos)}.")


def run_stage_1(
    video_path: Path,
    output_dir: Path,
    device: str,
    force_fresh: bool = False,
    auto_reuse: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute Stage 1 tracking or load existing tracking if approved."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "tracks_summary.json"
    history_path = output_dir / "track_history.json"

    can_reuse = summary_path.is_file() and history_path.is_file()
    reuse = False

    if can_reuse and not force_fresh:
        if auto_reuse:
            reuse = True
        else:
            try:
                ans = input(
                    f"\n[INFO] Found existing Stage 1 tracking in '{output_dir.name}/'. Reuse? [Y/n]: "
                ).strip().lower()
                reuse = ans in ("", "y", "yes")
            except (EOFError, KeyboardInterrupt):
                reuse = True

    if reuse:
        print(f"[INFO] Reusing existing Stage 1 tracking results from {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        return summary, history

    print(f"\n[INFO] Running Stage 1 BoT-SORT Vehicle Tracking on {video_path.name}...")
    pipeline = VehicleTrackingPipeline(
        model_path="yolo26s.pt",
        tracker_config=str(PROJECT_ROOT / "cv_pipeline" / "botsort_reid.yaml"),
        output_dir=str(output_dir),
        imgsz=1280,
        min_frames=15,
        conf_threshold=0.20,
        device=device,
    )
    video_stem = video_path.stem
    output_video_name = f"{video_stem}_tracked.mp4"
    summary = pipeline.run(video_path=str(video_path), output_video_name=output_video_name)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    return summary, history


def display_tracks_table(tracks: List[Dict[str, Any]], max_rows: int = 40):
    """Print compact summary table of detected tracks."""
    print("\n" + "=" * 95)
    print("  STEP 2 — DETECTED VEHICLE TRACKS SUMMARY")
    print("=" * 95)
    print(
        f"{'Track ID':>8} | {'Class':<12} | {'Category':<12} | {'Frames':>6} | {'Duration':>8} | {'Start Time':<10} | {'End Time':<10}"
    )
    print("-" * 95)

    displayed = tracks[:max_rows]
    for t in displayed:
        tid = t.get("id")
        cls = t.get("class_name", "vehicle")
        cat = t.get("category", "vehicle")
        fc = t.get("frame_count", 0)
        dur = t.get("duration_seconds", 0.0)
        st = t.get("start_timestamp", "--")
        et = t.get("end_timestamp", "--")
        print(f"{tid:>8} | {cls:<12} | {cat:<12} | {fc:>6} | {dur:>7.2f}s | {st:<10} | {et:<10}")

    if len(tracks) > max_rows:
        print(f"... and {len(tracks) - max_rows} more tracks (total {len(tracks)} clean tracks).")
    print("=" * 95)


def run_stage_2_full(
    video_path: Path,
    track_history: Dict[str, Any],
    target_id: int,
    output_root: Path,
    device: str,
    vlm_provider: str = "auto",
    skip_vlm: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 2.1 through Stage 2.4 end-to-end for a target vehicle track."""
    t0_stage2 = time.time()
    track_dir = output_root / f"track_{target_id:03d}"

    print("\n" + "=" * 80)
    print(f"  EXECUTING FULL STAGE 2 (2.1 -> 2.4) FOR TRACK {target_id:03d}")
    print(f"  Target Track Directory: {track_dir}")
    print("=" * 80)

    # Stage 2.1: Extract lossless vehicle crops
    print(f"[Stage 2.1] Extracting vehicle crops for Track {target_id}...")
    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=target_id,
        output_dir=output_root,
        padding_pct=0.08,
    )
    raw_dir_str = manifest.get("raw_crops_dir", "")
    crop_manifest_path = Path(raw_dir_str).parent / "crop_manifest.json"

    # Stage 2.2: Dual-Detector License Plate Cropping & Dynamic Readability Selection
    print("[Stage 2.2] Running dual-detector plate localization & readability scoring...")
    plate_manifest_path = track_dir / "plate_manifest.json"
    run_dual_plate_detection(crop_manifest_path=crop_manifest_path, device=device)

    if not plate_manifest_path.is_file():
        raise FileNotFoundError(f"Plate manifest was not generated at {plate_manifest_path}")

    # Stage 2.3: Dual-Pipeline OCR & Slot Arbitration
    print("[Stage 2.3] Running RapidOCR + PARSeq dual-pipeline OCR and slot arbitration...")
    ocr_result = run_ocr_track(str(plate_manifest_path), device=device)

    # Stage 2.4: Blind Multi-Tier VLM Verification
    vlm_result = None
    if not skip_vlm:
        print(f"[Stage 2.4] Running blind multi-tier VLM verification (provider={vlm_provider})...")
        vlm_result = verify_track_vlm(track_dir=track_dir, provider=vlm_provider)

    elapsed_s2 = round(time.time() - t0_stage2, 2)
    return {
        "track_id": target_id,
        "track_dir": track_dir,
        "elapsed_seconds": elapsed_s2,
        "crop_manifest": manifest,
        "ocr_result": ocr_result,
        "vlm_result": vlm_result,
    }


def get_clearest_crop_path(track_dir: Path, ocr_result: Dict[str, Any]) -> Optional[Path]:
    """Identify the single clearest / highest-weight unenhanced license plate crop."""
    # 1. Inspect vlm_evidence_pack[0]
    vlm_pack = ocr_result.get("vlm_evidence_pack") or []
    if vlm_pack and vlm_pack[0].get("unenhanced_crop_path"):
        p = Path(vlm_pack[0]["unenhanced_crop_path"])
        if not p.is_file():
            p = track_dir / "plate_crops_selected" / "unenhanced" / p.name
        if p.is_file():
            return p

    # 2. Inspect plate_manifest.json by peak readability score
    manifest_p = track_dir / "plate_manifest.json"
    if manifest_p.is_file():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                pm = json.load(f)
            plates = [p for p in pm.get("plates", []) if p.get("is_selected_for_ocr")]
            if not plates:
                plates = pm.get("plates", [])
            if plates:
                plates_by_score = sorted(
                    plates, key=lambda x: x.get("readability_score", 0), reverse=True
                )
                top = plates_by_score[0]
                u_path = top.get("selected_unenhanced_path") or top.get("all_unenhanced_path")
                if u_path:
                    p = Path(u_path)
                    if not p.is_file():
                        p = track_dir / "plate_crops_selected" / "unenhanced" / p.name
                    if p.is_file():
                        return p
        except Exception:
            pass

    # 3. Fallback: first png in plate_crops_selected/unenhanced
    sel_unenh = track_dir / "plate_crops_selected" / "unenhanced"
    if sel_unenh.is_dir():
        pngs = sorted(sel_unenh.glob("*.png"))
        if pngs:
            return pngs[0]

    return None


def print_forensic_report(s2_res: Dict[str, Any]):
    """Print complete forensic terminal report with slot table and ghost predictions."""
    track_id = s2_res["track_id"]
    track_dir = s2_res["track_dir"]
    elapsed = s2_res["elapsed_seconds"]
    ocr_res = s2_res["ocr_result"]
    vlm_res = s2_res.get("vlm_result") or {}

    final_plate = (
        vlm_res.get("final_plate")
        or vlm_res.get("reconciled_plate")
        or ocr_res.get("recovered_plate", "??????????")
    )
    overall_conf = vlm_res.get(
        "overall_confidence", ocr_res.get("overall_confidence", 0.0)
    )
    state_valid = vlm_res.get("state_code_valid", ocr_res.get("state_code_valid", False))
    provider = vlm_res.get("provider_used", "none")
    model = vlm_res.get("model_used", "none")

    print("\n" + "=" * 95)
    print(f"  FORENSIC LICENSE PLATE RECONSTRUCTION REPORT | TRACK {track_id:03d}")
    print("=" * 95)
    print(f"  Hard Reconciled Plate (`final_plate`):  {final_plate}")
    print(f"  Overall Confidence:                    {overall_conf:.4f}")
    print(f"  Valid Indian RTO State Code:           {state_valid}")
    print(f"  VLM Provider / Model:                  {provider} ({model})")
    print(f"  Total Stage 2 Execution Time:          {elapsed:.2f}s")
    print("=" * 95)

    # Slot-by-slot table
    reconciled_slots = vlm_res.get("reconciled_slots") or []
    ocr_slots = ocr_res.get("slots") or []
    n_slots = max(len(reconciled_slots), len(ocr_slots))

    print(
        f"{'Slot':<4} | {'Type':<4} | {'Pipe A':<14} | {'Pipe B':<14} | {'Blind VLM':<12} | {'Tone':<7} | {'Final':<5} | {'Ghost Stack'}"
    )
    print("-" * 95)

    for i in range(n_slots):
        s_vlm = reconciled_slots[i] if i < len(reconciled_slots) else {}
        s_ocr = ocr_slots[i] if i < len(ocr_slots) else {}

        slot_type = s_ocr.get("slot_type", "?")
        pa_top = s_ocr.get("pipe_a_top", "?")
        pa_conf = s_ocr.get("pipe_a_conf", 0.0)
        pa_str = f"{pa_top} ({pa_conf:.2f})" if pa_top != "?" else "--"

        pb_top = s_ocr.get("pipe_b_top", "?")
        pb_conf = s_ocr.get("pipe_b_conf", 0.0)
        pb_str = f"{pb_top} ({pb_conf:.2f})" if pb_top != "?" else "--"

        v_raw = s_vlm.get("vlm_blind_raw_char") or s_vlm.get("vlm_char") or "--"
        v_conf = s_vlm.get("vlm_conf", 0.0)
        v_str = f"{v_raw} ({v_conf:.2f})" if v_raw != "--" else "--"

        tone = s_vlm.get("ui_tile_tone", "--")
        fc = s_vlm.get("final_char", s_ocr.get("char", "?"))

        ghosts = s_vlm.get("ghost_stack") or []
        ghost_str = (
            ", ".join(f"{g['char']}({g['soft_score']:.2f})" for g in ghosts)
            if ghosts
            else "-"
        )

        print(
            f"{i:<4} | {slot_type:<4} | {pa_str:<14} | {pb_str:<14} | {v_str:<12} | {tone:<7} | {fc:<5} | {ghost_str}"
        )
    print("=" * 95)

    # Ghost predictions
    ghost_preds = vlm_res.get("ghost_predictions") or []
    if ghost_preds:
        print("\n  Top-3 Ghost Predictions (Advisory Operator Re-Ranking):")
        print(f"  {'Rank':<4} | {'Plate':<12} | {'Soft Score':<10} | {'Valid RTO':<9} | {'Ghost Slot Indices'}")
        print("  " + "-" * 65)
        for gp in ghost_preds[:3]:
            r = gp.get("rank", 1)
            p = gp.get("plate", "??????????")
            sc = gp.get("soft_score", 0.0)
            vr = gp.get("state_code_valid", False)
            g_slots = gp.get("ghost_slot_indices", [])
            print(f"  {r:<4} | {p:<12} | {sc:<10.3f} | {str(vr):<9} | {g_slots}")
        print("=" * 95)

    # Asset paths
    grid_path = track_dir / "vlm_evidence_unenhanced_grid.png"
    clearest_path = get_clearest_crop_path(track_dir, ocr_res)
    print("\n  Forensic Visual Assets:")
    if clearest_path and clearest_path.is_file():
        print(f"  -> Clearest Plate Image: {clearest_path.resolve()}")
    if grid_path.is_file():
        print(f"  -> 2x2 Evidence Grid:   {grid_path.resolve()}")
    print("=" * 95)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Full-Stack Vehicle Traceability Tester (Stage 1 + Stage 2)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to specific video file (bypasses video picker).",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Target Track ID to test directly (bypasses track selection).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Compute device ('0', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "groq", "nvidia", "ollama"],
        help="VLM provider for Stage 2.4.",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip Stage 2.4 VLM and evaluate Stage 2.3 OCR only.",
    )
    parser.add_argument(
        "--reuse-tracking",
        action="store_true",
        help="Automatically reuse existing Stage 1 tracking if found without prompting.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh Stage 1 execution even if cached tracking exists.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open image or video files in OS viewer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("  INVICTUS VEHICLE TRACE — FULL-STACK PIPELINE TESTER")
    print(f"  Device: {args.device} (CUDA Available: {torch.cuda.is_available()})")
    print(f"  Output Base: {PROJECT_ROOT / 'test_outputs'}")
    print("=" * 80)

    # Main interactive loop across videos
    while True:
        if args.video:
            selected_video = Path(args.video).resolve()
            if not selected_video.is_file():
                print(f"[ERROR] Specified video file not found: {selected_video}")
                sys.exit(1)
        else:
            test_videos = find_test_videos()
            selected_video = pick_video_interactively(test_videos)
            if selected_video is None:
                print("Session terminated.")
                break

        clip_stem = selected_video.stem
        output_dir = PROJECT_ROOT / "test_outputs" / clip_stem

        # Run Stage 1
        summary, track_history = run_stage_1(
            video_path=selected_video,
            output_dir=output_dir,
            device=args.device,
            force_fresh=args.fresh,
            auto_reuse=args.reuse_tracking,
        )

        tracks = summary.get("tracks", [])
        if not tracks:
            print(f"[WARN] No clean vehicle tracks detected in {selected_video.name}.")
            if args.video:
                break
            continue

        annotated_video_path = summary.get("output_video_path")
        available_ids = {int(t["id"]): t for t in tracks}

        # Track selection loop for current video
        current_target_id = args.track_id
        while True:
            display_tracks_table(tracks)

            if current_target_id is None:
                if annotated_video_path:
                    print(f"  [v] Open Stage 1 annotated video to spot target vehicle ID")
                print("  [b] Switch to a different video clip")
                print("  [q] Quit tester\n")

                try:
                    user_inp = input("Enter Track ID to trace: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    return

                if user_inp == "q":
                    print("Exiting.")
                    return
                elif user_inp == "b":
                    break  # go back to video picker
                elif user_inp == "v":
                    if annotated_video_path:
                        print(f"[INFO] Opening annotated video: {annotated_video_path}")
                        open_file_in_os(annotated_video_path)
                    else:
                        print("[WARN] Annotated video path not found.")
                    continue
                elif user_inp.isdigit():
                    tid = int(user_inp)
                    if tid not in available_ids:
                        print(f"[ERROR] Track ID {tid} not in detected tracks list.")
                        continue
                    current_target_id = tid
                else:
                    print("Invalid input. Please enter a valid Track ID, 'v', 'b', or 'q'.")
                    continue

            # Run full Stage 2 on current_target_id
            try:
                s2_results = run_stage_2_full(
                    video_path=selected_video,
                    track_history=track_history,
                    target_id=current_target_id,
                    output_root=output_dir,
                    device=args.device,
                    vlm_provider=args.provider,
                    skip_vlm=args.skip_vlm,
                )

                # Print forensic report
                print_forensic_report(s2_results)

                # Auto-open clearest plate crop
                if not args.no_open:
                    clearest_crop = get_clearest_crop_path(
                        s2_results["track_dir"], s2_results["ocr_result"]
                    )
                    if clearest_crop and clearest_crop.is_file():
                        print(
                            f"\n[AUTO-OPEN] Opening clearest license plate image in default viewer: {clearest_crop.name}"
                        )
                        open_file_in_os(clearest_crop)

            except Exception as e:
                logger.error(f"Stage 2 failed for Track {current_target_id}: {e}", exc_info=True)

            # If user ran via non-interactive --track-id, exit once done
            if args.track_id is not None:
                return

            # Reset current_target_id to allow selecting next track
            current_target_id = None

            # Fast loop prompt
            try:
                loop_inp = input(
                    f"\nTrace another track from '{clip_stem}'? Enter Track ID, [b] change video, or [q] quit: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            if loop_inp == "q":
                print("Exiting.")
                return
            elif loop_inp == "b":
                break
            elif loop_inp.isdigit() and int(loop_inp) in available_ids:
                current_target_id = int(loop_inp)
            # Else loop will reprint table and prompt

        if args.video:
            break


if __name__ == "__main__":
    main()
