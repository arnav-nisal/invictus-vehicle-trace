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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from backend.database import (
    ensure_db_exists,
    get_all_registry_vehicles,
    get_sightings_by_plate,
    get_total_vehicles_count,
)
from backend.matcher import match_and_rank_vehicles
from backend.pdf_report import generate_forensic_pdf
from backend.schemas import BodyType, VehicleEvidence
from cv_pipeline.crop_extractor import extract_vehicle_crops, parse_target_ids
from cv_pipeline.ocr_engine import run_ocr_track
from cv_pipeline.pipeline import VehicleTrackingPipeline
from cv_pipeline.plate_detector import run_dual_plate_detection
from cv_pipeline.vehicle_crop_selector import run_stage3_vehicle_attributes
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
    imgsz: int = 960,
    stride: Optional[int] = None,
    write_video: bool = True,
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

    print(f"\n[INFO] Running Stage 1 BoT-SORT Vehicle Tracking on {video_path.name} (imgsz={imgsz}, stride={stride}, write_video={write_video})...")
    pipeline = VehicleTrackingPipeline(
        model_path="yolo26s.pt",
        tracker_config=str(PROJECT_ROOT / "cv_pipeline" / "botsort_reid.yaml"),
        output_dir=str(output_dir),
        imgsz=imgsz,
        min_frames=15,
        conf_threshold=0.20,
        device=device,
        stride=stride,
        write_video=write_video,
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
    target_id: Union[int, str, List[Any]],
    output_root: Path,
    device: str,
    vlm_provider: str = "auto",
    skip_vlm: bool = False,
    skip_attributes: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 2.1 through Stage 2.4 + Stage 3 end-to-end for a target vehicle track."""
    t0_stage2 = time.time()
    target_ids = parse_target_ids(target_id)
    primary_id = target_ids[0]
    is_merged = len(target_ids) > 1

    if is_merged:
        track_folder_name = f"track_{'_'.join(f'{tid:03d}' for tid in target_ids)}"
    else:
        track_folder_name = f"track_{primary_id:03d}"

    track_dir = output_root / track_folder_name

    target_desc = f"{target_ids} (Merged Single Vehicle)" if is_merged else f"{primary_id:03d}"
    print("\n" + "=" * 80)
    print(f"  EXECUTING FULL STAGE 2 & STAGE 3 FOR TRACK {target_desc}")
    print(f"  Target Track Directory: {track_dir}")
    print("=" * 80)

    # Stage 2.1: Extract lossless vehicle crops
    print(f"[Stage 2.1] Extracting vehicle crops for Track {target_desc}...")
    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=target_ids,
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

    # Stage 3: Vehicle Attribute Extraction (Top-3 sharpest vehicle crops)
    vehicle_attributes = None
    if not skip_attributes:
        print("[Stage 3] Running vehicle attribute extraction (Top-3 sharpest crops)...")
        coarse_cls = manifest.get("class_name") or manifest.get("category")
        try:
            vehicle_attributes = run_stage3_vehicle_attributes(
                track_dir=track_dir,
                coarse_class=coarse_cls,
                track_id=primary_id,
                top_k=3,
            )
        except Exception as e:
            logger.error(f"Stage 3 vehicle attribute extraction failed: {e}")

    elapsed_s2 = round(time.time() - t0_stage2, 2)
    return {
        "track_id": primary_id,
        "merged_track_ids": target_ids,
        "is_merged_track": is_merged,
        "track_dir": track_dir,
        "elapsed_seconds": elapsed_s2,
        "crop_manifest": manifest,
        "ocr_result": ocr_result,
        "vlm_result": vlm_result,
        "vehicle_attributes": vehicle_attributes,
    }


def get_clearest_crop_path(track_dir: Path, ocr_result: Dict[str, Any]) -> Optional[Path]:
    """Identify the single clearest / highest-weight unenhanced license plate crop."""
    # 1. Inspect vlm_evidence_pack[0]
    vlm_pack = ocr_result.get("vlm_evidence_pack") or []
    if vlm_pack and vlm_pack[0].get("unenhanced_crop_path"):
        p = Path(vlm_pack[0]["unenhanced_crop_path"])
        for sub in ("plate_crops_selected/unenhanced", "plate_crops_all/unenhanced"):
            local_p = track_dir / Path(sub) / p.name
            if local_p.is_file():
                return local_p
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
                    for sub in ("plate_crops_selected/unenhanced", "plate_crops_all/unenhanced"):
                        local_p = track_dir / Path(sub) / p.name
                        if local_p.is_file():
                            return local_p
                    if p.is_file():
                        return p
        except Exception:
            pass

    # 3. Fallback: first png in plate_crops_selected/unenhanced or plate_crops_all/unenhanced
    for sub in ("plate_crops_selected/unenhanced", "plate_crops_all/unenhanced"):
        sel_unenh = track_dir / Path(sub)
        if sel_unenh.is_dir():
            pngs = sorted(sel_unenh.glob("*.png"))
            if pngs:
                return pngs[0]

    return None


def map_to_body_type(val: Optional[str]) -> BodyType:
    """Safely map arbitrary vehicle class/category string to BodyType enum."""
    if not val:
        return BodyType.CAR
    s = str(val).lower().strip().replace("_", "-").replace(" ", "-")
    if s in ("motorcycle", "scooter", "bike", "two-wheeler", "bicycle"):
        return BodyType.TWO_WHEELER
    if s in ("auto", "auto-rickshaw", "rickshaw", "tuk-tuk"):
        return BodyType.AUTO_RICKSHAW
    if s in ("suv", "crossover"):
        return BodyType.SUV
    if s in ("bus",):
        return BodyType.BUS
    if s in ("truck", "lorry", "pickup"):
        return BodyType.TRUCK
    return BodyType.CAR


def run_stage_4_registry_and_report(
    s2_res: Dict[str, Any],
    clip_stem: str,
    video_path: Path,
    skip_registry: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 4 SQLite registry comparison and forensic PDF generation."""
    t0_s4 = time.time()
    track_dir = Path(s2_res["track_dir"])
    primary_id = s2_res["track_id"]
    target_ids = s2_res.get("merged_track_ids", [primary_id])
    is_merged = s2_res.get("is_merged_track", False)
    track_key = track_dir.name

    ocr_res = s2_res.get("ocr_result") or {}
    vlm_res = s2_res.get("vlm_result") or {}
    manifest = s2_res.get("crop_manifest") or {}
    veh_attrs = s2_res.get("vehicle_attributes") or {}

    final_plate = (
        vlm_res.get("final_plate")
        or vlm_res.get("reconciled_plate")
        or ocr_res.get("recovered_plate", "??????????")
    )
    overall_conf = float(
        vlm_res.get("overall_confidence", ocr_res.get("overall_confidence", 0.0))
    )

    reconciled_slots = vlm_res.get("reconciled_slots", [])
    char_confs = [
        float(s.get("final_char_soft_score", s.get("vlm_conf", 0.5)))
        for s in reconciled_slots
    ]
    if not char_confs:
        ocr_slots = ocr_res.get("slots", [])
        char_confs = [
            float(s.get("final_confidence", s.get("pipe_a_conf", overall_conf)))
            for s in ocr_slots
        ]
    if not char_confs:
        char_confs = [overall_conf] * len(final_plate)

    attrs = veh_attrs.get("attributes", {}) if veh_attrs else {}
    body_val = attrs.get("body_type") or manifest.get("class_name", "car")
    body_enum = map_to_body_type(body_val)
    colour_val = attrs.get("colour", "white")

    total_db_records = 0
    formatted_matches: List[Dict[str, Any]] = []

    if not skip_registry:
        print("\n[Stage 4] Querying local SQLite registry and ranking candidate vehicles...")
        db_path = ensure_db_exists()
        total_db_records = get_total_vehicles_count(db_path)
        reg_vehicles = get_all_registry_vehicles(db_path)

        evidence = VehicleEvidence(
            track_id=primary_id,
            consensus_plate=final_plate,
            char_confidences=char_confs,
            colour=colour_val,
            colour_conf=float(attrs.get("colour_conf", 0.85)),
            body_type=body_enum,
            body_type_conf=float(attrs.get("body_type_conf", 0.85)),
            make=attrs.get("make", "Unknown"),
            make_conf=float(attrs.get("make_conf", 0.0)),
            model=attrs.get("model", "Unknown"),
            model_conf=float(attrs.get("model_conf", 0.0)),
            plate_detected=(final_plate != "NO_PLATE"),
        )

        match_resp = match_and_rank_vehicles(
            evidence=evidence,
            registry_vehicles=reg_vehicles,
            top_k=5,
        )

        for cand in match_resp.top_matches:
            sightings = []
            try:
                sightings = [s.model_dump() for s in get_sightings_by_plate(cand.plate_number, db_path)]
            except Exception:
                pass

            formatted_matches.append({
                "rank": cand.rank,
                "plate_number": cand.plate_number,
                "total_confidence": cand.composite_score,
                "plate_score": cand.score_breakdown.plate_score,
                "visual_score": cand.score_breakdown.visual_score,
                "body_type": cand.body_type,
                "colour": cand.colour,
                "make": cand.make,
                "model": cand.model,
                "owner_name": cand.owner_name or "Unknown",
                "registered_rto": cand.registered_rto or cand.rto_city,
                "registration_year": cand.registration_year,
                "sightings": sightings,
            })

        # Save registry_matches.json
        matches_path = track_dir / "registry_matches.json"
        with open(matches_path, "w", encoding="utf-8") as f:
            json.dump(formatted_matches, f, indent=2)

    # Build evidence_urls
    clearest_p = get_clearest_crop_path(track_dir, ocr_res)
    clearest_url = None
    if clearest_p:
        try:
            rel_p = clearest_p.relative_to(track_dir).as_posix()
        except ValueError:
            rel_p = clearest_p.name
        clearest_url = f"/outputs/{clip_stem}/{track_key}/{rel_p}"

    best_veh_p = track_dir / "stage3_vehicle_crops_top3.png"
    if not best_veh_p.is_file():
        raw_veh_d = track_dir / "vehicle_crops_unenhanced"
        if raw_veh_d.is_dir():
            first_v = next(raw_veh_d.glob("*.png"), None)
            if first_v:
                best_veh_p = first_v

    evidence_urls = {
        "best_vehicle_crop_url": f"/outputs/{clip_stem}/{track_key}/{best_veh_p.name}" if best_veh_p and best_veh_p.is_file() else None,
        "vehicle_contact_strip_url": f"/outputs/{clip_stem}/{track_key}/stage3_vehicle_crops_top3.png" if (track_dir / "stage3_vehicle_crops_top3.png").is_file() else None,
        "clearest_plate_crop_url": clearest_url,
        "plate_grid_unenhanced_url": f"/outputs/{clip_stem}/{track_key}/vlm_evidence_unenhanced_grid.png" if (track_dir / "vlm_evidence_unenhanced_grid.png").is_file() else None,
        "plate_grid_enhanced_url": f"/outputs/{clip_stem}/{track_key}/vlm_evidence_enhanced_grid.png" if (track_dir / "vlm_evidence_enhanced_grid.png").is_file() else None,
    }

    elapsed_total = round(s2_res.get("elapsed_seconds", 0.0) + (time.time() - t0_s4), 2)
    ver_result = {
        "case_id": f"INV-{clip_stem.upper()}-{track_key.upper()}",
        "job_id": clip_stem,
        "track_key": track_key,
        "filename": video_path.name,
        "target_id": primary_id,
        "merged_track_ids": target_ids,
        "is_merged_track": is_merged,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processing_time_sec": elapsed_total,
        "plate": {
            "final_plate": final_plate,
            "overall_confidence": overall_conf,
            "valid_state_code": vlm_res.get("state_code_valid", ocr_res.get("state_code_valid", False)),
            "provider_used": vlm_res.get("provider_used", "none"),
            "model_used": vlm_res.get("model_used", "none"),
            "reconciled_slots": vlm_res.get("reconciled_slots", []),
            "ghost_predictions": vlm_res.get("ghost_predictions", []),
        },
        "vehicle": {
            "body_type": body_val,
            "body_type_conf": float(attrs.get("body_type_conf", 0.85)),
            "colour": colour_val,
            "colour_conf": float(attrs.get("colour_conf", 0.85)),
            "make": attrs.get("make", "Unknown"),
            "make_conf": float(attrs.get("make_conf", 0.0)),
            "model": attrs.get("model", "Unknown"),
            "model_conf": float(attrs.get("model_conf", 0.0)),
            "provider_used": veh_attrs.get("provider", "none") if veh_attrs else "none",
            "selected_crops": veh_attrs.get("selected_crop_names", []) if veh_attrs else [],
        },
        "registry_matches": formatted_matches,
        "evidence_urls": evidence_urls,
        "pdf_report_url": f"/api/reports/{clip_stem}/{track_key}/pdf",
    }

    # Save verification_result.json
    ver_path = track_dir / "verification_result.json"
    with open(ver_path, "w", encoding="utf-8") as f:
        json.dump(ver_result, f, indent=2)

    # Generate Forensic PDF
    pdf_path = track_dir / f"Invictus_Report_{clip_stem}_{track_key}.pdf"
    try:
        generate_forensic_pdf(ver_result, pdf_path, track_dir=track_dir)
    except Exception as e:
        logger.error(f"Failed to generate PDF for {track_key}: {e}", exc_info=True)

    s2_res["total_db_records"] = total_db_records
    s2_res["registry_matches"] = formatted_matches
    s2_res["verification_result"] = ver_result
    s2_res["pdf_path"] = pdf_path if pdf_path.is_file() else None
    s2_res["skip_registry"] = skip_registry
    return s2_res


def print_forensic_report(s2_res: Dict[str, Any]):
    """Print complete forensic terminal report with slot table and ghost predictions."""
    track_id = s2_res["track_id"]
    is_merged = s2_res.get("is_merged_track", False)
    merged_ids = s2_res.get("merged_track_ids", [track_id])
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

    target_header = (
        f"TRACK {track_id:03d} (Merged IDs: {', '.join(str(i) for i in merged_ids)})"
        if is_merged
        else f"TRACK {track_id:03d}"
    )

    print("\n" + "=" * 95)
    print(f"  FORENSIC LICENSE PLATE RECONSTRUCTION REPORT | {target_header}")
    print("=" * 95)
    if is_merged:
        print(f"  Target Vehicle: Track #{track_id} (Merged IDs: {', '.join(str(i) for i in merged_ids)} — treated as single vehicle)")
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

    # Stage 3 Vehicle Intelligence
    vehicle_attrs = s2_res.get("vehicle_attributes")
    if vehicle_attrs:
        attrs = vehicle_attrs.get("attributes", {})
        b_type = attrs.get("body_type", "Unknown")
        b_conf = attrs.get("body_type_conf", 0.0)
        colour = attrs.get("colour", "Unknown")
        c_conf = attrs.get("colour_conf", 0.0)
        make = attrs.get("make", "Unknown")
        m_conf = attrs.get("make_conf", 0.0)
        v_model = attrs.get("model", "Unknown")
        mod_conf = attrs.get("model_conf", 0.0)
        v_prov = vehicle_attrs.get("provider", "none")
        v_model_used = vehicle_attrs.get("model_used", "none")
        crop_names = vehicle_attrs.get("selected_crop_names", [])

        print("\n" + "=" * 95)
        print(f"  STAGE 3 VEHICLE INTELLIGENCE & ATTRIBUTES | TRACK {track_id:03d}")
        print("=" * 95)
        print(f"  Body Type:      {b_type:<15} (confidence: {b_conf:.2f})")
        print(f"  Colour:         {colour:<15} (confidence: {c_conf:.2f})")
        print(f"  Make:           {make:<15} (confidence: {m_conf:.2f})")
        print(f"  Model:          {v_model:<15} (confidence: {mod_conf:.2f})")
        print(f"  Provider/Model: {v_prov} ({v_model_used})")
        if crop_names:
            print(f"  Selected Crops: {', '.join(crop_names)}")
        print("=" * 95)

    # Stage 4 SQLite Registry Comparison
    skip_reg = s2_res.get("skip_registry", False)
    if not skip_reg:
        total_db = s2_res.get("total_db_records", 0)
        matches = s2_res.get("registry_matches", [])
        print("\n" + "=" * 95)
        print(f"  STAGE 4 — SQLITE REGISTRY COMPARISON (Total Records Searched: {total_db:,})")
        print("=" * 95)
        if matches:
            print(
                f"  {'Rank':<4} | {'Plate':<12} | {'Total %':<7} | {'Plate %':<7} | {'Visual %':<8} | {'Vehicle (Colour Make Model · Body)':<33} | {'Owner / RTO'}"
            )
            print("  " + "-" * 91)
            for cand in matches[:5]:
                r = f"#{cand.get('rank', 1)}"
                p = cand.get("plate_number", "Unknown")
                tot = f"{cand.get('total_confidence', 0.0):.0%}"
                p_sc = f"{cand.get('plate_score', 0.0):.0%}"
                v_sc = f"{cand.get('visual_score', 0.0):.0%}"
                c_m_m = f"{str(cand.get('colour', '')).capitalize()} {cand.get('make', '')} {cand.get('model', '')} · {str(cand.get('body_type', '')).capitalize()}".strip()
                if len(c_m_m) > 33:
                    c_m_m = c_m_m[:30] + "..."
                owner_rto = f"{cand.get('owner_name', 'Unknown')} / {cand.get('registered_rto', 'Unknown')}"
                print(f"  {r:<4} | {p:<12} | {tot:>7} | {p_sc:>7} | {v_sc:>8} | {c_m_m:<33} | {owner_rto}")

            top_m = matches[0]
            sightings = top_m.get("sightings", [])
            if sightings:
                print(f"\n  Chronological Sightings for Rank #1 ({top_m.get('plate_number')}):")
                for s in sightings:
                    ts = s.get("timestamp", "--")
                    cam = s.get("camera_id", "CAM_01")
                    conf = f"{s.get('confidence', 0.0):.0%}"
                    print(f"    • {ts} | Camera: {cam} | Confidence: {conf}")
        else:
            print("  No matching vehicle found in local SQLite registry.")
        print("=" * 95)

    # Asset paths
    grid_path = track_dir / "vlm_evidence_unenhanced_grid.png"
    clearest_path = get_clearest_crop_path(track_dir, ocr_res)
    contact_strip_path = track_dir / "stage3_vehicle_crops_top3.png"
    pdf_path = s2_res.get("pdf_path")
    print("\n  Forensic Visual Assets:")
    if clearest_path and clearest_path.is_file():
        print(f"  -> Clearest Plate Image:        {clearest_path.resolve()}")
    if grid_path.is_file():
        print(f"  -> 2x2 Plate Evidence Grid:     {grid_path.resolve()}")
    if contact_strip_path.is_file():
        print(f"  -> 1x3 Vehicle Contact Strip:   {contact_strip_path.resolve()}")
    if pdf_path and Path(pdf_path).is_file():
        p_obj = Path(pdf_path)
        sz_str = format_size(p_obj.stat().st_size)
        print(f"  -> Forensic PDF Report:         {p_obj.resolve()} ({sz_str})")
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
        type=str,
        default=None,
        help="Target Track ID or comma-separated IDs (e.g. 26 or '26, 64') to test directly.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO inference resolution (default: 960).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frame stride for video tracking (default: auto 2 for >=24 FPS, 1 otherwise).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing annotated tracking video (Pass 2) for maximum speed.",
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
        "--skip-attributes",
        action="store_true",
        help="Skip Stage 3 vehicle attribute extraction.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Skip Stage 4 SQLite registry comparison.",
    )
    parser.add_argument(
        "--open-pdf",
        action="store_true",
        help="Automatically open generated Forensic PDF Summary Report in OS viewer.",
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
            imgsz=args.imgsz,
            stride=args.stride,
            write_video=not args.no_video,
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
        current_target_id = (
            parse_target_ids(args.track_id) if args.track_id is not None else None
        )
        if current_target_id is not None and len(current_target_id) == 1:
            current_target_id = current_target_id[0]

        while True:
            display_tracks_table(tracks)

            if current_target_id is None:
                if annotated_video_path:
                    print(f"  [v] Open Stage 1 annotated video to spot target vehicle ID")
                print("  [b] Switch to a different video clip")
                print("  [q] Quit tester\n")

                try:
                    user_inp = input("Enter Track ID (or comma-separated IDs e.g. 26, 64) to trace: ").strip().lower()
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
                else:
                    try:
                        parsed = parse_target_ids(user_inp)
                        invalid = [t for t in parsed if t not in available_ids]
                        if invalid:
                            print(f"[ERROR] Track ID(s) {invalid} not in detected tracks list. Available IDs: {sorted(available_ids.keys())}")
                            continue
                        current_target_id = parsed if len(parsed) > 1 else parsed[0]
                    except Exception:
                        print("Invalid input. Please enter valid Track ID(s), 'v', 'b', or 'q'.")
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
                    skip_attributes=args.skip_attributes,
                )

                # Run Stage 4 SQLite Registry Comparison & Forensic PDF Generation
                s2_results = run_stage_4_registry_and_report(
                    s2_res=s2_results,
                    clip_stem=clip_stem,
                    video_path=selected_video,
                    skip_registry=args.skip_registry,
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

                # Auto-open PDF if --open-pdf requested
                if args.open_pdf and not args.no_open:
                    pdf_p = s2_results.get("pdf_path")
                    if pdf_p and Path(pdf_p).is_file():
                        print(
                            f"\n[AUTO-OPEN] Opening Forensic PDF Report in default viewer: {Path(pdf_p).name}"
                        )
                        open_file_in_os(pdf_p)

            except Exception as e:
                logger.error(f"Pipeline failed for Track {current_target_id}: {e}", exc_info=True)
                s2_results = None

            # If user ran via non-interactive --track-id, exit once done
            if args.track_id is not None:
                return

            # Reset current_target_id to allow selecting next track
            current_target_id = None
            pdf_p = s2_results.get("pdf_path") if s2_results else None
            track_dir = s2_results["track_dir"] if s2_results else None

            # Upgraded post-run interactive menu
            loop_action = None
            while True:
                print("\n  [p] Open Forensic PDF Summary Report")
                print("  [c] Open 1x3 Stage 3 Vehicle Contact Strip")
                print("  [v] Re-open Stage 1 tracked video")
                print("  [b] Switch to a different video clip")
                print("  [q] Quit tester\n")

                try:
                    loop_inp = input(
                        f"Trace another track from '{clip_stem}'? Enter Track ID(s), [p], [c], [v], [b], or [q]: "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    return

                if loop_inp in ("q", "quit", "exit"):
                    print("Exiting.")
                    return
                elif loop_inp == "b":
                    loop_action = "b"
                    break
                elif loop_inp == "p":
                    if pdf_p and Path(pdf_p).is_file():
                        print(f"[INFO] Opening Forensic PDF Report: {pdf_p}")
                        open_file_in_os(pdf_p)
                    else:
                        print("[WARN] Forensic PDF Report not found.")
                    continue
                elif loop_inp == "c":
                    strip_p = track_dir / "stage3_vehicle_crops_top3.png" if track_dir else None
                    if strip_p and strip_p.is_file():
                        print(f"[INFO] Opening 1x3 Vehicle Contact Strip: {strip_p}")
                        open_file_in_os(strip_p)
                    else:
                        print("[WARN] 1x3 Vehicle Contact Strip not found.")
                    continue
                elif loop_inp == "v":
                    if annotated_video_path and Path(annotated_video_path).is_file():
                        print(f"[INFO] Opening annotated video: {annotated_video_path}")
                        open_file_in_os(annotated_video_path)
                    else:
                        print("[WARN] Annotated video path not found.")
                    continue
                else:
                    try:
                        parsed = parse_target_ids(loop_inp)
                        invalid = [t for t in parsed if t not in available_ids]
                        if invalid:
                            print(f"[ERROR] Track ID(s) {invalid} not in detected tracks list. Available IDs: {sorted(available_ids.keys())}")
                            continue
                        current_target_id = parsed if len(parsed) > 1 else parsed[0]
                        break
                    except Exception:
                        print("Invalid input. Please enter valid Track ID(s), 'p', 'c', 'v', 'b', or 'q'.")
                        continue

            if loop_action == "b":
                break

        if args.video:
            break


if __name__ == "__main__":
    main()
