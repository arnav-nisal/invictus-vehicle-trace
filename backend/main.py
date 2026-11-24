"""FastAPI Backend Service for Vehicle Identification and Traceability.

Exposes RESTful endpoints for video scanning, vehicle evidence analysis,
database health checks, and pre-configured realistic demo scenarios.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.database import (
    ensure_db_exists,
    get_all_registry_vehicles,
    get_sightings_by_plate,
    get_total_vehicles_count,
    record_sighting,
)
from backend.demo_data import DEMO_SCENARIOS, MOCK_VIDEO_SCAN_RESPONSE
from backend.cv_bridge import (
    get_vehicle_evidence_for_track,
    scan_video_with_cv_pipeline,
)
from backend.matcher import match_and_rank_vehicles
from backend.pdf_report import generate_forensic_pdf
from backend.report import generate_forensic_html_report
from backend.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    BodyType,
    DemoScenario,
    SightingRecord,
    VehicleEvidence,
    VideoScanResponse,
)
from cv_pipeline.crop_extractor import extract_vehicle_crops, parse_target_ids
from cv_pipeline.ocr_engine import run_ocr_track
from cv_pipeline.pipeline import VehicleTrackingPipeline
from cv_pipeline.plate_detector import run_dual_plate_detection
from cv_pipeline.vehicle_crop_selector import run_stage3_vehicle_attributes
from cv_pipeline.vlm_verifier import verify_track_vlm
from cv_pipeline.web_artifacts import build_web_artifacts, parse_resolution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vehicle_trace.main")

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CLIPS_DIR = BASE_DIR / "data" / "Final"
if not CLIPS_DIR.exists():
    CLIPS_DIR = BASE_DIR / "data" / "final"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTOR_PROVIDER = os.getenv("EXTRACTOR_PROVIDER", "opensource").lower()

# Thread pool for asynchronous GPU Stage 1 jobs (single-worker to prevent GPU collisions)
STAGE1_EXECUTOR = ThreadPoolExecutor(max_workers=1)
JOBS: Dict[str, Dict[str, Any]] = {}


def ensure_output_directories():
    """Create outputs/crops, outputs/tracks, and outputs/clips folders if missing."""
    for sub in ("tracks", "crops", "clips"):
        (OUTPUTS_DIR / sub).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup and shutdown lifecycle management."""
    logger.info("Initializing Vehicle Traceability Backend...")
    ensure_output_directories()
    # Startup check: auto-seed data/registry.db if absent
    db_path = ensure_db_exists()
    total_vehicles = get_total_vehicles_count(db_path)
    logger.info(f"Database online ({db_path}): {total_vehicles} vehicles loaded.")
    yield
    logger.info("Shutting down Vehicle Traceability Backend.")


# Initialize FastAPI application
app = FastAPI(
    title="Vehicle Identification & Traceability API",
    description="Probabilistic vehicle matching, CCTV OCR confusion resolution, and visual trace analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local React / Vite frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directory for saved crops, thumbnails, and demo clips
ensure_output_directories()
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/media/clips", StaticFiles(directory=str(CLIPS_DIR)), name="media_clips")


# ---------------------------------------------------------------------------
# API Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check payload detailing service status and database state."""
    status: str = Field(default="ok", examples=["ok"])
    service: str = Field(
        default="Vehicle Identification & Traceability Backend",
        examples=["Vehicle Identification & Traceability Backend"],
    )
    version: str = Field(default="1.0.0", examples=["1.0.0"])
    database_connected: bool = Field(default=True)
    registry_vehicles_count: int = Field(..., ge=0, examples=[1020])


class SelectLocalRequest(BaseModel):
    filename: str = Field(..., description="Name of the video file in data/Final", examples=["file_07.mp4"])
    force_refresh: bool = Field(default=False, description="Whether to bypass cache and re-run Stage 1")


class PipelineVerifyRequest(BaseModel):
    job_id: str = Field(..., description="Job identifier or video stem (e.g. file_07)", examples=["file_07"])
    track_ids: Optional[List[int]] = Field(default=None, description="List of track IDs to trace (e.g. [20] or [20, 33])", examples=[[20]])
    track_id: Optional[int] = Field(default=None, description="Single track ID (fallback for track_ids)", examples=[20])
    force_refresh: bool = Field(default=False, description="Whether to bypass disk cache and re-run verification")


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _map_to_body_type(val: Optional[str]) -> BodyType:
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


def _find_video_path(job_id: str, filename: Optional[str] = None) -> Path:
    """Find video file on disk in data/Final or data/clips."""
    if filename:
        p = CLIPS_DIR / filename
        if p.is_file():
            return p

    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        p = CLIPS_DIR / f"{job_id}{ext}"
        if p.is_file():
            return p

    data_clips = BASE_DIR / "data" / "clips"
    if data_clips.is_dir():
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            p = data_clips / f"{job_id}{ext}"
            if p.is_file():
                return p

    return CLIPS_DIR / f"{job_id}.mp4"


def _get_clearest_crop_path(track_dir: Path, ocr_result: Dict[str, Any]) -> Optional[Path]:
    """Identify the clearest unenhanced plate crop from OCR evidence pack or manifest."""
    vlm_pack = ocr_result.get("vlm_evidence_pack") or []
    if vlm_pack and vlm_pack[0].get("unenhanced_crop_path"):
        p = Path(vlm_pack[0]["unenhanced_crop_path"])
        if not p.is_file():
            p = track_dir / "plate_crops_selected" / "unenhanced" / p.name
        if p.is_file():
            return p

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
                        p = track_dir / "plate_crops_all" / "unenhanced" / p.name
                    if p.is_file():
                        return p
        except Exception:
            pass

    unenh_dir = track_dir / "plate_crops_selected" / "unenhanced"
    if unenh_dir.is_dir():
        first_p = next(unenh_dir.glob("*.png"), None)
        if first_p:
            return first_p
    return None


def _build_stage1_result(clip_stem: str, video_path: Path, tracks_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Format tracks_summary into the exact result structure required by Screen 2 frontend."""
    width, height = parse_resolution(tracks_summary.get("video_resolution"))
    tracks_list = [
        {
            "track_id": int(t["id"]),
            "vehicle_class": t.get("class_name", "vehicle"),
            "category": t.get("category", t.get("class_name", "vehicle")),
            "crop_url": f"/outputs/{clip_stem}/thumbnails/track_{int(t['id']):03d}.jpg",
            "start_sec": round(float(t.get("start_time_seconds", 0.0)), 2),
            "end_sec": round(float(t.get("end_time_seconds", 0.0)), 2),
            "frame_count": int(t.get("frame_count", 0)),
        }
        for t in tracks_summary.get("tracks", [])
        if "id" in t
    ]
    return {
        "job_id": clip_stem,
        "filename": video_path.name,
        "video_url": f"/media/clips/{video_path.name}",
        "fps": round(float(tracks_summary.get("video_fps", 30.0)), 2),
        "width": width,
        "height": height,
        "total_frames": int(tracks_summary.get("total_frames", 0)),
        "boxes_url": f"/outputs/{clip_stem}/boxes.json",
        "tracks": tracks_list,
    }


def _print_forensic_console_report(res: Dict[str, Any]):
    """Print the exact forensic report table directly to the backend terminal."""
    case_id = res.get("case_id", "UNKNOWN")
    job_id = res.get("job_id", "")
    track_key = res.get("track_key", "")
    target_id = res.get("target_id", 0)
    merged_ids = res.get("merged_track_ids", [target_id])
    is_merged = res.get("is_merged_track", False)
    plate_data = res.get("plate", {})
    final_plate = plate_data.get("final_plate", "??????????")
    overall_conf = plate_data.get("overall_confidence", 0.0)
    prov = plate_data.get("provider_used", "none")
    model = plate_data.get("model_used", "none")
    veh = res.get("vehicle", {})
    matches = res.get("registry_matches", [])

    print("\n" + "=" * 95, flush=True)
    print(f"  INVICTUS FORENSIC EVIDENCE REPORT | {case_id}", flush=True)
    if is_merged:
        print(f"  Target: Track #{target_id} (Merged IDs: {', '.join(str(x) for x in merged_ids)}) | Video: {job_id}", flush=True)
    else:
        print(f"  Target: Track #{target_id} | Video: {job_id}", flush=True)
    print(f"  Hard Reconciled Plate:  {final_plate}  (Overall Confidence: {overall_conf:.2f}, Provider: {prov}/{model})", flush=True)
    print("=" * 95, flush=True)
    print(f"  {'Slot':<4} | {'Type':<6} | {'Pipe A':<14} | {'Pipe B':<14} | {'Blind VLM':<14} | {'Tone':<7} | {'Final':<5} | {'Ghost Stack'}", flush=True)
    print("  " + "-" * 91, flush=True)

    slots = plate_data.get("reconciled_slots", [])
    for s in slots:
        idx = s.get("slot_idx", s.get("slot_index", 0))
        st = s.get("slot_type", s.get("expected_type", "?"))
        pa = s.get("pipe_a_candidate") or s.get("pipe_a_char") or "--"
        pa_conf = s.get("pipe_a_conf", 0.0)
        pa_str = f"{pa} ({pa_conf:.2f})" if pa != "--" and pa != "?" else (pa if pa != "?" else "--")

        pb = s.get("pipe_b_candidate") or s.get("pipe_b_char") or "--"
        pb_conf = s.get("pipe_b_conf", 0.0)
        pb_str = f"{pb} ({pb_conf:.2f})" if pb != "--" and pb != "?" else (pb if pb != "?" else "--")

        vlm_c = s.get("vlm_blind_raw_char") or s.get("vlm_char") or "--"
        vlm_conf = s.get("vlm_conf", 0.0)
        vlm_str = f"{vlm_c} ({vlm_conf:.2f})" if vlm_c != "--" and vlm_c != "?" else (vlm_c if vlm_c != "?" else "--")

        tone = s.get("ui_tile_tone", "--")
        fc = s.get("final_char", "?")

        ghosts = s.get("ghost_stack", [])
        ghost_str = ", ".join(f"{g.get('char')}({g.get('soft_score', 0):.2f})" for g in ghosts) if ghosts else "-"
        print(f"  {idx:<4} | {st:<6} | {pa_str:<14} | {pb_str:<14} | {vlm_str:<14} | {tone:<7} | {fc:<5} | {ghost_str}", flush=True)

    print("=" * 95, flush=True)

    ghost_preds = plate_data.get("ghost_predictions", [])
    if ghost_preds:
        print("  Top-3 Ghost Predictions (Advisory Operator Re-Ranking):", flush=True)
        print(f"  {'Rank':<4} | {'Plate':<12} | {'Soft Score':<10} | {'Valid RTO':<9} | {'Ghost Slot Indices'}", flush=True)
        print("  " + "-" * 65, flush=True)
        for gp in ghost_preds[:3]:
            r = gp.get("rank", 1)
            p = gp.get("plate", "??????????")
            sc = gp.get("soft_score", 0.0)
            vr = gp.get("state_code_valid", False)
            g_slots = gp.get("ghost_slot_indices", [])
            print(f"  {r:<4} | {p:<12} | {sc:<10.3f} | {str(vr):<9} | {g_slots}", flush=True)
        print("=" * 95, flush=True)

    print(f"  STAGE 3 VEHICLE ATTRIBUTES: {veh.get('colour', 'Unknown')} {veh.get('make', '')} {veh.get('model', '')} ({veh.get('body_type', 'car')})", flush=True)
    if matches:
        top_m = matches[0]
        print(f"  TOP REGISTRY MATCH: #{top_m.get('rank')} {top_m.get('plate_number')} · {round(top_m.get('total_confidence', 0)*100)}% Match · {top_m.get('owner_name')} ({top_m.get('registered_rto')})", flush=True)
    print("=" * 95 + "\n", flush=True)


def _run_stage1_background_job(clip_stem: str, video_path: Path):
    """Worker function running fast Stage 1 tracking in background thread."""
    try:
        print(f"\n[STAGE 1] Ingesting video: {video_path.name} (Job: {clip_stem})...", flush=True)
        clip_out_dir = OUTPUTS_DIR / clip_stem
        clip_out_dir.mkdir(parents=True, exist_ok=True)

        JOBS[clip_stem] = {
            "job_id": clip_stem,
            "status": "running",
            "stage_step": 1,
            "progress": 15,
            "message": "Ingesting video stream...",
            "error": None,
            "result": None,
        }

        device = "0" if torch.cuda.is_available() else "cpu"
        pipeline = VehicleTrackingPipeline(
            model_path="yolo11s.pt",
            output_dir=clip_out_dir,
            imgsz=960,
            device=device,
            write_video=False,
        )

        JOBS[clip_stem].update({
            "stage_step": 2,
            "progress": 40,
            "message": "Tracking vehicles with YOLO + BoT-SORT...",
        })
        print(f"[STAGE 1] Running BoT-SORT GPU tracking on {video_path.name}...", flush=True)
        summary = pipeline.run(str(video_path), write_video=False)
        print(f"[STAGE 1] Tracking complete: {len(summary.get('tracks', []))} clean vehicles recorded.", flush=True)

        JOBS[clip_stem].update({
            "stage_step": 3,
            "progress": 85,
            "message": "Generating bounding boxes & thumbnails...",
        })
        print(f"[STAGE 1] Generating web bounding boxes and thumbnails...", flush=True)
        build_web_artifacts(clip_stem, clip_out_dir, video_path)
        print(f"[STAGE 1] Web artifacts ready -> boxes.json + thumbnails generated.", flush=True)

        summ_path = clip_out_dir / "tracks_summary.json"
        with open(summ_path, "r", encoding="utf-8") as f:
            summ_data = json.load(f)

        result = _build_stage1_result(clip_stem, video_path, summ_data)

        JOBS[clip_stem].update({
            "status": "ready",
            "stage_step": 3,
            "progress": 100,
            "message": "Stage 1 tracking and web artifacts ready",
            "error": None,
            "result": result,
        })
        print(f"[STAGE 1] Job '{clip_stem}' completed successfully.\n", flush=True)
    except Exception as e:
        logger.error(f"Background Stage 1 job for {clip_stem} failed: {e}", exc_info=True)
        print(f"[STAGE 1 ERROR] Job '{clip_stem}' failed: {e}", flush=True)
        JOBS[clip_stem] = {
            "job_id": clip_stem,
            "status": "error",
            "stage_step": 0,
            "progress": 0,
            "message": f"Processing failed: {str(e)}",
            "error": str(e),
            "result": None,
        }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Service Health and Database Connectivity",
    tags=["System"],
)
async def get_health() -> HealthResponse:
    """Return system health status and total registered vehicles count."""
    try:
        count = get_total_vehicles_count()
        return HealthResponse(
            status="ok",
            service="Vehicle Identification & Traceability Backend",
            version="1.0.0",
            database_connected=True,
            registry_vehicles_count=count,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="degraded",
            service="Vehicle Identification & Traceability Backend",
            version="1.0.0",
            database_connected=False,
            registry_vehicles_count=0,
        )


@app.get(
    "/api/demo-scenarios",
    response_model=List[DemoScenario],
    summary="Get 4 Ready-to-Run Test Scenarios",
    tags=["Scenarios"],
)
async def get_demo_scenarios() -> List[DemoScenario]:
    """Provide 4 realistic VehicleEvidence test payloads:

    1. GT-1: Blurry partial plate MH12??1432 (white scooter).
    2. GT-2: OCR-confused plate DL03C880?1 (red SUV).
    3. GT-4: Partial plate MH02??5820 (yellow auto-rickshaw).
    4. Out-of-Registry: GJ05XX9999 (blue car) triggering 'No Confident Match'.
    """
    return list(DEMO_SCENARIOS.values())


class VideoScanRequest(BaseModel):
    """Optional request payload for scanning a video or selecting a demo clip."""
    clip_id: Optional[str] = Field(
        default=None,
        description="Demo clip ID (e.g. 'demo-traffic-1', 'clip-01').",
        examples=["demo-traffic-1"],
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Relative or absolute path to a video file on disk.",
        examples=["outputs/clips/demo_traffic.mp4"],
    )


@app.post(
    "/api/videos/scan",
    response_model=VideoScanResponse,
    summary="Scan Uploaded Video or Demo Clip for Vehicle Tracks",
    tags=["Video Processing"],
)
async def scan_video(
    request: Request,
    payload: Optional[VideoScanRequest] = Body(None),
) -> VideoScanResponse:
    """Scan an uploaded video or demo clip to detect and track vehicles.

    Hooks into Arnav's YOLO26s + BoT-SORT CV pipeline; maps tracked vehicles
    to TrackSummary models and returns detailed vehicle tracks.
    Supports JSON payloads {"clip_id": "..."} as well as raw binary video streams.
    """
    content_type = request.headers.get("content-type", "")
    target_clip_id = payload.clip_id if payload else None
    target_video_path = payload.video_path if payload else None

    if not target_clip_id and "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                target_clip_id = body.get("clip_id") or body.get("video_id")
                if not target_video_path:
                    target_video_path = body.get("video_path")
        except Exception:
            pass

    # Save uploaded body if binary video content is streamed
    if "video/" in content_type or "octet-stream" in content_type:
        upload_path = OUTPUTS_DIR / "clips" / "uploaded_scan.mp4"
        try:
            content = await request.body()
            if content:
                with open(upload_path, "wb") as f:
                    f.write(content)
                logger.info(f"Saved binary uploaded video clip to: {upload_path}")
                target_video_path = str(upload_path)
        except Exception as e:
            logger.warning(f"Could not save uploaded video: {e}")

    # Use cv_bridge to scan or load existing CV pipeline tracks
    return scan_video_with_cv_pipeline(
        clip_id=target_clip_id,
        video_path=target_video_path,
    )


@app.post(
    "/api/vehicles/analyze",
    response_model=AnalysisResponse,
    summary="Probabilistic Vehicle Matching & Shortlist Ranking",
    tags=["Matching Engine"],
)
async def analyze_vehicle(
    payload: AnalyzeRequest = Body(...),
) -> AnalysisResponse:
    """Analyze vehicle visual and OCR evidence against data/registry.db.

    Accepts either:
    1. A full `evidence` (VehicleEvidence) payload.
    2. A `scenario_id` (e.g., 'scenario-1', 'scenario-2', etc.) for instant preset analysis.
    3. A `track_id` (e.g., 1, 2, 27) connecting directly to the CV bridge.
    With optional `known_partial_plate` witness hint.

    Returns the ranked Top-5 shortlist of matching candidates, feature score
    breakdown, explainable match reasons, and calibrated confidence gating.
    """
    evidence: Optional[VehicleEvidence] = payload.evidence
    known_hint: Optional[str] = payload.known_partial_plate

    # If scenario_id provided, look up evidence and run through active extractor provider
    if not evidence and payload.scenario_id:
        scenario = DEMO_SCENARIOS.get(payload.scenario_id)
        if not scenario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario '{payload.scenario_id}' not found. Available: {list(DEMO_SCENARIOS.keys())}",
            )
        target_track_id = scenario.evidence.track_id
        try:
            evidence = get_vehicle_evidence_for_track(target_track_id)
        except Exception as e:
            logger.warning(f"CV bridge extraction for scenario {payload.scenario_id} (track {target_track_id}) failed: {e}")
            evidence = scenario.evidence.model_copy(deep=True)

    # If track_id provided without direct evidence, extract via CV bridge
    if not evidence and payload.track_id is not None:
        try:
            evidence = get_vehicle_evidence_for_track(payload.track_id)
        except Exception as e:
            logger.warning(f"CV bridge extraction for track {payload.track_id} failed: {e}")

    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either 'evidence', a valid 'scenario_id', or a 'track_id' must be provided.",
        )

    # Load registry records
    try:
        registry_vehicles = get_all_registry_vehicles()
    except Exception as e:
        logger.error(f"Failed to query vehicle registry: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query SQLite vehicle registry.",
        )

    # Execute custom multi-signal probabilistic matching engine
    rto_priors = payload.rto_priors or getattr(evidence, "rto_priors", None)
    if rto_priors and getattr(evidence, "rto_priors", None) is None:
        evidence.rto_priors = rto_priors

    result = match_and_rank_vehicles(
        evidence=evidence,
        registry_vehicles=registry_vehicles,
        known_partial_plate=known_hint,
        rto_priors=rto_priors,
        top_k=5,
    )

    # Propagate any VLM outage/fallback warnings to the API response
    if evidence.warning:
        result.warning = evidence.warning

    # Automatically log sighting if a confident match was produced
    if not result.no_confident_match and result.top_matches:
        top_cand = result.top_matches[0]
        crop_path = None
        if evidence.frames_used and len(evidence.frames_used) > 0:
            crop_path = evidence.frames_used[0].image_path or evidence.frames_used[0].plate_crop_path
        camera_id = payload.camera_id or "CAM_01"
        try:
            record_sighting(
                track_id=evidence.track_id,
                camera_id=camera_id,
                top_matched_plate=top_cand.plate_number,
                confidence=top_cand.composite_score,
                crop_path=crop_path,
            )
        except Exception as e:
            logger.warning(f"Failed to auto-record sighting for plate {top_cand.plate_number}: {e}")

    return result


@app.get(
    "/api/sightings/{plate_number}",
    response_model=List[SightingRecord],
    summary="Get Multi-Camera Sighting History for Plate",
    tags=["Sightings"],
)
async def get_plate_sightings(plate_number: str) -> List[SightingRecord]:
    """Retrieve all historical multi-camera sightings for a license plate ordered by timestamp ASC."""
    try:
        return get_sightings_by_plate(plate_number)
    except Exception as e:
        logger.error(f"Error querying sightings for plate '{plate_number}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query sightings for plate: {plate_number}",
        )


@app.post(
    "/api/reports/export",
    response_class=HTMLResponse,
    summary="Export Clean, Formal HTML Forensic Case Report",
    tags=["Reporting"],
)
async def export_forensic_report(request: Request) -> HTMLResponse:
    """Takes an AnalysisResponse payload and generates a clean, formal, self-contained
    HTML forensic case report (text/html) with print-optimized @media print CSS and
    a built-in 'Print / Save as PDF' toolbar.

    Strictly compliant with AIML-01 requirements:
    - Case timestamp & target tracking ID
    - Keyframes and cropped plate images (base64 embedded with SVG fallback)
    - Per-frame OCR recovery table with character confidences
    - Final consensus partial plate highlighting ? unknown slots
    - Detected vehicle colour & body type with confidence scores
    - Ranked Top-5 shortlist or high-visibility 'No Confident Match' notice
    - Plain-English match reasoning and chain-of-custody audit notes
    """
    try:
        body = await request.json()
        if not body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body cannot be empty.",
            )

        # Support direct AnalysisResponse or {"analysis": AnalysisResponse}
        if "analysis" in body and isinstance(body["analysis"], dict):
            analysis_data = body["analysis"]
        else:
            analysis_data = body

        analysis = AnalysisResponse.model_validate(analysis_data)
        html_report = generate_forensic_html_report(analysis)
        return HTMLResponse(
            content=html_report,
            media_type="text/html",
            headers={"Content-Disposition": f'inline; filename="forensic-report-track-{analysis.track_id:03d}.html"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate forensic report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid AnalysisResponse payload for report generation: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Screen 1, 2, and 3 Endpoints (Frontend Bridge & Forensic Dossier)
# ---------------------------------------------------------------------------

@app.get(
    "/api/pipeline/clips",
    summary="List Available Video Clips",
    tags=["Pipeline"],
)
async def list_available_clips() -> List[Dict[str, Any]]:
    """List available video clips in data/Final for selection."""
    if not CLIPS_DIR.exists():
        return []

    files_info = {}
    info_json = CLIPS_DIR / "files_info.json"
    if info_json.is_file():
        try:
            with open(info_json, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if "new_filename" in item:
                        files_info[item["new_filename"]] = item
        except Exception:
            pass

    clips = []
    for p in sorted(CLIPS_DIR.glob("*.mp4")):
        info = files_info.get(p.name, {})
        clips.append({
            "id": p.stem,
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "video_url": f"/media/clips/{p.name}",
            "duration_seconds": info.get("final_duration_seconds"),
            "resolution": info.get("resolution"),
        })
    return clips


@app.get(
    "/api/registry/stats",
    summary="Get Total Records in Vehicle Registry",
    tags=["System"],
)
async def get_registry_stats() -> Dict[str, int]:
    """Return total records count in the vehicle registry database."""
    try:
        count = get_total_vehicles_count()
        return {"total_records": count}
    except Exception as e:
        logger.error(f"Error reading registry stats: {e}")
        return {"total_records": 0}


@app.post(
    "/api/videos/select-local",
    summary="Select Local Video & Run Stage 1",
    tags=["Video Processing"],
)
async def select_local_video(payload: SelectLocalRequest = Body(...)) -> Dict[str, Any]:
    """Select a video from data/Final and initiate fast Stage 1 tracking or return instant cache hit."""
    fn = payload.filename.strip()
    if not fn or ".." in fn or "/" in fn or "\\" in fn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal not allowed.",
        )

    video_path = CLIPS_DIR / fn
    if not video_path.is_file():
        fallback = BASE_DIR / "data" / "clips" / fn
        if fallback.is_file():
            video_path = fallback
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video clip '{fn}' not found in {CLIPS_DIR.name}.",
            )

    clip_stem = video_path.stem
    clip_out_dir = OUTPUTS_DIR / clip_stem
    boxes_path = clip_out_dir / "boxes.json"
    summ_path = clip_out_dir / "tracks_summary.json"

    print(f"\n[API] select-local: '{fn}' (cached: {boxes_path.is_file() and summ_path.is_file()})", flush=True)

    # Instant Cache Hit
    if boxes_path.is_file() and summ_path.is_file() and not payload.force_refresh:
        try:
            with open(summ_path, "r", encoding="utf-8") as f:
                summ_data = json.load(f)
            result = _build_stage1_result(clip_stem, video_path, summ_data)
            ready_state = {
                "job_id": clip_stem,
                "status": "ready",
                "stage_step": 3,
                "progress": 100,
                "message": "Ready (cached)",
                "error": None,
                "result": result,
            }
            JOBS[clip_stem] = ready_state
            return ready_state
        except Exception as e:
            logger.warning(f"Failed reading cached summary for {clip_stem}: {e}")

    # Return running state if job in progress
    existing = JOBS.get(clip_stem)
    if existing and existing.get("status") in ("queued", "running") and not payload.force_refresh:
        return existing

    init_state = {
        "job_id": clip_stem,
        "status": "queued",
        "stage_step": 1,
        "progress": 5,
        "message": "Queued for processing...",
        "error": None,
        "result": None,
    }
    JOBS[clip_stem] = init_state
    STAGE1_EXECUTOR.submit(_run_stage1_background_job, clip_stem, video_path)
    return init_state


@app.get(
    "/api/videos/{job_id}/status",
    summary="Get Stage 1 Processing Status & Tracks Result",
    tags=["Video Processing"],
)
async def get_video_status(job_id: str) -> Dict[str, Any]:
    """Retrieve status, progress, and completed tracks summary for a video processing job."""
    job_stem = Path(job_id).stem
    if job_stem in JOBS:
        return JOBS[job_stem]

    # Check on disk
    clip_out_dir = OUTPUTS_DIR / job_stem
    boxes_path = clip_out_dir / "boxes.json"
    summ_path = clip_out_dir / "tracks_summary.json"

    if boxes_path.is_file() and summ_path.is_file():
        v_candidates = list(CLIPS_DIR.glob(f"{job_stem}.*"))
        v_path = v_candidates[0] if v_candidates else CLIPS_DIR / f"{job_stem}.mp4"
        try:
            with open(summ_path, "r", encoding="utf-8") as f:
                summ_data = json.load(f)
            result = _build_stage1_result(job_stem, v_path, summ_data)
            ready_state = {
                "job_id": job_stem,
                "status": "ready",
                "stage_step": 3,
                "progress": 100,
                "message": "Ready (cached)",
                "error": None,
                "result": result,
            }
            JOBS[job_stem] = ready_state
            return ready_state
        except Exception as e:
            logger.warning(f"Failed loading on-disk state for {job_stem}: {e}")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Video job '{job_id}' not found.",
    )


@app.post(
    "/api/pipeline/verify",
    summary="Run Stage 2, 3, and 4 Full-Pipeline Verification",
    tags=["Pipeline"],
)
async def verify_pipeline_track(payload: PipelineVerifyRequest = Body(...)) -> Dict[str, Any]:
    """Run full Stage 2 -> 3 -> 4 pipeline on selected track(s) with disk caching."""
    t0 = time.time()
    raw_ids = payload.track_ids if payload.track_ids is not None else ([payload.track_id] if payload.track_id is not None else None)
    if not raw_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'track_ids' (list of ints) or 'track_id' (int) must be provided.",
        )

    target_ids = sorted(list(set(raw_ids)))
    is_merged = len(target_ids) > 1
    job_id = Path(payload.job_id).stem

    if is_merged:
        track_key = f"track_{'_'.join(f'{tid:03d}' for tid in target_ids)}"
    else:
        track_key = f"track_{target_ids[0]:03d}"

    primary_id = target_ids[0]
    clip_dir = OUTPUTS_DIR / job_id
    track_dir = clip_dir / track_key
    ver_path = track_dir / "verification_result.json"
    pdf_path = track_dir / f"Invictus_Report_{job_id}_{track_key}.pdf"

    # Disk Cache Check
    if ver_path.is_file() and not payload.force_refresh:
        try:
            with open(ver_path, "r", encoding="utf-8") as f:
                cached_res = json.load(f)
            if not pdf_path.is_file():
                try:
                    generate_forensic_pdf(cached_res, pdf_path, track_dir=track_dir)
                except Exception as e:
                    logger.warning(f"Could not generate PDF on cache hit: {e}")
            print(f"[CACHE HIT] Returning verified results for {job_id}/{track_key} (<0.04s)", flush=True)
            _print_forensic_console_report(cached_res)
            return cached_res
        except Exception as e:
            logger.warning(f"Failed to read cached verification result: {e}")

    # Locate source video
    video_path = _find_video_path(job_id)
    if not video_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source video for job '{job_id}' not found.",
        )

    # Locate track_history
    hist_path = clip_dir / "track_history.json"
    if not hist_path.is_file():
        logger.info(f"track_history.json missing for {job_id}; running Stage 1 tracking...")
        device_st1 = "0" if torch.cuda.is_available() else "cpu"
        pipeline = VehicleTrackingPipeline(
            model_path="yolo11s.pt",
            output_dir=clip_dir,
            imgsz=960,
            device=device_st1,
            write_video=False,
        )
        pipeline.run(str(video_path), write_video=False)
        build_web_artifacts(job_id, clip_dir, video_path)

    with open(hist_path, "r", encoding="utf-8") as f:
        track_history = json.load(f)

    device = "0" if torch.cuda.is_available() else "cpu"

    # Stage 2.1: Vehicle crops extraction
    logger.info(f"[Stage 2.1] Extracting crops for {job_id} {target_ids}...")
    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=target_ids,
        output_dir=clip_dir,
        padding_pct=0.08,
    )
    crop_manifest_path = track_dir / "crop_manifest.json"

    # Stage 2.2: Dual plate detection
    logger.info(f"[Stage 2.2] Dual-detector plate detection for {track_key}...")
    run_dual_plate_detection(crop_manifest_path=crop_manifest_path, device=device)
    plate_manifest_path = track_dir / "plate_manifest.json"

    # Stage 2.3: OCR Engine arbitration
    logger.info(f"[Stage 2.3] Dual-pipeline OCR & slot arbitration for {track_key}...")
    ocr_result = run_ocr_track(str(plate_manifest_path), device=device)

    # Stage 2.4: Blind VLM verification
    logger.info(f"[Stage 2.4] Blind multi-tier VLM triple-check for {track_key}...")
    vlm_result = verify_track_vlm(track_dir=track_dir, provider="auto")

    # Stage 3: Vehicle attribute extraction
    logger.info(f"[Stage 3] Vehicle attribute extraction for {track_key}...")
    coarse_cls = manifest.get("class_name") or manifest.get("category", "car")
    vehicle_attrs = None
    try:
        vehicle_attrs = run_stage3_vehicle_attributes(
            track_dir=track_dir,
            coarse_class=coarse_cls,
            track_id=primary_id,
            top_k=3,
        )
    except Exception as e:
        logger.error(f"Stage 3 vehicle attribute extraction failed: {e}")

    # Stage 4: SQLite Registry Matching
    logger.info(f"[Stage 4] Forensic registry matching for {track_key}...")
    final_plate = (
        vlm_result.get("final_plate")
        or vlm_result.get("reconciled_plate")
        or ocr_result.get("recovered_plate", "??????????")
    )
    overall_conf = float(vlm_result.get("overall_confidence", ocr_result.get("overall_confidence", 0.0)))
    char_confs = [
        float(s.get("final_char_soft_score", s.get("vlm_conf", 0.5)))
        for s in vlm_result.get("reconciled_slots", [])
    ]
    if not char_confs:
        char_confs = [overall_conf] * len(final_plate)

    attrs = vehicle_attrs.get("attributes", {}) if vehicle_attrs else {}
    body_val = attrs.get("body_type") or manifest.get("class_name", "car")
    body_enum = _map_to_body_type(body_val)
    colour_val = attrs.get("colour", "white")

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

    reg_vehicles = get_all_registry_vehicles()
    match_resp = match_and_rank_vehicles(
        evidence=evidence,
        registry_vehicles=reg_vehicles,
        top_k=5,
    )

    formatted_matches = []
    for cand in match_resp.top_matches:
        sightings = []
        try:
            sightings = [s.model_dump() for s in get_sightings_by_plate(cand.plate_number)]
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

    # Evidence URLs
    clearest_p = _get_clearest_crop_path(track_dir, ocr_result)
    clearest_url = (
        f"/outputs/{job_id}/{track_key}/{clearest_p.relative_to(track_dir).as_posix()}"
        if clearest_p
        else None
    )

    best_veh_p = track_dir / "stage3_vehicle_crops_top3.png"
    if not best_veh_p.is_file():
        raw_veh_d = track_dir / "vehicle_crops_unenhanced"
        if raw_veh_d.is_dir():
            first_v = next(raw_veh_d.glob("*.png"), None)
            if first_v:
                best_veh_p = first_v

    evidence_urls = {
        "best_vehicle_crop_url": f"/outputs/{job_id}/{track_key}/{best_veh_p.name}" if best_veh_p and best_veh_p.is_file() else None,
        "vehicle_contact_strip_url": f"/outputs/{job_id}/{track_key}/stage3_vehicle_crops_top3.png" if (track_dir / "stage3_vehicle_crops_top3.png").is_file() else None,
        "clearest_plate_crop_url": clearest_url,
        "plate_grid_unenhanced_url": f"/outputs/{job_id}/{track_key}/vlm_evidence_unenhanced_grid.png" if (track_dir / "vlm_evidence_unenhanced_grid.png").is_file() else None,
        "plate_grid_enhanced_url": f"/outputs/{job_id}/{track_key}/vlm_evidence_enhanced_grid.png" if (track_dir / "vlm_evidence_enhanced_grid.png").is_file() else None,
    }

    elapsed_total = round(time.time() - t0, 2)
    ver_result = {
        "case_id": f"INV-{job_id.upper()}-{track_key.upper()}",
        "job_id": job_id,
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
            "valid_state_code": vlm_result.get("state_code_valid", ocr_result.get("state_code_valid", False)),
            "provider_used": vlm_result.get("provider_used", "none"),
            "model_used": vlm_result.get("model_used", "none"),
            "reconciled_slots": vlm_result.get("reconciled_slots", []),
            "ghost_predictions": vlm_result.get("ghost_predictions", []),
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
            "provider_used": vehicle_attrs.get("provider", "none") if vehicle_attrs else "none",
            "selected_crops": vehicle_attrs.get("selected_crop_names", []) if vehicle_attrs else [],
        },
        "registry_matches": formatted_matches,
        "evidence_urls": evidence_urls,
        "pdf_report_url": f"/api/reports/{job_id}/{track_key}/pdf",
    }

    # Save to disk
    with open(ver_path, "w", encoding="utf-8") as f:
        json.dump(ver_result, f, indent=2)

    # Generate forensic PDF dossier
    try:
        generate_forensic_pdf(ver_result, pdf_path, track_dir=track_dir)
    except Exception as e:
        logger.error(f"Failed to generate PDF for {track_key}: {e}", exc_info=True)

    _print_forensic_console_report(ver_result)
    return ver_result


@app.get(
    "/api/reports/{job_id}/{track_key}/pdf",
    summary="Download Forensic PDF Summary Dossier",
    tags=["Reporting"],
)
async def get_forensic_pdf_report(job_id: str, track_key: str):
    """Serve publication-grade forensic PDF report for a verified vehicle track."""
    clean_job = Path(job_id).stem
    clean_track = Path(track_key).name
    track_dir = OUTPUTS_DIR / clean_job / clean_track
    ver_path = track_dir / "verification_result.json"
    pdf_path = track_dir / f"Invictus_Report_{clean_job}_{clean_track}.pdf"

    if not ver_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification result for '{clean_job}/{clean_track}' not found. Run POST /api/pipeline/verify first.",
        )

    try:
        with open(ver_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not pdf_path.is_file():
            generate_forensic_pdf(data, pdf_path, track_dir=track_dir)

        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename=f"Invictus_Report_{clean_job}_{clean_track}.pdf",
        )
    except Exception as e:
        logger.error(f"Failed serving PDF for {clean_job}/{clean_track}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forensic PDF: {str(e)}",
        )


