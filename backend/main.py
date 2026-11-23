"""FastAPI Backend Service for Vehicle Identification and Traceability.

Exposes RESTful endpoints for video scanning, vehicle evidence analysis,
database health checks, and pre-configured realistic demo scenarios.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.database import ensure_db_exists, get_all_registry_vehicles, get_total_vehicles_count
from backend.demo_data import DEMO_SCENARIOS, MOCK_VIDEO_SCAN_RESPONSE
from backend.cv_bridge import (
    get_vehicle_evidence_for_track,
    scan_video_with_cv_pipeline,
)
from backend.matcher import match_and_rank_vehicles
from backend.report import generate_forensic_html_report
from backend.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    DemoScenario,
    VehicleEvidence,
    VideoScanResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vehicle_trace.main")

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


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

    # If scenario_id provided, look up evidence from pre-configured scenarios (instant presets)
    if not evidence and payload.scenario_id:
        scenario = DEMO_SCENARIOS.get(payload.scenario_id)
        if not scenario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario '{payload.scenario_id}' not found. Available: {list(DEMO_SCENARIOS.keys())}",
            )
        evidence = scenario.evidence
        # If user didn't provide a known_partial_plate hint, check scenario default
        if not known_hint and scenario.suggested_known_partial_plate:
            pass

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
    result = match_and_rank_vehicles(
        evidence=evidence,
        registry_vehicles=registry_vehicles,
        known_partial_plate=known_hint,
        top_k=5,
    )

    return result


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

