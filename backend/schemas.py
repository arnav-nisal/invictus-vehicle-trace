"""Shared CV ↔ Backend Pydantic Data Contract.

Defines the exact data schemas exchanged between the Computer Vision
tracking/OCR pipeline (Arnav) and the FastAPI Backend / React UI (Samarth).
Compatible with Pydantic V2 and FastAPI 0.100+.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Supported Enums
# ---------------------------------------------------------------------------

class BodyType(str, Enum):
    """Supported vehicle body types for identification and registry matching."""
    CAR = "car"
    SUV = "suv"
    TWO_WHEELER = "two-wheeler"
    AUTO_RICKSHAW = "auto-rickshaw"
    TRUCK = "truck"
    BUS = "bus"


# ---------------------------------------------------------------------------
# Initial Video Scan Contract
# ---------------------------------------------------------------------------

class TrackSummary(BaseModel):
    """Summary of an individual tracked vehicle detected in a video clip."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    track_id: int = Field(
        ...,
        alias="id",
        description="Unique integer tracking ID assigned by BoT-SORT / ByteTrack.",
        examples=[1, 2, 42],
    )
    coarse_class: str = Field(
        ...,
        alias="class_name",
        description="Coarse vehicle class (e.g., motorcycle, car, bus, truck, two-wheeler).",
        examples=["two-wheeler", "motorcycle", "car"],
    )
    first_sec: float = Field(
        ...,
        alias="start_time_seconds",
        ge=0.0,
        description="First timestamp in seconds where the vehicle appears.",
        examples=[1.25],
    )
    last_sec: float = Field(
        ...,
        alias="end_time_seconds",
        ge=0.0,
        description="Last timestamp in seconds where the vehicle appears.",
        examples=[7.80],
    )
    frames_seen: int = Field(
        ...,
        alias="frame_count",
        gt=0,
        description="Total number of frames in which this vehicle track was observed.",
        examples=[145],
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        alias="saved_image_path",
        description="Path or URL to the highest-sharpness thumbnail crop for this track.",
        examples=["outputs/tracks/track_001_motorcycle.jpg"],
    )

    @model_validator(mode="before")
    @classmethod
    def map_cv_pipeline_fields(cls, data: Any) -> Any:
        """Harmonize fields from raw CV pipeline tracks_summary.json entries."""
        if isinstance(data, dict):
            # Map alternative naming conventions if present
            if "track_id" not in data and "id" in data:
                data["track_id"] = data["id"]
            if "coarse_class" not in data:
                if "category" in data and data["category"]:
                    data["coarse_class"] = data["category"]
                elif "class_name" in data and data["class_name"]:
                    data["coarse_class"] = data["class_name"]
            if "first_sec" not in data and "start_time_seconds" in data:
                data["first_sec"] = float(data["start_time_seconds"])
            if "last_sec" not in data and "end_time_seconds" in data:
                data["last_sec"] = float(data["end_time_seconds"])
            if "frames_seen" not in data and "frame_count" in data:
                data["frames_seen"] = int(data["frame_count"])
            if "thumbnail_url" not in data and "saved_image_path" in data:
                data["thumbnail_url"] = data["saved_image_path"]
        return data


class VideoScanResponse(BaseModel):
    """Response payload returned after scanning an uploaded video clip."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    video_id: Optional[str] = Field(
        default=None,
        description="Unique identifier or hash for the processed video.",
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Relative or absolute path to the processed video file.",
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total duration of the processed video in seconds.",
        examples=[15.5],
    )
    total_frames: int = Field(
        ...,
        gt=0,
        description="Total number of frames in the video.",
        examples=[465],
    )
    fps: float = Field(
        default=30.0,
        gt=0.0,
        description="Framerate of the video.",
        examples=[30.0],
    )
    tracks_count: int = Field(
        ...,
        ge=0,
        description="Total number of qualified vehicle tracks detected.",
        examples=[8],
    )
    tracks: List[TrackSummary] = Field(
        default_factory=list,
        description="List of all detected and tracked vehicles in the clip.",
    )


# ---------------------------------------------------------------------------
# Extracted Vehicle Evidence Contract
# ---------------------------------------------------------------------------

class FrameEvidence(BaseModel):
    """Per-frame visual and OCR evidence for a tracked vehicle."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    frame_idx: int = Field(
        ...,
        ge=0,
        description="Zero-indexed frame number from the video stream.",
        examples=[124],
    )
    timestamp_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Video timestamp in seconds for this frame.",
        examples=[4.133],
    )
    image_path: str = Field(
        ...,
        description="File path or URL to the vehicle bounding-box crop image.",
        examples=["outputs/crops/track_001_frame_0124_vehicle.jpg"],
    )
    plate_crop_path: Optional[str] = Field(
        default=None,
        description="File path or URL to the isolated license plate crop image.",
        examples=["outputs/crops/track_001_frame_0124_plate.jpg"],
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Raw OCR text reading for this specific frame (may contain ?).",
        examples=["MH12DE1432"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this frame's OCR / detection.",
        examples=[0.92],
    )


class VehicleEvidence(BaseModel):
    """Detailed visual and OCR evidence extracted for a single selected vehicle."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    track_id: int = Field(
        ...,
        description="Track ID of the selected vehicle.",
        examples=[1],
    )
    consensus_plate: str = Field(
        ...,
        description="Multi-frame consensus plate text using '?' for unreadable/ambiguous chars.",
        examples=["MH12DE1432", "MH12??1432"],
    )
    char_confidences: List[float] = Field(
        default_factory=list,
        description="Per-character confidence scores (0.0 to 1.0) aligned with consensus_plate.",
        examples=[[0.98, 0.97, 0.95, 0.93, 0.91, 0.89, 0.99, 0.97, 0.94, 0.96]],
    )
    colour: str = Field(
        ...,
        description="Predicted vehicle dominant colour.",
        examples=["white", "red", "silver", "black"],
    )
    colour_conf: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 - 1.0) of the colour prediction.",
        examples=[0.94],
    )
    body_type: BodyType = Field(
        ...,
        description="Body type classification: car, suv, two-wheeler, auto-rickshaw, truck, bus.",
        examples=[BodyType.TWO_WHEELER],
    )
    body_type_conf: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 - 1.0) of the body type classification.",
        examples=[0.96],
    )
    make: str = Field(
        default="Unknown",
        description="Predicted vehicle manufacturer or company (e.g. Honda, Tata, Maruti Suzuki, Bajaj).",
        examples=["Honda", "Tata", "Unknown"],
    )
    make_conf: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 - 1.0) of the make/company prediction.",
        examples=[0.92, 0.0],
    )
    model: str = Field(
        default="Unknown",
        description="Predicted vehicle model name (e.g. Activa 6G, Nexon, Swift, Dzire).",
        examples=["Activa 6G", "Nexon", "Unknown"],
    )
    model_conf: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 - 1.0) of the model prediction.",
        examples=[0.89, 0.0],
    )
    frames_used: List[FrameEvidence] = Field(
        default_factory=list,
        description="List of best per-frame crops and OCR predictions used for consensus.",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Warning or outage notice from attribute extraction (e.g. VLM cascade failure).",
        examples=["All open-source VLM tiers failed: Tier 1 (NVIDIA NIM): ..."],
    )
    plate_detected: bool = Field(
        default=True,
        description="Whether a license plate was detected on the vehicle (False triggers VISUAL_ONLY mode).",
        examples=[True],
    )
    rto_priors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional RTO prior probability distribution (e.g. {'MH14': 0.80, 'MH12': 0.12}).",
        examples=[{"MH14": 0.80, "MH12": 0.12}],
    )

    @field_validator("consensus_plate")
    @classmethod
    def validate_plate_format(cls, v: str) -> str:
        """Normalize plate format: uppercase and allow alphanumeric plus '?' placeholders."""
        cleaned = v.strip().upper().replace(" ", "").replace("-", "")
        if not cleaned or cleaned in ("NO_PLATE", "UNKNOWN", "NONE"):
            return cleaned or "NO_PLATE"
        # Ensure only alphanumeric and '?', '_' are present
        for ch in cleaned:
            if not (ch.isalnum() or ch in ("?", "_")):
                raise ValueError(f"Invalid character '{ch}' in consensus_plate '{v}'. Use '?' for unreadable characters.")
        return cleaned


# ---------------------------------------------------------------------------
# Registry Matching & Analysis Output Contract
# ---------------------------------------------------------------------------

class RegistryVehicle(BaseModel):
    """Representation of an official synthetic vehicle record in the registry."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    plate_number: str = Field(
        ...,
        description="Standardized Indian registration plate number.",
        examples=["MH12DE1432"],
    )
    body_type: str = Field(
        ...,
        description="Vehicle body type category.",
        examples=["two-wheeler"],
    )
    colour: str = Field(
        ...,
        description="Registered vehicle exterior colour.",
        examples=["white"],
    )
    make: str = Field(
        ...,
        description="Vehicle manufacturer.",
        examples=["Honda"],
    )
    model: str = Field(
        ...,
        description="Vehicle model name.",
        examples=["Activa 6G"],
    )
    rto_city: str = Field(
        ...,
        description="RTO registration city and state.",
        examples=["Pune, Maharashtra"],
    )
    synthetic_owner_id: str = Field(
        ...,
        description="100% synthetic privacy-preserving owner identifier (NEVER real names).",
        examples=["SYN-OWN-1001"],
    )
    owner_name: str = Field(
        default="Unknown",
        description="100% synthetic vehicle owner name (NEVER real personal data).",
        examples=["Aarav Sharma"],
    )
    registered_rto: str = Field(
        default="",
        description="Official registered RTO office authority.",
        examples=["MH-12 Pune RTO, Maharashtra"],
    )
    registration_year: int = Field(
        default=2021,
        ge=1990,
        le=2030,
        description="Year of initial vehicle registration.",
        examples=[2021],
    )
    watchlist_status: str = Field(
        default="CLEAR",
        description="Law enforcement surveillance watchlist status: CLEAR, STOLEN, FLAGGED.",
        examples=["CLEAR"],
    )


class ScoreBreakdown(BaseModel):
    """Detailed per-feature score components making up the composite match score."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    plate_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="License plate similarity score (considering edit distance & OCR ambiguities).",
        examples=[1.0],
    )
    colour_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Colour match similarity score between detected and registered colour.",
        examples=[0.95],
    )
    body_type_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Body type compatibility score between detected and registered body type.",
        examples=[1.0],
    )
    make_model_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Secondary heuristic score for make/model consistency if detected.",
        examples=[0.8],
    )
    rto_prior_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="RTO prefix prior probability score.",
        examples=[0.82],
    )
    visual_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Combined visual attribute score (body, colour, make, model).",
        examples=[0.95],
    )


class MatchCandidate(BaseModel):
    """A ranked potential match candidate from the synthetic vehicle registry."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    rank: int = Field(
        ...,
        ge=1,
        le=5,
        description="Shortlist rank (1 = best candidate, up to 5).",
        examples=[1],
    )
    plate_number: str = Field(
        ...,
        description="Registry license plate number.",
        examples=["MH12DE1432"],
    )
    body_type: str = Field(
        ...,
        description="Registry vehicle body type.",
        examples=["two-wheeler"],
    )
    colour: str = Field(
        ...,
        description="Registry vehicle colour.",
        examples=["white"],
    )
    make: str = Field(
        ...,
        description="Registry vehicle make.",
        examples=["Honda"],
    )
    model: str = Field(
        ...,
        description="Registry vehicle model.",
        examples=["Activa 6G"],
    )
    rto_city: str = Field(
        ...,
        description="Registry RTO jurisdiction.",
        examples=["Pune, Maharashtra"],
    )
    synthetic_owner_id: str = Field(
        ...,
        description="Synthetic owner identifier (NEVER real names).",
        examples=["SYN-OWN-1001"],
    )
    owner_name: Optional[str] = Field(
        default=None,
        description="100% synthetic owner name for candidate.",
        examples=["Aarav Sharma"],
    )
    registered_rto: Optional[str] = Field(
        default=None,
        description="Registered RTO jurisdiction office.",
        examples=["MH-12 Pune RTO, Maharashtra"],
    )
    registration_year: Optional[int] = Field(
        default=None,
        description="Year of vehicle registration.",
        examples=[2021],
    )
    watchlist_status: str = Field(
        default="CLEAR",
        description="Watchlist status: CLEAR, STOLEN, FLAGGED.",
        examples=["CLEAR"],
    )
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted composite confidence score (0.0 to 1.0).",
        examples=[0.985],
    )
    total_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Total forensic matching confidence score (0.0 to 1.0).",
        examples=[0.985],
    )
    character_diff: Optional[str] = Field(
        default=None,
        description="Character-by-character alignment diff (e.g. 'M H 1 2 [?->D] [?->E] 1 4 3 2').",
        examples=["M H 1 2 [?->D] [?->E] 1 4 3 2"],
    )
    anomaly_flags: List[str] = Field(
        default_factory=list,
        description="Forensic anomaly flags, e.g. ['SUSPECTED_CLONED_PLATE'].",
        examples=[["SUSPECTED_CLONED_PLATE"]],
    )
    rto_prior_score: Optional[float] = Field(
        default=None,
        description="Candidate RTO prior score.",
        examples=[0.82],
    )
    visual_score: Optional[float] = Field(
        default=None,
        description="Candidate visual score.",
        examples=[0.95],
    )
    score_breakdown: ScoreBreakdown = Field(
        ...,
        description="Per-feature score breakdown.",
    )
    match_reasons: List[str] = Field(
        default_factory=list,
        description="Plain-English explanation of why this vehicle matches or differs.",
        examples=[
            [
                "Exact match on plate number characters (MH12DE1432).",
                "Body type strictly matches: two-wheeler (Honda Activa 6G).",
                "Colour matches: white (confidence: 94%).",
            ]
        ],
    )


class AnalysisResponse(BaseModel):
    """Complete backend response for vehicle evidence analysis and registry matching."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    track_id: int = Field(
        ...,
        description="Track ID of the analyzed vehicle.",
        examples=[1],
    )
    evidence: VehicleEvidence = Field(
        ...,
        description="Detailed extracted evidence for the vehicle.",
    )
    top_matches: List[MatchCandidate] = Field(
        default_factory=list,
        description="Ranked Top-5 shortlist of registry candidates.",
    )
    no_confident_match: bool = Field(
        default=False,
        description="True if no registry candidate exceeds the minimum confidence threshold.",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Plain-English summary of matching outcome or reason for low confidence.",
        examples=[
            "Strong match found: Rank 1 vehicle (MH12DE1432) has 98.5% confidence with exact plate match."
        ],
    )
    warning: Optional[str] = Field(
        default=None,
        description="Visible warning or error notification (e.g., VLM tier failover to local fallback).",
        examples=["All open-source VLM tiers failed: Tier 1 (NVIDIA NIM): ..."],
    )


class AnalyzeRequest(BaseModel):
    """Request payload for vehicle evidence analysis and registry matching."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    scenario_id: Optional[str] = Field(
        default=None,
        description="Optional ID of a pre-configured demo scenario (e.g., 'scenario-1', 'scenario-2').",
        examples=["scenario-1"],
    )
    track_id: Optional[int] = Field(
        default=None,
        description="Optional tracked vehicle integer ID to analyze via the CV bridge.",
        examples=[1, 2, 27],
    )
    evidence: Optional[VehicleEvidence] = Field(
        default=None,
        description="Extracted visual and OCR evidence for a tracked vehicle.",
    )
    known_partial_plate: Optional[str] = Field(
        default=None,
        description="Optional investigator witness hint or known partial plate characters.",
        examples=["MH12DE"],
    )
    camera_id: Optional[str] = Field(
        default="CAM_01",
        description="Surveillance camera identifier where vehicle was sighted.",
        examples=["CAM_01", "CAM_04"],
    )
    rto_priors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional RTO prior probability distribution (e.g. {'MH14': 0.80, 'MH12': 0.12}).",
        examples=[{"MH14": 0.80, "MH12": 0.12}],
    )


class SightingRecord(BaseModel):
    """Timeline sighting entry across distributed CCTV camera nodes."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    sighting_id: int = Field(
        ...,
        description="Unique auto-incrementing sighting identifier.",
        examples=[1],
    )
    track_id: Optional[int] = Field(
        default=None,
        description="Track identifier from the CCTV video processor.",
        examples=[101],
    )
    camera_id: str = Field(
        default="CAM_01",
        description="Identifier of camera node capturing sighting.",
        examples=["CAM_01"],
    )
    timestamp: str = Field(
        ...,
        description="Timestamp string of the vehicle sighting.",
        examples=["2026-11-23 08:30:15"],
    )
    top_matched_plate: str = Field(
        ...,
        description="Top matched license plate number registered in sighting.",
        examples=["MH12DE1432"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of top plate match.",
        examples=[0.985],
    )
    crop_path: Optional[str] = Field(
        default=None,
        description="Relative file path to cropped vehicle evidence image.",
        examples=["outputs/crops/track_001_frame_0042_vehicle.jpg"],
    )


class DemoScenario(BaseModel):
    """Pre-configured realistic demo scenario for rapid evaluation and testing."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str = Field(..., description="Unique scenario slug or identifier.")
    title: str = Field(..., description="Short descriptive title of the scenario.")
    description: str = Field(..., description="Detailed briefing of the test scenario.")
    evidence: VehicleEvidence = Field(..., description="Extracted vehicle evidence payload.")
    suggested_known_partial_plate: Optional[str] = Field(
        default=None,
        description="Optional investigator partial plate hint.",
    )
    expected_outcome: str = Field(..., description="Summary of expected matching outcome.")

