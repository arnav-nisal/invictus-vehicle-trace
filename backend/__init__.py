"""Backend package for Vehicle Identification and Traceability."""

from .schemas import (
    BodyType,
    TrackSummary,
    VideoScanResponse,
    FrameEvidence,
    VehicleEvidence,
    RegistryVehicle,
    ScoreBreakdown,
    MatchCandidate,
    AnalysisResponse,
    AnalyzeRequest,
    DemoScenario,
)

__all__ = [
    "BodyType",
    "TrackSummary",
    "VideoScanResponse",
    "FrameEvidence",
    "VehicleEvidence",
    "RegistryVehicle",
    "ScoreBreakdown",
    "MatchCandidate",
    "AnalysisResponse",
    "AnalyzeRequest",
    "DemoScenario",
]
