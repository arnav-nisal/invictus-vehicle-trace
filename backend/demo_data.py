"""Pre-configured realistic demo scenarios and mock CV scan tracking results.

Provides 4 ready-to-run VehicleEvidence test payloads and realistic mock track
summaries so the frontend and API work seamlessly end-to-end.
"""

from __future__ import annotations

from typing import Dict, List

from backend.schemas import (
    BodyType,
    DemoScenario,
    FrameEvidence,
    TrackSummary,
    VehicleEvidence,
    VideoScanResponse,
)

# ---------------------------------------------------------------------------
# 4 Ready-to-Run Realistic VehicleEvidence Demo Scenarios
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: Dict[str, DemoScenario] = {
    "scenario-1": DemoScenario(
        id="scenario-1",
        title="Scenario 1: Blurry Partial Plate (White Scooter GT-1)",
        description=(
            "CCTV frame capture under heavy motion blur obscuring the series letters: "
            "'MH12??1432'. Vehicle is a white two-wheeler (Honda Activa 6G). "
            "Tests wildcard handling and ability to beat 3 hard near-miss decoys."
        ),
        evidence=VehicleEvidence(
            track_id=1,
            consensus_plate="MH12??1432",
            char_confidences=[0.98, 0.97, 0.95, 0.94, 0.20, 0.20, 0.99, 0.98, 0.96, 0.97],
            colour="white",
            colour_conf=0.94,
            body_type=BodyType.TWO_WHEELER,
            body_type_conf=0.96,
            make="Honda",
            make_conf=0.92,
            model="Activa 6G",
            model_conf=0.90,
            frames_used=[
                FrameEvidence(
                    frame_idx=42,
                    timestamp_sec=1.40,
                    image_path="outputs/crops/track_001_frame_0042_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_001_frame_0042_plate.jpg",
                    ocr_text="MH12??1432",
                    confidence=0.92,
                ),
                FrameEvidence(
                    frame_idx=48,
                    timestamp_sec=1.60,
                    image_path="outputs/crops/track_001_frame_0048_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_001_frame_0048_plate.jpg",
                    ocr_text="MH12DE1432",
                    confidence=0.88,
                ),
            ],
        ),
        suggested_known_partial_plate="MH12DE",
        expected_outcome=(
            "GT-1 (MH12DE1432) correctly ranks #1, outranking its 3 hard near-miss decoys "
            "(MH120E1432, MH12DEI432, MH12DE143Z)."
        ),
    ),
    "scenario-2": DemoScenario(
        id="scenario-2",
        title="Scenario 2: OCR-Confused Red SUV (GT-2)",
        description=(
            "CCTV motion blur caused OCR character confusion (detected '8' instead of 'B', "
            "and '?' for '5'): 'DL03C880?1'. Vehicle is a red compact SUV (Tata Nexon). "
            "Tests the domain visual confusion matrix (8↔B)."
        ),
        evidence=VehicleEvidence(
            track_id=2,
            consensus_plate="DL03C880?1",
            char_confidences=[0.98, 0.97, 0.95, 0.94, 0.92, 0.72, 0.96, 0.95, 0.15, 0.97],
            colour="red",
            colour_conf=0.95,
            body_type=BodyType.SUV,
            body_type_conf=0.96,
            make="Tata",
            make_conf=0.94,
            model="Nexon",
            model_conf=0.91,
            frames_used=[
                FrameEvidence(
                    frame_idx=88,
                    timestamp_sec=2.93,
                    image_path="outputs/crops/track_002_frame_0088_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_002_frame_0088_plate.jpg",
                    ocr_text="DL03C880?1",
                    confidence=0.89,
                )
            ],
        ),
        suggested_known_partial_plate=None,
        expected_outcome=(
            "OCR confusion matrix accommodates 8↔B swap and ranks GT-2 (DL03CB8051) #1, "
            "penalizing syntactically invalid decoys."
        ),
    ),
    "scenario-3": DemoScenario(
        id="scenario-3",
        title="Scenario 3: Partial Yellow Auto-Rickshaw (GT-4)",
        description=(
            "Rear-angle CCTV view on a three-wheeler with obscured series: 'MH02??5820'. "
            "Vehicle is a yellow auto-rickshaw (Bajaj Compact 4S). "
            "Tests partial plate matching fused with distinct body type and colour."
        ),
        evidence=VehicleEvidence(
            track_id=3,
            consensus_plate="MH02??5820",
            char_confidences=[0.97, 0.96, 0.98, 0.97, 0.25, 0.25, 0.99, 0.98, 0.97, 0.98],
            colour="yellow",
            colour_conf=0.93,
            body_type=BodyType.AUTO_RICKSHAW,
            body_type_conf=0.97,
            make="Bajaj",
            make_conf=0.95,
            model="Compact 4S",
            model_conf=0.90,
            frames_used=[
                FrameEvidence(
                    frame_idx=15,
                    timestamp_sec=0.50,
                    image_path="outputs/crops/track_003_frame_0015_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_003_frame_0015_plate.jpg",
                    ocr_text="MH02??5820",
                    confidence=0.91,
                )
            ],
        ),
        suggested_known_partial_plate=None,
        expected_outcome=(
            "Partial plate characters ('MH02', '5820') combined with yellow auto-rickshaw "
            "attributes rank GT-4 (MH02AZ5820) #1."
        ),
    ),
    "scenario-4": DemoScenario(
        id="scenario-4",
        title="Scenario 4: Out-of-Registry Vehicle",
        description=(
            "High-confidence CCTV reading for an unregistered/fictitious vehicle from Gujarat: "
            "'GJ05XX9999', blue car/sedan. Does not exist in data/registry.db."
        ),
        evidence=VehicleEvidence(
            track_id=4,
            consensus_plate="GJ05XX9999",
            char_confidences=[0.99, 0.99, 0.98, 0.98, 0.96, 0.96, 0.99, 0.99, 0.99, 0.99],
            colour="blue",
            colour_conf=0.91,
            body_type=BodyType.CAR,
            body_type_conf=0.94,
            make="Unknown",
            make_conf=0.0,
            model="Unknown",
            model_conf=0.0,
            frames_used=[
                FrameEvidence(
                    frame_idx=110,
                    timestamp_sec=3.67,
                    image_path="outputs/crops/track_004_frame_0110_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_004_frame_0110_plate.jpg",
                    ocr_text="GJ05XX9999",
                    confidence=0.95,
                )
            ],
        ),
        suggested_known_partial_plate=None,
        expected_outcome=(
            "Triggers the calibrated 'No Confident Match' gate (no_confident_match=True) "
            "because the observed plate strongly contradicts all registered vehicles."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Realistic Mock Video Scanned Tracks (Fallback when CV pipeline is offline)
# ---------------------------------------------------------------------------

MOCK_VIDEO_SCAN_RESPONSE = VideoScanResponse(
    video_id="demo-scan-junction-01",
    video_path="outputs/clips/demo_traffic.mp4",
    duration_seconds=15.0,
    total_frames=450,
    fps=30.0,
    tracks_count=4,
    tracks=[
        TrackSummary(
            track_id=1,
            coarse_class="two-wheeler",
            first_sec=1.20,
            last_sec=6.50,
            frames_seen=159,
            thumbnail_url="outputs/tracks/track_001_two_wheeler.jpg",
        ),
        TrackSummary(
            track_id=2,
            coarse_class="suv",
            first_sec=2.10,
            last_sec=8.40,
            frames_seen=189,
            thumbnail_url="outputs/tracks/track_002_suv.jpg",
        ),
        TrackSummary(
            track_id=3,
            coarse_class="auto-rickshaw",
            first_sec=0.40,
            last_sec=7.10,
            frames_seen=201,
            thumbnail_url="outputs/tracks/track_003_autorickshaw.jpg",
        ),
        TrackSummary(
            track_id=4,
            coarse_class="car",
            first_sec=3.50,
            last_sec=11.20,
            frames_seen=231,
            thumbnail_url="outputs/tracks/track_004_car.jpg",
        ),
    ],
)
