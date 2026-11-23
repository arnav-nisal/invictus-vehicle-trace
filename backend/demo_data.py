"""Pre-configured realistic demo scenarios and mock CV scan tracking results.

Provides 4 ready-to-run VehicleEvidence test payloads and realistic mock track
summaries so the frontend and API work seamlessly end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.schemas import (
    BodyType,
    DemoScenario,
    FrameEvidence,
    TrackSummary,
    VehicleEvidence,
    VideoScanResponse,
)

# ---------------------------------------------------------------------------
# 6 Curated Ground-Truth Dummy Benchmark Cases (Seeded into Registry)
# ---------------------------------------------------------------------------

DUMMY_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "CASE-1",
        "title": "Clean High-Confidence Plate Verification",
        "plate_number": "MH12DE1432",
        "body_type": "two-wheeler",
        "colour": "white",
        "make": "Honda",
        "model": "Activa 6G",
        "rto_city": "Pune, Maharashtra",
        "registered_rto": "MH-12 Pune RTO, Maharashtra",
        "registration_year": 2021,
        "owner_name": "Aarav Sharma",
        "synthetic_owner_id": "SYN-OWN-1001",
        "watchlist_status": "CLEAR",
        "simulated_evidence": VehicleEvidence(
            track_id=1,
            consensus_plate="MH12DE1432",
            char_confidences=[0.99] * 10,
            colour="white",
            colour_conf=0.96,
            body_type=BodyType.TWO_WHEELER,
            body_type_conf=0.98,
            make="Honda",
            make_conf=0.95,
            model="Activa 6G",
            model_conf=0.92,
        ),
    },
    {
        "case_id": "CASE-2",
        "title": "OCR Confusion Recovery on Stolen Vehicle",
        "plate_number": "DL03CB8051",
        "body_type": "suv",
        "colour": "red",
        "make": "Tata",
        "model": "Nexon",
        "rto_city": "Delhi Central, Delhi",
        "registered_rto": "DL-03 Sheikh Sarai RTO, Delhi",
        "registration_year": 2022,
        "owner_name": "Vikram Malhotra",
        "synthetic_owner_id": "SYN-OWN-1005",
        "watchlist_status": "STOLEN",  # Deliberately marked STOLEN
        "simulated_evidence": VehicleEvidence(
            track_id=2,
            consensus_plate="DL03C88051",  # OCR read '8' instead of series letter 'B'
            char_confidences=[0.98, 0.97, 0.95, 0.94, 0.92, 0.35, 0.96, 0.95, 0.94, 0.97],
            colour="red",
            colour_conf=0.95,
            body_type=BodyType.SUV,
            body_type_conf=0.96,
            make="Tata",
            make_conf=0.94,
            model="Nexon",
            model_conf=0.91,
        ),
    },
    {
        "case_id": "CASE-3",
        "title": "Visual-Only Tracking When Plate Is Undetected",
        "plate_number": "MH02AZ5820",
        "body_type": "auto-rickshaw",
        "colour": "yellow",
        "make": "Bajaj",
        "model": "Compact 4S",
        "rto_city": "Mumbai West, Maharashtra",
        "registered_rto": "MH-02 Andheri RTO, Maharashtra",
        "registration_year": 2019,
        "owner_name": "Ramesh Patil",
        "synthetic_owner_id": "SYN-OWN-1013",
        "watchlist_status": "FLAGGED",  # Deliberately marked FLAGGED
        "simulated_evidence": VehicleEvidence(
            track_id=3,
            consensus_plate="NO_PLATE",
            plate_detected=False,
            char_confidences=[],
            colour="yellow",
            colour_conf=0.95,
            body_type=BodyType.AUTO_RICKSHAW,
            body_type_conf=0.98,
            make="Bajaj",
            make_conf=0.95,
            model="Compact 4S",
            model_conf=0.92,
        ),
    },
    {
        "case_id": "CASE-4",
        "title": "Visual Attribute Disambiguation Over Hard Lookalikes",
        "plate_number": "KA01MJ4096",
        "body_type": "car",
        "colour": "silver",
        "make": "Maruti Suzuki",
        "model": "Dzire",
        "rto_city": "Bengaluru Central, Karnataka",
        "registered_rto": "KA-01 Koramangala RTO, Karnataka",
        "registration_year": 2020,
        "owner_name": "Priya Nair",
        "synthetic_owner_id": "SYN-OWN-1009",
        "watchlist_status": "CLEAR",
        "simulated_evidence": VehicleEvidence(
            track_id=4,
            consensus_plate="KA01??4096",  # Series letters occluded
            char_confidences=[0.98, 0.98, 0.97, 0.97, 0.15, 0.15, 0.99, 0.98, 0.99, 0.98],
            colour="silver",
            colour_conf=0.95,
            body_type=BodyType.CAR,
            body_type_conf=0.96,
            make="Maruti Suzuki",
            make_conf=0.94,
            model="Dzire",
            model_conf=0.92,
        ),
    },
    {
        "case_id": "CASE-5",
        "title": "Cloned Plate Anomaly Detection (Truck vs Two-Wheeler)",
        "plate_number": "TN09BY7104",
        "body_type": "truck",
        "colour": "blue",
        "make": "Tata",
        "model": "407 Gold",
        "rto_city": "Chennai South, Tamil Nadu",
        "registered_rto": "TN-09 K.K. Nagar RTO, Tamil Nadu",
        "registration_year": 2018,
        "owner_name": "Karthik Subramanian",
        "synthetic_owner_id": "SYN-OWN-1017",
        "watchlist_status": "CLEAR",
        "simulated_evidence": VehicleEvidence(
            track_id=5,
            consensus_plate="TN09BY7104",  # High-confidence plate match
            char_confidences=[0.99] * 10,
            colour="red",  # High-confidence visual contradiction (red two-wheeler vs blue truck)
            colour_conf=0.95,
            body_type=BodyType.TWO_WHEELER,
            body_type_conf=0.95,
            make="Honda",
            make_conf=0.92,
            model="Activa",
            model_conf=0.90,
        ),
    },
    {
        "case_id": "CASE-6",
        "title": "Om's RTO Prior Boost for Occluded Prefix",
        "plate_number": "MH14EU8899",
        "body_type": "suv",
        "colour": "black",
        "make": "Mahindra",
        "model": "Thar",
        "rto_city": "Pimpri-Chinchwad, Maharashtra",
        "registered_rto": "MH-14 Pimpri-Chinchwad RTO, Maharashtra",
        "registration_year": 2023,
        "owner_name": "Siddharth Deshmukh",
        "synthetic_owner_id": "SYN-OWN-1025",
        "watchlist_status": "CLEAR",
        "rto_priors": {"MH14": 0.82, "MH12": 0.10, "KA12": 0.08},
        "simulated_evidence": VehicleEvidence(
            track_id=6,
            consensus_plate="??14EU8899",  # State prefix occluded
            char_confidences=[0.10, 0.10, 0.98, 0.98, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99],
            colour="black",
            colour_conf=0.95,
            body_type=BodyType.SUV,
            body_type_conf=0.97,
            make="Mahindra",
            make_conf=0.94,
            model="Thar",
            model_conf=0.92,
            rto_priors={"MH14": 0.82, "MH12": 0.10, "KA12": 0.08},
        ),
    },
]

# ---------------------------------------------------------------------------
# Ready-to-Run Realistic VehicleEvidence Demo Scenarios
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
    "scenario-5": DemoScenario(
        id="scenario-5",
        title="Scenario 5: Silver Sedan Lookalike (GT-3)",
        description=(
            "Clean rear capture of a silver sedan with minor dust: 'KA01MJ4096'. "
            "Vehicle is a Maruti Suzuki Dzire."
        ),
        evidence=VehicleEvidence(
            track_id=5,
            consensus_plate="KA01MJ4096",
            char_confidences=[0.99, 0.98, 0.99, 0.97, 0.96, 0.98, 0.99, 0.99, 0.98, 0.99],
            colour="silver",
            colour_conf=0.95,
            body_type=BodyType.CAR,
            body_type_conf=0.95,
            make="Maruti Suzuki",
            make_conf=0.92,
            model="Dzire",
            model_conf=0.90,
            frames_used=[
                FrameEvidence(
                    frame_idx=150,
                    timestamp_sec=5.0,
                    image_path="outputs/crops/track_005_frame_0150_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_005_frame_0150_plate.jpg",
                    ocr_text="KA01MJ4096",
                    confidence=0.96,
                )
            ],
        ),
        suggested_known_partial_plate=None,
        expected_outcome="GT-3 (KA01MJ4096) correctly ranks #1 with high confidence.",
    ),
    "scenario-6": DemoScenario(
        id="scenario-6",
        title="Scenario 6: Commercial Cargo Truck (GT-5)",
        description=(
            "Toll booth surveillance capture of a medium commercial truck: 'TN09BY7104'. "
            "Vehicle is a blue Tata 407 Gold."
        ),
        evidence=VehicleEvidence(
            track_id=6,
            consensus_plate="TN09BY7104",
            char_confidences=[0.97, 0.98, 0.99, 0.96, 0.95, 0.97, 0.99, 0.98, 0.97, 0.98],
            colour="blue",
            colour_conf=0.92,
            body_type=BodyType.TRUCK,
            body_type_conf=0.94,
            make="Tata",
            make_conf=0.90,
            model="407 Gold",
            model_conf=0.88,
            frames_used=[
                FrameEvidence(
                    frame_idx=210,
                    timestamp_sec=7.0,
                    image_path="outputs/crops/track_006_frame_0210_vehicle.jpg",
                    plate_crop_path="outputs/crops/track_006_frame_0210_plate.jpg",
                    ocr_text="TN09BY7104",
                    confidence=0.94,
                )
            ],
        ),
        suggested_known_partial_plate=None,
        expected_outcome="GT-5 (TN09BY7104) ranks #1 as commercial vehicle.",
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
