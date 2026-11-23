"""Pytest Suite for Registry Upgrade, Hard Negatives, and Multi-Camera Sightings.

Tests:
1. Database schema and size (1,000+ vehicles, synthetic owner names, watchlist tags).
2. Ground-truth 6 dummy benchmark cases (STOLEN & FLAGGED presence).
3. Deliberate hard-negative lookalikes (0<->D, 8<->B, and last-4 digits across MH12, MH14, KA12).
4. Auto-logging of sightings upon confident match via POST /api/vehicles/analyze.
5. GET /api/sightings/{plate_number} multi-camera timeline ordered by timestamp ASC.
6. Preservation of frontend API contracts and candidate metadata fields.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient

from backend.database import (
    DEFAULT_DB_PATH,
    ensure_db_exists,
    get_all_registry_vehicles,
    get_db_connection,
    get_sightings_by_plate,
    get_total_vehicles_count,
    get_vehicle_by_plate,
)
from backend.demo_data import DUMMY_CASES
from backend.main import app

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure database is present and seeded."""
    ensure_db_exists(DEFAULT_DB_PATH)


def test_registry_database_schema_and_size():
    """Verify database has 1,000+ vehicles and all upgraded columns exist."""
    count = get_total_vehicles_count()
    assert count >= 1000, f"Expected at least 1,000 registered vehicles, found {count}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check vehicles columns
        cursor.execute("PRAGMA table_info(vehicles);")
        columns = {row["name"] for row in cursor.fetchall()}
        required_cols = {
            "id", "plate_number", "body_type", "colour", "make", "model",
            "rto_city", "synthetic_owner_id", "owner_name", "registered_rto",
            "registration_year", "watchlist_status", "created_at"
        }
        assert required_cols.issubset(columns), f"Missing columns in vehicles table: {required_cols - columns}"

        # Check sightings table
        cursor.execute("PRAGMA table_info(sightings);")
        sighting_cols = {row["name"] for row in cursor.fetchall()}
        required_sighting_cols = {
            "sighting_id", "track_id", "camera_id", "timestamp",
            "top_matched_plate", "confidence", "crop_path"
        }
        assert required_sighting_cols.issubset(sighting_cols), (
            f"Missing columns in sightings table: {required_sighting_cols - sighting_cols}"
        )


def test_ground_truth_6_dummy_cases():
    """Verify all 6 dummy cases from backend/demo_data.py are seeded with exact plates and watchlist tags."""
    assert len(DUMMY_CASES) == 6, "Expected 6 dummy cases in backend/demo_data.py"

    stolen_count = 0
    flagged_count = 0

    for case in DUMMY_CASES:
        plate = case["plate_number"]
        v = get_vehicle_by_plate(plate)
        assert v is not None, f"Dummy case plate {plate} not found in database"
        assert v.make == case["make"]
        assert v.model == case["model"]
        assert v.colour == case["colour"]
        assert v.body_type == case["body_type"]
        assert v.owner_name == case["owner_name"]
        assert v.watchlist_status in ("CLEAR", "STOLEN", "FLAGGED")

        if v.watchlist_status == "STOLEN":
            stolen_count += 1
        elif v.watchlist_status == "FLAGGED":
            flagged_count += 1

    assert stolen_count == 1, f"Expected exactly 1 STOLEN vehicle among 6 dummy cases, got {stolen_count}"
    assert flagged_count == 1, f"Expected exactly 1 FLAGGED vehicle among 6 dummy cases, got {flagged_count}"


def test_hard_negative_ocr_confusion_presence():
    """Verify presence of OCR confusion lookalikes (0<->D, 8<->B, etc.)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number FROM vehicles;")
        all_plates = {row["plate_number"] for row in cursor.fetchall()}

    # 1. MH12DE1432 lookalikes (0<->D)
    assert "MH120E1432" in all_plates, "Lookalike MH120E1432 (0<->D) missing"

    # 2. DL03CB8051 lookalikes (8<->B)
    assert "DL03C88051" in all_plates or "DL03CBB051" in all_plates, "Lookalike with 8<->B missing for DL03CB8051"

    # 3. MH02AZ5820 lookalikes (8<->B)
    assert "MH02AZ5B20" in all_plates, "Lookalike MH02AZ5B20 (8<->B) missing"

    # 4. TN09BY7104 lookalikes (8<->B)
    assert "TN098Y7104" in all_plates, "Lookalike TN098Y7104 (8<->B) missing"


def test_hard_negative_last_4_digits_across_jurisdictions():
    """Verify vehicles sharing exact same last 4 digits exist across MH12, MH14, and KA12."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number FROM vehicles;")
        all_plates = {row["plate_number"] for row in cursor.fetchall()}

    # Check for all 6 tail numbers across MH12, MH14, and KA12
    for case in DUMMY_CASES:
        tail = case["plate_number"][-4:]
        has_mh12 = any(p.startswith("MH12") and p.endswith(tail) for p in all_plates)
        has_mh14 = any(p.startswith("MH14") and p.endswith(tail) for p in all_plates)
        has_ka12 = any(p.startswith("KA12") and p.endswith(tail) for p in all_plates)

        assert has_mh12, f"Missing MH12 vehicle with tail {tail}"
        assert has_mh14, f"Missing MH14 vehicle with tail {tail}"
        assert has_ka12, f"Missing KA12 vehicle with tail {tail}"


def test_preseeded_historical_sightings():
    """Verify pre-seeded historical sightings exist for primary plate across multiple cameras."""
    primary_plate = "MH12DE1432"
    sightings = get_sightings_by_plate(primary_plate)
    assert len(sightings) >= 3, f"Expected at least 3 historical sightings for {primary_plate}, found {len(sightings)}"

    cameras = {s.camera_id for s in sightings}
    assert len(cameras) >= 3, f"Expected sightings across >=3 distinct cameras, found: {cameras}"
    assert "CAM_01" in cameras
    assert "CAM_04" in cameras
    assert "CAM_07" in cameras


def test_get_sightings_api_endpoint():
    """Verify GET /api/sightings/{plate_number} returns sightings ordered by timestamp ASC."""
    resp = client.get("/api/sightings/MH12DE1432")
    assert resp.status_code == 200
    records = resp.json()
    assert isinstance(records, list)
    assert len(records) >= 3

    # Verify fields in SightingRecord schema
    for r in records:
        assert "sighting_id" in r
        assert "camera_id" in r
        assert "timestamp" in r
        assert "top_matched_plate" in r
        assert "confidence" in r

    # Verify chronological ordering (timestamp ASC)
    timestamps = [r["timestamp"] for r in records]
    assert timestamps == sorted(timestamps), "Sightings are not sorted chronologically ascending"


def test_sighting_auto_logging_on_confident_match():
    """Verify that analyzing a vehicle automatically logs a sighting when confident match is found."""
    # Count initial sightings for MH12DE1432
    initial_sightings = len(get_sightings_by_plate("MH12DE1432"))

    # Execute analysis with custom camera_id
    payload = {
        "scenario_id": "scenario-1",
        "camera_id": "CAM_99_OVERPASS",
    }
    resp = client.post("/api/vehicles/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert not data["no_confident_match"]
    assert len(data["top_matches"]) > 0
    top_plate = data["top_matches"][0]["plate_number"]
    assert top_plate == "MH12DE1432"

    # Query sightings again
    after_sightings = get_sightings_by_plate("MH12DE1432")
    assert len(after_sightings) == initial_sightings + 1

    latest_sighting = after_sightings[-1]
    assert latest_sighting.camera_id == "CAM_99_OVERPASS"
    assert latest_sighting.top_matched_plate == "MH12DE1432"
    assert latest_sighting.confidence > 0.90


def test_no_sighting_logged_for_unregistered_vehicle():
    """Verify that unconfident matches (e.g. out-of-registry scenario-4) do not log sightings."""
    unregistered_plate = "GJ05XX9999"
    initial_sightings = len(get_sightings_by_plate(unregistered_plate))

    payload = {
        "scenario_id": "scenario-4",
        "camera_id": "CAM_01",
    }
    resp = client.post("/api/vehicles/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["no_confident_match"] is True

    after_sightings = len(get_sightings_by_plate(unregistered_plate))
    assert after_sightings == initial_sightings == 0


def test_candidate_metadata_and_frontend_contract():
    """Verify that candidate responses include owner_name, registered_rto, registration_year, and watchlist_status."""
    resp = client.post("/api/vehicles/analyze", json={"scenario_id": "scenario-2"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["top_matches"]) > 0

    top_cand = data["top_matches"][0]
    assert top_cand["plate_number"] == "DL03CB8051"
    assert "owner_name" in top_cand and top_cand["owner_name"] is not None
    assert "registered_rto" in top_cand and top_cand["registered_rto"] is not None
    assert "registration_year" in top_cand and top_cand["registration_year"] is not None
    assert top_cand["watchlist_status"] == "STOLEN"  # Ground truth DL03CB8051 is STOLEN
