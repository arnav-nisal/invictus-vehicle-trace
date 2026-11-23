"""Comprehensive Forensic Vehicle Matcher & Ranking Engine Tests.

Exercises all 6 forensic scenarios:
1. Clean high-confidence plate verification (MH12DE1432) -> Rank #1 (>95%).
2. Om's RTO prior boost on occluded prefix (??14EU8899) -> beats MH12/KA12 lookalikes.
3. CCTV OCR confusion matrix recovery (DL03C88051 with 8<->B swap) on stolen vehicle.
4. Visual attribute disambiguation (KA01??4096) beating hard tail lookalikes.
5. Cloned plate anomaly detection (TN09BY7104 clean plate on contradictory visual body/colour).
6. Visual-only tracking mode when plate_detected=False (MH02AZ5820).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.database import DEFAULT_DB_PATH, ensure_db_exists, get_all_registry_vehicles
from backend.demo_data import DUMMY_CASES
from backend.main import app
from backend.matcher import match_and_rank_vehicles
from backend.schemas import BodyType, VehicleEvidence

client = TestClient(app)


@pytest.fixture(scope="session")
def vehicles():
    """Load all registered vehicles from SQLite."""
    db_path = ensure_db_exists(DEFAULT_DB_PATH)
    return get_all_registry_vehicles(db_path)


def test_scenario_1_clean_high_confidence_plate(vehicles):
    """Case 1: Clean high-confidence plate -> Rank #1 with total_confidence > 0.95."""
    case = next(c for c in DUMMY_CASES if c["case_id"] == "CASE-1")
    evidence: VehicleEvidence = case["simulated_evidence"]

    response = match_and_rank_vehicles(evidence, vehicles)
    assert not response.no_confident_match, "Should find confident match for clean plate"
    assert len(response.top_matches) > 0

    top = response.top_matches[0]
    assert top.plate_number == "MH12DE1432"
    assert top.rank == 1
    assert top.total_confidence >= 0.95, f"Expected total_confidence >= 0.95, got {top.total_confidence}"
    assert top.composite_score == top.total_confidence
    assert top.score_breakdown.plate_score >= 0.95
    assert top.character_diff is not None
    assert "M H 1 2 D E 1 4 3 2" in top.character_diff


def test_scenario_2_rto_prior_boost_beats_lookalikes(vehicles):
    """Case 2: Occluded RTO prefix ??14EU8899 + rto_priors -> beats MH12 and KA12 8899 lookalikes."""
    case = next(c for c in DUMMY_CASES if c["case_id"] == "CASE-6")
    evidence: VehicleEvidence = case["simulated_evidence"]
    priors = case["rto_priors"]

    response = match_and_rank_vehicles(evidence, vehicles, rto_priors=priors, top_k=20)
    assert not response.no_confident_match
    assert len(response.top_matches) > 0

    top = response.top_matches[0]
    assert top.plate_number == "MH14EU8899", f"Expected MH14EU8899 at Rank 1, got {top.plate_number}"

    scores = {m.plate_number: m.composite_score for m in response.top_matches}
    rto_scores = {m.plate_number: m.rto_prior_score for m in response.top_matches}

    # MH14EU8899 must beat the cross-jurisdiction 8899 lookalikes
    assert scores["MH14EU8899"] > scores.get("MH12AB8899", 0.0), "MH14EU8899 must beat MH12 lookalike"
    assert scores["MH14EU8899"] > scores.get("KA12AB8899", 0.0), "MH14EU8899 must beat KA12 lookalike"
    assert rto_scores["MH14EU8899"] == 0.82
    assert rto_scores.get("MH12AB8899", 0.0) == 0.10
    assert rto_scores.get("KA12AB8899", 0.0) == 0.08


def test_scenario_3_ocr_confusion_recovers_stolen_vehicle(vehicles):
    """Case 3: OCR confusion DL03C88051 (8 in series slot) recovers DL03CB8051 over corrupt lookalike."""
    case = next(c for c in DUMMY_CASES if c["case_id"] == "CASE-2")
    evidence: VehicleEvidence = case["simulated_evidence"]

    response = match_and_rank_vehicles(evidence, vehicles)
    assert not response.no_confident_match
    assert len(response.top_matches) > 0

    top = response.top_matches[0]
    assert top.plate_number == "DL03CB8051", f"Expected DL03CB8051 at Rank 1, got {top.plate_number}"
    assert top.watchlist_status == "STOLEN"

    scores = {m.plate_number: m.composite_score for m in response.top_matches}
    assert scores["DL03CB8051"] > scores.get("DL03C88051", 0.0), "DL03CB8051 must beat syntax-corrupt 8 lookalike"

    # Verify character diff records the OCR swap
    assert "[8->B]" in top.character_diff or "[8->8]" in top.character_diff


def test_scenario_4_visual_attribute_disambiguation(vehicles):
    """Case 4: Missing middle series KA01??4096 -> visual cues beat 4096 lookalikes."""
    case = next(c for c in DUMMY_CASES if c["case_id"] == "CASE-4")
    evidence: VehicleEvidence = case["simulated_evidence"]

    response = match_and_rank_vehicles(evidence, vehicles)
    assert not response.no_confident_match
    assert len(response.top_matches) > 0

    top = response.top_matches[0]
    assert top.plate_number == "KA01MJ4096", f"Expected KA01MJ4096 at Rank 1, got {top.plate_number}"
    assert top.score_breakdown.visual_score >= 0.80

    scores = {m.plate_number: m.composite_score for m in response.top_matches}
    assert scores["KA01MJ4096"] > scores.get("KA01BJ4096", 0.0)


def test_scenario_5_suspected_cloned_plate_anomaly(vehicles):
    """Case 5: Clean plate TN09BY7104 on contradictory visual evidence (red scooter vs blue truck)."""
    case = next(c for c in DUMMY_CASES if c["case_id"] == "CASE-5")
    evidence: VehicleEvidence = case["simulated_evidence"]

    response = match_and_rank_vehicles(evidence, vehicles)
    assert len(response.top_matches) > 0

    # Look for candidate TN09BY7104 in top matches
    cand = next((m for m in response.top_matches if m.plate_number == "TN09BY7104"), None)
    assert cand is not None, "TN09BY7104 should be among top matches due to plate similarity"
    assert cand.score_breakdown.plate_score >= 0.95
    assert "SUSPECTED_CLONED_PLATE" in cand.anomaly_flags, (
        f"Expected 'SUSPECTED_CLONED_PLATE' flag, got {cand.anomaly_flags}"
    )

    # Verify report export renders the anomaly badge
    resp = client.post("/api/reports/export", json=response.model_dump())
    assert resp.status_code == 200
    html_content = resp.text
    assert "SUSPECTED CLONED PLATE" in html_content


def test_scenario_6_visual_only_tracking_no_plate(vehicles):
    """Case 6: plate_detected=False -> valid VISUAL_ONLY Top-5 shortlist without crashing, capped at 0.75."""
    case = next(c for c in DUMMY_CASES if c["case_id"] == "CASE-3")
    evidence: VehicleEvidence = case["simulated_evidence"]

    response = match_and_rank_vehicles(evidence, vehicles)
    assert len(response.top_matches) == 5
    for m in response.top_matches:
        assert m.total_confidence <= 0.75, f"total_confidence {m.total_confidence} exceeded 0.75 cap in VISUAL_ONLY mode"
        assert m.score_breakdown.plate_score == 0.0
        assert m.score_breakdown.visual_score > 0.0

    # Rank 1 should match the yellow auto-rickshaw attributes
    top = response.top_matches[0]
    assert top.plate_number == "MH02AZ5820"
    assert top.body_type == "auto-rickshaw"
    assert top.colour == "yellow"
    assert "Visual-only tracking shortlist" in response.explanation
