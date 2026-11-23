#!/usr/bin/env python3
"""Comprehensive Verification Test Suite for Probabilistic Matcher & Ranking Engine.

Validates:
1. Scenario 1: GT-1 (MH12DE1432) beats its 3 hard near-miss decoys on blurry plate MH12??1432.
2. Scenario 2: OCR confusion (8↔B) still ranks GT-2 (DL03CB8051) #1 on observed DL03C880?1.
3. Scenario 3: Partial plate (MH02??5820) + yellow auto-rickshaw attributes rank GT-4 (MH02AZ5820) #1.
4. Scenario 4: Out-of-registry vehicle (GJ05XX9999) correctly triggers no_confident_match = True.
5. Investigator Witness Hint: known_partial_plate merges seamlessly and boosts confidence.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import ensure_db_exists, get_all_registry_vehicles
from backend.demo_data import DEMO_SCENARIOS
from backend.matcher import match_and_rank_vehicles
from backend.schemas import AnalysisResponse, BodyType, VehicleEvidence


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def print_matches_table(response: AnalysisResponse):
    print(f"\nTarget Track ID: {response.track_id} | Observed: '{response.evidence.consensus_plate}' | "
          f"{response.evidence.colour} {response.evidence.body_type.value}")
    print(f"No Confident Match Gate: {response.no_confident_match}")
    print(f"Outcome Explanation: {response.explanation}\n")
    print(f"{'Rank':<5} | {'Plate':<12} | {'Composite':<10} | {'Plate':<8} | {'Colour':<8} | {'Body':<8} | {'Vehicle Make/Model':<22}")
    print("-" * 85)
    for c in response.top_matches:
        print(
            f"{c.rank:<5} | {c.plate_number:<12} | {c.composite_score * 100:>8.2f}% | "
            f"{c.score_breakdown.plate_score * 100:>6.1f}% | "
            f"{c.score_breakdown.colour_score * 100:>6.1f}% | "
            f"{c.score_breakdown.body_type_score * 100:>6.1f}% | "
            f"{c.make} {c.model:<15}"
        )
    print("-" * 85)
    if response.top_matches:
        print("Top Candidate Match Reasons:")
        for reason in response.top_matches[0].match_reasons:
            print(f"  • {reason}")


try:
    import pytest

    @pytest.fixture
    def vehicles():
        db_path = ensure_db_exists()
        return get_all_registry_vehicles(db_path)
except ImportError:
    pass


def test_scenario_1_blurry_scooter(vehicles):
    print_header("Test 1: Blurry Partial Plate (GT-1 White Scooter vs 3 Hard Decoys)")
    scenario = DEMO_SCENARIOS["scenario-1"]
    response = match_and_rank_vehicles(scenario.evidence, vehicles)
    print_matches_table(response)

    assert not response.no_confident_match, "Scenario 1 should find a confident match"
    top = response.top_matches[0]
    assert top.plate_number == "MH12DE1432", f"Expected GT-1 (MH12DE1432) at Rank 1, got {top.plate_number}"

    # Verify GT-1 beats all 3 hard decoys
    scores = {m.plate_number: m.composite_score for m in response.top_matches}
    assert scores["MH12DE1432"] > scores.get("MH120E1432", 0.0), "GT-1 must beat Decoy 1B (D vs 0)"
    assert scores["MH12DE1432"] > scores.get("MH12DEI432", 0.0), "GT-1 must beat Decoy 1A (1 vs I)"
    assert scores["MH12DE1432"] > scores.get("MH12DE143Z", 0.0), "GT-1 must beat Decoy 1C (2 vs Z)"

    # Test with investigator witness hint 'MH12DE'
    print("\n[+] Testing with investigator witness hint 'MH12DE'...")
    response_hint = match_and_rank_vehicles(scenario.evidence, vehicles, known_partial_plate="MH12DE")
    print_matches_table(response_hint)
    assert response_hint.top_matches[0].plate_number == "MH12DE1432"
    assert any("witness hint" in r.lower() for r in response_hint.top_matches[0].match_reasons)

    print("\n>>> TEST 1 PASSED: GT-1 ranked #1 and convincingly beat all 3 hard decoys!")


def test_scenario_2_ocr_confusion(vehicles):
    print_header("Test 2: CCTV OCR Visual Confusion Matrix (GT-2 Red SUV with 8↔B Confusion)")
    scenario = DEMO_SCENARIOS["scenario-2"]
    response = match_and_rank_vehicles(scenario.evidence, vehicles)
    print_matches_table(response)

    assert not response.no_confident_match, "Scenario 2 should find a confident match"
    top = response.top_matches[0]
    assert top.plate_number == "DL03CB8051", f"Expected GT-2 (DL03CB8051) at Rank 1, got {top.plate_number}"

    # Verify 8↔B confusion was tolerated and flagged
    reasons_str = " ".join(top.match_reasons)
    assert "8" in reasons_str and "B" in reasons_str, "Match reasons must explicitly flag 8↔B confusion"

    # Verify GT-2 ranks above corrupt decoys
    scores = {m.plate_number: m.composite_score for m in response.top_matches}
    assert scores["DL03CB8051"] > scores.get("DL03C88051", 0.0), "GT-2 must beat Decoy 2A (B vs 8)"
    assert scores["DL03CB8051"] > scores.get("DL03CBO051", 0.0), "GT-2 must beat Decoy 2B (0 vs O)"
    assert scores["DL03CB8051"] > scores.get("DL03CB80S1", 0.0), "GT-2 must beat Decoy 2C (5 vs S)"

    print("\n>>> TEST 2 PASSED: GT-2 ranked #1 via 8↔B OCR confusion tolerance and RTO syntax rules!")


def test_scenario_3_partial_auto_rickshaw(vehicles):
    print_header("Test 3: Partial Plate + Distinct Attributes (GT-4 Yellow Auto-Rickshaw)")
    scenario = DEMO_SCENARIOS["scenario-3"]
    response = match_and_rank_vehicles(scenario.evidence, vehicles)
    print_matches_table(response)

    assert not response.no_confident_match, "Scenario 3 should find a confident match"
    top = response.top_matches[0]
    assert top.plate_number == "MH02AZ5820", f"Expected GT-4 (MH02AZ5820) at Rank 1, got {top.plate_number}"

    scores = {m.plate_number: m.composite_score for m in response.top_matches}
    assert scores["MH02AZ5820"] > scores.get("MH02AZ5B20", 0.0), "GT-4 must beat Decoy 4A"
    assert scores["MH02AZ5820"] > scores.get("MH02AZS820", 0.0), "GT-4 must beat Decoy 4B"
    assert scores["MH02AZ5820"] > scores.get("MH02AZ582D", 0.0), "GT-4 must beat Decoy 4C"

    print("\n>>> TEST 3 PASSED: GT-4 ranked #1 combining partial plate hits and vehicle attributes!")


def test_scenario_4_out_of_registry(vehicles):
    print_header("Test 4: Calibrated 'No Confident Match' Gate (GJ05XX9999)")
    scenario = DEMO_SCENARIOS["scenario-4"]
    response = match_and_rank_vehicles(scenario.evidence, vehicles)
    print_matches_table(response)

    assert response.no_confident_match is True, "Scenario 4 MUST trigger no_confident_match = True"
    assert "no confident match" in response.explanation.lower(), "Explanation must indicate lack of confident match"

    print("\n>>> TEST 4 PASSED: Unregistered vehicle correctly gated as 'No Confident Match'!")


def test_fastapi_endpoints(vehicles):
    print_header("Test 5: FastAPI Service Endpoints Integration (ASGI)")
    import asyncio
    import json
    from backend.main import app

    async def call_api(method, path, body_data=None):
        headers = []
        body_bytes = b""
        if body_data is not None:
            body_bytes = json.dumps(body_data).encode()
            headers = [(b"content-type", b"application/json")]

        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "headers": headers,
            "query_string": b"",
        }
        status_code = None
        response_body = bytearray()

        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        async def send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        await app(scope, receive, send)
        decoded = response_body.decode("utf-8") if response_body else ""
        try:
            data = json.loads(decoded)
        except Exception:
            data = decoded
        return status_code, data

    async def run_tests():
        # 1. Health check
        code, health = await call_api("GET", "/api/health")
        assert code == 200, f"Health check returned {code}"
        assert health["status"] == "ok"
        assert health["registry_vehicles_count"] >= 1000
        print(f"  [✓] GET /api/health -> 200 OK (vehicles: {health['registry_vehicles_count']})")

        # 2. Demo scenarios
        code, scenarios = await call_api("GET", "/api/demo-scenarios")
        assert code == 200, f"Demo scenarios returned {code}"
        assert len(scenarios) >= 4
        print(f"  [✓] GET /api/demo-scenarios -> 200 OK ({len(scenarios)} scenarios available)")

        # 3. Video scan
        code, scan = await call_api("POST", "/api/videos/scan", {"clip_id": "demo-traffic-01"})
        assert code == 200, f"Video scan returned {code}"
        assert scan["tracks_count"] > 0
        print(f"  [✓] POST /api/videos/scan -> 200 OK ({scan['tracks_count']} vehicle tracks detected)")

        # 4. Analyze endpoint with scenario_id
        code, analysis = await call_api("POST", "/api/vehicles/analyze", {"scenario_id": "scenario-1"})
        assert code == 200, f"Analyze returned {code}"
        assert analysis["top_matches"][0]["plate_number"] == "MH12DE1432"
        print(f"  [✓] POST /api/vehicles/analyze (scenario-1) -> 200 OK (Rank 1: {analysis['top_matches'][0]['plate_number']})")

        # 4b. Analyze endpoint with track_id via CV bridge
        code, analysis_track = await call_api("POST", "/api/vehicles/analyze", {"track_id": 1})
        assert code == 200, f"Analyze with track_id returned {code}"
        assert analysis_track["top_matches"][0]["plate_number"] == "MH12DE1432"
        print(f"  [✓] POST /api/vehicles/analyze (track_id=1 via CV bridge) -> 200 OK (Rank 1: {analysis_track['top_matches'][0]['plate_number']})")

        # 5. Analyze endpoint with scenario-4 (out of registry)
        code, analysis4 = await call_api("POST", "/api/vehicles/analyze", {"scenario_id": "scenario-4"})
        assert code == 200, f"Analyze returned {code}"
        assert analysis4["no_confident_match"] is True
        print(f"  [✓] POST /api/vehicles/analyze (scenario-4) -> 200 OK (no_confident_match: True)")

        # 6. Report export endpoint with AnalysisResponse
        code, html_content = await call_api("POST", "/api/reports/export", analysis)
        assert code == 200, f"Report export returned {code}"
        assert "FORENSIC VEHICLE TRACEABILITY REPORT" in html_content
        assert "MH12??1432" in html_content
        print(f"  [✓] POST /api/reports/export -> 200 OK ({len(html_content)} bytes self-contained HTML)")

    asyncio.run(run_tests())
    print("\n>>> TEST 5 PASSED: All FastAPI endpoints operational and returning valid schemas!")


def main():
    print("=" * 80)
    print(" INVICTUS VEHICLE TRACE - STEP 2 MATCHER & RANKING VERIFICATION")
    print("=" * 80)

    db_path = ensure_db_exists()
    vehicles = get_all_registry_vehicles(db_path)
    print(f"[✓] Connected to SQLite database: {db_path} ({len(vehicles)} vehicle records loaded)")

    tests = [
        ("Scenario 1 (GT-1 vs 3 Decoys)", test_scenario_1_blurry_scooter),
        ("Scenario 2 (GT-2 8↔B Confusion)", test_scenario_2_ocr_confusion),
        ("Scenario 3 (GT-4 Partial Auto-Rickshaw)", test_scenario_3_partial_auto_rickshaw),
        ("Scenario 4 (Out-of-Registry Gating)", test_scenario_4_out_of_registry),
        ("FastAPI Service Endpoints", test_fastapi_endpoints),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            test_fn(vehicles)
        except AssertionError as e:
            print(f"\n[FAILED] {name}: {e}")
            all_passed = False
        except Exception as e:
            print(f"\n[ERROR] {name}: Unexpected error {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print(" ALL 5 VERIFICATION TESTS PASSED PERFECTLY (100% SUCCESS)!")
        print("=" * 80)
        sys.exit(0)
    else:
        print(" SOME VERIFICATION TESTS FAILED.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
