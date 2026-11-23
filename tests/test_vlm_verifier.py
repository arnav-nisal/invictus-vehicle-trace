"""Unit tests for Stage 2.4 — Blind Multi-Tier VLM Triple-Check & Slot-by-Slot Tie-Breaker."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from cv_pipeline.vlm_verifier import (
    DEFAULT_CELL_SIZE,
    _extract_json_payload,
    build_evidence_grids,
    compute_ghost_predictions,
    encode_image_b64,
    reconcile_vlm_with_ocr,
)


@pytest.fixture
def dummy_track_env(tmp_path: Path):
    """Creates synthetic image files and mock OCR result for testing."""
    track_dir = tmp_path / "track_test"
    track_dir.mkdir(parents=True, exist_ok=True)

    unenh_dir = track_dir / "plate_crops_selected" / "unenhanced"
    enh_dir = track_dir / "plate_crops_selected" / "enhanced"
    unenh_dir.mkdir(parents=True, exist_ok=True)
    enh_dir.mkdir(parents=True, exist_ok=True)

    evidence_pack = []
    for i in range(4):
        img = np.full((60, 180, 3), 100 + i * 20, dtype=np.uint8)
        cv2.putText(img, f"P{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        u_path = unenh_dir / f"plate_{i:06d}.png"
        e_path = enh_dir / f"plate_{i:06d}.png"
        cv2.imwrite(str(u_path), img)
        cv2.imwrite(str(e_path), img)

        evidence_pack.append({
            "frame_idx": i + 10,
            "weight": 1.25 - i * 0.1,
            "readability_score": 2500 - i * 100,
            "verification_status": "DUAL_VERIFIED",
            "unenhanced_crop_path": str(u_path),
            "enhanced_crop_path": str(e_path),
        })

    mock_ocr = {
        "schema_version": "2.3.0",
        "track_id": 99,
        "recovered_plate": "UP15BE5580",
        "plate_length": 10,
        "overall_confidence": 0.82,
        "slots": [
            {"slot": 0, "slot_type": "L", "char": "U", "bucket": "LOCKED", "pipe_a_top": "U", "pipe_a_conf": 0.95, "pipe_b_top": "U", "pipe_b_conf": 0.95},
            {"slot": 1, "slot_type": "L", "char": "P", "bucket": "LOCKED", "pipe_a_top": "P", "pipe_a_conf": 0.92, "pipe_b_top": "P", "pipe_b_conf": 0.91},
            {"slot": 2, "slot_type": "D", "char": "1", "bucket": "LOCKED", "pipe_a_top": "1", "pipe_a_conf": 0.88, "pipe_b_top": "1", "pipe_b_conf": 0.89},
            {"slot": 3, "slot_type": "D", "char": "5", "bucket": "LOCKED", "pipe_a_top": "5", "pipe_a_conf": 0.85, "pipe_b_top": "5", "pipe_b_conf": 0.84},
            {"slot": 4, "slot_type": "L", "char": "B", "bucket": "CONTRADICTORY", "pipe_a_top": "B", "pipe_a_conf": 0.55, "pipe_a_runner_up": "D", "pipe_a_runner_up_conf": 0.35, "pipe_b_top": "8", "pipe_b_conf": 0.52},
            {"slot": 5, "slot_type": "L", "char": "E", "bucket": "LOCKED", "pipe_a_top": "E", "pipe_a_conf": 0.90, "pipe_b_top": "E", "pipe_b_conf": 0.92},
            {"slot": 6, "slot_type": "D", "char": "5", "bucket": "LOCKED", "pipe_a_top": "5", "pipe_a_conf": 0.88, "pipe_b_top": "5", "pipe_b_conf": 0.87},
            {"slot": 7, "slot_type": "D", "char": "5", "bucket": "LOCKED", "pipe_a_top": "5", "pipe_a_conf": 0.86, "pipe_b_top": "5", "pipe_b_conf": 0.89},
            {"slot": 8, "slot_type": "D", "char": "8", "bucket": "CONTRADICTORY", "pipe_a_top": "8", "pipe_a_conf": 0.48, "pipe_b_top": "0", "pipe_b_conf": 0.45},
            {"slot": 9, "slot_type": "D", "char": "?", "bucket": "UNREADABLE", "pipe_a_top": "?", "pipe_a_conf": 0.15, "pipe_b_top": "?", "pipe_b_conf": 0.10},
        ],
        "vlm_evidence_pack": evidence_pack,
    }

    return track_dir, evidence_pack, mock_ocr


def test_evidence_grid_builder(dummy_track_env):
    """Test 2x2 grid image stitching and resolution."""
    track_dir, evidence_pack, _ = dummy_track_env
    unenh_path, enh_path = build_evidence_grids(track_dir, evidence_pack)

    assert unenh_path.is_file(), "Unenhanced grid missing"
    assert enh_path.is_file(), "Enhanced grid missing"

    img_unenh = cv2.imread(str(unenh_path))
    img_enh = cv2.imread(str(enh_path))

    expected_h = DEFAULT_CELL_SIZE[1] * 2  # 240
    expected_w = DEFAULT_CELL_SIZE[0] * 2  # 640

    assert img_unenh.shape == (expected_h, expected_w, 3), f"Wrong unenh shape: {img_unenh.shape}"
    assert img_enh.shape == (expected_h, expected_w, 3), f"Wrong enh shape: {img_enh.shape}"

    # Test base64 encoding
    b64_unenh = encode_image_b64(unenh_path)
    assert b64_unenh.startswith("data:image/png;base64,")


def test_json_payload_extraction():
    """Test extraction of JSON from raw text, code fences, and chatter."""
    raw1 = '{"plate_text": "UP15BE5580", "plate_layout": "single_line", "slot_reads": []}'
    res1 = _extract_json_payload(raw1)
    assert res1 is not None and res1["plate_text"] == "UP15BE5580"

    raw2 = "Here is your analysis:\n```json\n{\"plate_text\": \"MH01BG7184\", \"plate_layout\": \"two_line\", \"slot_reads\": []}\n```\nDone."
    res2 = _extract_json_payload(raw2)
    assert res2 is not None and res2["plate_text"] == "MH01BG7184"

    raw3 = "Inspected crops.\nResult: {\"plate_text\": \"DL3CC1234\", \"plate_layout\": \"single_line\"}\nConfidence is high."
    res3 = _extract_json_payload(raw3)
    assert res3 is not None and res3["plate_text"] == "DL3CC1234"


def test_vlm_pre_disambiguation(dummy_track_env):
    """Test pre-reconciliation disambiguation turns false disputes (S->5 on digit slot) into confirmations."""
    _, _, mock_ocr = dummy_track_env

    # Slot 6 is digit slot 'D', OCR locked '5'. VLM outputs 'S' at 0.90 conf.
    vlm_output = {
        "plate_text": "UP15BES580",
        "plate_layout": "two_line",
        "slot_reads": [
            {"slot_idx": 6, "char": "S", "confidence": 0.90, "legible": True, "note": "looks like S"},
        ],
    }

    res = reconcile_vlm_with_ocr(mock_ocr, vlm_output)
    slots = {s["slot_idx"]: s for s in res["reconciled_slots"]}

    # Slot 6 disambiguates S -> 5, confirming locked OCR
    s6 = slots[6]
    assert s6["vlm_blind_raw_char"] == "S"
    assert s6["vlm_disambiguated_char"] == "5"
    assert s6["final_char"] == "5"
    assert s6["action"] == "TRIPLE_CHECK_CONFIRMED"
    assert s6["ui_tile_tone"] == "green"
    assert s6["vlm_conf"] == pytest.approx(0.90 * 0.92, abs=1e-3)
    assert any("vlm_disambiguated: 'S'->'5'" in n for n in s6["reasoning_notes"])


def test_reconciliation_locked_slots(dummy_track_env):
    """Test LOCKED slot triple-check confirmation and protection from single-VLM override."""
    _, _, mock_ocr = dummy_track_env

    # VLM agrees with OCR on slot 0 ("U") and disagrees on slot 1 ("P")
    vlm_output = {
        "plate_text": "UR15BE5580",
        "plate_layout": "single_line",
        "slot_reads": [
            {"slot_idx": 0, "char": "U", "confidence": 0.95, "legible": True, "note": "clear U"},
            {"slot_idx": 1, "char": "R", "confidence": 0.60, "legible": False, "note": "blurry stroke"},
        ],
    }

    result = reconcile_vlm_with_ocr(mock_ocr, vlm_output)
    slots = {s["slot_idx"]: s for s in result["reconciled_slots"]}

    # Slot 0: agreed -> TRIPLE_CHECK_CONFIRMED
    assert slots[0]["final_char"] == "U"
    assert slots[0]["action"] == "TRIPLE_CHECK_CONFIRMED"
    assert slots[0]["ui_tile_tone"] == "green"

    # Slot 1: VLM disagreed ("R") -> locked OCR held firm, disputed tag with penalty
    assert slots[1]["final_char"] == "P"
    assert slots[1]["action"] == "LOCKED_VLM_DISPUTED"
    assert slots[1]["ui_tile_tone"] == "black"
    assert slots[1]["confidence_penalty"] == -0.05

    # High certainty disagreement (confidence >= 0.90) still does NOT override 2-OCR consensus
    vlm_dispute_high = {
        "plate_text": "UB15BE5580",
        "slot_reads": [
            {"slot_idx": 1, "char": "B", "confidence": 0.95, "legible": True, "note": "unmistakable B"},
        ],
    }
    res_dispute = reconcile_vlm_with_ocr(mock_ocr, vlm_dispute_high)
    slots_d = {s["slot_idx"]: s for s in res_dispute["reconciled_slots"]}
    assert slots_d[1]["final_char"] == "P"
    assert slots_d[1]["action"] == "LOCKED_VLM_DISPUTED"
    assert slots_d[1]["ui_tile_tone"] == "black"
    assert slots_d[1]["confidence_penalty"] == -0.05


def test_reconciliation_contradictory_tie_break(dummy_track_env):
    """Test CONTRADICTORY slot tie-breaking restricted to OCR candidates only."""
    _, _, mock_ocr = dummy_track_env

    # Slot 4: OCR Pipe A="B", Pipe B="8"
    # Case A: VLM breaks tie in favor of Pipe A ("B")
    vlm_tie_a = {
        "slot_reads": [
            {"slot_idx": 4, "char": "B", "confidence": 0.85, "legible": True, "note": "vertical stem with two closed loops"},
        ]
    }
    res_a = reconcile_vlm_with_ocr(mock_ocr, vlm_tie_a)
    slots_a = {s["slot_idx"]: s for s in res_a["reconciled_slots"]}
    assert slots_a[4]["final_char"] == "B"
    assert slots_a[4]["action"] == "TIE_BREAK_PIPE_A"
    assert slots_a[4]["ui_tile_tone"] == "black"

    # Case B: VLM breaks tie on slot 8 (digit slot, Pipe A="8", Pipe B="0") in favor of Pipe B ("0")
    vlm_tie_b = {
        "slot_reads": [
            {"slot_idx": 8, "char": "0", "confidence": 0.88, "legible": True, "note": "round 0"},
        ]
    }
    res_b = reconcile_vlm_with_ocr(mock_ocr, vlm_tie_b)
    slots_b = {s["slot_idx"]: s for s in res_b["reconciled_slots"]}
    assert slots_b[8]["final_char"] == "0"
    assert slots_b[8]["action"] == "TIE_BREAK_PIPE_B"
    assert slots_b[8]["ui_tile_tone"] == "black"

    # Case C: VLM selects a 3rd candidate NOT in OCR candidates -> VLM_ABSTAINED -> '?'
    vlm_tie_c = {
        "slot_reads": [
            {"slot_idx": 4, "char": "Z", "confidence": 0.95, "legible": True, "note": "diagonal stroke Z"},
        ]
    }
    res_c = reconcile_vlm_with_ocr(mock_ocr, vlm_tie_c)
    slots_c = {s["slot_idx"]: s for s in res_c["reconciled_slots"]}
    assert slots_c[4]["final_char"] == "?"
    assert slots_c[4]["action"] == "VLM_ABSTAINED"
    assert slots_c[4]["ui_tile_tone"] == "unknown"


def test_reconciliation_unreadable_anti_hallucination(dummy_track_env):
    """Test UNREADABLE hard-block: any unreadable slot stays '?' regardless of VLM confidence."""
    _, _, mock_ocr = dummy_track_env

    # Slot 9 is UNREADABLE in mock_ocr
    vlm_guess = {
        "slot_reads": [
            {"slot_idx": 9, "char": "4", "confidence": 0.60, "legible": False, "note": "might be 4 or 1"},
        ]
    }
    res_guess = reconcile_vlm_with_ocr(mock_ocr, vlm_guess)
    slots_g = {s["slot_idx"]: s for s in res_guess["reconciled_slots"]}
    assert slots_g[9]["final_char"] == "?"
    assert slots_g[9]["action"] == "FORBIDDEN_KEEP_UNKNOWN"
    assert slots_g[9]["ui_tile_tone"] == "unknown"
    assert slots_g[9]["vlm_blind_raw_char"] == "4"

    # Even high-confidence VLM guess (0.95) is hard-blocked from final_plate
    vlm_high_guess = {
        "slot_reads": [
            {"slot_idx": 9, "char": "0", "confidence": 0.95, "legible": True, "note": "clear round 0"},
        ]
    }
    res_high = reconcile_vlm_with_ocr(mock_ocr, vlm_high_guess)
    slots_h = {s["slot_idx"]: s for s in res_high["reconciled_slots"]}
    assert slots_h[9]["final_char"] == "?"
    assert slots_h[9]["action"] == "FORBIDDEN_KEEP_UNKNOWN"
    assert slots_h[9]["ui_tile_tone"] == "unknown"
    assert slots_h[9]["vlm_blind_raw_char"] == "0"


def test_ghost_stack_tagging_and_tone(dummy_track_env):
    """Test ghost_stack deduplication, soft_score filtering, and conflict_alt vs ghost_fill tagging."""
    _, _, mock_ocr = dummy_track_env

    # Slot 4 (CONTRADICTORY: A="B", B="8", A_runner="D")
    # Slot 9 (UNREADABLE: A="?", B="?")
    vlm_output = {
        "plate_text": "UP15BE5584",
        "slot_reads": [
            {"slot_idx": 4, "char": "B", "confidence": 0.85, "legible": True, "note": "B"},
            {"slot_idx": 9, "char": "4", "confidence": 0.90, "legible": True, "note": "4"},
        ],
    }

    res = reconcile_vlm_with_ocr(mock_ocr, vlm_output)
    slots = {s["slot_idx"]: s for s in res["reconciled_slots"]}

    # Slot 4: final_char is "B" (tone="black"). ghost_stack must EXCLUDE "B" and tag runners as "conflict_alt"
    s4 = slots[4]
    assert s4["final_char"] == "B"
    assert s4["ui_tile_tone"] == "black"
    g4_chars = [g["char"] for g in s4["ghost_stack"]]
    assert "B" not in g4_chars, "Winning final_char must be excluded from ghost_stack on black tile"
    for g in s4["ghost_stack"]:
        assert g["kind"] == "conflict_alt"
        assert g["soft_score"] >= 0.12

    # Slot 9: final_char is "?" (tone="unknown"). ghost_stack must contain "4" tagged as "ghost_fill"
    s9 = slots[9]
    assert s9["final_char"] == "?"
    assert s9["ui_tile_tone"] == "unknown"
    assert len(s9["ghost_stack"]) > 0
    assert s9["ghost_stack"][0]["char"] == "4"
    assert s9["ghost_stack"][0]["kind"] == "ghost_fill"


def test_beam_search_ghost_predictions():
    """Test beam search ghost_predictions ranks valid Indian state codes (MH) higher than invalid (KH)."""
    # Create slots representing a partially occluded plate: ????BG718?
    # Slot 0 has ghost candidates: 'M' (0.50) and 'K' (0.52)
    # Slot 1 has ghost candidate: 'H' (0.80)
    mock_slots = [
        {
            "slot_idx": 0, "final_char": "?", "ui_tile_tone": "unknown", "final_char_soft_score": 0.0,
            "ghost_stack": [
                {"char": "K", "soft_score": 0.52, "kind": "ghost_fill", "sources": ["pipe_a"]},
                {"char": "M", "soft_score": 0.50, "kind": "ghost_fill", "sources": ["vlm"]},
            ],
        },
        {
            "slot_idx": 1, "final_char": "?", "ui_tile_tone": "unknown", "final_char_soft_score": 0.0,
            "ghost_stack": [{"char": "H", "soft_score": 0.80, "kind": "ghost_fill", "sources": ["vlm"]}],
        },
        {
            "slot_idx": 2, "final_char": "0", "ui_tile_tone": "green", "final_char_soft_score": 0.90, "ghost_stack": [],
        },
        {
            "slot_idx": 3, "final_char": "1", "ui_tile_tone": "green", "final_char_soft_score": 0.90, "ghost_stack": [],
        },
        {
            "slot_idx": 4, "final_char": "B", "ui_tile_tone": "black", "final_char_soft_score": 0.80, "ghost_stack": [],
        },
        {
            "slot_idx": 5, "final_char": "G", "ui_tile_tone": "black", "final_char_soft_score": 0.80, "ghost_stack": [],
        },
        {
            "slot_idx": 6, "final_char": "7", "ui_tile_tone": "green", "final_char_soft_score": 0.95, "ghost_stack": [],
        },
        {
            "slot_idx": 7, "final_char": "1", "ui_tile_tone": "green", "final_char_soft_score": 0.95, "ghost_stack": [],
        },
        {
            "slot_idx": 8, "final_char": "8", "ui_tile_tone": "green", "final_char_soft_score": 0.90, "ghost_stack": [],
        },
        {
            "slot_idx": 9, "final_char": "?", "ui_tile_tone": "unknown", "final_char_soft_score": 0.0,
            "ghost_stack": [{"char": "4", "soft_score": 0.70, "kind": "ghost_fill", "sources": ["vlm"]}],
        },
    ]

    preds = compute_ghost_predictions(mock_slots, plate_len=10, beam_width=8, max_predictions=3)
    assert len(preds) >= 2

    # MH01... gets +0.08 valid state code bonus (Maharashtra) and should beat KH01...
    top_pred = preds[0]
    assert top_pred["plate"].startswith("MH01")
    assert top_pred["state_code_valid"] is True
    assert 0 in top_pred["ghost_slot_indices"]
    assert 1 in top_pred["ghost_slot_indices"]
    assert 9 in top_pred["ghost_slot_indices"]

    # Second prediction should be KH01...
    second_pred = preds[1]
    assert second_pred["plate"].startswith("KH01")
    assert second_pred["state_code_valid"] is False
    assert top_pred["soft_score"] > second_pred["soft_score"]


def test_fallback_when_vlm_fails(dummy_track_env):
    """Test graceful fallback to Stage 2.3 OCR if all VLM providers fail."""
    _, _, mock_ocr = dummy_track_env

    res = reconcile_vlm_with_ocr(
        ocr_result=mock_ocr,
        vlm_output=None,
        fallback_trace=[{"tier": 1, "provider": "groq", "status": "FAILED", "error": "Timeout"}],
    )

    assert res["vlm_status"] == "FALLBACK_TO_OCR"
    assert res["reconciled_plate"] == "UP15BE5580"
    assert res["provider_used"] == "none"
    assert len(res["reconciled_slots"]) == 10
    assert res["reconciled_slots"][0]["action"] == "FALLBACK_STAGE_2_3_OCR"
    assert "ghost_predictions" in res
