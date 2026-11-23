"""Unit tests for Stage 2.3 OCR arbitration logic.

Tests are pure-Python — no GPU, no model downloads, no disk I/O.
Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_ocr_arbitration.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_pipeline.ocr_engine import (
    CandidateVote,
    LOCKED_MIN_CONF,
    UNREADABLE_MAX_CONF,
    _apply_typed_swap,
    _fuse_readings,
    _top_char,
    aggregate_slot_votes,
    align_to_template,
    classify_slot,
    compute_frame_weight,
    sanitize_plate,
    select_template,
    validate_state_code,
)


# ---------------------------------------------------------------------------
# sanitize_plate
# ---------------------------------------------------------------------------
class TestSanitizePlate:
    def test_uppercase_strip(self):
        assert sanitize_plate("  mh12ab1234  ") == "MH12AB1234"

    def test_strip_ind_prefix_valid_length(self):
        assert sanitize_plate("INDMH12AB1234") == "MH12AB1234"

    def test_strip_in_prefix_valid_length(self):
        assert sanitize_plate("INMH12AB1234") == "MH12AB1234"

    def test_no_strip_if_result_too_short(self):
        # "INXX" → strip IN → "XX" (len 2) — not 9-10, so keep original
        result = sanitize_plate("INXX")
        assert result == "INXX"

    def test_punctuation_stripped(self):
        assert sanitize_plate("MH-12 AB 1234") == "MH12AB1234"


# ---------------------------------------------------------------------------
# select_template
# ---------------------------------------------------------------------------
class TestSelectTemplate:
    def test_prefers_10_when_majority(self):
        candidates = ["MH12AB1234", "MH12AB1234", "MH12AB123"]
        assert select_template(candidates) == 10

    def test_prefers_9_when_majority(self):
        candidates = ["MH12A1234", "MH12A1234", "MH12AB1234"]
        assert select_template(candidates) == 9

    def test_fallback_10_on_empty(self):
        assert select_template([]) == 10


# ---------------------------------------------------------------------------
# align_to_template
# ---------------------------------------------------------------------------
class TestAlignToTemplate:
    def test_exact_match_passthrough(self):
        txt = "MH12AB1234"
        aligned, confs, notes = align_to_template(txt, 10)
        assert aligned == "MH12AB1234"
        assert len(confs) == 10

    def test_one_short_inserts_question_at_mismatch(self):
        # 9 chars; template slot 2 is 'D', but if char is letter it inserts '?'
        txt = "MH12B1234"   # 9 chars, missing a digit
        aligned, confs, notes = align_to_template(txt, 10)
        assert len(aligned) == 10
        assert "?" in aligned

    def test_one_long_drops_bad_slot(self):
        # 11 chars; drop extra
        txt = "MMH12AB1234"
        aligned, confs, notes = align_to_template(txt, 10)
        assert len(aligned) == 10

    def test_too_short_padded_with_question(self):
        txt = "MH12"
        aligned, confs, notes = align_to_template(txt, 10)
        assert len(aligned) == 10
        assert aligned.endswith("??????")

    def test_dropped_edge_char_does_not_shift_all_slots(self):
        # MH12AB1234 with one char dropped from edge → should still recover most
        txt = "H12AB1234"  # 9 chars, missing leading M
        aligned, confs, notes = align_to_template(txt, 10)
        assert len(aligned) == 10
        # At least some core digits should survive
        assert any(c.isdigit() for c in aligned)


# ---------------------------------------------------------------------------
# Indian template disambiguation
# ---------------------------------------------------------------------------
class TestTemplateDisambiguation:
    def test_standard_10_char(self):
        candidates = ["MH12AB1234"] * 5
        assert select_template(candidates) == 10

    def test_standard_9_char(self):
        candidates = ["MH12A1234"] * 5
        assert select_template(candidates) == 9

    def test_bharat_series_is_10(self):
        # Bharat Series: 22BH1234XX — length 10
        candidates = ["22BH1234AB"] * 5
        assert select_template(candidates) == 10


# ---------------------------------------------------------------------------
# validate_state_code
# ---------------------------------------------------------------------------
class TestValidateStateCode:
    def test_valid_state(self):
        assert validate_state_code("MH12AB1234") is True

    def test_valid_state_dl(self):
        assert validate_state_code("DL01AA1234") is True

    def test_invalid_state(self):
        assert validate_state_code("ZZ99XX9999") is False

    def test_bharat_series(self):
        # positions 2-3 == BH
        assert validate_state_code("22BH1234AB") is True


# ---------------------------------------------------------------------------
# compute_frame_weight
# ---------------------------------------------------------------------------
class TestComputeFrameWeight:
    def test_dual_verified_bonus(self):
        w_dual = compute_frame_weight(1000, 1000, "DUAL_VERIFIED", 0.8)
        w_single = compute_frame_weight(1000, 1000, "SINGLE_CUSTOM_YOLO26", 0.8)
        assert w_dual == pytest.approx(w_single * 1.25, rel=1e-6)

    def test_zero_ocr_conf_gives_zero_weight(self):
        w = compute_frame_weight(1000, 1000, "DUAL_VERIFIED", 0.0)
        assert w == pytest.approx(0.0)

    def test_norm_capped_at_1_2(self):
        # readability_score >> track_p95 → norm capped at 1.2
        w = compute_frame_weight(9999, 100, "SINGLE_CUSTOM_YOLO26", 1.0)
        assert w == pytest.approx(1.2, rel=1e-6)


# ---------------------------------------------------------------------------
# LOCKED bucket
# ---------------------------------------------------------------------------
class TestLockedBucket:
    def _make_dist(self, char: str, conf: float) -> dict:
        return {char: conf}

    def test_locked_both_agree_high_conf_multi_frame(self):
        dist_a = {"M": 0.80}
        dist_b = {"M": 0.80}
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=6, n_supporting_b=6)
        assert result["bucket"] == "LOCKED"
        assert result["char"] == "M"
        assert result["vlm_action"] == "TRIPLE_CHECK"

    def test_locked_single_frame_high_conf(self):
        dist_a = {"M": 0.90}
        dist_b = {"M": 0.90}
        result = classify_slot(dist_a, dist_b, n_frames=1, n_supporting_a=1, n_supporting_b=1)
        # combined = 0.90, >= LOCKED_SINGLE_FRAME_CONF=0.85 → LOCKED
        assert result["bucket"] == "LOCKED"


# ---------------------------------------------------------------------------
# CONTRADICTORY bucket
# ---------------------------------------------------------------------------
class TestContradictoryBucket:
    def test_contradictory_pipes_disagree_both_high_conf(self):
        dist_a = {"M": 0.75}
        dist_b = {"N": 0.75}
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=5, n_supporting_b=5)
        assert result["bucket"] == "CONTRADICTORY"
        assert result["vlm_action"] == "TIE_BREAK"

    def test_unanimous_mid_conf_is_contradictory_not_locked(self):
        # A unanimous match but at mid raw OCR conf (~0.50) must land in CONTRADICTORY, NOT LOCKED
        dist_a = {"M": 0.50}
        dist_b = {"M": 0.50}
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=5, n_supporting_b=5)
        assert result["bucket"] == "CONTRADICTORY"
        assert result["char"] == "M"
        assert result["vlm_action"] == "TIE_BREAK"

        # Also verify with CandidateVote structured dict
        cv_a = {"M": CandidateVote({"weight": 5.0, "vote_share": 1.0, "mean_char_conf": 0.50, "effective_conf": 0.50})}
        cv_b = {"M": CandidateVote({"weight": 5.0, "vote_share": 1.0, "mean_char_conf": 0.50, "effective_conf": 0.50})}
        result2 = classify_slot(cv_a, cv_b, n_frames=10, n_supporting_a=5, n_supporting_b=5)
        assert result2["bucket"] == "CONTRADICTORY"
        assert result2["char"] == "M"
        assert result2["vlm_action"] == "TIE_BREAK"

    def test_contradictory_agree_but_mid_conf(self):
        # Both agree on 'M' but each pipe also has a competing char so combined conf < LOCKED_MIN_CONF
        # dist_a: M=0.5, X=0.5 → top=M at 50% conf_a=0.5; dist_b: M=0.6, Y=0.4 → top=M at 60%
        # combined = (0.5+0.6)/2 = 0.55, < 0.65 → CONTRADICTORY
        dist_a = {"M": 0.5, "X": 0.5}
        dist_b = {"M": 0.6, "Y": 0.4}
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=4, n_supporting_b=4)
        # combined ~ 0.55, both agree on M but conf < LOCKED_MIN_CONF=0.65
        # agree=True but falls through because combined < LOCKED_MIN_CONF
        # conf_a=0.5 >= CONTRADICTORY_MIN_CONF=0.35 → CONTRADICTORY
        assert result["bucket"] == "CONTRADICTORY"

    def test_contradictory_char_chosen_from_higher_conf_pipe(self):
        # Pipe A: M at 40% (also has X at 60%), Pipe B: N at 100%
        # They disagree: M vs N
        dist_a = {"M": 0.4, "X": 0.6}   # top = X at 60%, conf_a = 0.6
        dist_b = {"N": 1.0}              # top = N at 100%, conf_b = 1.0
        # conf_a=0.6 for X, conf_b=1.0 for N; they disagree (X ≠ N)
        # combined = (0.6 + 1.0) / 2 = 0.8 → but disagree → CONTRADICTORY
        # char_out = char_b (conf_b > conf_a)
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=3, n_supporting_b=5)
        assert result["bucket"] == "CONTRADICTORY"
        assert result["char"] == "N"  # higher conf pipe B wins


# ---------------------------------------------------------------------------
# UNREADABLE bucket
# ---------------------------------------------------------------------------
class TestUnreadableBucket:
    def test_unanimous_low_conf_is_unreadable_not_locked(self):
        # A unanimous match but at low raw OCR conf (~0.25) must land in UNREADABLE, NOT LOCKED
        dist_a = {"M": 0.25}
        dist_b = {"M": 0.25}
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=5, n_supporting_b=5)
        assert result["bucket"] == "UNREADABLE"
        assert result["char"] == "?"
        assert result["vlm_action"] == "FORBIDDEN_KEEP_UNKNOWN"

        # Also verify with CandidateVote structured dict
        cv_a = {"M": CandidateVote({"weight": 5.0, "vote_share": 1.0, "mean_char_conf": 0.25, "effective_conf": 0.25})}
        cv_b = {"M": CandidateVote({"weight": 5.0, "vote_share": 1.0, "mean_char_conf": 0.25, "effective_conf": 0.25})}
        result2 = classify_slot(cv_a, cv_b, n_frames=10, n_supporting_a=5, n_supporting_b=5)
        assert result2["bucket"] == "UNREADABLE"
        assert result2["char"] == "?"
        assert result2["vlm_action"] == "FORBIDDEN_KEEP_UNKNOWN"

    def test_unreadable_low_combined_conf(self):
        dist_a = {"M": 0.20}
        dist_b = {"M": 0.15}
        result = classify_slot(dist_a, dist_b, n_frames=10, n_supporting_a=1, n_supporting_b=1)
        assert result["bucket"] == "UNREADABLE"
        assert result["char"] == "?"
        assert result["vlm_action"] == "FORBIDDEN_KEEP_UNKNOWN"

    def test_unreadable_low_frame_coverage(self):
        dist_a = {"M": 0.80}
        dist_b = {"M": 0.80}
        # only 1 frame out of 100 — coverage 1% < 15%
        result = classify_slot(dist_a, dist_b, n_frames=100, n_supporting_a=1, n_supporting_b=0)
        assert result["bucket"] == "UNREADABLE"

    def test_unreadable_empty_dists(self):
        result = classify_slot({}, {}, n_frames=10, n_supporting_a=0, n_supporting_b=0)
        assert result["bucket"] == "UNREADABLE"
        assert result["char"] == "?"


# ---------------------------------------------------------------------------
# _apply_typed_swap
# ---------------------------------------------------------------------------
class TestTypedSwap:
    def test_O_to_0_in_digit_slot(self):
        ch, conf, note = _apply_typed_swap("O", "D", 1.0)
        assert ch == "0"
        assert conf == pytest.approx(0.92)
        assert note is not None

    def test_0_to_O_in_letter_slot(self):
        ch, conf, note = _apply_typed_swap("0", "L", 1.0)
        assert ch == "O"
        assert conf == pytest.approx(0.92)

    def test_no_swap_correct_type(self):
        ch, conf, note = _apply_typed_swap("M", "L", 1.0)
        assert ch == "M"
        assert conf == pytest.approx(1.0)
        assert note is None

    def test_S_to_5_digit_slot(self):
        ch, conf, note = _apply_typed_swap("S", "D", 0.8)
        assert ch == "5"
        assert conf == pytest.approx(0.736)


# ---------------------------------------------------------------------------
# aggregate_slot_votes
# ---------------------------------------------------------------------------
class TestAggregateSlotVotes:
    def test_weights_accumulate_correctly(self):
        reads = [
            {"text": "MH12AB1234", "weight": 1.0},
            {"text": "MH12AB1234", "weight": 2.0},
            {"text": "NH12AB1234", "weight": 0.5},
        ]
        votes = aggregate_slot_votes(reads, 10)
        # Slot 0: M=3.0, N=0.5
        assert votes[0]["M"] == pytest.approx(3.0)
        assert votes[0]["N"] == pytest.approx(0.5)

    def test_aggregate_tracks_vote_share_and_mean_conf(self):
        reads = [
            {"text": "MH12AB1234", "weight": 2.0, "char_confs": [0.8] * 10},
            {"text": "MH12AB1234", "weight": 2.0, "char_confs": [0.6] * 10},
        ]
        votes = aggregate_slot_votes(reads, 10)
        assert votes[0]["M"]["vote_share"] == pytest.approx(1.0)
        assert votes[0]["M"]["mean_char_conf"] == pytest.approx(0.7)
        assert votes[0]["M"]["effective_conf"] == pytest.approx(0.7)
        assert votes[0]["M"]["weight"] == pytest.approx(4.0)
        assert votes[0]["M"] == pytest.approx(4.0)

    def test_mismatched_length_frames_skipped(self):
        reads = [
            {"text": "MH12AB123", "weight": 5.0},   # length 9, not 10
            {"text": "MH12AB1234", "weight": 1.0},   # length 10 ✓
        ]
        votes = aggregate_slot_votes(reads, 10)
        # Only the 10-char frame contributes
        assert votes[0]["M"] == pytest.approx(1.0)
