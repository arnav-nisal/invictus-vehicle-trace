"""Custom Probabilistic Vehicle Matching & Ranking Engine.

Combines partial number plate similarity, CCTV OCR visual confusion matrix,
wildcard-aware character scoring, attribute similarity (colour & body type),
adaptive weighting, and calibrated confidence gating into a production-grade
probabilistic matcher.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.schemas import (
    AnalysisResponse,
    BodyType,
    MatchCandidate,
    RegistryVehicle,
    ScoreBreakdown,
    VehicleEvidence,
)

logger = logging.getLogger("vehicle_trace.matcher")

# ---------------------------------------------------------------------------
# CCTV OCR Visual Confusion Matrix
# Domain confusion groups for characters visually similar under CCTV motion blur
# (0↔O↔D↔Q, 8↔B, 5↔S, 1↔I↔L↔7, 2↔Z, 6↔G, M↔N, U↔V)
# ---------------------------------------------------------------------------
CONFUSION_GROUPS: List[Set[str]] = [
    {"0", "O", "D", "Q"},
    {"8", "B"},
    {"5", "S"},
    {"1", "I", "L", "7"},
    {"2", "Z"},
    {"6", "G"},
    {"M", "N"},
    {"U", "V"},
]

CONFUSION_LOOKUP: Dict[str, Set[str]] = {}
for group in CONFUSION_GROUPS:
    for char in group:
        CONFUSION_LOOKUP[char] = group

# Indian Standard Motor Vehicle Registration Format:
# 2 Letters (State) + 2 Digits (RTO) + 1-2 Letters (Series) + 4 Digits (Number)
STANDARD_INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")

# ---------------------------------------------------------------------------
# Visual Attribute Families
# ---------------------------------------------------------------------------
COLOUR_FAMILIES: List[Tuple[Set[str], float]] = [
    ({"white", "silver", "grey"}, 0.70),
    ({"red", "maroon", "brown", "orange"}, 0.75),
    ({"yellow", "black-yellow"}, 0.80),
    ({"blue", "cyan", "navy"}, 0.70),
    ({"black", "dark-grey", "grey"}, 0.65),
]

BODY_TYPE_COMPATIBILITY: Dict[Tuple[str, str], float] = {
    ("car", "suv"): 0.70,
    ("suv", "car"): 0.70,
}

# Calibrated Threshold: below this, we do not force a bad match
CALIBRATED_CONFIDENCE_THRESHOLD = 0.65


def clean_plate_string(text: str) -> str:
    """Normalize plate strings by uppercasing and stripping separators."""
    return text.strip().upper().replace(" ", "").replace("-", "")


def merge_investigator_hint(
    observed_plate: str,
    char_confidences: List[float],
    known_partial_plate: Optional[str],
) -> Tuple[str, List[float], Optional[str]]:
    """Merge investigator known_partial_plate witness hint into observed evidence."""
    if not known_partial_plate:
        return observed_plate, char_confidences, None

    hint = clean_plate_string(known_partial_plate)
    if not hint:
        return observed_plate, char_confidences, None

    obs_list = list(observed_plate)
    conf_list = list(char_confidences)
    # Ensure confidence list matches length
    while len(conf_list) < len(obs_list):
        conf_list.append(0.90)

    applied_changes = []

    if len(hint) == len(obs_list):
        # Direct positional overlay
        for i in range(len(obs_list)):
            if obs_list[i] == "?" and hint[i] != "?":
                obs_list[i] = hint[i]
                conf_list[i] = 1.0
                applied_changes.append(f"slot {i + 1} -> '{hint[i]}'")
            elif hint[i] != "?" and hint[i] != obs_list[i]:
                # If investigator provided a confirmed reading for an ambiguous character
                obs_list[i] = hint[i]
                conf_list[i] = 1.0
                applied_changes.append(f"slot {i + 1} updated to '{hint[i]}'")
    else:
        # Substring or prefix overlay (e.g., 'MH12DE' or 'DE')
        # Check if hint matches the start of observed
        prefix_len = len(hint)
        if prefix_len <= len(obs_list):
            # Check if this could be series characters (indices 4..5 or 4)
            if prefix_len == 2 and obs_list[4:6] == ["?", "?"] and hint.isalpha():
                obs_list[4] = hint[0]
                obs_list[5] = hint[1]
                conf_list[4] = 1.0
                conf_list[5] = 1.0
                applied_changes.append(f"series slots 5-6 confirmed as '{hint}'")
            elif all(obs_list[i] in ("?", hint[i]) for i in range(prefix_len)):
                for i in range(prefix_len):
                    if obs_list[i] == "?":
                        obs_list[i] = hint[i]
                        conf_list[i] = 1.0
                        applied_changes.append(f"slot {i + 1} -> '{hint[i]}'")

    merged_plate = "".join(obs_list)
    hint_summary = (
        f"Investigator witness hint '{hint}' applied ({', '.join(applied_changes)})"
        if applied_changes
        else f"Investigator witness hint '{hint}' provided"
    )
    return merged_plate, conf_list, hint_summary


def score_plate(
    observed_plate: str,
    candidate_plate: str,
    char_confidences: List[float],
) -> Tuple[float, List[str]]:
    """Wildcard & Confidence-Aware Plate Scorer with CCTV OCR Confusion Matrix.

    Handles '?' unknown slots without penalizing unreadable characters,
    weights readable characters by OCR confidence, accommodates visually
    confusable CCTV motion-blur characters (soft penalty), and rewards
    standard Indian RTO plate format plausibility.
    """
    obs = clean_plate_string(observed_plate)
    cand = clean_plate_string(candidate_plate)

    confs = list(char_confidences)
    while len(confs) < len(obs):
        confs.append(0.90)

    reasons: List[str] = []
    L_obs = len(obs)
    L_cand = len(cand)

    # Check candidate adherence to official Indian RTO plate syntax
    cand_has_valid_syntax = bool(STANDARD_INDIAN_PLATE_REGEX.match(cand))

    # Calculate positional match score
    total_weight = 0.0
    accumulated_score = 0.0
    exact_matches = 0
    wildcards_handled = 0
    confusions_tolerated = 0

    max_len = max(L_obs, L_cand)
    min_len = min(L_obs, L_cand)

    for i in range(min_len):
        c_obs = obs[i]
        c_cand = cand[i]
        w = confs[i] if i < len(confs) else 0.90

        if c_obs == "?":
            # Wildcard slot: do not penalize unreadable characters
            wildcards_handled += 1
            continue

        total_weight += w

        if c_obs == c_cand:
            accumulated_score += 1.0 * w
            exact_matches += 1
        elif c_cand in CONFUSION_LOOKUP.get(c_obs, set()):
            # CCTV OCR blur confusion match: soft partial penalty (0.80)
            confusions_tolerated += 1
            confusion_factor = 0.85
            # If the confusion is resolving an OCR digit in a structurally expected letter slot
            # (such as '8' in a series slot matching 'B' in a valid registry plate)
            if i in (4, 5) and c_obs.isdigit() and c_cand.isalpha() and cand_has_valid_syntax:
                confusion_factor = 0.95

            accumulated_score += confusion_factor * w
            reasons.append(
                f"CCTV OCR confusion tolerated: observed '{c_obs}' vs candidate '{c_cand}' at position {i + 1} ({c_obs}↔{c_cand} blur family)."
            )
        else:
            # Outright mismatch
            accumulated_score += 0.0

    # If observed and candidate lengths differ, apply length mismatch penalty
    length_penalty = 0.0
    if L_obs != L_cand:
        length_diff = abs(L_obs - L_cand)
        length_penalty = 0.15 * length_diff

    # Base plate similarity
    if total_weight > 0:
        raw_plate_score = accumulated_score / total_weight
    else:
        # All slots were wildcards
        raw_plate_score = 0.50

    raw_plate_score = max(0.0, raw_plate_score - length_penalty)

    # RTO Syntax Plausibility factor:
    # A candidate in the official registry conforming to Indian RTO syntax gets full score.
    # Decoys with syntax errors (like digits in series slots or letters in number slots)
    # receive a syntax penalty so real RTO vehicles rank correctly over corrupt decoys.
    if cand_has_valid_syntax:
        syntax_multiplier = 1.0
    else:
        syntax_multiplier = 0.85
        reasons.append(
            f"Candidate plate '{cand}' contains non-standard Indian RTO character structure."
        )

    final_plate_score = round(min(1.0, max(0.0, raw_plate_score * syntax_multiplier)), 4)

    # Explainable match reasons
    legible_count = L_obs - wildcards_handled
    if exact_matches > 0 or wildcards_handled > 0:
        reason_summary = (
            f"Plate comparison: matched {exact_matches}/{legible_count} legible characters "
            f"against '{cand}' (exact: {exact_matches}"
        )
        if confusions_tolerated > 0:
            reason_summary += f", OCR blur confusions tolerated: {confusions_tolerated}"
        reason_summary += ")."
        reasons.insert(0, reason_summary)

    if wildcards_handled > 0:
        reasons.append(f"Handled {wildcards_handled} unreadable/wildcard '?' slots without penalty.")

    return final_plate_score, reasons


def score_colour(observed_colour: str, candidate_colour: str) -> Tuple[float, List[str]]:
    """Score colour similarity including adjacent visual families."""
    c_obs = observed_colour.strip().lower()
    c_cand = candidate_colour.strip().lower()
    reasons: List[str] = []

    if c_obs == c_cand:
        reasons.append(f"Exact vehicle colour match: '{c_cand}'.")
        return 1.0, reasons

    # Check visually adjacent colour families
    for family, score in COLOUR_FAMILIES:
        if c_obs in family and c_cand in family:
            reasons.append(
                f"Colour family compatibility: observed '{c_obs}' is visually adjacent to '{c_cand}' under CCTV lighting."
            )
            return score, reasons

    reasons.append(f"Colour mismatch: observed '{c_obs}' vs registered '{c_cand}'.")
    return 0.0, reasons


def score_body_type(observed_body: BodyType, candidate_body: str) -> Tuple[float, List[str]]:
    """Score body type compatibility including adjacent types (car ↔ suv)."""
    b_obs = observed_body.value.lower()
    b_cand = candidate_body.strip().lower()
    reasons: List[str] = []

    if b_obs == b_cand:
        reasons.append(f"Exact body type match: '{b_cand}'.")
        return 1.0, reasons

    if (b_obs, b_cand) in BODY_TYPE_COMPATIBILITY:
        score = BODY_TYPE_COMPATIBILITY[(b_obs, b_cand)]
        reasons.append(
            f"Adjacent body type compatibility: observed '{b_obs}' vs registered '{b_cand}' (similarity score: {score:.2f})."
        )
        return score, reasons

    reasons.append(f"Body type mismatch: observed '{b_obs}' vs registered '{b_cand}'.")
    return 0.0, reasons


def score_candidate(
    evidence: VehicleEvidence,
    candidate: RegistryVehicle,
    known_partial_plate: Optional[str] = None,
) -> MatchCandidate:
    """Evaluate a single registry vehicle against the extracted visual evidence."""
    # 1. Merge witness hint if provided
    merged_plate, merged_confs, hint_note = merge_investigator_hint(
        evidence.consensus_plate,
        evidence.char_confidences,
        known_partial_plate,
    )

    # 2. Score Plate
    plate_score, plate_reasons = score_plate(merged_plate, candidate.plate_number, merged_confs)
    if hint_note:
        plate_reasons.append(hint_note)

    # 3. Score Colour
    colour_score, colour_reasons = score_colour(evidence.colour, candidate.colour)

    # 4. Score Body Type
    body_score, body_reasons = score_body_type(evidence.body_type, candidate.body_type)

    # 5. Adaptive Weights:
    # Dynamically shift weight toward colour and body type when 3 or fewer plate characters are legible
    legible_chars = sum(1 for ch in merged_plate if ch != "?")
    if legible_chars <= 3:
        w_plate = 0.30
        w_colour = 0.35
        w_body = 0.35
    else:
        w_plate = 0.65
        w_colour = 0.20
        w_body = 0.15

    # Secondary make/model heuristic score (0.0 if not matched or unknown)
    make_model_score = 0.0

    # Calculate composite score incorporating detector confidences
    # Attributes are scaled by detection confidence
    c_conf = evidence.colour_conf
    b_conf = evidence.body_type_conf

    effective_colour = colour_score * (0.80 + 0.20 * c_conf)
    effective_body = body_score * (0.80 + 0.20 * b_conf)

    composite = (
        w_plate * plate_score
        + w_colour * effective_colour
        + w_body * effective_body
    )
    composite_score = round(min(1.0, max(0.0, composite)), 4)

    # Aggregate match reasons
    all_reasons = plate_reasons + body_reasons + colour_reasons

    score_breakdown = ScoreBreakdown(
        plate_score=plate_score,
        colour_score=round(colour_score, 4),
        body_type_score=round(body_score, 4),
        make_model_score=round(make_model_score, 4),
    )

    return MatchCandidate(
        rank=1,  # Placeholder, assigned after sorting
        plate_number=candidate.plate_number,
        body_type=candidate.body_type,
        colour=candidate.colour,
        make=candidate.make,
        model=candidate.model,
        rto_city=candidate.rto_city,
        synthetic_owner_id=candidate.synthetic_owner_id,
        composite_score=composite_score,
        score_breakdown=score_breakdown,
        match_reasons=all_reasons,
    )


def match_and_rank_vehicles(
    evidence: VehicleEvidence,
    registry_vehicles: List[RegistryVehicle],
    known_partial_plate: Optional[str] = None,
    top_k: int = 5,
) -> AnalysisResponse:
    """Run multi-signal probabilistic matching across registry vehicles.

    Returns the Top 5 ranked MatchCandidate entries and evaluates the
    calibrated "No Confident Match" gate.
    """
    if not registry_vehicles:
        return AnalysisResponse(
            track_id=evidence.track_id,
            evidence=evidence,
            top_matches=[],
            no_confident_match=True,
            explanation="Registry database is empty. No matching records available.",
        )

    # Score all candidates
    candidates: List[MatchCandidate] = [
        score_candidate(evidence, vehicle, known_partial_plate)
        for vehicle in registry_vehicles
    ]

    # Sort descending by composite_score, breaking ties with plate_score
    candidates.sort(
        key=lambda c: (c.composite_score, c.score_breakdown.plate_score),
        reverse=True,
    )

    # Assign ranks 1 to top_k
    top_matches = candidates[:top_k]
    for idx, cand in enumerate(top_matches, start=1):
        cand.rank = idx

    # Calibrated "No Confident Match" Gate
    # Check if top candidate exceeds threshold or strongly contradicts evidence
    no_confident_match = False
    explanation = ""

    if not top_matches:
        no_confident_match = True
        explanation = "No matching candidates found in vehicle registry."
    else:
        top_cand = top_matches[0]
        # Gate 1: Composite score falls below calibrated threshold
        if top_cand.composite_score < CALIBRATED_CONFIDENCE_THRESHOLD:
            no_confident_match = True
            explanation = (
                f"No confident match found. The highest-ranked candidate ({top_cand.plate_number}) "
                f"achieved a composite confidence of {top_cand.composite_score:.1%}, which is below our "
                f"calibrated threshold of {CALIBRATED_CONFIDENCE_THRESHOLD:.0%}. Observed evidence "
                f"(plate: '{evidence.consensus_plate}', colour: '{evidence.colour}', body: '{evidence.body_type.value}') "
                f"does not strongly align with any registered vehicle."
            )
        # Gate 2: High-confidence observed plate characters strongly contradict registry
        elif (
            sum(1 for ch in evidence.consensus_plate if ch != "?") >= 5
            and top_cand.score_breakdown.plate_score < 0.50
        ):
            no_confident_match = True
            explanation = (
                f"No confident match found. Although visual attributes align, the plate number "
                f"strongly contradicts registry records (plate similarity: {top_cand.score_breakdown.plate_score:.1%})."
            )
        else:
            explanation = (
                f"Confident match identified: Rank 1 candidate {top_cand.plate_number} "
                f"({top_cand.make} {top_cand.model}, {top_cand.colour} {top_cand.body_type}) "
                f"with {top_cand.composite_score:.1%} composite confidence."
            )

    return AnalysisResponse(
        track_id=evidence.track_id,
        evidence=evidence,
        top_matches=top_matches,
        no_confident_match=no_confident_match,
        explanation=explanation,
    )
