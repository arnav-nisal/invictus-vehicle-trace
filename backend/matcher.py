"""Custom Probabilistic Vehicle Matching & Forensic Ranking Engine.

Features:
1. Indian Plate Alignment & OCR-Aware Scorer:
   - Normalizes and aligns standard Indian plates (AA00AA0000), single-letter series (AA00A0000),
     and BH series (YYBH0000AA).
   - Character-by-character scoring via CCTV OCR confusion matrix:
     small penalty (0.20) for classic swaps (0<->O<->D, 8<->B, 5<->S, 1<->I<->L, 2<->Z, 6<->G, 4<->A)
     and full penalty (1.0) for unrelated mismatches.
   - Treats '?' or any character with confidence < 0.40 as a soft wildcard.
2. Om's RTO Prior Boost:
   - Ingests optional rto_priors (e.g. {"MH14": 0.82, "MH12": 0.10, "KA12": 0.08}).
   - When prefix positions 0-3 contain '?' or low-confidence characters, scores matching candidate
     RTO prefixes proportional to their prior probability.
3. Dynamic Confidence-Weighted Fusion:
   - Clear plate (>=80% confident, no '?'): 65% Plate+RTO / 35% Visual VLM.
   - Partial/blurry plate ('?' or low conf present): 45% Plate+RTO / 55% Visual VLM.
   - No plate (plate_detected == False): 10% RTO prior / 90% Visual VLM (VISUAL_ONLY mode, capped at 0.75).
   - Visual sub-weights: body_type (30%), colour (30%, with silver<->grey, white<->off-white, maroon<->red partials),
     make (25%), model (15%) scaled by Stage 3 confidences.
4. Cloned-Plate Anomaly Detection & Explainable Breakdown:
   - High plate score (>=0.95) with high-confidence visual contradiction (>=0.85 conf on body or colour)
     triggers 'SUSPECTED_CLONED_PLATE' in anomaly_flags.
   - Full explainable breakdown: total_confidence, plate_score, rto_prior_score, visual_score,
     character_diff (e.g. 'M H 1 2 [?->D] [?->E] 1 4 3 2'), and anomaly_flags.
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
# Classic swap families under motion blur and resolution degradation
# 0↔O↔D, 8↔B, 5↔S, 1↔I↔L, 2↔Z, 6↔G, 4↔A
# ---------------------------------------------------------------------------
CLASSIC_SWAP_GROUPS: List[Set[str]] = [
    {"0", "O", "D", "Q"},
    {"8", "B"},
    {"5", "S"},
    {"1", "I", "L", "7"},
    {"2", "Z"},
    {"6", "G"},
    {"4", "A"},
    {"M", "N"},
    {"U", "V"},
]

CONFUSION_LOOKUP: Dict[str, Set[str]] = {}
for group in CLASSIC_SWAP_GROUPS:
    for char in group:
        CONFUSION_LOOKUP[char] = group

# Indian Standard Motor Vehicle Registration Format:
# 2 Letters (State) + 2 Digits (RTO) + 1-2 Letters (Series) + 4 Digits (Number)
STANDARD_INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")
BH_SERIES_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# ---------------------------------------------------------------------------
# Visual Attribute Families (Partial Match Matrix)
# ---------------------------------------------------------------------------
COLOUR_PARTIAL_PAIRS: Dict[Tuple[str, str], float] = {
    ("silver", "grey"): 0.75,
    ("grey", "silver"): 0.75,
    ("white", "off-white"): 0.75,
    ("off-white", "white"): 0.75,
    ("maroon", "red"): 0.75,
    ("red", "maroon"): 0.75,
    ("brown", "red"): 0.70,
    ("red", "brown"): 0.70,
    ("yellow", "black-yellow"): 0.80,
    ("black-yellow", "yellow"): 0.80,
    ("blue", "navy"): 0.75,
    ("navy", "blue"): 0.75,
    ("blue", "cyan"): 0.70,
    ("cyan", "blue"): 0.70,
    ("black", "dark-grey"): 0.70,
    ("dark-grey", "black"): 0.70,
}

BODY_TYPE_COMPATIBILITY: Dict[Tuple[str, str], float] = {
    ("car", "suv"): 0.70,
    ("suv", "car"): 0.70,
}

# Calibrated Threshold: below this, we do not force a bad match
CALIBRATED_CONFIDENCE_THRESHOLD = 0.65


def clean_plate_string(text: str) -> str:
    """Normalize plate strings by uppercasing and stripping separators."""
    if not text:
        return ""
    return text.strip().upper().replace(" ", "").replace("-", "").replace(".", "")


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
    while len(conf_list) < len(obs_list):
        conf_list.append(0.90)

    applied_changes = []

    if len(hint) == len(obs_list):
        for i in range(len(obs_list)):
            if obs_list[i] == "?" and hint[i] != "?":
                obs_list[i] = hint[i]
                conf_list[i] = 1.0
                applied_changes.append(f"slot {i + 1} -> '{hint[i]}'")
            elif hint[i] != "?" and hint[i] != obs_list[i]:
                obs_list[i] = hint[i]
                conf_list[i] = 1.0
                applied_changes.append(f"slot {i + 1} updated to '{hint[i]}'")
    else:
        prefix_len = len(hint)
        if prefix_len <= len(obs_list):
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


def align_and_score_plate(
    observed_plate: str,
    candidate_plate: str,
    char_confidences: List[float],
) -> Tuple[float, str, List[str]]:
    """Align Indian plates (standard AA00AA0000, single-letter AA00A0000, BH series)
    and score character-by-character using the CCTV OCR confusion matrix.

    Returns:
        (plate_score, character_diff, match_reasons)
    """
    obs = clean_plate_string(observed_plate)
    cand = clean_plate_string(candidate_plate)

    confs = list(char_confidences)
    while len(confs) < len(obs):
        confs.append(0.90)

    reasons: List[str] = []
    L_obs = len(obs)
    L_cand = len(cand)

    cand_has_valid_syntax = bool(STANDARD_INDIAN_PLATE_REGEX.match(cand) or BH_SERIES_REGEX.match(cand))

    # --- Alignment Strategy ---
    # Case 1: Identical length (10 vs 10 or 9 vs 9) -> positional alignment
    # Case 2: Length 9 vs 10 (Single-letter series vs Two-letter series) -> align State(2), RTO(2), Series, Tail(4)
    if L_obs == L_cand:
        aligned_pairs: List[Tuple[str, str, float]] = [
            (obs[i], cand[i], confs[i]) for i in range(L_obs)
        ]
    elif L_obs == 9 and L_cand == 10:
        # Candidate has 2 series letters (cand[4:6]); observed has 1 (obs[4])
        # Try aligning obs[4] with cand[4] (gap at 5) vs with cand[5] (gap at 4)
        opt_a = (
            [(obs[i], cand[i], confs[i]) for i in range(4)]
            + [(obs[4], cand[4], confs[4])]
            + [("?", cand[5], 0.30)]
            + [(obs[5 + j], cand[6 + j], confs[5 + j]) for j in range(4)]
        )
        opt_b = (
            [(obs[i], cand[i], confs[i]) for i in range(4)]
            + [("?", cand[4], 0.30)]
            + [(obs[4], cand[5], confs[4])]
            + [(obs[5 + j], cand[6 + j], confs[5 + j]) for j in range(4)]
        )
        # Select option with higher matching agreement
        score_a = sum(1.0 for o, c, _ in opt_a if o == c or c in CONFUSION_LOOKUP.get(o, set()))
        score_b = sum(1.0 for o, c, _ in opt_b if o == c or c in CONFUSION_LOOKUP.get(o, set()))
        aligned_pairs = opt_a if score_a >= score_b else opt_b
    elif L_obs == 10 and L_cand == 9:
        # Observed has 2 series characters (obs[4:6]); candidate has 1 (cand[4])
        opt_a = (
            [(obs[i], cand[i], confs[i]) for i in range(4)]
            + [(obs[4], cand[4], confs[4])]
            + [(obs[5], "?", confs[5])]
            + [(obs[6 + j], cand[5 + j], confs[6 + j]) for j in range(4)]
        )
        opt_b = (
            [(obs[i], cand[i], confs[i]) for i in range(4)]
            + [(obs[4], "?", confs[4])]
            + [(obs[5], cand[4], confs[5])]
            + [(obs[6 + j], cand[5 + j], confs[6 + j]) for j in range(4)]
        )
        score_a = sum(1.0 for o, c, _ in opt_a if o == c or c in CONFUSION_LOOKUP.get(o, set()))
        score_b = sum(1.0 for o, c, _ in opt_b if o == c or c in CONFUSION_LOOKUP.get(o, set()))
        aligned_pairs = opt_a if score_a >= score_b else opt_b
    else:
        # Fallback for unexpected length differences: direct alignment up to min length
        min_l = min(L_obs, L_cand)
        aligned_pairs = [(obs[i], cand[i], confs[i]) for i in range(min_l)]

    total_weight = 0.0
    accumulated_score = 0.0
    exact_matches = 0
    wildcards_handled = 0
    confusions_tolerated = 0
    diff_tokens: List[str] = []

    for idx, (c_obs, c_cand, w) in enumerate(aligned_pairs):
        is_soft_wildcard = (c_obs == "?" or c_cand == "?" or w < 0.40)

        if is_soft_wildcard:
            wildcards_handled += 1
            # Soft wildcard: does not heavily penalize ambiguous/unreadable character
            diff_tokens.append(f"[{c_obs}->{c_cand}]")
            total_weight += w
            accumulated_score += 0.85 * w  # Soft neutral credit
            continue

        total_weight += w

        if c_obs == c_cand:
            accumulated_score += 1.0 * w
            exact_matches += 1
            diff_tokens.append(c_cand)
        elif c_cand in CONFUSION_LOOKUP.get(c_obs, set()):
            # Classic OCR swap penalty (0.20 penalty -> 0.80 score)
            confusions_tolerated += 1
            penalty = 0.20
            # If confusion is in series slot resolving a digit into a valid letter
            if idx in (4, 5) and c_obs.isdigit() and c_cand.isalpha() and cand_has_valid_syntax:
                penalty = 0.05

            score_val = max(0.0, 1.0 - penalty)
            accumulated_score += score_val * w
            diff_tokens.append(f"[{c_obs}->{c_cand}]")
            reasons.append(
                f"CCTV OCR confusion tolerated: observed '{c_obs}' vs candidate '{c_cand}' at slot {idx + 1} ({c_obs}↔{c_cand} family)."
            )
        else:
            # Full mismatch penalty (1.0 penalty -> 0.0 score)
            accumulated_score += 0.0
            diff_tokens.append(f"[{c_obs}->{c_cand}]")

    diff_str = " ".join(diff_tokens)

    # Length mismatch penalty if overall sizes differed significantly
    length_penalty = 0.15 * abs(L_obs - L_cand) if abs(L_obs - L_cand) > 1 else 0.0

    raw_plate_score = (accumulated_score / total_weight) if total_weight > 0 else 0.50
    raw_plate_score = max(0.0, raw_plate_score - length_penalty)

    # Syntax factor: candidates matching standard syntax get full score; corrupt decoys get 0.85
    syntax_multiplier = 1.0 if cand_has_valid_syntax else 0.85
    if not cand_has_valid_syntax:
        reasons.append(f"Candidate plate '{cand}' contains non-standard Indian RTO character structure.")

    final_plate_score = round(min(1.0, max(0.0, raw_plate_score * syntax_multiplier)), 4)

    # Explainable reason summary
    legible_count = len(aligned_pairs) - wildcards_handled
    if exact_matches > 0 or wildcards_handled > 0:
        summary_msg = f"Plate comparison: matched {exact_matches}/{legible_count} legible characters against '{cand}' (exact: {exact_matches}"
        if confusions_tolerated > 0:
            summary_msg += f", OCR confusions tolerated: {confusions_tolerated}"
        summary_msg += ")."
        reasons.insert(0, summary_msg)

    if wildcards_handled > 0:
        reasons.append(f"Handled {wildcards_handled} unreadable/wildcard '?' slots without penalty.")

    return final_plate_score, diff_str, reasons


def score_rto_prior(
    observed_plate: str,
    char_confidences: List[float],
    candidate_plate: str,
    registered_rto: str,
    rto_priors: Optional[Dict[str, float]],
) -> Tuple[float, bool]:
    """Score matching candidate RTO prefix proportional to Om's RTO prior probability distribution.

    Returns:
        (rto_prior_score, is_prefix_occluded)
    """
    obs = clean_plate_string(observed_plate)
    cand = clean_plate_string(candidate_plate)
    cand_prefix = cand[:4] if len(cand) >= 4 else cand

    # Check if positions 0-3 of observed plate contain '?' or low-confidence characters
    is_occluded = False
    for i in range(min(4, len(obs))):
        c = obs[i]
        conf = char_confidences[i] if i < len(char_confidences) else 0.90
        if c == "?" or conf < 0.40:
            is_occluded = True
            break

    if not rto_priors:
        # When no priors are configured, default to neutral 1.0 if matching or occluded
        if is_occluded:
            return 1.0, is_occluded
        else:
            return (1.0 if cand_prefix == obs[:4] else 0.0), False

    # Lookup prior for candidate prefix (e.g. 'MH14')
    matched_prior = rto_priors.get(cand_prefix)
    if matched_prior is None:
        # Check against registered_rto string (e.g. 'MH-14' or 'MH14')
        clean_rto = registered_rto.upper().replace("-", "").replace(" ", "")
        for k, v in rto_priors.items():
            if k.upper().replace("-", "") in clean_rto:
                matched_prior = v
                break

    prior_score = matched_prior if matched_prior is not None else 0.0
    return round(float(prior_score), 4), is_occluded


def score_colour(observed_colour: str, candidate_colour: str) -> Tuple[float, List[str]]:
    """Score vehicle colour including exact match, partial pairs, and visual families."""
    c_obs = observed_colour.strip().lower()
    c_cand = candidate_colour.strip().lower()
    reasons: List[str] = []

    if c_obs == c_cand:
        reasons.append(f"Exact vehicle colour match: '{c_cand}'.")
        return 1.0, reasons

    # Specific partial 0.75 pairs requested: silver<->grey, white<->off-white, maroon<->red
    if (c_obs, c_cand) in COLOUR_PARTIAL_PAIRS:
        score = COLOUR_PARTIAL_PAIRS[(c_obs, c_cand)]
        reasons.append(
            f"Partial colour compatibility: observed '{c_obs}' matches registered '{c_cand}' (score: {score:.2f})."
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
            f"Adjacent body type compatibility: observed '{b_obs}' vs registered '{b_cand}' (score: {score:.2f})."
        )
        return score, reasons

    reasons.append(f"Body type mismatch: observed '{b_obs}' vs registered '{b_cand}'.")
    return 0.0, reasons


def score_make_model(
    evidence: VehicleEvidence,
    candidate: RegistryVehicle,
) -> Tuple[float, float, List[str]]:
    """Score make and model independently to allow fine-grained visual sub-weighting."""
    obs_make = (getattr(evidence, "make", None) or "Unknown").strip()
    obs_model = (getattr(evidence, "model", None) or "Unknown").strip()
    cand_make = (candidate.make or "").strip()
    cand_model = (candidate.model or "").strip()

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    nm_obs_make = norm(obs_make)
    nm_cand_make = norm(cand_make)
    nm_obs_model = norm(obs_model)
    nm_cand_model = norm(cand_model)

    reasons: List[str] = []
    make_score = 0.0
    if nm_obs_make and nm_obs_make not in ("unknown", "none"):
        if nm_obs_make == nm_cand_make:
            make_score = 1.0
            reasons.append(f"Company/Make match: observed manufacturer '{obs_make}' matches registered '{cand_make}'.")
        elif nm_obs_make in nm_cand_make or nm_cand_make in nm_obs_make:
            make_score = 0.85
            reasons.append(f"Company/Make similarity: observed manufacturer '{obs_make}' closely aligns with '{cand_make}'.")

    model_score = 0.0
    if nm_obs_model and nm_obs_model not in ("unknown", "none"):
        if nm_obs_model == nm_cand_model:
            model_score = 1.0
            reasons.append(f"Model match: observed vehicle model '{obs_model}' matches registered '{cand_model}'.")
        elif nm_obs_model in nm_cand_model or nm_cand_model in nm_obs_model:
            model_score = 0.80
            reasons.append(f"Model family match: observed vehicle model '{obs_model}' closely shares series with '{cand_model}'.")

    return round(make_score, 4), round(model_score, 4), reasons


def score_candidate(
    evidence: VehicleEvidence,
    candidate: RegistryVehicle,
    known_partial_plate: Optional[str] = None,
    rto_priors: Optional[Dict[str, float]] = None,
) -> MatchCandidate:
    """Evaluate a single registry vehicle against extracted evidence using dynamic fusion."""
    plate_detected = getattr(evidence, "plate_detected", True)
    consensus_plate = getattr(evidence, "consensus_plate", "") or ""
    clean_obs = clean_plate_string(consensus_plate)

    # 1. Determine Fusion Mode
    if not plate_detected or clean_obs in ("", "NO_PLATE", "UNKNOWN", "NONE"):
        fusion_mode = "VISUAL_ONLY"
    elif "?" not in clean_obs and sum(1 for c in evidence.char_confidences[:len(clean_obs)] if c >= 0.40) / max(1, len(clean_obs)) >= 0.80:
        fusion_mode = "CLEAR_PLATE"
    else:
        fusion_mode = "PARTIAL_PLATE"

    # 2. Merge Investigator Hint if provided
    merged_plate, merged_confs, hint_note = merge_investigator_hint(
        consensus_plate,
        evidence.char_confidences,
        known_partial_plate,
    )

    # 3. Score Plate & RTO Prior
    if fusion_mode != "VISUAL_ONLY":
        plate_score, char_diff, plate_reasons = align_and_score_plate(
            merged_plate, candidate.plate_number, merged_confs
        )
        if hint_note:
            plate_reasons.append(hint_note)
    else:
        plate_score = 0.0
        char_diff = "NO_PLATE_DETECTED"
        plate_reasons = ["License plate was not detected or unreadable (VISUAL_ONLY mode)."]

    # RTO Prior score
    priors_to_use = rto_priors or getattr(evidence, "rto_priors", None)
    rto_score, is_prefix_occluded = score_rto_prior(
        merged_plate, merged_confs, candidate.plate_number, candidate.registered_rto, priors_to_use
    )
    if priors_to_use and is_prefix_occluded and rto_score > 0:
        plate_reasons.append(f"Om's RTO Prior Boost: matched prefix '{candidate.plate_number[:4]}' with probability {rto_score:.2f}.")

    # 4. Score Visual Attributes (Body, Colour, Make, Model)
    colour_score, colour_reasons = score_colour(evidence.colour, candidate.colour)
    body_score, body_reasons = score_body_type(evidence.body_type, candidate.body_type)
    make_score, model_score, mm_reasons = score_make_model(evidence, candidate)

    # Visual sub-weights: body (30%), colour (30%), make (25%), model (15%) scaled by confidences
    b_conf = getattr(evidence, "body_type_conf", 0.90) or 0.90
    c_conf = getattr(evidence, "colour_conf", 0.90) or 0.90
    m_conf = getattr(evidence, "make_conf", 0.0) or 0.0
    mod_conf = getattr(evidence, "model_conf", 0.0) or 0.0

    eff_w_body = 0.30 * (0.80 + 0.20 * b_conf)
    eff_w_colour = 0.30 * (0.80 + 0.20 * c_conf)
    eff_w_make = 0.25 * m_conf
    eff_w_model = 0.15 * mod_conf
    total_v_weight = eff_w_body + eff_w_colour + eff_w_make + eff_w_model

    visual_score = (
        eff_w_body * body_score
        + eff_w_colour * colour_score
        + eff_w_make * make_score
        + eff_w_model * model_score
    ) / max(0.001, total_v_weight)
    visual_score = round(min(1.0, max(0.0, visual_score)), 4)

    # 5. Dynamic Confidence-Weighted Fusion
    if fusion_mode == "CLEAR_PLATE":
        # 65% Plate+RTO / 35% Visual VLM
        if priors_to_use and is_prefix_occluded:
            plate_rto_score = 0.80 * plate_score + 0.20 * rto_score
        else:
            plate_rto_score = plate_score
        composite = 0.65 * plate_rto_score + 0.35 * visual_score
    elif fusion_mode == "PARTIAL_PLATE":
        # 45% Plate+RTO / 55% Visual VLM
        if priors_to_use and is_prefix_occluded:
            plate_rto_score = 0.65 * plate_score + 0.35 * rto_score
        else:
            plate_rto_score = plate_score
        composite = 0.45 * plate_rto_score + 0.55 * visual_score
    else:  # VISUAL_ONLY
        # 10% RTO prior / 90% Visual VLM, capped at 0.75
        composite = min(0.75, 0.10 * rto_score + 0.90 * visual_score)

    composite_score = round(min(1.0, max(0.0, composite)), 4)

    # 6. Cloned-Plate Anomaly Detection
    anomaly_flags: List[str] = []
    if plate_score >= 0.95:
        # High confidence visual contradiction
        body_contradicts = bool(b_conf >= 0.85 and body_score < 0.50)
        colour_contradicts = bool(c_conf >= 0.85 and colour_score < 0.50)
        if body_contradicts or colour_contradicts:
            anomaly_flags.append("SUSPECTED_CLONED_PLATE")
            plate_reasons.append(
                f"ALERT: Suspected cloned plate detected - high plate similarity ({plate_score:.1%}) "
                f"strongly contradicts high-confidence visual evidence ({evidence.colour} {evidence.body_type.value} "
                f"vs registered {candidate.colour} {candidate.body_type})."
            )

    all_reasons = plate_reasons + body_reasons + colour_reasons + mm_reasons

    score_breakdown = ScoreBreakdown(
        plate_score=plate_score,
        colour_score=round(colour_score, 4),
        body_type_score=round(body_score, 4),
        make_model_score=round((make_score + model_score) / 2.0, 4),
        rto_prior_score=round(rto_score, 4),
        visual_score=visual_score,
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
        owner_name=candidate.owner_name,
        registered_rto=candidate.registered_rto,
        registration_year=candidate.registration_year,
        watchlist_status=candidate.watchlist_status,
        composite_score=composite_score,
        total_confidence=composite_score,
        character_diff=char_diff,
        anomaly_flags=anomaly_flags,
        rto_prior_score=round(rto_score, 4),
        visual_score=visual_score,
        score_breakdown=score_breakdown,
        match_reasons=all_reasons,
    )


def match_and_rank_vehicles(
    evidence: VehicleEvidence,
    registry_vehicles: List[RegistryVehicle],
    known_partial_plate: Optional[str] = None,
    rto_priors: Optional[Dict[str, float]] = None,
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

    priors = rto_priors or getattr(evidence, "rto_priors", None)

    # Score all candidates
    candidates: List[MatchCandidate] = [
        score_candidate(evidence, vehicle, known_partial_plate, priors)
        for vehicle in registry_vehicles
    ]

    # Sort descending by composite_score, breaking ties with plate_score and visual_score
    candidates.sort(
        key=lambda c: (c.composite_score, c.score_breakdown.plate_score, c.score_breakdown.visual_score),
        reverse=True,
    )

    top_matches = candidates[:top_k]
    for idx, cand in enumerate(top_matches, start=1):
        cand.rank = idx

    # Calibrated "No Confident Match" Gate
    no_confident_match = False
    explanation = ""

    if not top_matches:
        no_confident_match = True
        explanation = "No matching candidates found in vehicle registry."
    else:
        top_cand = top_matches[0]
        plate_detected = getattr(evidence, "plate_detected", True)
        obs_clean = clean_plate_string(evidence.consensus_plate)

        if not plate_detected or obs_clean in ("", "NO_PLATE", "UNKNOWN", "NONE"):
            # VISUAL_ONLY Mode
            explanation = (
                f"Visual-only tracking shortlist: plate not detected. Top candidate {top_cand.plate_number} "
                f"({top_cand.make} {top_cand.model}, {top_cand.colour} {top_cand.body_type}) "
                f"achieved {top_cand.total_confidence:.1%} visual confidence."
            )
        elif top_cand.composite_score < CALIBRATED_CONFIDENCE_THRESHOLD:
            no_confident_match = True
            explanation = (
                f"No confident match found. The highest-ranked candidate ({top_cand.plate_number}) "
                f"achieved a composite confidence of {top_cand.composite_score:.1%}, which is below our "
                f"calibrated threshold of {CALIBRATED_CONFIDENCE_THRESHOLD:.0%}. Observed evidence "
                f"(plate: '{evidence.consensus_plate}', colour: '{evidence.colour}', body: '{evidence.body_type.value}') "
                f"does not strongly align with any registered vehicle."
            )
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
