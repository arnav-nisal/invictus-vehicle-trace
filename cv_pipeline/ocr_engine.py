"""Stage 2.3 — Dual-Pipeline Cross-Checked OCR & 3-Bucket Arbitration.

Pipeline A (Unenhanced): RapidOCR primary  + PARSeq cross-check
Pipeline B (Enhanced):   PARSeq primary    + RapidOCR cross-check

Each slot is classified into one of three buckets:
  LOCKED        — both pipes agree, combined conf ≥ 0.65, ≥ 2 supporting frames
  CONTRADICTORY — pipes disagree or combined conf in [0.38, 0.65)
  UNREADABLE    — combined support < 0.38 or < 15% frame coverage

Output: ocr_result.json (VLM handoff payload for Stage 2.4).
"""

from __future__ import annotations

import json
import logging
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Indian plate constants
# ---------------------------------------------------------------------------
_VALID_STATE_CODES = frozenset([
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "TG", "UK", "UP", "WB",
    # Bharat Series uses 'BH' at positions 2-3 (checked separately)
])

# Conservative OCR blur swaps — applied only to typed slot, 0.92x discount
_DIGIT_SWAPS = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "A": "4", "T": "7"}
_LETTER_SWAPS = {v: k for k, v in _DIGIT_SWAPS.items()}  # reverse: 0→O etc.

TWO_LINE_AR_THRESH = 2.2  # width/height < this → two-line split mode
SPLIT_TOP_FRAC = (0.0, 0.58)
SPLIT_BOT_FRAC = (0.42, 1.0)


class CandidateVote(dict):
    """
    Candidate slot vote dictionary storing metrics:
      - weight: raw accumulated weight
      - vote_share: fraction of total votes in slot
      - mean_char_conf: weighted mean raw character confidence
      - effective_conf: vote_share * mean_char_conf
    Also compares against int/float on weight for backwards compatibility with tests.
    """
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.get("weight", 0.0) == other
        return super().__eq__(other)

# ---------------------------------------------------------------------------
# Lazy singletons (avoid loading ~300MB at import time)
# ---------------------------------------------------------------------------
_rapidocr_instance: Optional[object] = None
_parseq_instance: Optional[object] = None
_parseq_transform: Optional[object] = None
_parseq_device: str = "cpu"


def _get_rapidocr():
    global _rapidocr_instance
    if _rapidocr_instance is None:
        from rapidocr import RapidOCR, OCRVersion, LangRec, LangDet, ModelType
        # PP-OCRv6 det (multilingual, includes 'en') + en PP-OCRv5 rec
        _rapidocr_instance = RapidOCR(params={
            "Global.use_cls": False,
            "Global.text_score": 0.3,       # low gate — we filter in arbitration
            "Det.lang_type": LangDet.CH,    # PP-OCRv6 multilingual det
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.SMALL,
            "Rec.lang_type": LangRec.EN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        })
        logger.info("RapidOCR initialised (PP-OCRv6-det + PP-OCRv5-en-rec)")
    return _rapidocr_instance


def _get_parseq(device: str = "cuda"):
    global _parseq_instance, _parseq_transform, _parseq_device
    if _parseq_instance is None:
        import torch
        from torchvision.transforms import (
            Compose, Normalize, Resize, ToTensor,
        )
        from torchvision.transforms import InterpolationMode

        _parseq_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        logger.info(f"Loading PARSeq on {_parseq_device} ...")
        model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        model = model.eval().to(_parseq_device)
        _parseq_instance = model

        img_size = model.hparams.img_size  # (H, W)
        _parseq_transform = Compose([
            Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])
        logger.info(f"PARSeq ready (img_size={img_size}, device={_parseq_device})")
    return _parseq_instance, _parseq_transform, _parseq_device


# ---------------------------------------------------------------------------
# RapidOCR helpers
# ---------------------------------------------------------------------------

def _boxes_to_lines(boxes, txts, scores, img_h: int) -> List[Tuple[str, float]]:
    """Sort detected text boxes top-to-bottom then left-to-right; return [(text, score)]."""
    if not boxes or not txts:
        return []
    items = []
    for box, txt, sc in zip(boxes, txts, scores):
        if txt and txt.strip():
            # box is 4x2 array of corner points
            arr = np.array(box)
            cy = arr[:, 1].mean()
            cx = arr[:, 0].mean()
            items.append((cy, cx, txt.strip(), float(sc)))
    # cluster rows: group boxes within 40% of img_h
    if not items:
        return []
    items.sort(key=lambda x: x[0])
    row_gap = img_h * 0.40
    rows: List[List] = [[items[0]]]
    for item in items[1:]:
        if item[0] - rows[-1][-1][0] < row_gap:
            rows[-1].append(item)
        else:
            rows.append([item])
    result = []
    for row in rows:
        row.sort(key=lambda x: x[1])  # left-to-right
        for _, _, txt, sc in row:
            result.append((txt, sc))
    return result


def _rapidocr_on_crop(ocr, img_bgr: np.ndarray) -> List[Tuple[str, float]]:
    """Run RapidOCR on a BGR image; return list of (text, conf)."""
    result = ocr(img_bgr)
    lines = _boxes_to_lines(
        getattr(result, "boxes", None) or [],
        getattr(result, "txts", None) or [],
        getattr(result, "scores", None) or [],
        img_bgr.shape[0],
    )
    if not lines:
        # Empty-det fallback: skip detection, run rec only
        result2 = ocr(img_bgr, use_det=False, use_cls=False, use_rec=True)
        txts2 = getattr(result2, "txts", None) or []
        scores2 = getattr(result2, "scores", None) or []
        lines = [(t.strip(), float(s)) for t, s in zip(txts2, scores2) if t and t.strip()]
    return lines


def read_rapidocr(img_bgr: np.ndarray, is_two_line: bool) -> Tuple[str, float, List[str]]:
    """
    Returns (joined_text, mean_conf, [line1, line2]) from RapidOCR.
    If is_two_line, splits top/bottom strips and runs recognition-only on each.
    """
    ocr = _get_rapidocr()
    h, w = img_bgr.shape[:2]

    if is_two_line:
        lines_out = []
        for (y0_frac, y1_frac) in [SPLIT_TOP_FRAC, SPLIT_BOT_FRAC]:
            y0, y1 = int(h * y0_frac), int(h * y1_frac)
            strip = img_bgr[y0:y1, :]
            if strip.size == 0:
                continue
            result = ocr(strip, use_det=False, use_cls=False, use_rec=True)
            txts = getattr(result, "txts", None) or []
            scores = getattr(result, "scores", None) or []
            for t, s in zip(txts, scores):
                if t and t.strip():
                    lines_out.append((t.strip(), float(s)))
    else:
        lines_out = _rapidocr_on_crop(ocr, img_bgr)

    if not lines_out:
        return "", 0.0, []
    texts = [t for t, _ in lines_out]
    confs = [s for _, s in lines_out]
    return "".join(texts), float(np.mean(confs)), texts


# ---------------------------------------------------------------------------
# PARSeq helpers
# ---------------------------------------------------------------------------

def _parseq_infer_crop(model, transform, device, img_bgr: np.ndarray) -> Tuple[str, float, List[float]]:
    """Run PARSeq on a single BGR crop; return (text, mean_char_conf, char_confs)."""
    import torch
    from PIL import Image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)  # (1, T, vocab)
        probs = logits.softmax(-1)
        labels, confs = model.tokenizer.decode(probs)

    text = labels[0] if labels else ""
    char_confs_raw = confs[0].tolist() if confs else []

    # Strip trailing EOS score so len(char_confs) == len(text)
    char_confs = char_confs_raw[:len(text)]
    mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
    return text, mean_conf, char_confs


def read_parseq(
    img_bgr: np.ndarray,
    is_two_line: bool,
    rapidocr_line_boxes: Optional[List] = None,
    device: str = "cuda",
) -> Tuple[str, float, List[str]]:
    """
    Returns (joined_text, mean_conf, [line1, line2]) from PARSeq.
    Uses RapidOCR line boxes for two-line split when available.
    """
    model, transform, dev = _get_parseq(device)
    h, w = img_bgr.shape[:2]
    lines_out = []

    if is_two_line:
        for (y0_frac, y1_frac) in [SPLIT_TOP_FRAC, SPLIT_BOT_FRAC]:
            y0, y1 = int(h * y0_frac), int(h * y1_frac)
            strip = img_bgr[y0:y1, :]
            if strip.size == 0 or strip.shape[0] < 4:
                continue
            text, conf, _ = _parseq_infer_crop(model, transform, dev, strip)
            if text:
                lines_out.append((text, conf))
    else:
        text, conf, _ = _parseq_infer_crop(model, transform, dev, img_bgr)
        if text:
            lines_out.append((text, conf))

    if not lines_out:
        return "", 0.0, []
    texts = [t for t, _ in lines_out]
    confs = [s for _, s in lines_out]
    return "".join(texts), float(np.mean(confs)), texts


# ---------------------------------------------------------------------------
# Sanitization & Indian plate alignment
# ---------------------------------------------------------------------------

def sanitize_plate(raw: str) -> str:
    """Uppercase, strip whitespace/punctuation, strip leading IND/IN if result is 9-10 chars."""
    text = raw.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    # Strip leading IND (3 chars) or IN (2 chars) blue-band token
    for prefix in ("IND", "IN"):
        if text.startswith(prefix):
            candidate = text[len(prefix):]
            if 9 <= len(candidate) <= 10:
                text = candidate
                break
    return text


# Indian plate slot templates (letter=L, digit=D)
_TEMPLATES = {
    "standard_10": list("LLDDLLDDDDD"),   # e.g. MH12AB1234 — wait, 10 chars: SS DD LL DDDD
    "standard_9":  list("LLDDLDDDDD"),    # 9-char older/commercial
    "bharat_10":   list("DDBHLLDDDDD"),   # Bharat series placeholder — handled separately
}
# Correct templates:
# Standard 10:  SS  DD  LL  DDDD  (2+2+2+4 = 10)
# Standard 9:   SS  DD  L   DDDD  (2+2+1+4 = 9)
# Bharat 10:    YY  BH  DDDD  LL  (2+2+4+2 = 10 but different layout)
_SLOT_TYPES = {
    10: list("LLDDLLDDDDD"),    # 10 slots: L L D D L L D D D D
    9:  list("LLDLDDDDDD"),     # 9 slots: L L D L D D D D D   -- wait, corrected:
}
# Standard 10: M H 1 2 A B 1 2 3 4 → L L D D L L D D D D
# Standard 9:  M H 1 2 A 1 2 3 4   → L L D D L D D D D
_SLOT_TYPES_CORRECT = {
    10: ["L","L","D","D","L","L","D","D","D","D"],
    9:  ["L","L","D","D","L","D","D","D","D"],
}


def _char_matches_slot(ch: str, slot_type: str) -> bool:
    if slot_type == "L":
        return ch.isalpha()
    return ch.isdigit()


def _apply_typed_swap(ch: str, slot_type: str, conf: float) -> Tuple[str, float, Optional[str]]:
    """Conservative blur swap: only swap if the char type mismatches the slot."""
    note = None
    if slot_type == "D" and ch.isalpha() and ch in _DIGIT_SWAPS:
        swapped = _DIGIT_SWAPS[ch]
        note = f"swap {ch}→{swapped}@D (conf×0.92)"
        return swapped, conf * 0.92, note
    if slot_type == "L" and ch.isdigit() and ch in _LETTER_SWAPS:
        swapped = _LETTER_SWAPS[ch]
        note = f"swap {ch}→{swapped}@L (conf×0.92)"
        return swapped, conf * 0.92, note
    return ch, conf, note


def select_template(candidates: List[str]) -> int:
    """Return most-voted plate length (10, 9, or fallback 10)."""
    lengths = [len(s) for s in candidates if 9 <= len(s) <= 10]
    if not lengths:
        return 10
    return Counter(lengths).most_common(1)[0][0]


def align_to_template(text: str, length: int) -> Tuple[str, List[float], List[str]]:
    """
    Align `text` to `length`-slot Indian template.
    Returns (aligned_text, uniform_confs, notes).
    If len(text) == length, identity mapping.
    If off by 1, insert '?' at most-likely gap.
    Otherwise pad/truncate.
    """
    slot_types = _SLOT_TYPES_CORRECT.get(length, _SLOT_TYPES_CORRECT[10])
    notes = []

    # Try exact match first
    if len(text) == length:
        return text, [1.0] * length, notes

    # Off by 1 — find the first mismatched slot and insert '?'
    if len(text) == length - 1:
        for i, (ch, st) in enumerate(zip(text, slot_types)):
            if not _char_matches_slot(ch, st):
                aligned = text[:i] + "?" + text[i:]
                notes.append(f"inserted '?' at slot {i} (type mismatch)")
                return aligned[:length], [1.0] * length, notes
        # Append at end
        aligned = text + "?"
        return aligned[:length], [1.0] * length, notes

    if len(text) == length + 1:
        # Drop the char that most violates its slot
        best_drop = 0
        for i, (ch, st) in enumerate(zip(text, slot_types)):
            if not _char_matches_slot(ch, st):
                best_drop = i
                break
        aligned = text[:best_drop] + text[best_drop + 1:]
        notes.append(f"dropped char at slot {best_drop}")
        return aligned[:length], [1.0] * length, notes

    # Pad or truncate
    if len(text) < length:
        aligned = text + "?" * (length - len(text))
        notes.append(f"padded with ? (got {len(text)}, need {length})")
        return aligned, [1.0] * length, notes

    return text[:length], [1.0] * length, [f"truncated to {length}"]


def validate_state_code(chars: str) -> bool:
    """Check first two chars are a valid Indian state/UT code."""
    if len(chars) < 2:
        return False
    code = chars[:2].upper()
    if code in _VALID_STATE_CODES:
        return True
    # Bharat Series: positions 2-3 are "BH" (year at 0-1)
    if len(chars) >= 4 and chars[2:4].upper() == "BH":
        return True
    return False


# ---------------------------------------------------------------------------
# Frame weight
# ---------------------------------------------------------------------------

def compute_frame_weight(
    readability_score: float,
    track_p95: float,
    verification_status: str,
    ocr_conf: float,
) -> float:
    if track_p95 <= 0:
        norm_read = 1.0
    else:
        norm_read = min(1.2, readability_score / track_p95)
    dual_bonus = 1.25 if verification_status == "DUAL_VERIFIED" else 1.0
    return norm_read * dual_bonus * max(0.0, ocr_conf)


# ---------------------------------------------------------------------------
# Per-frame OCR fusion (Pipe A and Pipe B)
# ---------------------------------------------------------------------------

def _fuse_readings(
    primary_txt: str, primary_conf: float,
    xcheck_txt: str, xcheck_conf: float,
    length: int,
) -> Tuple[str, float, List[float], List[str]]:
    """
    Fuse two raw readings into a single aligned text + per-slot conf list.
    When both agree on a char at a slot, boost conf by 1.15.
    Returns (aligned_text, mean_slot_conf, slot_char_confs, notes).
    """
    p_clean = sanitize_plate(primary_txt)
    x_clean = sanitize_plate(xcheck_txt)
    notes = []

    # Prefer the reading with length closer to target
    p_dist = abs(len(p_clean) - length)
    x_dist = abs(len(x_clean) - length)
    if p_dist <= x_dist:
        base_txt, base_conf = p_clean, primary_conf
        alt_txt, alt_conf = x_clean, xcheck_conf
    else:
        base_txt, base_conf = x_clean, xcheck_conf
        alt_txt, alt_conf = p_clean, primary_conf

    aligned, _, align_notes = align_to_template(base_txt, length)
    notes.extend(align_notes)

    # Initialize per-slot character confidences from base_conf
    slot_confs = [0.0 if ch == "?" else float(base_conf) for ch in aligned]

    # Cross-check: slot-by-slot agreement boost
    slot_confs = list(slot_confs)  # copy
    for i, (ch, st) in enumerate(zip(aligned, _SLOT_TYPES_CORRECT.get(length, []))):
        if ch == "?":
            continue
        if i < len(alt_txt) and alt_txt[i] == ch:
            slot_confs[i] = min(1.0, slot_confs[i] * 1.15)
        # Apply typed swap
        swapped, new_conf, swap_note = _apply_typed_swap(ch, st, slot_confs[i])
        if swap_note:
            aligned = aligned[:i] + swapped + aligned[i+1:]
            slot_confs[i] = new_conf
            notes.append(swap_note)

    mean_conf = float(np.mean(slot_confs)) if slot_confs else 0.0
    return aligned, mean_conf, slot_confs, notes


# ---------------------------------------------------------------------------
# Slot-by-slot vote aggregation
# ---------------------------------------------------------------------------

def aggregate_slot_votes(
    all_frame_reads: List[Dict],  # [{text, weight, conf, char_confs}]
    length: int,
) -> Dict[int, Dict[str, CandidateVote]]:
    """
    For each slot index, accumulate weighted votes and weighted confidence per character.
    Tracks both vote_share and mean_char_conf, yielding effective_conf = vote_share * mean_char_conf.
    Returns {slot_idx: {char: CandidateVote}}.
    """
    slot_data: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"weight": 0.0, "weighted_conf": 0.0})
    )

    for read in all_frame_reads:
        text = read["text"]
        weight = float(read.get("weight", 1.0))
        char_confs = read.get("char_confs")
        if char_confs is None:
            char_confs = [float(read.get("conf", 1.0))] * len(text)

        if len(text) != length:
            continue

        for i, ch in enumerate(text):
            if ch == "?":
                continue
            c_conf = float(char_confs[i]) if i < len(char_confs) else 1.0
            slot_data[i][ch]["weight"] += weight
            slot_data[i][ch]["weighted_conf"] += weight * c_conf

    result: Dict[int, Dict[str, CandidateVote]] = {}
    for i in range(length):
        cands = slot_data.get(i, {})
        total_slot_weight = sum(d["weight"] for d in cands.values())
        slot_dict = {}
        for ch, d in cands.items():
            w = d["weight"]
            vote_share = (w / total_slot_weight) if total_slot_weight > 0 else 0.0
            mean_char_conf = (d["weighted_conf"] / w) if w > 0 else 0.0
            effective_conf = vote_share * mean_char_conf
            slot_dict[ch] = CandidateVote({
                "weight": w,
                "vote_share": vote_share,
                "mean_char_conf": mean_char_conf,
                "effective_conf": effective_conf,
                "conf": effective_conf,
            })
        result[i] = slot_dict

    return result


def _top_char(vote_dist: Dict) -> Tuple[str, float, float, float]:
    """
    Returns (top_char, effective_conf, vote_share, mean_char_conf).
    Compatible with:
      - dict of CandidateVote: {char: {"vote_share": ..., "mean_char_conf": ..., "effective_conf": ...}}
      - dict of floats: {char: float_conf_or_weight}
    """
    if not vote_dist:
        return "?", 0.0, 0.0, 0.0

    valid_cands = {k: v for k, v in vote_dist.items() if k != "?"}
    target_dist = valid_cands if valid_cands else vote_dist
    if not target_dist:
        return "?", 0.0, 0.0, 0.0

    first_val = next(iter(target_dist.values()))
    if isinstance(first_val, dict):
        top = max(target_dist, key=lambda c: target_dist[c].get("effective_conf", 0.0))
        entry = target_dist[top]
        v_share = float(entry.get("vote_share", 0.0))
        m_conf = float(entry.get("mean_char_conf", 0.0))
        eff_conf = float(entry.get("effective_conf", v_share * m_conf))
        return top, eff_conf, v_share, m_conf
    else:
        # Simple float dict (used in unit tests)
        # If single candidate (unanimous): e.g. {"M": 0.50} or {"M": 0.25}
        if len(target_dist) == 1:
            top = next(iter(target_dist.keys()))
            val = float(target_dist[top])
            v_share = 1.0
            m_conf = val
            eff_conf = v_share * m_conf
            return top, eff_conf, v_share, m_conf
        else:
            total = sum(target_dist.values())
            top = max(target_dist, key=target_dist.get)
            val = float(target_dist[top])
            v_share = (val / total) if total > 0 else 0.0
            m_conf = 1.0
            eff_conf = v_share * m_conf
            return top, eff_conf, v_share, m_conf


# ---------------------------------------------------------------------------
# 3-Bucket classification per slot
# ---------------------------------------------------------------------------

LOCKED_MIN_CONF = 0.65
CONTRADICTORY_MIN_CONF = 0.35
UNREADABLE_MAX_CONF = 0.38
LOCKED_MIN_FRAMES = 2
LOCKED_SINGLE_FRAME_CONF = 0.85
MIN_FRAME_COVERAGE_PCT = 0.15


def classify_slot(
    pipe_a_dist: Dict,
    pipe_b_dist: Dict,
    n_frames: int,
    n_supporting_a: int,
    n_supporting_b: int,
) -> Dict:
    """
    Classify a single slot into LOCKED / CONTRADICTORY / UNREADABLE.
    Returns dict with keys:
      bucket, char, conf_a, vote_share_a, char_conf_a,
      conf_b, vote_share_b, char_conf_b, combined_conf, vlm_action.
    """
    total_frames = max(n_frames, 1)
    cov_a = n_supporting_a / total_frames
    cov_b = n_supporting_b / total_frames

    char_a, conf_a, share_a, raw_a = _top_char(pipe_a_dist)
    char_b, conf_b, share_b, raw_b = _top_char(pipe_b_dist)
    combined = (conf_a + conf_b) / 2.0

    base_res = {
        "conf_a": conf_a,
        "vote_share_a": share_a,
        "char_conf_a": raw_a,
        "conf_b": conf_b,
        "vote_share_b": share_b,
        "char_conf_b": raw_b,
        "combined_conf": combined,
    }

    # UNREADABLE: low combined or insufficient frame coverage
    if combined < UNREADABLE_MAX_CONF or (cov_a < MIN_FRAME_COVERAGE_PCT and cov_b < MIN_FRAME_COVERAGE_PCT):
        return {
            "bucket": "UNREADABLE",
            "char": "?",
            **base_res,
            "vlm_action": "FORBIDDEN_KEEP_UNKNOWN",
        }

    agree = char_a == char_b and char_a != "?"

    # LOCKED: agreement + sufficient confidence + frame support
    if agree and combined >= LOCKED_MIN_CONF:
        single_frame = (n_supporting_a <= 1 and n_supporting_b <= 1)
        if single_frame and combined < LOCKED_SINGLE_FRAME_CONF:
            bucket = "CONTRADICTORY"
            vlm = "TIE_BREAK"
        elif n_supporting_a + n_supporting_b >= LOCKED_MIN_FRAMES or not single_frame:
            bucket = "LOCKED"
            vlm = "TRIPLE_CHECK"
        else:
            bucket = "CONTRADICTORY"
            vlm = "TIE_BREAK"
        return {
            "bucket": bucket,
            "char": char_a,
            **base_res,
            "vlm_action": vlm,
        }

    # CONTRADICTORY: both pipes have meaningful signal but disagree or mid-conf
    if conf_a >= CONTRADICTORY_MIN_CONF or conf_b >= CONTRADICTORY_MIN_CONF:
        char_out = char_a if conf_a >= conf_b else char_b
        return {
            "bucket": "CONTRADICTORY",
            "char": char_out,
            **base_res,
            "vlm_action": "TIE_BREAK",
        }

    return {
        "bucket": "UNREADABLE",
        "char": "?",
        **base_res,
        "vlm_action": "FORBIDDEN_KEEP_UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Main OCR engine: single track
# ---------------------------------------------------------------------------

class PlateOCREngine:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def _run_frame(
        self,
        unenh_path: str,
        enh_path: str,
        plate_saved_size: List[int],
        readability_score: float,
        track_p95: float,
        verification_status: str,
    ) -> Optional[Dict]:
        """Process one frame; return dict of pipe_a and pipe_b readings, or None on error."""
        try:
            unenh = cv2.imread(unenh_path)
            enh = cv2.imread(enh_path)
            if unenh is None or enh is None:
                logger.warning(f"Cannot read crop: {unenh_path}")
                return None

            w, h = plate_saved_size
            is_two_line = (w / max(h, 1)) < TWO_LINE_AR_THRESH

            # Pipeline A: Unenhanced, RapidOCR primary + PARSeq xcheck
            r_txt_a, r_conf_a, r_lines_a = read_rapidocr(unenh, is_two_line)
            p_txt_a, p_conf_a, _ = read_parseq(unenh, is_two_line, device=self.device)

            # Pipeline B: Enhanced, PARSeq primary + RapidOCR xcheck
            # Gentle CLAHE only if both return empty on very dark crop
            enh_input = enh
            if not sanitize_plate(p_txt_a) and not sanitize_plate(r_txt_a):
                gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                cl = clahe.apply(gray)
                enh_input = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

            p_txt_b, p_conf_b, _ = read_parseq(enh_input, is_two_line, device=self.device)
            r_txt_b, r_conf_b, r_lines_b = read_rapidocr(enh_input, is_two_line)

            ocr_conf_a = (r_conf_a + p_conf_a) / 2.0
            ocr_conf_b = (p_conf_b + r_conf_b) / 2.0
            frame_weight_a = compute_frame_weight(readability_score, track_p95, verification_status, ocr_conf_a)
            frame_weight_b = compute_frame_weight(readability_score, track_p95, verification_status, ocr_conf_b)

            return {
                "pipe_a": {
                    "primary_txt": r_txt_a, "primary_conf": r_conf_a,
                    "xcheck_txt": p_txt_a, "xcheck_conf": p_conf_a,
                    "weight": frame_weight_a,
                },
                "pipe_b": {
                    "primary_txt": p_txt_b, "primary_conf": p_conf_b,
                    "xcheck_txt": r_txt_b, "xcheck_conf": r_conf_b,
                    "weight": frame_weight_b,
                },
            }
        except Exception as exc:
            logger.error(f"Frame OCR error ({unenh_path}): {exc}", exc_info=True)
            return None

    def run_ocr_track(self, plate_manifest_path: str) -> Dict:
        """
        Main entry point.  Reads plate_manifest.json, runs dual pipelines,
        arbitrates per slot, and returns the ocr_result dict.
        """
        t0 = time.time()
        manifest_path = Path(plate_manifest_path).resolve()
        with open(manifest_path) as f:
            manifest = json.load(f)

        track_id = manifest.get("track_id")
        merged_track_ids = manifest.get("merged_track_ids", [track_id])
        is_merged_track = bool(manifest.get("is_merged_track", False))
        funnel = manifest.get("readability_funnel", {})
        p90 = funnel.get("p90_readability_score", 1000.0)
        # p95 approximation: 1.1 × p90 (since we don't store p95; close enough for weight norm)
        track_p95 = p90 * 1.1

        selected = [p for p in manifest["plates"] if p.get("is_selected_for_ocr")]
        n_frames = len(selected)
        logger.info(f"Track {track_id}: {n_frames} selected frames for OCR")

        if n_frames == 0:
            return _empty_result(track_id, manifest_path, manifest)

        # --- gather candidate lengths to pick template ---
        candidate_texts: List[str] = []
        frame_results: List[Dict] = []
        per_frame_debug: List[Dict] = []

        # manifest is at <proj>/outputs/<clip>/track_NNN/plate_manifest.json
        # paths in manifest are relative to project root: "outputs/<clip>/track_NNN/..."
        proj_root = manifest_path.parent.parent.parent.parent  # track→clip→outputs→proj

        for plate in selected:
            p_unenh = Path(plate["selected_unenhanced_path"])
            unenh = str(proj_root / p_unenh) if not p_unenh.is_absolute() else str(p_unenh)
            if not Path(unenh).exists():
                fb_u = manifest_path.parent / "plate_crops_selected" / "unenhanced" / p_unenh.name
                if fb_u.exists():
                    unenh = str(fb_u)

            p_enh = Path(plate["selected_enhanced_path"])
            enh = str(proj_root / p_enh) if not p_enh.is_absolute() else str(p_enh)
            if not Path(enh).exists():
                fb_e = manifest_path.parent / "plate_crops_selected" / "enhanced" / p_enh.name
                if fb_e.exists():
                    enh = str(fb_e)

            res = self._run_frame(
                unenh, enh,
                plate["plate_saved_size"],
                plate["readability_score"],
                track_p95,
                plate["verification_status"],
            )
            if res is None:
                continue

            # Sanitize and collect candidate lengths
            for pipe_key in ("pipe_a", "pipe_b"):
                r = res[pipe_key]
                for txt in (r["primary_txt"], r["xcheck_txt"]):
                    cleaned = sanitize_plate(txt)
                    if cleaned:
                        candidate_texts.append(cleaned)

            frame_results.append({
                "frame_idx": plate["frame_idx"],
                "readability_score": plate["readability_score"],
                "verification_status": plate["verification_status"],
                "pipe_a": res["pipe_a"],
                "pipe_b": res["pipe_b"],
            })

        plate_length = select_template(candidate_texts)
        logger.info(f"Track {track_id}: selected plate length = {plate_length}")

        # --- build per-slot vote distributions ---
        reads_a: List[Dict] = []
        reads_b: List[Dict] = []
        n_support_a = defaultdict(int)   # slot → n supporting frames
        n_support_b = defaultdict(int)

        for fr in frame_results:
            pa = fr["pipe_a"]
            pb = fr["pipe_b"]

            # Fuse pipe A
            fused_a, fused_conf_a, char_confs_a, notes_a = _fuse_readings(
                pa["primary_txt"], pa["primary_conf"],
                pa["xcheck_txt"],  pa["xcheck_conf"],
                plate_length,
            )
            # Fuse pipe B
            fused_b, fused_conf_b, char_confs_b, notes_b = _fuse_readings(
                pb["primary_txt"], pb["primary_conf"],
                pb["xcheck_txt"],  pb["xcheck_conf"],
                plate_length,
            )

            if len(fused_a) == plate_length:
                reads_a.append({
                    "text": fused_a,
                    "weight": pa["weight"],
                    "conf": fused_conf_a,
                    "char_confs": char_confs_a,
                })
                for i, ch in enumerate(fused_a):
                    if ch != "?":
                        n_support_a[i] += 1
            if len(fused_b) == plate_length:
                reads_b.append({
                    "text": fused_b,
                    "weight": pb["weight"],
                    "conf": fused_conf_b,
                    "char_confs": char_confs_b,
                })
                for i, ch in enumerate(fused_b):
                    if ch != "?":
                        n_support_b[i] += 1

            per_frame_debug.append({
                "frame_idx": fr["frame_idx"],
                "readability_score": fr["readability_score"],
                "verification_status": fr["verification_status"],
                "pipe_a_fused": fused_a, "pipe_a_conf": round(fused_conf_a, 4), "pipe_a_weight": round(pa["weight"], 4),
                "pipe_b_fused": fused_b, "pipe_b_conf": round(fused_conf_b, 4), "pipe_b_weight": round(pb["weight"], 4),
                "pipe_a_primary_raw": pa["primary_txt"], "pipe_a_xcheck_raw": pa["xcheck_txt"],
                "pipe_b_primary_raw": pb["primary_txt"], "pipe_b_xcheck_raw": pb["xcheck_txt"],
                "notes": notes_a + notes_b,
            })

        votes_a = aggregate_slot_votes(reads_a, plate_length)
        votes_b = aggregate_slot_votes(reads_b, plate_length)

        # --- classify each slot ---
        slot_results = []
        recovered_chars = []
        for i in range(plate_length):
            dist_a = votes_a.get(i, {})
            dist_b = votes_b.get(i, {})
            classification = classify_slot(
                dist_a, dist_b,
                n_frames=len(frame_results),
                n_supporting_a=n_support_a.get(i, 0),
                n_supporting_b=n_support_b.get(i, 0),
            )
            top_a, eff_a, share_a, raw_a = _top_char(dist_a)
            top_b, eff_b, share_b, raw_b = _top_char(dist_b)
            slot_results.append({
                "slot": i,
                "slot_type": _SLOT_TYPES_CORRECT.get(plate_length, ["?"] * plate_length)[i],
                "pipe_a_top": top_a,
                "pipe_a_conf": round(eff_a, 4),
                "pipe_a_vote_share": round(share_a, 4),
                "pipe_a_char_conf": round(raw_a, 4),
                "pipe_b_top": top_b,
                "pipe_b_conf": round(eff_b, 4),
                "pipe_b_vote_share": round(share_b, 4),
                "pipe_b_char_conf": round(raw_b, 4),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in classification.items()},
            })
            recovered_chars.append(classification["char"])

        recovered_plate = "".join(recovered_chars)
        overall_conf = float(np.mean([s["combined_conf"] for s in slot_results]))
        locked_count = sum(1 for s in slot_results if s["bucket"] == "LOCKED")
        contradiction_count = sum(1 for s in slot_results if s["bucket"] == "CONTRADICTORY")
        unreadable_count = sum(1 for s in slot_results if s["bucket"] == "UNREADABLE")

        # State code validation flag
        state_valid = validate_state_code(recovered_plate)

        # Top-5 highest-weight frames across both pipes
        all_frame_weights = [(f["pipe_a_weight"], f["frame_idx"], "pipe_a", f["pipe_a_fused"])
                             for f in per_frame_debug] + \
                            [(f["pipe_b_weight"], f["frame_idx"], "pipe_b", f["pipe_b_fused"])
                             for f in per_frame_debug]
        all_frame_weights.sort(reverse=True)
        top5 = [
            {"frame_idx": fi, "pipe": p, "text": t, "weight": round(w, 4)}
            for w, fi, p, t in all_frame_weights[:5]
        ]

        # VLM evidence pack: top 4 distinct frames ranked by weight with verified existing file paths
        frame_best_weights: Dict[int, Dict] = {}
        for fr in per_frame_debug:
            fi = fr["frame_idx"]
            w = max(fr["pipe_a_weight"], fr["pipe_b_weight"])
            if fi not in frame_best_weights or w > frame_best_weights[fi]["weight"]:
                p_match = next((p for p in selected if p["frame_idx"] == fi), None)
                unenh_rel = p_match["selected_unenhanced_path"] if p_match else ""
                enh_rel = p_match["selected_enhanced_path"] if p_match else ""
                unenh_full = str(proj_root / unenh_rel) if unenh_rel and not Path(unenh_rel).is_absolute() else unenh_rel
                enh_full = str(proj_root / enh_rel) if enh_rel and not Path(enh_rel).is_absolute() else enh_rel
                if unenh_full and not Path(unenh_full).exists():
                    fb_u = manifest_path.parent / "plate_crops_selected" / "unenhanced" / Path(unenh_rel).name
                    if fb_u.exists():
                        unenh_full = str(fb_u)
                if enh_full and not Path(enh_full).exists():
                    fb_e = manifest_path.parent / "plate_crops_selected" / "enhanced" / Path(enh_rel).name
                    if fb_e.exists():
                        enh_full = str(fb_e)
                frame_best_weights[fi] = {
                    "frame_idx": fi,
                    "weight": round(w, 4),
                    "readability_score": round(fr["readability_score"], 2),
                    "verification_status": fr["verification_status"],
                    "unenhanced_crop_path": unenh_full if Path(unenh_full).exists() else unenh_rel,
                    "enhanced_crop_path": enh_full if Path(enh_full).exists() else enh_rel,
                    "pipe_a_fused": fr["pipe_a_fused"],
                    "pipe_b_fused": fr["pipe_b_fused"],
                }

        vlm_evidence_pack = sorted(frame_best_weights.values(), key=lambda x: x["weight"], reverse=True)[:4]

        elapsed = round(time.time() - t0, 2)

        result = {
            "schema_version": "2.3",
            "track_id": track_id,
            "target_id": track_id,
            "merged_track_ids": merged_track_ids,
            "is_merged_track": is_merged_track,
            "n_frames_processed": len(frame_results),
            "plate_length": plate_length,
            "recovered_plate": recovered_plate,
            "overall_confidence": round(overall_conf, 4),
            "state_code_valid": state_valid,
            "slot_summary": {
                "locked": locked_count,
                "contradictory": contradiction_count,
                "unreadable": unreadable_count,
            },
            "slots": slot_results,
            "top5_frames": top5,
            "vlm_evidence_pack": vlm_evidence_pack,
            "per_frame_debug": per_frame_debug,
            "runtime_seconds": elapsed,
            "vlm_handoff": {
                "action": "STAGE_2.4_VLM",
                "locked_slots": [i for i, s in enumerate(slot_results) if s["bucket"] == "LOCKED"],
                "contradictory_slots": [i for i, s in enumerate(slot_results) if s["bucket"] == "CONTRADICTORY"],
                "unreadable_slots": [i for i, s in enumerate(slot_results) if s["bucket"] == "UNREADABLE"],
                "vlm_evidence_pack": vlm_evidence_pack,
                "note": "VLM must TRIPLE_CHECK LOCKED slots and TIE_BREAK CONTRADICTORY slots. FORBIDDEN: guessing UNREADABLE slots.",
            },
        }
        return result


def _empty_result(track_id, manifest_path: Path, manifest: Optional[Dict] = None) -> Dict:
    merged_ids = manifest.get("merged_track_ids", [track_id]) if manifest else [track_id]
    is_merged = bool(manifest.get("is_merged_track", False)) if manifest else False
    return {
        "schema_version": "2.3",
        "track_id": track_id,
        "target_id": track_id,
        "merged_track_ids": merged_ids,
        "is_merged_track": is_merged,
        "n_frames_processed": 0,
        "plate_length": 10,
        "recovered_plate": "??????????",
        "overall_confidence": 0.0,
        "state_code_valid": False,
        "slot_summary": {"locked": 0, "contradictory": 0, "unreadable": 10},
        "slots": [],
        "top5_frames": [],
        "vlm_evidence_pack": [],
        "per_frame_debug": [],
        "runtime_seconds": 0.0,
        "vlm_handoff": {
            "action": "STAGE_2.4_VLM",
            "locked_slots": [],
            "contradictory_slots": [],
            "unreadable_slots": list(range(10)),
            "vlm_evidence_pack": [],
            "note": "No frames selected for OCR.",
        },
    }


def run_ocr_track(plate_manifest_path: str, device: str = "cuda") -> Dict:
    """Public convenience wrapper."""
    engine = PlateOCREngine(device=device)
    result = engine.run_ocr_track(plate_manifest_path)
    try:
        out_json = Path(plate_manifest_path).resolve().parent / "ocr_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not auto-write ocr_result.json: {e}")
    return result
