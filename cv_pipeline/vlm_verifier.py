"""Stage 2.4 — Blind Multi-Tier VLM Triple-Check & Slot-by-Slot Tie-Breaker.

3-Tier OpenAI-compatible fallback chain:
  Tier 1 — NVIDIA Build NIM (nvidia/nemotron-3.5-super-vl-120b-a12b) with multi-key failover
  Tier 2 — Groq (qwen/qwen3.8-27b)
  Tier 3 — Ollama local (qwen3-vl:4b-instruct / qwen3-vl:2b-instruct)

Visual evidence: 2x2 labeled stitched grids (unenhanced & enhanced) from Stage 2.3 evidence pack.
Protocol: Blind prompt (no OCR text leakage) -> strict JSON extraction -> slot-by-slot 3-bucket reconciliation:
  - LOCKED: Triple-check confirmation (override only on >=0.90 visual certainty).
  - CONTRADICTORY: Tie-breaker between Pipe A and Pipe B (or independent choice on >=0.70 confidence).
  - UNREADABLE: Anti-hallucination guard (FORBIDDEN_KEEP_UNKNOWN unless clear visual resolution >=0.85).

Emits vlm_result.json in outputs/<clip>/track_<ID>/.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from cv_pipeline.api_keys import get_nvidia_api_keys
from cv_pipeline.ocr_engine import (
    _SLOT_TYPES_CORRECT,
    _VALID_STATE_CODES,
    _apply_typed_swap,
    validate_state_code,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------
DEFAULT_CELL_SIZE = (320, 120)  # (width, height) in pixels
DEFAULT_HEADER_HEIGHT = 24

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_TIMEOUT = 15.0

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
NVIDIA_TIMEOUT = 25.0

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODELS = ["qwen3-vl:4b-instruct", "qwen3-vl:2b-instruct"]
OLLAMA_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Visual Evidence Grid Builder
# ---------------------------------------------------------------------------
def _create_labeled_cell(
    img: Optional[np.ndarray],
    label: str,
    target_size: Tuple[int, int] = DEFAULT_CELL_SIZE,
    header_height: int = DEFAULT_HEADER_HEIGHT,
    header_color: Tuple[int, int, int] = (35, 35, 35),
) -> np.ndarray:
    """Resize image and prepend a clean banner header."""
    cell_w, cell_h = target_size
    img_h = max(10, cell_h - header_height)

    # Header canvas
    header = np.full((header_height, cell_w, 3), header_color, dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (8, int(header_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Content canvas
    if img is None or img.size == 0:
        content = np.zeros((img_h, cell_w, 3), dtype=np.uint8)
        cv2.putText(
            content,
            "N/A",
            (cell_w // 2 - 15, img_h // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            1,
            cv2.LINE_AA,
        )
    else:
        content = cv2.resize(img, (cell_w, img_h), interpolation=cv2.INTER_AREA)
        if len(content.shape) == 2:
            content = cv2.cvtColor(content, cv2.COLOR_GRAY2BGR)

    return np.vstack([header, content])


def build_evidence_grids(
    track_dir: str | Path,
    evidence_pack: List[Dict[str, Any]],
    target_cell_size: Tuple[int, int] = DEFAULT_CELL_SIZE,
    header_height: int = DEFAULT_HEADER_HEIGHT,
) -> Tuple[Path, Path]:
    """
    Build 2x2 labeled stitched grids for unenhanced and enhanced crops.
    Saves:
      - <track_dir>/vlm_evidence_unenhanced_grid.png
      - <track_dir>/vlm_evidence_enhanced_grid.png
    """
    t_dir = Path(track_dir).resolve()
    t_dir.mkdir(parents=True, exist_ok=True)
    proj_root = t_dir.parent.parent.parent  # .../outputs/<clip>/<track> -> proj_root

    # Select top 4 frames
    pack = list(evidence_pack)[:4]

    unenh_cells: List[np.ndarray] = []
    enh_cells: List[np.ndarray] = []

    for i in range(4):
        if i < len(pack):
            item = pack[i]
            frame_idx = item.get("frame_idx", i)
            score = item.get("readability_score", 0.0)
            weight = item.get("weight", 0.0)
            label = f"Frame {frame_idx:03d} | Score: {score:.0f} | w={weight:.2f}"

            # Resolve paths
            unenh_p = Path(item.get("unenhanced_crop_path", ""))
            enh_p = Path(item.get("enhanced_crop_path", ""))

            if not unenh_p.is_file() and not unenh_p.is_absolute():
                unenh_p = proj_root / unenh_p
            if not unenh_p.is_file():
                fallback_u = t_dir / "plate_crops_selected" / "unenhanced" / unenh_p.name
                if fallback_u.is_file():
                    unenh_p = fallback_u

            if not enh_p.is_file() and not enh_p.is_absolute():
                enh_p = proj_root / enh_p
            if not enh_p.is_file():
                fallback_e = t_dir / "plate_crops_selected" / "enhanced" / enh_p.name
                if fallback_e.is_file():
                    enh_p = fallback_e

            img_unenh = cv2.imread(str(unenh_p)) if unenh_p.is_file() else None
            img_enh = cv2.imread(str(enh_p)) if enh_p.is_file() else None

            unenh_cells.append(
                _create_labeled_cell(
                    img_unenh,
                    label,
                    target_cell_size,
                    header_height,
                    header_color=(45, 30, 30),
                )
            )
            enh_cells.append(
                _create_labeled_cell(
                    img_enh,
                    label,
                    target_cell_size,
                    header_height,
                    header_color=(20, 45, 35),
                )
            )
        else:
            unenh_cells.append(
                _create_labeled_cell(None, "Slot Empty", target_cell_size, header_height)
            )
            enh_cells.append(
                _create_labeled_cell(None, "Slot Empty", target_cell_size, header_height)
            )

    # 2x2 grid assembly
    unenh_top = np.hstack([unenh_cells[0], unenh_cells[1]])
    unenh_bot = np.hstack([unenh_cells[2], unenh_cells[3]])
    unenh_grid = np.vstack([unenh_top, unenh_bot])

    enh_top = np.hstack([enh_cells[0], enh_cells[1]])
    enh_bot = np.hstack([enh_cells[2], enh_cells[3]])
    enh_grid = np.vstack([enh_top, enh_bot])

    unenh_path = t_dir / "vlm_evidence_unenhanced_grid.png"
    enh_path = t_dir / "vlm_evidence_enhanced_grid.png"

    cv2.imwrite(str(unenh_path), unenh_grid)
    cv2.imwrite(str(enh_path), enh_grid)

    logger.info(f"Built evidence grids: {unenh_path.name}, {enh_path.name}")
    return unenh_path, enh_path


def encode_image_b64(image_path: str | Path) -> str:
    """Encode an image to a base64 data URL."""
    p = Path(image_path)
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Blind Prompt Formulation & Parsing
# ---------------------------------------------------------------------------
def _build_blind_vlm_messages(
    unenh_b64: str,
    enh_b64: str,
    plate_len: int = 10,
) -> List[Dict[str, Any]]:
    """Construct blind VLM prompt (zero OCR text leakage)."""
    system_prompt = (
        "You are an expert forensic vehicle license plate reading AI specializing in Indian vehicle license plates.\n"
        "You are provided with two 2x2 composite image grids representing 4 distinct camera captures of the exact same vehicle plate:\n"
        "1. Image 1: Unenhanced original camera crops across the 4 frames.\n"
        "2. Image 2: Enhanced contrast-normalized crops across the same 4 frames.\n\n"
        "Examine both grids thoroughly across all 4 frames to cross-verify every single character and cancel out glare, motion blur, and sensor noise.\n\n"
        "CRITICAL RULES:\n"
        f"- Target plate format is typically standard Indian registration ({plate_len} characters: 2-letter state code, 2-digit district code, 1-2 letters series, 4 digits number).\n"
        "- Layout: determine if the plate is 'single_line' (horizontal) or 'two_line' (stacked).\n"
        "  * If two-line stacked: read the TOP line first (state code and district code), followed by the BOTTOM line (series and 4-digit number), left to right.\n"
        "- Read characters position-by-position (slot 0 to N-1).\n"
        "- For each slot:\n"
        "  * slot_idx: integer index 0..N-1\n"
        "  * char: uppercase alphanumeric char ('A'-'Z', '0'-'9'), or '?' if the character is blurry, occluded, cut off, or illegible.\n"
        "  * confidence: float between 0.0 and 1.0 representing visual certainty.\n"
        "  * legible: boolean (true if clearly discernable, false if degraded/blurry/unknown).\n"
        "  * note: brief visual justification (e.g., 'clear curved stroke of 5 on frame 253', 'vertical stroke ambiguous between 1 and I', etc.).\n"
        "- ANTI-HALLUCINATION GUARD: If a character is unclear or cannot be positively identified from visual evidence alone, set char='?' and legible=false. NEVER guess or invent characters.\n\n"
        "You MUST respond ONLY with a single valid JSON object following this exact schema:\n"
        "{\n"
        '  "plate_text": "...",\n'
        '  "plate_layout": "single_line" | "two_line",\n'
        '  "slot_reads": [\n'
        '    {"slot_idx": 0, "char": "U", "confidence": 0.95, "legible": true, "note": "..."}\n'
        "  ],\n"
        '  "image_quality_note": "..."\n'
        "}"
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Read the Indian vehicle license plate across both 2x2 grids (4 distinct camera frames). "
                f"Identify layout, read characters slot-by-slot, and return the strict JSON schema. "
                f"Preserve '?' for any illegible or uncertain character."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": unenh_b64},
        },
        {
            "type": "image_url",
            "image_url": {"url": enh_b64},
        },
    ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _extract_json_payload(content: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON dictionary from raw model text."""
    if not content:
        return None
    cleaned = content.strip()

    # Direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Markdown block extraction: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Outermost curly braces
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# 3-Tier Multi-Provider Fallback Caller
# ---------------------------------------------------------------------------
class VLMClient:
    """Manages multi-tier VLM calls with failover tracing."""

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()

    def _call_groq(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, float]:
        """Tier 2: Groq qwen/qwen3.8-27b."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        client = OpenAI(base_url=GROQ_BASE_URL, api_key=self.groq_api_key, max_retries=0)
        start_t = time.time()

        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=600,
                extra_body={"reasoning_effort": "low", "reasoning_format": "hidden"},
                timeout=GROQ_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Groq with reasoning_format failed ({e}); falling back to reasoning_effort=none")
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=600,
                extra_body={"reasoning_effort": "none"},
                timeout=GROQ_TIMEOUT,
            )

        latency = time.time() - start_t
        raw_text = resp.choices[0].message.content or ""
        parsed = _extract_json_payload(raw_text)
        if not parsed:
            raise ValueError(f"Groq did not return valid JSON: {raw_text[:200]}")

        return parsed, GROQ_MODEL, latency

    def _call_nvidia(
        self,
        messages: List[Dict[str, Any]],
        api_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, float]:
        """Tier 1: NVIDIA Build NIM nvidia/nemotron-3.5-super-vl-120b-a12b."""
        effective_key = api_key or self.nvidia_api_key
        if not effective_key:
            keys = get_nvidia_api_keys()
            effective_key = keys[0][1] if keys else None
        if not effective_key:
            raise ValueError("NVIDIA_API_KEY is not set.")

        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=effective_key, max_retries=0)
        start_t = time.time()

        resp = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=600,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=NVIDIA_TIMEOUT,
        )

        latency = time.time() - start_t
        raw_text = resp.choices[0].message.content or ""
        parsed = _extract_json_payload(raw_text)
        if not parsed:
            raise ValueError(f"NVIDIA NIM did not return valid JSON: {raw_text[:200]}")

        return parsed, NVIDIA_MODEL, latency

    def _call_ollama(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, float]:
        """Tier 3: Local Ollama instance."""
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        last_err: Optional[Exception] = None

        for model_name in OLLAMA_MODELS:
            start_t = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    timeout=OLLAMA_TIMEOUT,
                )
                latency = time.time() - start_t
                raw_text = resp.choices[0].message.content or ""
                parsed = _extract_json_payload(raw_text)
                if parsed:
                    return parsed, model_name, latency
                else:
                    raise ValueError(f"Ollama {model_name} did not return valid JSON")
            except Exception as e:
                last_err = e
                continue

        raise ValueError(f"Ollama local models failed: {last_err}")

    def execute_blind_query(
        self,
        unenh_b64: str,
        enh_b64: str,
        plate_len: int = 10,
        provider: str = "auto",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], float, List[Dict[str, Any]]]:
        """
        Execute query against specified provider or auto-fallback chain.
        Returns: (parsed_json, provider_used, model_used, latency, fallback_trace)
        """
        messages = _build_blind_vlm_messages(unenh_b64, enh_b64, plate_len=plate_len)
        fallback_trace: List[Dict[str, Any]] = []

        # Determine target tiers
        provider_norm = provider.lower().strip()
        if provider_norm == "auto":
            tiers = [
                (1, "nvidia", self._call_nvidia),
                (2, "groq", self._call_groq),
                (3, "ollama", self._call_ollama),
            ]
        elif provider_norm == "nvidia":
            tiers = [(1, "nvidia", self._call_nvidia)]
        elif provider_norm == "groq":
            tiers = [(2, "groq", self._call_groq)]
        elif provider_norm == "ollama":
            tiers = [(3, "ollama", self._call_ollama)]
        else:
            raise ValueError(f"Unknown VLM provider: {provider}")

        for tier_num, prov_name, call_fn in tiers:
            # Check prerequisites
            if prov_name == "groq" and not self.groq_api_key:
                fallback_trace.append({
                    "tier": tier_num,
                    "provider": prov_name,
                    "model": GROQ_MODEL,
                    "status": "SKIPPED",
                    "error": "GROQ_API_KEY missing",
                    "latency_sec": 0.0,
                })
                continue
            if prov_name == "nvidia":
                nv_keys = get_nvidia_api_keys()
                if self.nvidia_api_key and not any(k[1] == self.nvidia_api_key for k in nv_keys):
                    nv_keys.insert(0, ("NVIDIA_API_KEY", self.nvidia_api_key))

                if not nv_keys:
                    fallback_trace.append({
                        "tier": tier_num,
                        "provider": "nvidia",
                        "model": NVIDIA_MODEL,
                        "status": "SKIPPED",
                        "error": "NVIDIA_API_KEY missing",
                        "latency_sec": 0.0,
                    })
                    continue

                for key_label, secret_val in nv_keys:
                    prov_label = f"nvidia[{key_label}]"
                    try:
                        parsed_json, model_used, latency = self._call_nvidia(messages, api_key=secret_val)
                        fallback_trace.append({
                            "tier": tier_num,
                            "provider": prov_label,
                            "model": model_used,
                            "status": "SUCCESS",
                            "error": None,
                            "latency_sec": round(latency, 3),
                        })
                        logger.info(f"VLM call succeeded with {prov_label} ({model_used}) in {latency:.2f}s")
                        return parsed_json, prov_label, model_used, latency, fallback_trace
                    except Exception as e:
                        err_msg = str(e)
                        fallback_trace.append({
                            "tier": tier_num,
                            "provider": prov_label,
                            "model": getattr(e, "model", NVIDIA_MODEL),
                            "status": "FAILED",
                            "error": err_msg,
                            "latency_sec": 0.0,
                        })
                        logger.warning(f"VLM Tier {tier_num} ({prov_label}) failed: {err_msg}")
                continue

            try:
                parsed_json, model_used, latency = call_fn(messages)
                fallback_trace.append({
                    "tier": tier_num,
                    "provider": prov_name,
                    "model": model_used,
                    "status": "SUCCESS",
                    "error": None,
                    "latency_sec": round(latency, 3),
                })
                logger.info(f"VLM call succeeded with {prov_name} ({model_used}) in {latency:.2f}s")
                return parsed_json, prov_name, model_used, latency, fallback_trace
            except Exception as e:
                err_msg = str(e)
                fallback_trace.append({
                    "tier": tier_num,
                    "provider": prov_name,
                    "model": getattr(e, "model", prov_name),
                    "status": "FAILED",
                    "error": err_msg,
                    "latency_sec": 0.0,
                })
                logger.warning(f"VLM Tier {tier_num} ({prov_name}) failed: {err_msg}")

        logger.error("All requested VLM providers failed; initiating graceful OCR fallback.")
        return None, None, None, 0.0, fallback_trace


# ---------------------------------------------------------------------------
# Beam Search Ghost Predictions (Advisory / UI Hover Re-Ranking)
# ---------------------------------------------------------------------------
def compute_ghost_predictions(
    reconciled_slots: List[Dict[str, Any]],
    plate_len: int,
    beam_width: int = 8,
    max_predictions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Lightweight beam search generating top-3 distinct full-plate hypotheses for UI hover paths.
    Scores hypotheses by mean soft_score across slots plus a +0.08 Indian state code validity bonus.
    Never alters ground-truth final_plate.
    """
    candidate_pool: List[List[Tuple[str, float, bool]]] = []
    for s in reconciled_slots:
        tone = s.get("ui_tile_tone", "unknown")
        final_c = s.get("final_char", "?")
        f_score = float(s.get("final_char_soft_score", 0.0))
        g_stack = s.get("ghost_stack", [])

        if tone == "green":
            # Anchored strictly to final_char (no branching)
            candidate_pool.append([(final_c, f_score, False)])
        elif tone == "black":
            # final_char + up to top 2 conflict_alt candidates
            cands = [(final_c, f_score, False)]
            for g in g_stack[:2]:
                cands.append((g["char"], float(g["soft_score"]), True))
            candidate_pool.append(cands)
        else:  # "unknown" ('?')
            # top 3 ghost_fill candidates (if empty, keep '?')
            if not g_stack:
                candidate_pool.append([("?", 0.0, True)])
            else:
                candidate_pool.append([
                    (g["char"], float(g["soft_score"]), True) for g in g_stack[:3]
                ])

    beam: List[Tuple[str, float, List[int]]] = [("", 0.0, [])]  # (prefix, cumulative_score, ghost_indices)

    effective_len = min(plate_len, len(candidate_pool))
    for i in range(effective_len):
        slot_cands = candidate_pool[i]
        next_beam: List[Tuple[str, float, List[int]]] = []
        for prefix, cum_score, ghost_indices in beam:
            for char, score, is_ghost in slot_cands:
                new_ghosts = ghost_indices + ([i] if is_ghost else [])
                next_beam.append((prefix + char, cum_score + score, new_ghosts))
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]

    scored_hypotheses = []
    for plate_str, cum_score, ghost_indices in beam:
        base_score = cum_score / effective_len if effective_len > 0 else 0.0
        state_valid = validate_state_code(plate_str)
        bonus = 0.08 if state_valid else 0.0
        total_score = round(base_score + bonus, 3)
        scored_hypotheses.append({
            "plate": plate_str,
            "candidate_plate": plate_str,
            "soft_score": total_score,
            "combined_confidence": total_score,
            "state_code_valid": state_valid,
            "ghost_slot_indices": ghost_indices,
        })

    # Sort descending by soft_score, prioritizing valid state codes on ties
    scored_hypotheses.sort(key=lambda x: (x["soft_score"], x["state_code_valid"]), reverse=True)

    distinct: List[Dict[str, Any]] = []
    seen = set()
    for h in scored_hypotheses:
        if h["plate"] not in seen:
            seen.add(h["plate"])
            h["rank"] = len(distinct) + 1
            distinct.append(h)
            if len(distinct) == max_predictions:
                break

    return distinct


def _char_matches_slot_type(ch: str, slot_type: Optional[str]) -> bool:
    """Check if character matches the expected Indian format type (LETTER / DIGIT)."""
    if not ch or not ch.isalnum():
        return False
    st = str(slot_type).upper() if slot_type else ""
    if st in ("L", "LETTER"):
        return ch.isalpha()
    if st in ("D", "DIGIT"):
        return ch.isdigit()
    return False


# ---------------------------------------------------------------------------
# Slot-by-Slot 3-Bucket Reconciliation
# ---------------------------------------------------------------------------
def reconcile_vlm_with_ocr(
    ocr_result: Dict[str, Any],
    vlm_output: Optional[Dict[str, Any]],
    provider_used: Optional[str] = None,
    model_used: Optional[str] = None,
    runtime_seconds: float = 0.0,
    fallback_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Arbitrate between Stage 2.3 OCR and Stage 2.4 Blind VLM results.
    Enforces:
      - Pre-reconciliation VLM disambiguation (0.92x discount on blur-swapped chars).
      - LOCKED: Triple-check confirmation (LOCKED_VLM_DISPUTED on mismatch, -0.05 penalty).
      - CONTRADICTORY: Tie-break strictly restricted to OCR candidates (VLM_ABSTAINED if other).
      - UNREADABLE: Hard-block anti-hallucination (FORBIDDEN_KEEP_UNKNOWN, never promoted).
      - Per-slot ui_tile_tone ('green', 'black', 'unknown') and deduplicated ghost_stack.
      - Top-3 ghost_predictions via lightweight beam search.
    """
    recovered_ocr = ocr_result.get("recovered_plate", "")
    ocr_slots = ocr_result.get("slots", [])
    plate_len = ocr_result.get("plate_length", len(recovered_ocr) or 10)

    # Graceful fallback if VLM failed
    if not vlm_output:
        reconciled_slots = []
        for i, s in enumerate(ocr_slots):
            c = s.get("char", "?")
            bucket = s.get("bucket", "UNREADABLE")
            tone = (
                "green"
                if bucket == "LOCKED" and c != "?"
                else ("black" if bucket == "CONTRADICTORY" and c != "?" else "unknown")
            )
            reconciled_slots.append({
                "slot_idx": s.get("slot", i),
                "slot_index": s.get("slot", i),
                "slot_type": s.get("slot_type", "UNKNOWN"),
                "expected_type": s.get("slot_type", "UNKNOWN"),
                "ocr_char": c,
                "ocr_bucket": bucket,
                "pipe_a_candidate": s.get("pipe_a_top", "?"),
                "pipe_b_candidate": s.get("pipe_b_top", "?"),
                "pipe_a_char": s.get("pipe_a_top", "?"),
                "pipe_b_char": s.get("pipe_b_top", "?"),
                "vlm_char": None,
                "vlm_blind_raw_char": None,
                "vlm_disambiguated_char": None,
                "vlm_conf": 0.0,
                "vlm_legible": False,
                "final_char": c,
                "final_char_soft_score": float(s.get("combined_conf", 0.0)),
                "final_confidence": float(s.get("combined_conf", 0.0)),
                "action": "FALLBACK_STAGE_2_3_OCR",
                "ui_tile_tone": tone,
                "ghost_stack": [],
                "reasoning_notes": [],
                "note": "VLM unavailable; preserved Stage 2.3 arbitration.",
            })

        ghost_predictions = compute_ghost_predictions(reconciled_slots, plate_len)

        return {
            "schema_version": "2.4.0",
            "track_id": ocr_result.get("track_id", "unknown"),
            "target_id": ocr_result.get("target_id", ocr_result.get("track_id", "unknown")),
            "merged_track_ids": ocr_result.get("merged_track_ids", [ocr_result.get("track_id")]),
            "is_merged_track": bool(ocr_result.get("is_merged_track", False)),
            "vlm_status": "FALLBACK_TO_OCR",
            "provider_used": "none",
            "model_used": "none",
            "runtime_seconds": round(runtime_seconds, 3),
            "fallback_trace": fallback_trace or [],
            "stage_2_3_plate": recovered_ocr,
            "vlm_raw_plate": None,
            "final_plate": recovered_ocr,
            "reconciled_plate": recovered_ocr,
            "overall_confidence": float(ocr_result.get("overall_confidence", 0.0)),
            "state_code_valid": validate_state_code(recovered_ocr),
            "plate_layout": "unknown",
            "reconciled_slots": reconciled_slots,
            "ghost_predictions": ghost_predictions,
            "actions_summary": {"FALLBACK_STAGE_2_3_OCR": len(reconciled_slots)},
        }

    # Parse VLM slot readings
    vlm_raw_text = (vlm_output.get("plate_text") or "").strip().replace(" ", "").upper()
    vlm_layout = vlm_output.get("plate_layout", "single_line")
    vlm_slot_list = vlm_output.get("slot_reads", [])

    vlm_slot_map: Dict[int, Dict[str, Any]] = {}
    for item in vlm_slot_list:
        if isinstance(item, dict) and "slot_idx" in item:
            idx = int(item["slot_idx"])
            vlm_slot_map[idx] = item

    reconciled_slots = []
    actions_summary: Dict[str, int] = {}

    for i in range(plate_len):
        # OCR slot data
        if i < len(ocr_slots):
            s = ocr_slots[i]
            ocr_char = str(s.get("char", "?")).upper()
            ocr_bucket = s.get("bucket", "UNREADABLE")
            pipe_a_cand = str(s.get("pipe_a_top") or s.get("pipe_a_candidate") or "?").upper()
            pipe_b_cand = str(s.get("pipe_b_top") or s.get("pipe_b_candidate") or "?").upper()
            pa_conf_raw = s.get("pipe_a_conf") if s.get("pipe_a_conf") is not None else s.get("pipe_a_confidence", 0.0)
            pipe_a_conf = float(pa_conf_raw or 0.0)
            pb_conf_raw = s.get("pipe_b_conf") if s.get("pipe_b_conf") is not None else s.get("pipe_b_confidence", 0.0)
            pipe_b_conf = float(pb_conf_raw or 0.0)
            slot_type = s.get("slot_type")
        else:
            s = {}
            ocr_char = "?"
            ocr_bucket = "UNREADABLE"
            pipe_a_cand = "?"
            pipe_b_cand = "?"
            pipe_a_conf = 0.0
            pipe_b_conf = 0.0
            slot_type = None

        if not slot_type or slot_type == "?":
            slot_types_list = _SLOT_TYPES_CORRECT.get(plate_len, _SLOT_TYPES_CORRECT[10])
            slot_type = slot_types_list[i] if i < len(slot_types_list) else "D"

        # VLM slot data
        v_entry = vlm_slot_map.get(i)
        if v_entry:
            v_char = str(v_entry.get("char", "?")).strip().upper() or "?"
            v_conf = float(v_entry.get("confidence", 0.0))
            v_legible = bool(v_entry.get("legible", False))
            v_note = str(v_entry.get("note", ""))
        elif i < len(vlm_raw_text):
            v_char = vlm_raw_text[i]
            v_conf = 0.70 if v_char != "?" else 0.0
            v_legible = (v_char != "?")
            v_note = "Derived from VLM plate_text"
        else:
            v_char = "?"
            v_conf = 0.0
            v_legible = False
            v_note = "Missing in VLM response"

        # Sanitize single-char
        if len(v_char) > 1:
            v_char = v_char[0]

        vlm_blind_raw_char = v_char
        reasoning_notes: List[str] = []

        # 1. Pre-Reconciliation VLM Disambiguation
        if v_char.isalnum():
            swapped_char, discounted_conf, swap_note = _apply_typed_swap(v_char, slot_type, v_conf)
            if swap_note:
                vlm_disambiguated_char = swapped_char
                v_char = swapped_char
                v_conf = discounted_conf
                reasoning_notes.append(f"vlm_disambiguated: '{vlm_blind_raw_char}'->'{swapped_char}'")
            else:
                vlm_disambiguated_char = v_char
        else:
            vlm_disambiguated_char = v_char

        final_char = ocr_char
        action = "UNRESOLVED"
        ui_tile_tone = "unknown"

        # --- Rule 1: LOCKED Slots (Triple-Check Protection) ---
        if ocr_bucket == "LOCKED":
            if v_char == ocr_char:
                action = "TRIPLE_CHECK_CONFIRMED"
                final_char = ocr_char
                ui_tile_tone = "green"
            else:
                # Disagreed with 2-OCR consensus: keep OCR char, tag disputed with -0.05 penalty
                action = "LOCKED_VLM_DISPUTED"
                final_char = ocr_char
                ui_tile_tone = "black"

        # --- Rule 2: CONTRADICTORY Slots (Tie-Break to OCR Candidates Only) ---
        elif ocr_bucket == "CONTRADICTORY":
            ocr_cands = [c for c in [pipe_a_cand, pipe_b_cand] if c and c != "?"]
            if v_legible and v_conf >= 0.70 and v_char.isalnum() and v_char in ocr_cands:
                if v_char == pipe_a_cand:
                    action = "TIE_BREAK_PIPE_A"
                    final_char = pipe_a_cand
                    ui_tile_tone = "black"
                elif v_char == pipe_b_cand:
                    action = "TIE_BREAK_PIPE_B"
                    final_char = pipe_b_cand
                    ui_tile_tone = "black"
                else:
                    action = "VLM_ABSTAINED"
                    final_char = "?"
                    ui_tile_tone = "unknown"
            else:
                # VLM output '?' or character not in OCR candidates, or low confidence: demote to '?'
                action = "VLM_ABSTAINED"
                final_char = "?"
                ui_tile_tone = "unknown"

        # --- Rule 3: UNREADABLE Slots (Hard-Block Anti-Hallucination + Unanimous Low-Conf Rescue) ---
        else:  # UNREADABLE
            unanimous_char = vlm_disambiguated_char
            is_valid_type = _char_matches_slot_type(unanimous_char, slot_type)
            is_non_empty = bool(
                pipe_a_cand and pipe_b_cand and unanimous_char
                and pipe_a_cand != "?" and pipe_b_cand != "?" and unanimous_char != "?"
            )
            is_unanimous = is_non_empty and (pipe_a_cand == pipe_b_cand == unanimous_char)
            is_conf_ok = (pipe_a_conf >= 0.15 and pipe_b_conf >= 0.15 and v_conf >= 0.80)

            if is_unanimous and is_conf_ok and is_valid_type:
                action = "UNANIMOUS_LOW_CONF_RESCUE"
                final_char = unanimous_char
                ui_tile_tone = "black"
                reasoning_notes.append("unanimous_low_conf_rescue")
            else:
                # Zero hallucination: any UNREADABLE slot strictly stays '?'
                action = "FORBIDDEN_KEEP_UNKNOWN"
                final_char = "?"
                ui_tile_tone = "unknown"

        actions_summary[action] = actions_summary.get(action, 0) + 1

        # 2. Gather candidates and compute soft scores
        raw_cands: Dict[str, Dict[str, float]] = {}

        def _register_cand(c: str, src: str, conf_val: float):
            if not c or c == "?" or not c.isalnum():
                return
            c = c.upper()
            if c not in raw_cands:
                raw_cands[c] = {"vlm": 0.0, "pipe_a": 0.0, "pipe_b": 0.0}
            raw_cands[c][src] = max(raw_cands[c][src], float(conf_val))

        if vlm_disambiguated_char and vlm_disambiguated_char != "?":
            _register_cand(vlm_disambiguated_char, "vlm", v_conf)

        if pipe_a_cand and pipe_a_cand != "?":
            _register_cand(pipe_a_cand, "pipe_a", pipe_a_conf)
        pa_runner = s.get("pipe_a_runner_up") if i < len(ocr_slots) else None
        if pa_runner and pa_runner != "?":
            _register_cand(pa_runner, "pipe_a", s.get("pipe_a_runner_up_conf", 0.0))

        if pipe_b_cand and pipe_b_cand != "?":
            _register_cand(pipe_b_cand, "pipe_b", pipe_b_conf)
        pb_runner = s.get("pipe_b_runner_up") if i < len(ocr_slots) else None
        if pb_runner and pb_runner != "?":
            _register_cand(pb_runner, "pipe_b", s.get("pipe_b_runner_up_conf", 0.0))

        # Calculate soft score for each candidate:
        # soft_score = 0.45 * vlm_conf(c) + 0.30 * pipe_a_conf(c) + 0.25 * pipe_b_conf(c)
        scored_cands = []
        for c, confs in raw_cands.items():
            sc = round(0.45 * confs["vlm"] + 0.30 * confs["pipe_a"] + 0.25 * confs["pipe_b"], 3)
            src_list = [k for k, val in confs.items() if val > 0.0]
            scored_cands.append({
                "char": c,
                "soft_score": sc,
                "sources": src_list,
            })

        # Calculate final_char's soft_score (for beam search)
        if final_char != "?":
            fc_confs = raw_cands.get(final_char, {"vlm": 0.0, "pipe_a": 0.0, "pipe_b": 0.0})
            final_char_soft_score = round(
                0.45 * fc_confs["vlm"] + 0.30 * fc_confs["pipe_a"] + 0.25 * fc_confs["pipe_b"], 3
            )
            if ui_tile_tone == "green":
                final_char_soft_score = max(final_char_soft_score, 0.75)
            elif action == "UNANIMOUS_LOW_CONF_RESCUE":
                final_char_soft_score = min(final_char_soft_score, 0.70)
        else:
            final_char_soft_score = 0.0

        # Build ghost_stack:
        # - Drop candidates with soft_score < 0.05
        # - Exclude final_char if tone is green or black
        # - Tag kind: "ghost_fill" if tone == "unknown" else "conflict_alt"
        ghost_entries = []
        for cand in scored_cands:
            if cand["soft_score"] < 0.05:
                continue
            if ui_tile_tone in ("green", "black") and cand["char"] == final_char:
                continue
            kind = "ghost_fill" if ui_tile_tone == "unknown" else "conflict_alt"
            ghost_entries.append({
                "char": cand["char"],
                "soft_score": cand["soft_score"],
                "kind": kind,
                "sources": cand["sources"],
            })

        ghost_entries.sort(key=lambda x: x["soft_score"], reverse=True)
        ghost_stack = ghost_entries[:3]

        slot_data = {
            "slot_idx": i,
            "slot_index": i,
            "ocr_char": ocr_char,
            "ocr_bucket": ocr_bucket,
            "slot_type": slot_type,
            "expected_type": slot_type,
            "pipe_a_candidate": pipe_a_cand,
            "pipe_b_candidate": pipe_b_cand,
            "pipe_a_char": pipe_a_cand,
            "pipe_b_char": pipe_b_cand,
            "vlm_char": v_char,
            "vlm_blind_raw_char": vlm_blind_raw_char,
            "vlm_disambiguated_char": vlm_disambiguated_char,
            "vlm_conf": round(v_conf, 3),
            "vlm_legible": v_legible,
            "final_char": final_char,
            "final_char_soft_score": final_char_soft_score,
            "final_confidence": final_char_soft_score,
            "action": action,
            "ui_tile_tone": ui_tile_tone,
            "ghost_stack": ghost_stack,
            "reasoning_notes": reasoning_notes,
            "note": v_note,
        }
        if action == "LOCKED_VLM_DISPUTED":
            slot_data["confidence_penalty"] = -0.05

        reconciled_slots.append(slot_data)

    reconciled_plate = "".join(s["final_char"] for s in reconciled_slots)
    state_valid = validate_state_code(reconciled_plate)
    ocr_overall_conf = float(ocr_result.get("overall_confidence", 0.0))
    disputed_count = actions_summary.get("LOCKED_VLM_DISPUTED", 0)
    adjusted_conf = max(0.0, round(ocr_overall_conf - (disputed_count * 0.05), 4))

    ghost_predictions = compute_ghost_predictions(reconciled_slots, plate_len)

    return {
        "schema_version": "2.4.0",
        "track_id": ocr_result.get("track_id", "unknown"),
        "target_id": ocr_result.get("target_id", ocr_result.get("track_id", "unknown")),
        "merged_track_ids": ocr_result.get("merged_track_ids", [ocr_result.get("track_id")]),
        "is_merged_track": bool(ocr_result.get("is_merged_track", False)),
        "vlm_status": "SUCCESS",
        "provider_used": provider_used or "unknown",
        "model_used": model_used or "unknown",
        "runtime_seconds": round(runtime_seconds, 3),
        "fallback_trace": fallback_trace or [],
        "stage_2_3_plate": recovered_ocr,
        "vlm_raw_plate": vlm_raw_text or None,
        "final_plate": reconciled_plate,
        "reconciled_plate": reconciled_plate,
        "overall_confidence": adjusted_conf,
        "confidence_penalty_total": round(disputed_count * -0.05, 2),
        "state_code_valid": state_valid,
        "plate_layout": vlm_layout,
        "reconciled_slots": reconciled_slots,
        "ghost_predictions": ghost_predictions,
        "actions_summary": actions_summary,
    }


# ---------------------------------------------------------------------------
# High-Level Orchestrator
# ---------------------------------------------------------------------------
def verify_track_vlm(
    track_dir: str | Path,
    provider: str = "auto",
) -> Dict[str, Any]:
    """
    Stage 2.4 End-to-End Entry Point:
    1. Reads Stage 2.3 ocr_result.json.
    2. Builds 2x2 unenhanced & enhanced evidence grids.
    3. Runs blind VLM call with multi-tier failover.
    4. Arbitrates slot-by-slot and emits vlm_result.json.
    """
    t_dir = Path(track_dir).resolve()
    ocr_json_path = t_dir / "ocr_result.json"
    if not ocr_json_path.is_file():
        raise FileNotFoundError(f"Missing ocr_result.json in {t_dir}")

    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_result = json.load(f)

    # Retrieve evidence pack
    evidence_pack = ocr_result.get("vlm_evidence_pack")
    if not evidence_pack and "vlm_handoff" in ocr_result:
        evidence_pack = ocr_result["vlm_handoff"].get("vlm_evidence_pack", [])

    # Step 1: Build labeled grids
    unenh_grid_path, enh_grid_path = build_evidence_grids(t_dir, evidence_pack or [])

    # Step 2: Base64 encode
    unenh_b64 = encode_image_b64(unenh_grid_path)
    enh_b64 = encode_image_b64(enh_grid_path)

    # Step 3: Call blind VLM with failover
    plate_len = ocr_result.get("plate_length", 10)
    client = VLMClient()
    parsed_json, prov_used, mod_used, latency, trace = client.execute_blind_query(
        unenh_b64=unenh_b64,
        enh_b64=enh_b64,
        plate_len=plate_len,
        provider=provider,
    )

    # Step 4: Reconcile results
    result = reconcile_vlm_with_ocr(
        ocr_result=ocr_result,
        vlm_output=parsed_json,
        provider_used=prov_used,
        model_used=mod_used,
        runtime_seconds=latency,
        fallback_trace=trace,
    )

    result["grid_images"] = {
        "unenhanced_grid_path": str(unenh_grid_path),
        "enhanced_grid_path": str(enh_grid_path),
    }

    # Step 5: Save vlm_result.json and final_plate_result.json
    out_vlm = t_dir / "vlm_result.json"
    out_final = t_dir / "final_plate_result.json"
    with open(out_vlm, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out_final, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Stage 2.4 complete for {t_dir.name} -> {result['reconciled_plate']} "
        f"(saved to {out_vlm.name} and {out_final.name})"
    )
    return result
