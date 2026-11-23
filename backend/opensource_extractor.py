"""Vehicle Attribute Extractor using Open-Source Multimodal LLMs via OpenAI-compatible endpoints.

Exposes the exact same extract_vehicle_attributes(images, ...) signature and return type
(ExtractedVehicleAttributes) as backend/attribute_extractor.py, plus a helper
extract_vehicle_attributes_with_meta(...) returning provider, model_used, and latency_sec.

Implements a 4-tier high-resilience cascade:
1. Primary (NVIDIA NIM):
   - Model: nvidia/nemotron-3.5-super-vl-120b-a12b
   - Endpoint: https://integrate.api.nvidia.com/v1
   - Thinking disabled: extra_body={"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
   - timeout=25.0s, max_retries=0

2. Fallback 1 (OpenRouter — Gemma 4):
   - Model: google/gemma-4-26b-a4b-it:free
   - Endpoint: https://openrouter.ai/api/v1
   - Reasoning disabled: extra_body={"reasoning": {"effort": "none", "enabled": False}}
   - timeout=20.0s, max_retries=0

3. Fallback 2 (OpenRouter — Nemotron):
   - Model: nvidia/nemotron-3.5-super-vl:free
   - Endpoint: https://openrouter.ai/api/v1
   - Reasoning disabled: extra_body={"reasoning": {"effort": "none", "enabled": False}}
   - timeout=20.0s, max_retries=0

4. Tier 4 (Explicit Error + Safe Fallback):
   - If Tiers 1–3 all fail, logs a loud ERROR with exact failure reasons from all 3 tiers
     and attaches an explicit warning message in extract_vehicle_attributes_with_meta
     and the returned schema/warning field so the UI/API surfaces the error explicitly.

Image processing:
Loads up to 3 images with Pillow, converts to RGB JPEG in memory, resizing if wider
than 1024px to keep payloads fast, and sends as data:image/jpeg;base64,{b64}.

All tiers are wrapped in try/except, logging elapsed time and errors, guaranteeing
the server never crashes.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Ensure environment variables from .env are loaded
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Re-use schema, constants, and local fallback from existing attribute extractor
from backend.attribute_extractor import (
    INDIAN_CCTV_PROMPT,
    VALID_BODY_TYPES,
    ExtractedVehicleAttributes,
    local_fallback_extraction,
)

logger = logging.getLogger("vehicle_trace.opensource_extractor")
logger.setLevel(logging.INFO)

# Formulate prompt enforcing raw JSON output matching exact schema keys
OPENSOURCE_CCTV_PROMPT = f"""{INDIAN_CCTV_PROMPT}

CRITICAL OUTPUT INSTRUCTIONS:
Return ONLY raw valid JSON matching these exact schema keys:
- "body_type": strictly one of "two-wheeler", "car", "suv", "auto-rickshaw", "truck", "bus"
- "body_type_conf": float between 0.0 and 1.0
- "colour": dominant vehicle colour in lowercase (e.g., "white", "black", "silver", "red", "yellow", "blue", "grey", "green", "orange", "brown")
- "colour_conf": float between 0.0 and 1.0
- "make": vehicle manufacturer/brand in India (e.g., "Honda", "Tata", "Maruti Suzuki", "Hyundai", "Mahindra", "Bajaj", "TVS", "Toyota", "Kia", "Hero", "Royal Enfield") or "Unknown"
- "make_conf": float between 0.0 and 1.0 (must be 0.0 if make is "Unknown")
- "model": vehicle model name in India (e.g., "Activa 6G", "Nexon", "Swift", "Dzire", "Creta", "Scorpio", "Jupiter", "Access 125", "Compact 4S") or "Unknown"
- "model_conf": float between 0.0 and 1.0 (must be 0.0 if model is "Unknown")

Do NOT include markdown formatting, code block fences, notes, or explanation.
Output ONLY the raw JSON object."""


class ExtractedVehicleAttributesWithMeta(ExtractedVehicleAttributes):
    """Pydantic model containing extracted attributes plus extraction metadata."""

    provider: str
    model_used: str
    latency_sec: float
    warning: Optional[str] = None
    tier_errors: Optional[List[str]] = None

    def to_base_attributes(self) -> ExtractedVehicleAttributes:
        """Return base ExtractedVehicleAttributes instance."""
        return ExtractedVehicleAttributes(
            body_type=self.body_type,
            body_type_conf=self.body_type_conf,
            colour=self.colour,
            colour_conf=self.colour_conf,
            make=self.make,
            make_conf=self.make_conf,
            model=self.model,
            model_conf=self.model_conf,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of attributes and metadata."""
        return self.model_dump()

    def __iter__(self) -> Iterator[Any]:
        """Support 2-tuple unpacking: attrs, meta = extract_vehicle_attributes_with_meta(...)."""
        yield self.to_base_attributes()
        yield {
            "provider": self.provider,
            "model_used": self.model_used,
            "latency_sec": self.latency_sec,
            "warning": self.warning,
            "tier_errors": self.tier_errors,
        }

    def __getitem__(self, item: Union[int, str]) -> Any:
        """Support index-based (for tuple unpack) or key-based (for dict-like) access."""
        if isinstance(item, int):
            return list(self)[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


def _clean_and_parse_json(raw_text: str) -> dict:
    """Strip markdown code fences (```json), commentary, and parse JSON string."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Locate first '{' and last '}' to isolate JSON block if surrounded by prose
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return json.loads(text)


def _normalize_and_validate(raw_data: dict) -> ExtractedVehicleAttributes:
    """Normalize raw model dictionary to match validation and normalization rules."""
    # 1. Normalize and enforce valid body type
    raw_body = str(raw_data.get("body_type", "")).lower().strip()
    if raw_body not in VALID_BODY_TYPES:
        if any(w in raw_body for w in ("motorcycle", "scooter", "bike", "two")):
            raw_body = "two-wheeler"
        elif any(w in raw_body for w in ("rickshaw", "auto", "three")):
            raw_body = "auto-rickshaw"
        elif any(w in raw_body for w in ("sedan", "hatchback")):
            raw_body = "car"
        elif "suv" in raw_body:
            raw_body = "suv"
        elif "truck" in raw_body:
            raw_body = "truck"
        elif "bus" in raw_body:
            raw_body = "bus"
        else:
            raw_body = "car"
    raw_data["body_type"] = raw_body

    # 2. Normalize colour
    raw_data["colour"] = str(raw_data.get("colour", "white")).lower().strip()

    # 3. Normalize confidence bounds (0.0 <= conf <= 1.0)
    def parse_conf(val: Any, default: float) -> float:
        try:
            f = float(val)
            if 1.0 < f <= 100.0:
                f = f / 100.0
            return max(0.0, min(1.0, round(f, 2)))
        except (ValueError, TypeError):
            return default

    raw_data["body_type_conf"] = parse_conf(raw_data.get("body_type_conf"), 0.90)
    raw_data["colour_conf"] = parse_conf(raw_data.get("colour_conf"), 0.85)

    # 4. Normalize make and make_conf
    make = str(raw_data.get("make", "Unknown")).strip()
    make_conf = parse_conf(raw_data.get("make_conf"), 0.0)
    if make.lower() in ("unknown", "", "none", "null") or make_conf == 0.0:
        make = "Unknown"
        make_conf = 0.0
    raw_data["make"] = make
    raw_data["make_conf"] = make_conf

    # 5. Normalize model and model_conf
    model_name = str(raw_data.get("model", "Unknown")).strip()
    model_conf = parse_conf(raw_data.get("model_conf"), 0.0)
    if model_name.lower() in ("unknown", "", "none", "null") or model_conf == 0.0:
        model_name = "Unknown"
        model_conf = 0.0
    raw_data["model"] = model_name
    raw_data["model_conf"] = model_conf

    return ExtractedVehicleAttributes.model_validate(raw_data)


def _prepare_images(
    images: Union[str, Path, np.ndarray, bytes, Image.Image, List[Any], Tuple[Any, ...], None]
) -> Tuple[List[str], Optional[np.ndarray]]:
    """Convert input images into base64-encoded JPEG strings (up to 3 images).

    Resizes images if wider than 1024px to keep payloads fast.
    Also returns the first image as a numpy BGR ndarray for local fallback.
    """
    if images is None:
        return [], None

    img_list = images if isinstance(images, (list, tuple)) else [images]
    img_list = img_list[:3]

    b64_list: List[str] = []
    first_crop_np: Optional[np.ndarray] = None

    for item in img_list:
        if item is None:
            continue

        pil_img: Optional[Image.Image] = None

        if isinstance(item, Image.Image):
            pil_img = item.convert("RGB")
            if first_crop_np is None:
                first_crop_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        elif isinstance(item, (str, Path)):
            p = Path(item)
            if not p.is_absolute():
                p = BASE_DIR / p
            if not p.exists():
                logger.warning(f"Image path does not exist: {p}")
                continue
            try:
                pil_img = Image.open(p).convert("RGB")
                if first_crop_np is None:
                    first_crop_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"Failed to open image file {p}: {e}")
                continue

        elif isinstance(item, bytes):
            if not item:
                continue
            try:
                pil_img = Image.open(io.BytesIO(item)).convert("RGB")
                if first_crop_np is None:
                    first_crop_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"Failed to decode image bytes: {e}")
                continue

        elif isinstance(item, np.ndarray):
            if item.size == 0:
                continue
            if first_crop_np is None:
                first_crop_np = item
            try:
                # Assuming OpenCV BGR
                pil_img = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.warning(f"Failed to convert numpy ndarray to image: {e}")
                continue

        if pil_img is not None:
            # Resize if wider than 1024px to keep payload fast
            if pil_img.width > 1024:
                new_width = 1024
                new_height = max(1, int(pil_img.height * (1024 / pil_img.width)))
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            b64_list.append(b64_str)

    return b64_list, first_crop_np


def extract_vehicle_attributes_with_meta(
    images: Union[str, Path, np.ndarray, bytes, Image.Image, List[Any], None] = None,
    coarse_class: Optional[str] = None,
    track_id: Optional[int] = None,
    image_paths: Optional[Union[str, Path, np.ndarray, bytes, Image.Image, List[Any]]] = None,
    **kwargs: Any,
) -> ExtractedVehicleAttributesWithMeta:
    """Extract vehicle attributes using a 4-tier cascade and return with metadata.

    Returns an ExtractedVehicleAttributesWithMeta instance containing:
    - all ExtractedVehicleAttributes fields (body_type, colour, make, model, and confidences)
    - provider: 'nvidia' | 'openrouter' | 'local_fallback'
    - model_used: string ID of the model utilized
    - latency_sec: total execution time in seconds

    Also supports unpacking as: attrs, meta = extract_vehicle_attributes_with_meta(...)
    """
    target = images if images is not None else image_paths
    b64_images, first_crop_np = _prepare_images(target)

    # Track errors across tiers for explicit outage surfacing
    tier_1_err: Optional[str] = None
    tier_2_err: Optional[str] = None
    tier_3_err: Optional[str] = None

    # If no valid images can be loaded, fall back immediately to local heuristic
    if not b64_images:
        warning_msg = "No valid vehicle images provided; using Tier 4 local heuristic fallback."
        logger.warning(warning_msg)
        t0 = time.perf_counter()
        fallback_attrs = local_fallback_extraction(coarse_class, first_crop_np, track_id)
        latency = round(time.perf_counter() - t0, 4)
        return ExtractedVehicleAttributesWithMeta(
            body_type=fallback_attrs.body_type,
            body_type_conf=fallback_attrs.body_type_conf,
            colour=fallback_attrs.colour,
            colour_conf=fallback_attrs.colour_conf,
            make=fallback_attrs.make,
            make_conf=fallback_attrs.make_conf,
            model=fallback_attrs.model,
            model_conf=fallback_attrs.model_conf,
            provider="local_fallback",
            model_used="local_heuristic",
            latency_sec=latency,
            warning=warning_msg,
            tier_errors=["No input images provided to VLM extractor."],
        )

    # Build multimodal messages payload
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": OPENSOURCE_CCTV_PROMPT}
    ]
    for b64 in b64_images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    messages = [{"role": "user", "content": user_content}]

    # Tier 1: Primary (NVIDIA NIM)
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if nvidia_key:
        model_name = "nvidia/nemotron-3.5-super-vl-120b-a12b"
        t0 = time.perf_counter()
        try:
            logger.info(f"Querying Tier 1 Primary (NVIDIA NIM - {model_name}) with {len(b64_images)} crop(s)...")
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key,
                timeout=25.0,
                max_retries=0,
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}},
                temperature=0.0,
                max_tokens=300,
                timeout=25.0,
            )
            raw_content = resp.choices[0].message.content or ""
            parsed = _clean_and_parse_json(raw_content)
            attrs = _normalize_and_validate(parsed)
            latency = round(time.perf_counter() - t0, 3)
            logger.info(
                f"Tier 1 (NVIDIA NIM - {model_name}) succeeded in {latency}s: "
                f"{attrs.make} {attrs.model} ({attrs.colour} {attrs.body_type})"
            )
            return ExtractedVehicleAttributesWithMeta(
                body_type=attrs.body_type,
                body_type_conf=attrs.body_type_conf,
                colour=attrs.colour,
                colour_conf=attrs.colour_conf,
                make=attrs.make,
                make_conf=attrs.make_conf,
                model=attrs.model,
                model_conf=attrs.model_conf,
                provider="nvidia",
                model_used=model_name,
                latency_sec=latency,
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 3)
            tier_1_err = f"{type(e).__name__}: {e} (after {elapsed}s)"
            logger.warning(
                f"Tier 1 (NVIDIA NIM - {model_name}) failed after {elapsed}s: {e}. "
                f"Falling back to Tier 2."
            )
    else:
        tier_1_err = "NVIDIA_API_KEY not configured"
        logger.info("NVIDIA_API_KEY not set. Skipping Tier 1 and proceeding to Tier 2.")

    # Tier 2: Fallback 1 (OpenRouter — Gemma 4)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        model_name = "google/gemma-4-26b-a4b-it:free"
        t0 = time.perf_counter()
        try:
            logger.info(f"Querying Tier 2 Fallback 1 (OpenRouter Gemma 4 - {model_name}) with {len(b64_images)} crop(s)...")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                timeout=20.0,
                max_retries=0,
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"reasoning": {"effort": "none", "enabled": False}},
                temperature=0.0,
                max_tokens=300,
                timeout=20.0,
            )
            raw_content = resp.choices[0].message.content or ""
            parsed = _clean_and_parse_json(raw_content)
            attrs = _normalize_and_validate(parsed)
            latency = round(time.perf_counter() - t0, 3)
            logger.info(
                f"Tier 2 (OpenRouter Gemma 4 - {model_name}) succeeded in {latency}s: "
                f"{attrs.make} {attrs.model} ({attrs.colour} {attrs.body_type})"
            )
            return ExtractedVehicleAttributesWithMeta(
                body_type=attrs.body_type,
                body_type_conf=attrs.body_type_conf,
                colour=attrs.colour,
                colour_conf=attrs.colour_conf,
                make=attrs.make,
                make_conf=attrs.make_conf,
                model=attrs.model,
                model_conf=attrs.model_conf,
                provider="openrouter",
                model_used=model_name,
                latency_sec=latency,
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 3)
            tier_2_err = f"{type(e).__name__}: {e} (after {elapsed}s)"
            logger.warning(
                f"Tier 2 (OpenRouter Gemma 4 - {model_name}) failed after {elapsed}s: {e}. "
                f"Falling back to Tier 3."
            )
    else:
        tier_2_err = "OPENROUTER_API_KEY not configured"
        logger.info("OPENROUTER_API_KEY not set. Skipping Tier 2 & Tier 3.")

    # Tier 3: Fallback 2 (OpenRouter — Nemotron)
    if openrouter_key:
        model_name = "nvidia/nemotron-3.5-super-vl:free"
        t0 = time.perf_counter()
        try:
            logger.info(f"Querying Tier 3 Fallback 2 (OpenRouter Nemotron - {model_name}) with {len(b64_images)} crop(s)...")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                timeout=20.0,
                max_retries=0,
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"reasoning": {"effort": "none", "enabled": False}},
                temperature=0.0,
                max_tokens=300,
                timeout=20.0,
            )
            raw_content = resp.choices[0].message.content or ""
            parsed = _clean_and_parse_json(raw_content)
            attrs = _normalize_and_validate(parsed)
            latency = round(time.perf_counter() - t0, 3)
            logger.info(
                f"Tier 3 (OpenRouter Nemotron - {model_name}) succeeded in {latency}s: "
                f"{attrs.make} {attrs.model} ({attrs.colour} {attrs.body_type})"
            )
            return ExtractedVehicleAttributesWithMeta(
                body_type=attrs.body_type,
                body_type_conf=attrs.body_type_conf,
                colour=attrs.colour,
                colour_conf=attrs.colour_conf,
                make=attrs.make,
                make_conf=attrs.make_conf,
                model=attrs.model,
                model_conf=attrs.model_conf,
                provider="openrouter",
                model_used=model_name,
                latency_sec=latency,
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 3)
            tier_3_err = f"{type(e).__name__}: {e} (after {elapsed}s)"
            logger.warning(
                f"Tier 3 (OpenRouter Nemotron - {model_name}) failed after {elapsed}s: {e}. "
                f"Falling back to Tier 4 (local safety net)."
            )
    else:
        tier_3_err = "OPENROUTER_API_KEY not configured"

    # Tier 4: Explicit Error + Safe Fallback (Local Heuristic)
    failure_details = []
    if tier_1_err:
        failure_details.append(f"Tier 1 (NVIDIA NIM): {tier_1_err}")
    if tier_2_err:
        failure_details.append(f"Tier 2 (OpenRouter Gemma 4): {tier_2_err}")
    if tier_3_err:
        failure_details.append(f"Tier 3 (OpenRouter Nemotron): {tier_3_err}")
    error_summary = "; ".join(failure_details) if failure_details else "All open-source VLM tiers unavailable"
    warning_msg = f"All open-source VLM tiers failed: {error_summary}"

    logger.error(
        f"\n{'=' * 80}\n"
        f"!!! VLM OUTAGE / CRITICAL CASCADING FAILURE ALERT !!!\n"
        f"Details: {warning_msg}\n"
        f"Engaging Tier 4: Local Heuristic Safe Fallback.\n"
        f"{'=' * 80}"
    )

    t0 = time.perf_counter()
    try:
        fallback_attrs = local_fallback_extraction(coarse_class, first_crop_np, track_id)
    except Exception as e:
        logger.error(f"Local fallback extraction error: {e}. Using deterministic default attributes.")
        fallback_attrs = ExtractedVehicleAttributes(
            body_type="car",
            body_type_conf=0.5,
            colour="white",
            colour_conf=0.5,
            make="Unknown",
            make_conf=0.0,
            model="Unknown",
            model_conf=0.0,
        )
    latency = round(time.perf_counter() - t0, 4)
    return ExtractedVehicleAttributesWithMeta(
        body_type=fallback_attrs.body_type,
        body_type_conf=fallback_attrs.body_type_conf,
        colour=fallback_attrs.colour,
        colour_conf=fallback_attrs.colour_conf,
        make=fallback_attrs.make,
        make_conf=fallback_attrs.make_conf,
        model=fallback_attrs.model,
        model_conf=fallback_attrs.model_conf,
        provider="local_fallback",
        model_used="local_heuristic",
        latency_sec=latency,
        warning=warning_msg,
        tier_errors=failure_details,
    )


def extract_vehicle_attributes(
    images: Union[str, Path, np.ndarray, bytes, Image.Image, List[Any], None] = None,
    coarse_class: Optional[str] = None,
    track_id: Optional[int] = None,
    image_paths: Optional[Union[str, Path, np.ndarray, bytes, Image.Image, List[Any]]] = None,
    **kwargs: Any,
) -> ExtractedVehicleAttributes:
    """Extract vehicle body_type, colour, make, and model with confidence scores.

    Matches the exact signature and return type of backend/attribute_extractor.py.
    Uses a 4-tier cascade:
      1. Primary: NVIDIA NIM (nvidia/nemotron-3.5-super-vl-120b-a12b)
      2. Fallback 1: OpenRouter (google/gemma-4-26b-a4b-it:free)
      3. Fallback 2: OpenRouter (nvidia/nemotron-3.5-super-vl:free)
      4. Tier 4 Safety Net: Local heuristic analyzer / demo scenario fallback.
    """
    target = images if images is not None else image_paths
    result_with_meta = extract_vehicle_attributes_with_meta(
        images=target,
        coarse_class=coarse_class,
        track_id=track_id,
        **kwargs,
    )
    return result_with_meta.to_base_attributes()
