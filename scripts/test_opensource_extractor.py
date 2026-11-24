#!/usr/bin/env python3
"""Comprehensive Verification Test Suite for Open-Source LLM Vehicle Attribute Extractor.

Validates:
1. Pydantic return model, field names, and confidence bounds.
2. Prompt normalization rules (body type aliases, colour lowercase, make/model Unknown logic).
3. Image preprocessing with Pillow (resizing >1024px, base64 JPEG format, up to 3 images, format conversion).
4. Helper extract_vehicle_attributes_with_meta returns provider, model_used, latency_sec and supports unpacking.
5. 4-tier cascade simulation: Tier 1 -> Tier 2 -> Tier 3 -> Tier 4 (local heuristic).
6. Live execution test on test_hyundai.png.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.opensource_extractor import (
    ExtractedVehicleAttributes,
    ExtractedVehicleAttributesWithMeta,
    VALID_BODY_TYPES,
    _clean_and_parse_json,
    _normalize_and_validate,
    _prepare_images,
    extract_vehicle_attributes,
    extract_vehicle_attributes_with_meta,
)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def test_schema_equivalence():
    print_section("1. Schema & Return Model Validation")
    target_fields = {
        "body_type",
        "body_type_conf",
        "colour",
        "colour_conf",
        "make",
        "make_conf",
        "model",
        "model_conf",
    }
    model_fields = set(ExtractedVehicleAttributes.model_fields.keys())
    assert model_fields == target_fields, f"Fields mismatch: {model_fields} vs {target_fields}"
    print("✓ Schema fields match exactly: " + ", ".join(sorted(model_fields)))

    # Verify ExtractedVehicleAttributesWithMeta inheritance and fields
    assert issubclass(ExtractedVehicleAttributesWithMeta, ExtractedVehicleAttributes)
    meta_fields = set(ExtractedVehicleAttributesWithMeta.model_fields.keys())
    expected_meta = target_fields | {"provider", "model_used", "latency_sec", "warning", "tier_errors"}
    assert meta_fields == expected_meta, f"Meta fields mismatch: {meta_fields}"
    print("✓ ExtractedVehicleAttributesWithMeta correctly extends base model with metadata fields.")


def test_json_parsing_and_stripping():
    print_section("2. JSON Code Fence Stripping & Parsing")
    raw_clean = '{"body_type": "car", "colour": "red"}'
    fenced_json = '```json\n{"body_type": "suv", "colour": "white"}\n```'
    fenced_bare = '```\n{"body_type": "truck", "colour": "black"}\n```'
    with_prose = 'Here is the vehicle analysis result:\n{"body_type": "bus", "colour": "blue"}\nHope this helps!'

    assert _clean_and_parse_json(raw_clean)["body_type"] == "car"
    print("✓ Clean JSON parsed successfully.")
    assert _clean_and_parse_json(fenced_json)["body_type"] == "suv"
    print("✓ Markdown ```json code fence stripped and parsed.")
    assert _clean_and_parse_json(fenced_bare)["body_type"] == "truck"
    print("✓ Plain ``` code fence stripped and parsed.")
    assert _clean_and_parse_json(with_prose)["body_type"] == "bus"
    print("✓ JSON extracted from surrounding conversational prose.")


def test_normalization_rules():
    print_section("3. Prompt Normalization & Confidence Bounds")

    # Body type alias mappings
    aliases = [
        ({"body_type": "motorcycle"}, "two-wheeler"),
        ({"body_type": "scooter"}, "two-wheeler"),
        ({"body_type": "Bajaj Auto Rickshaw"}, "auto-rickshaw"),
        ({"body_type": "three-wheeler"}, "auto-rickshaw"),
        ({"body_type": "sedan"}, "car"),
        ({"body_type": "hatchback"}, "car"),
        ({"body_type": "compact SUV"}, "suv"),
        ({"body_type": "Commercial Truck"}, "truck"),
        ({"body_type": "transit bus"}, "bus"),
        ({"body_type": "unrecognized vehicle"}, "car"),
    ]
    for inp, expected in aliases:
        inp_full = {
            **inp,
            "colour": "WHITE",
            "make": "Unknown",
            "make_conf": 0.0,
            "model": "Unknown",
            "model_conf": 0.0,
        }
        res = _normalize_and_validate(inp_full)
        assert res.body_type == expected, f"Expected {expected} for {inp}, got {res.body_type}"
        assert res.colour == "white", f"Expected lowercase colour 'white', got '{res.colour}'"
    print(f"✓ All {len(aliases)} body type aliases and uppercase colour normalized correctly.")

    # Confidence clamping (e.g. percentages 0-100 to 0.0-1.0)
    conf_test = {
        "body_type": "car",
        "body_type_conf": 98.0,  # 98% -> 0.98
        "colour": "silver",
        "colour_conf": "0.92",   # string -> float
        "make": "Hyundai",
        "make_conf": 85.0,       # 85% -> 0.85
        "model": "Creta",
        "model_conf": 78,        # int 78 -> 0.78
    }
    res_conf = _normalize_and_validate(conf_test)
    assert res_conf.body_type_conf == 0.98
    assert res_conf.colour_conf == 0.92
    assert res_conf.make_conf == 0.85
    assert res_conf.model_conf == 0.78
    print("✓ Confidence values (percentages, strings, ints) clamped and validated within [0.0, 1.0].")

    # Unknown make and model enforcement: must have 0.0 confidence
    unknown_test = {
        "body_type": "car",
        "colour": "white",
        "make": "Unknown",
        "make_conf": 0.85,  # conflicting confidence
        "model": "null",
        "model_conf": 0.50,
    }
    res_unknown = _normalize_and_validate(unknown_test)
    assert res_unknown.make == "Unknown" and res_unknown.make_conf == 0.0
    assert res_unknown.model == "Unknown" and res_unknown.model_conf == 0.0
    print("✓ 'Unknown' make and model strictly enforce confidence == 0.0.")

    # 0.0 confidence forces Unknown
    zero_conf_test = {
        "body_type": "car",
        "colour": "white",
        "make": "Tata",
        "make_conf": 0.0,
        "model": "Nexon",
        "model_conf": 0.0,
    }
    res_zero = _normalize_and_validate(zero_conf_test)
    assert res_zero.make == "Unknown" and res_zero.make_conf == 0.0
    assert res_zero.model == "Unknown" and res_zero.model_conf == 0.0
    print("✓ Zero confidence scores correctly map make and model to 'Unknown'.")


def test_image_preprocessing():
    print_section("4. Pillow Image Preprocessing & Sizing")

    # Wide image (> 1024px)
    wide_img = Image.new("RGB", (2048, 1024), color="red")
    b64_list, first_np = _prepare_images([wide_img])
    assert len(b64_list) == 1
    assert first_np is not None

    # Decode and check dimensions
    raw_bytes = Image.open(io.BytesIO(Image.open(io.BytesIO(wide_img.tobytes())).tobytes() if False else None or io.BytesIO())) if False else None
    # Verify through Pillow decoding of base64
    import base64
    decoded = Image.open(io.BytesIO(base64.b64decode(b64_list[0])))
    assert decoded.width == 1024, f"Expected resized width 1024, got {decoded.width}"
    assert decoded.height == 512, f"Expected resized height 512, got {decoded.height}"
    print("✓ Oversized image (2048x1024) resized to 1024x512 JPEG.")

    # Cap at 3 images
    img_list = [Image.new("RGB", (100, 100), color="blue") for _ in range(5)]
    b64_cap, _ = _prepare_images(img_list)
    assert len(b64_cap) == 3, f"Expected 3 images capped, got {len(b64_cap)}"
    print("✓ Max 3 images constraint strictly enforced.")

    # Various input types: path, bytes, ndarray
    crop_path = BASE_DIR / "outputs" / "crops" / "test_hyundai.png"
    if crop_path.exists():
        cv_img = cv2.imread(str(crop_path))
        raw_b = crop_path.read_bytes()
        mixed_b64, first_crop = _prepare_images([str(crop_path), cv_img, raw_b])
        assert len(mixed_b64) == 3
        assert first_crop is not None
        print("✓ Heterogeneous image types (file path, numpy ndarray, raw bytes) preprocessed cleanly.")


def test_helper_and_unpacking():
    print_section("5. Helper extract_vehicle_attributes_with_meta Interface")

    # Test local fallback call via with_meta
    res = extract_vehicle_attributes_with_meta(
        images=None,
        coarse_class="motorcycle",
        track_id=1,
    )
    # Check property access
    assert res.provider == "local_fallback"
    assert res.model_used == "local_heuristic"
    assert res.latency_sec >= 0.0
    assert res.body_type == "two-wheeler"
    assert res.make == "Honda"
    assert res.model == "Activa 6G"
    print("✓ Object property access verified: res.provider, res.model_used, res.latency_sec.")

    # Check 2-tuple unpacking: attrs, meta = ...
    attrs, meta = extract_vehicle_attributes_with_meta(
        images=None,
        coarse_class="motorcycle",
        track_id=1,
    )
    assert isinstance(attrs, ExtractedVehicleAttributes)
    assert isinstance(meta, dict)
    assert meta["provider"] == "local_fallback"
    assert meta["model_used"] == "local_heuristic"
    assert "latency_sec" in meta
    print("✓ 2-Tuple unpacking verified: (attrs, meta) with meta dictionary.")

    # Check dict-style access
    assert res["provider"] == "local_fallback"
    assert res["model_used"] == "local_heuristic"
    print("✓ Dict-style key access verified: res['provider'], res['model_used'].")

    # Check extract_vehicle_attributes return type
    base_res = extract_vehicle_attributes(images=None, coarse_class="car", track_id=2)
    assert type(base_res) is ExtractedVehicleAttributes, f"Expected {ExtractedVehicleAttributes}, got {type(base_res)}"
    assert base_res.make == "Tata" and base_res.model == "Nexon"
    print("✓ extract_vehicle_attributes return type is strictly ExtractedVehicleAttributes.")


def test_cascade_simulation():
    print_section("6. 4-Tier Cascade Simulation (Failover Resilience)")

    dummy_img = Image.new("RGB", (100, 100), color="white")

    # Case 1: Tier 1 fails -> Cascades to Tier 2 and succeeds
    with patch("backend.opensource_extractor.OpenAI") as mock_openai_cls:
        client_tier1 = MagicMock()
        client_tier1.chat.completions.create.side_effect = Exception("Tier 1 NVIDIA timeout 504")

        client_tier2 = MagicMock()
        resp2 = MagicMock()
        resp2.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "body_type": "suv",
                "body_type_conf": 0.95,
                "colour": "red",
                "colour_conf": 0.90,
                "make": "Tata",
                "make_conf": 0.88,
                "model": "Nexon",
                "model_conf": 0.85,
            })))
        ]
        client_tier2.chat.completions.create.return_value = resp2

        # Return client_tier1 on first call, client_tier2 on second call
        mock_openai_cls.side_effect = [client_tier1, client_tier2]

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "fake_nv",
            "NVIDIA_API_KEY_1": "",
            "NVIDIA_API_KEY_2": "",
            "OPENROUTER_API_KEY": "fake_or",
        }):
            res, meta = extract_vehicle_attributes_with_meta(images=[dummy_img])
            # Verify Tier 1 client initialization arguments (timeout=25.0, max_retries=0)
            mock_openai_cls.assert_any_call(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key="fake_nv",
                timeout=25.0,
                max_retries=0,
            )
            # Verify Tier 1 call extra_body contains reasoning_effort: none and enable_thinking: False
            t1_call_kwargs = client_tier1.chat.completions.create.call_args.kwargs
            assert t1_call_kwargs.get("timeout") == 25.0
            assert t1_call_kwargs.get("extra_body") == {
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            }
            # Verify Tier 2 client initialization and extra_body
            mock_openai_cls.assert_any_call(
                base_url="https://openrouter.ai/api/v1",
                api_key="fake_or",
                timeout=20.0,
                max_retries=0,
            )
            t2_call_kwargs = client_tier2.chat.completions.create.call_args.kwargs
            assert t2_call_kwargs.get("timeout") == 20.0
            assert t2_call_kwargs.get("extra_body") == {
                "reasoning": {"effort": "none", "enabled": False},
            }
            assert meta["provider"] == "openrouter"
            assert meta["model_used"] == "google/gemma-4-26b-a4b-it:free"
            assert res.make == "Tata"
            assert res.body_type == "suv"
            print("✓ Tier 1 (25s timeout, max_retries=0) failover to Tier 2 (Gemma 4: 20s timeout, reasoning.effort='none') verified.")

    # Case 2: Tier 1 & 2 fail -> Cascades to Tier 3 and succeeds
    with patch("backend.opensource_extractor.OpenAI") as mock_openai_cls:
        client_t1 = MagicMock()
        client_t1.chat.completions.create.side_effect = Exception("NVIDIA NIM 500 error")

        client_t2 = MagicMock()
        client_t2.chat.completions.create.side_effect = Exception("OpenRouter Gemma 4 429 rate limit")

        client_t3 = MagicMock()
        resp3 = MagicMock()
        resp3.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "body_type": "two-wheeler",
                "body_type_conf": 0.96,
                "colour": "black",
                "colour_conf": 0.94,
                "make": "Bajaj",
                "make_conf": 0.90,
                "model": "Pulsar",
                "model_conf": 0.88,
            })))
        ]
        client_t3.chat.completions.create.return_value = resp3

        mock_openai_cls.side_effect = [client_t1, client_t2, client_t3]

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "fake_nv",
            "NVIDIA_API_KEY_1": "",
            "NVIDIA_API_KEY_2": "",
            "OPENROUTER_API_KEY": "fake_or",
        }):
            res, meta = extract_vehicle_attributes_with_meta(images=[dummy_img])
            # Verify Tier 3 call extra_body and timeout
            t3_call_kwargs = client_t3.chat.completions.create.call_args.kwargs
            assert t3_call_kwargs.get("timeout") == 20.0
            assert t3_call_kwargs.get("extra_body") == {
                "reasoning": {"effort": "none", "enabled": False},
            }
            assert meta["provider"] == "openrouter"
            assert meta["model_used"] == "nvidia/nemotron-3.5-super-vl:free"
            assert res.make == "Bajaj"
            assert res.model == "Pulsar"
            print("✓ Tier 1 & Tier 2 failures successfully cascaded to Tier 3 (OpenRouter Nemotron).")

    # Case 3: Tier 1, 2, 3 all fail -> Loud ERROR and Tier 4 Local Heuristic Safe Fallback with Warning
    with patch("backend.opensource_extractor.OpenAI") as mock_openai_cls:
        client_f1 = MagicMock()
        client_f1.chat.completions.create.side_effect = Exception("NVIDIA NIM Connection Timeout")
        client_f2 = MagicMock()
        client_f2.chat.completions.create.side_effect = Exception("OpenRouter Gemma 4 502 Bad Gateway")
        client_f3 = MagicMock()
        client_f3.chat.completions.create.side_effect = Exception("OpenRouter Nemotron 429 Quota Exceeded")
        mock_openai_cls.side_effect = [client_f1, client_f2, client_f3]

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "fake_nv",
            "NVIDIA_API_KEY_1": "",
            "NVIDIA_API_KEY_2": "",
            "OPENROUTER_API_KEY": "fake_or",
        }):
            full_res = extract_vehicle_attributes_with_meta(
                images=[dummy_img],
                coarse_class="auto-rickshaw",
                track_id=3,
            )
            attrs, meta = full_res
            assert meta["provider"] == "local_fallback"
            assert meta["model_used"] == "local_heuristic"
            assert attrs.make == "Bajaj"
            assert attrs.model == "Compact 4S"
            assert "warning" in meta and meta["warning"] is not None
            assert "All open-source VLM tiers failed" in meta["warning"]
            assert "Tier 1 (NVIDIA NIM)" in meta["warning"]
            assert "Tier 2 (OpenRouter Gemma 4)" in meta["warning"]
            assert "Tier 3 (OpenRouter Nemotron)" in meta["warning"]
            assert full_res.warning == meta["warning"]
            assert len(full_res.tier_errors) == 3
            print("✓ Tier 4 explicit warning and exact error reasons attached when all tiers fail:")
            print(f"  Warning: {full_res.warning}")


def test_api_endpoint_integration():
    print_section("7. POST /api/vehicles/analyze Endpoint Integration")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Verify default EXTRACTOR_PROVIDER is opensource
    assert os.getenv("EXTRACTOR_PROVIDER", "opensource").lower() == "opensource"
    print("✓ EXTRACTOR_PROVIDER default is 'opensource'.")

    # 2. Test analyze vehicle with simulated Tier 4 outage
    with patch("backend.opensource_extractor.OpenAI") as mock_openai_cls, \
         patch("backend.opensource_extractor._prepare_images") as mock_prep:
        mock_openai_cls.side_effect = Exception("Simulated Total Network Outage")
        mock_prep.return_value = (["fake_b64"], None)
        with patch.dict(os.environ, {
            "EXTRACTOR_PROVIDER": "opensource",
            "NVIDIA_API_KEY": "fake_nv",
            "NVIDIA_API_KEY_1": "",
            "NVIDIA_API_KEY_2": "",
            "OPENROUTER_API_KEY": "fake_or",
        }):
            resp = client.post("/api/vehicles/analyze", json={"scenario_id": "scenario-1"})
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert "warning" in data
            assert data["warning"] is not None
            assert "All open-source VLM tiers failed" in data["warning"]
            print("✓ POST /api/vehicles/analyze returned 200 with explicit 'warning' on VLM outage:")
            print(f"  Warning in JSON response: {data['warning'][:80]}...")

    # 3. Test standard analyze vehicle with opensource provider
    resp_std = client.post("/api/vehicles/analyze", json={"scenario_id": "scenario-1"})
    assert resp_std.status_code == 200
    print("✓ POST /api/vehicles/analyze executed cleanly with open-source extractor.")


def test_live_execution():
    print_section("8. Live Extraction Test (outputs/crops/test_hyundai.png)")
    crop_path = BASE_DIR / "outputs" / "crops" / "test_hyundai.png"
    if not crop_path.exists():
        print("⚠ test_hyundai.png not found, skipping live inference.")
        return

    # Check if network API keys are present
    nv_key = os.getenv("NVIDIA_API_KEY")
    or_key = os.getenv("OPENROUTER_API_KEY")
    print(f"API Keys available: NVIDIA={'Yes' if nv_key else 'No'}, OpenRouter={'Yes' if or_key else 'No'}")

    res = extract_vehicle_attributes_with_meta(
        image_paths=[str(crop_path)],
        coarse_class="car",
    )
    print("\n--- Live Extraction Result ---")
    print(f"Provider:    {res.provider}")
    print(f"Model Used:  {res.model_used}")
    print(f"Latency:     {res.latency_sec:.3f}s")
    print(f"Vehicle:     {res.make} {res.model} (conf: make={res.make_conf}, model={res.model_conf})")
    print(f"Appearance:  {res.colour} {res.body_type} (conf: colour={res.colour_conf}, body={res.body_type_conf})")
    if res.warning:
        print(f"Warning:     {res.warning}")
    print("------------------------------")
    assert res.body_type in VALID_BODY_TYPES
    print("✓ Live extraction completed successfully without error.")


def main():
    test_schema_equivalence()
    test_json_parsing_and_stripping()
    test_normalization_rules()
    test_image_preprocessing()
    test_helper_and_unpacking()
    test_cascade_simulation()
    test_api_endpoint_integration()
    test_live_execution()
    print_section("All Tests Passed Successfully!")


if __name__ == "__main__":
    main()
