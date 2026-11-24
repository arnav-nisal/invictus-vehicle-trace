"""Unit tests for Multi-Key NVIDIA NIM Failover and Stage 3 Vehicle Intelligence."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from cv_pipeline.api_keys import get_nvidia_api_keys
from cv_pipeline.vehicle_crop_selector import (
    _compute_crop_quality_score,
    build_vehicle_crops_contact_strip,
    select_top_vehicle_crops,
)
from cv_pipeline.vlm_verifier import VLMClient


def test_get_nvidia_api_keys_ordering_and_dedup():
    """Verify key ordering, deduplication by secret value, and ignoring empty keys."""
    env_mock = {
        "NVIDIA_API_KEY": "secret_alpha",
        "NVIDIA_API_KEY_1": "secret_beta",
        "NVIDIA_API_KEY_2": "secret_alpha",  # Duplicate of first key
        "NVIDIA_API_KEY_3": "",              # Empty key
        "NVIDIA_API_KEY_10": "secret_gamma", # High index
        "UNRELATED_KEY": "something_else",
    }
    with patch.dict(os.environ, env_mock, clear=True):
        keys = get_nvidia_api_keys()
        labels = [k[0] for k in keys]
        secrets = [k[1] for k in keys]

        # NVIDIA_API_KEY, then NVIDIA_API_KEY_1, then NVIDIA_API_KEY_10 (KEY_2 skipped because duplicate)
        assert labels == ["NVIDIA_API_KEY", "NVIDIA_API_KEY_1", "NVIDIA_API_KEY_10"]
        assert secrets == ["secret_alpha", "secret_beta", "secret_gamma"]
        assert len(keys) == 3


def test_vlm_verifier_multi_key_failover():
    """Verify that VLMClient cycles through NVIDIA NIM keys on failure until one succeeds."""
    env_mock = {
        "NVIDIA_API_KEY": "nv_fail_key",
        "NVIDIA_API_KEY_1": "nv_success_key",
        "GROQ_API_KEY": "",  # Skip groq
    }

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "plate_length": 10,
                    "characters": [
                        {"slot_idx": 0, "char": "U", "confidence": 0.95, "legible": True, "note": ""},
                        {"slot_idx": 1, "char": "P", "confidence": 0.95, "legible": True, "note": ""},
                        {"slot_idx": 2, "char": "1", "confidence": 0.95, "legible": True, "note": ""},
                        {"slot_idx": 3, "char": "5", "confidence": 0.95, "legible": True, "note": ""},
                        {"slot_idx": 4, "char": "B", "confidence": 0.92, "legible": True, "note": ""},
                        {"slot_idx": 5, "char": "E", "confidence": 0.92, "legible": True, "note": ""},
                        {"slot_idx": 6, "char": "5", "confidence": 0.90, "legible": True, "note": ""},
                        {"slot_idx": 7, "char": "5", "confidence": 0.90, "legible": True, "note": ""},
                        {"slot_idx": 8, "char": "8", "confidence": 0.90, "legible": True, "note": ""},
                        {"slot_idx": 9, "char": "0", "confidence": 0.90, "legible": True, "note": ""},
                    ],
                    "layout": "single_line",
                    "overall_visual_clarity": "clear",
                })
            )
        )
    ]

    with patch.dict(os.environ, env_mock, clear=True):
        client = VLMClient()

        # Mock OpenAI so first key call raises 429, second key call succeeds
        with patch("cv_pipeline.vlm_verifier.OpenAI") as mock_openai_cls:
            mock_client_fail = MagicMock()
            mock_client_fail.chat.completions.create.side_effect = Exception("429 Too Many Requests")

            mock_client_success = MagicMock()
            mock_client_success.chat.completions.create.return_value = mock_resp

            mock_openai_cls.side_effect = [mock_client_fail, mock_client_success]

            parsed, prov_used, model_used, lat, trace = client.execute_blind_query(
                unenh_b64="fake_u", enh_b64="fake_e", plate_len=10, provider="nvidia"
            )

            assert parsed is not None
            assert prov_used == "nvidia[NVIDIA_API_KEY_1]"
            assert len(trace) == 2
            assert trace[0]["provider"] == "nvidia[NVIDIA_API_KEY]"
            assert trace[0]["status"] == "FAILED"
            assert "429" in trace[0]["error"]
            assert trace[1]["provider"] == "nvidia[NVIDIA_API_KEY_1]"
            assert trace[1]["status"] == "SUCCESS"


def test_opensource_extractor_multi_key_failover():
    """Verify that Stage 3 opensource_extractor cycles through NVIDIA keys on 429."""
    from backend.opensource_extractor import extract_vehicle_attributes_with_meta

    env_mock = {
        "NVIDIA_API_KEY": "nv_key_0",
        "NVIDIA_API_KEY_1": "nv_key_1",
        "OPENROUTER_API_KEY": "fake_or",
    }

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "body_type": "two-wheeler",
                    "body_type_conf": 0.97,
                    "colour": "white",
                    "colour_conf": 0.95,
                    "make": "Honda",
                    "make_conf": 0.94,
                    "model": "Activa 6G",
                    "model_conf": 0.92,
                })
            )
        )
    ]

    with patch.dict(os.environ, env_mock, clear=True):
        with patch("backend.opensource_extractor.OpenAI") as mock_openai_cls:
            mock_c1 = MagicMock()
            mock_c1.chat.completions.create.side_effect = Exception("429 Rate Limit Exceeded")

            mock_c2 = MagicMock()
            mock_c2.chat.completions.create.return_value = mock_resp

            mock_openai_cls.side_effect = [mock_c1, mock_c2]

            attrs_meta = extract_vehicle_attributes_with_meta(images=[dummy_img])

            assert attrs_meta.provider == "nvidia[NVIDIA_API_KEY_1]"
            assert attrs_meta.make == "Honda"
            assert attrs_meta.model == "Activa 6G"
            assert attrs_meta.colour == "white"
            assert attrs_meta.body_type == "two-wheeler"


def test_select_top_vehicle_crops():
    """Verify crop quality scoring, selection, and contact strip stitching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        t_dir = Path(tmp_dir) / "track_001"
        crops_dir = t_dir / "vehicle_crops_unenhanced"
        crops_dir.mkdir(parents=True)

        # 1. Blurry low-res image (low score)
        blurry = np.full((50, 50, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(crops_dir / "frame_000010_crop.png"), blurry)

        # 2. High-contrast sharp large image (high score)
        sharp_large = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(sharp_large, (50, 50), (250, 250), (255, 255, 255), 4)
        cv2.putText(sharp_large, "HONDA", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(str(crops_dir / "frame_000020_crop.png"), sharp_large)

        # 3. Medium sharp image
        sharp_med = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(sharp_med, (100, 100), 60, (200, 200, 200), 2)
        cv2.imwrite(str(crops_dir / "frame_000030_crop.png"), sharp_med)

        selected = select_top_vehicle_crops(t_dir, top_k=2)
        assert len(selected) == 2
        # Highest score should be among selected
        selected_names = [p.name for p in selected]
        assert "frame_000020_crop.png" in selected_names

        # Test contact strip generation
        strip_path = t_dir / "contact_strip.png"
        out_strip = build_vehicle_crops_contact_strip(selected, strip_path)
        assert out_strip is not None
        assert out_strip.is_file()
        assert out_strip.stat().st_size > 0

        # Verify image dimensions
        img_strip = cv2.imread(str(out_strip))
        assert img_strip is not None
        assert img_strip.shape[0] == 320  # Default target_height
        assert img_strip.shape[1] > 100


def test_vlm_verifier_auto_hierarchy():
    """Verify that auto mode executes Tier 1 (NVIDIA), then Tier 2 (Groq), then Tier 3 (Ollama)."""
    env_mock = {
        "NVIDIA_API_KEY": "nv_key",
        "GROQ_API_KEY": "groq_key",
    }

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "plate_length": 10,
                    "characters": [
                        {"slot_idx": i, "char": "A", "confidence": 0.9, "legible": True, "note": ""}
                        for i in range(10)
                    ],
                    "layout": "single_line",
                    "overall_visual_clarity": "clear",
                })
            )
        )
    ]

    with patch.dict(os.environ, env_mock, clear=True):
        client = VLMClient()

        # Case 1: Tier 1 (NVIDIA) succeeds directly
        with patch("cv_pipeline.vlm_verifier.OpenAI") as mock_openai_cls:
            mock_client_nv = MagicMock()
            mock_client_nv.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client_nv

            parsed, prov_used, model_used, lat, trace = client.execute_blind_query(
                unenh_b64="fake_u", enh_b64="fake_e", plate_len=10, provider="auto"
            )

            assert parsed is not None
            assert "nvidia" in prov_used
            assert trace[0]["tier"] == 1
            assert "nvidia" in trace[0]["provider"]
            assert trace[0]["status"] == "SUCCESS"

        # Case 2: Tier 1 (NVIDIA) fails -> falls back to Tier 2 (Groq)
        with patch("cv_pipeline.vlm_verifier.OpenAI") as mock_openai_cls:
            mock_client_nv = MagicMock()
            mock_client_nv.chat.completions.create.side_effect = Exception("NVIDIA 500 error")

            mock_client_groq = MagicMock()
            mock_client_groq.chat.completions.create.return_value = mock_resp

            mock_openai_cls.side_effect = [mock_client_nv, mock_client_groq]

            parsed, prov_used, model_used, lat, trace = client.execute_blind_query(
                unenh_b64="fake_u", enh_b64="fake_e", plate_len=10, provider="auto"
            )

            assert parsed is not None
            assert prov_used == "groq"
            assert trace[0]["tier"] == 1
            assert "nvidia" in trace[0]["provider"]
            assert trace[0]["status"] == "FAILED"
            assert trace[1]["tier"] == 2
            assert trace[1]["provider"] == "groq"
            assert trace[1]["status"] == "SUCCESS"

