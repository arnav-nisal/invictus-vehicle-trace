"""Pytest Suite for Screen 1, 2, and 3 FastAPI Backend Endpoints and Forensic PDF Generator.

Tests:
1. GET /api/pipeline/clips - list clips in data/Final with metadata & video URLs.
2. GET /api/registry/stats - database vehicle record counts.
3. POST /api/videos/select-local - security (path traversal rejection) & instant cache hit.
4. GET /api/videos/{job_id}/status - job polling schema and ready results.
5. backend.pdf_report: generate_forensic_pdf produces valid publication-grade multi-page PDF.
6. POST /api/pipeline/verify - parameter validation and cache hit execution.
7. GET /api/reports/{job_id}/{track_key}/pdf - PDF download serving & error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.main import app, OUTPUTS_DIR
from backend.pdf_report import generate_forensic_pdf, _resolve_image_path
from cv_pipeline.video_renderer import ensure_tracked_web_video

client = TestClient(app)


def test_list_available_clips(tmp_path, monkeypatch):
    """Verify GET /api/pipeline/clips returns clips from data/Final (or configured clips dir)."""
    if not list(backend_main.CLIPS_DIR.glob("*.mp4")):
        clips_dir = tmp_path / "Final"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / "sample_01.mp4").write_bytes(b"\x00" * 1024)
        monkeypatch.setattr(backend_main, "CLIPS_DIR", clips_dir)

    response = client.get("/api/pipeline/clips")
    assert response.status_code == 200
    clips = response.json()
    assert isinstance(clips, list)
    assert len(clips) >= 1

    first = clips[0]
    assert "id" in first
    assert "filename" in first
    assert "size_bytes" in first
    assert "video_url" in first
    assert first["video_url"].startswith("/media/clips/")
    assert first["size_bytes"] > 0


def test_get_registry_stats():
    """Verify GET /api/registry/stats returns total records count from SQLite."""
    response = client.get("/api/registry/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert isinstance(data["total_records"], int)
    assert data["total_records"] >= 1000


def test_select_local_path_traversal_rejection():
    """Verify POST /api/videos/select-local rejects path traversal attempts with HTTP 400."""
    # Dot-dot traversal
    r1 = client.post("/api/videos/select-local", json={"filename": "../secret.mp4"})
    assert r1.status_code == 400
    assert "traversal" in r1.json()["detail"].lower()

    # Forward slash
    r2 = client.post("/api/videos/select-local", json={"filename": "sub/secret.mp4"})
    assert r2.status_code == 400

    # Backslash
    r3 = client.post("/api/videos/select-local", json={"filename": "sub\\secret.mp4"})
    assert r3.status_code == 400

    # Empty filename
    r4 = client.post("/api/videos/select-local", json={"filename": "   "})
    assert r4.status_code == 400


def test_select_local_missing_file():
    """Verify POST /api/videos/select-local returns 404 for non-existent video."""
    response = client.post("/api/videos/select-local", json={"filename": "non_existent_clip_9999.mp4"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def _create_cached_stage1_fixture(tmp_path: Path, stem: str = "mock_traffic_clip") -> tuple[Path, Path]:
    """Create isolated clips and outputs directories with cached Stage 1 H.264 web artifacts."""
    clips_dir = tmp_path / "Final"
    outputs_dir = tmp_path / "outputs"
    clip_out_dir = outputs_dir / stem
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    (clips_dir / f"{stem}.mp4").write_bytes(b"\x00" * 2048)
    (clip_out_dir / f"{stem}_tracked_web.mp4").write_bytes(b"\x00" * 2048)
    summary = {
        "video_fps": 30.0,
        "video_resolution": "1920x1080",
        "total_frames": 90,
        "tracks": [
            {
                "id": 7,
                "class_name": "car",
                "category": "car",
                "start_time_seconds": 0.2,
                "end_time_seconds": 2.4,
                "frame_count": 66,
            }
        ],
    }
    (clip_out_dir / "tracks_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return clips_dir, outputs_dir


def test_select_local_instant_cache_hit(tmp_path, monkeypatch):
    """Verify POST /api/videos/select-local returns instant ready state when tracked H.264 video exists."""
    clips_dir, outputs_dir = _create_cached_stage1_fixture(tmp_path, "mock_traffic_clip")
    monkeypatch.setattr(backend_main, "CLIPS_DIR", clips_dir)
    monkeypatch.setattr(backend_main, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(backend_main, "JOBS", {})

    response = client.post("/api/videos/select-local", json={"filename": "mock_traffic_clip.mp4"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["stage_step"] == 3
    assert data["progress"] == 100
    assert data["job_id"] == "mock_traffic_clip"

    res = data["result"]
    assert res is not None
    assert res["job_id"] == "mock_traffic_clip"
    assert res["tracked_video_url"] == "/outputs/mock_traffic_clip/mock_traffic_clip_tracked_web.mp4"
    assert res["video_url"] == "/outputs/mock_traffic_clip/mock_traffic_clip_tracked_web.mp4"
    assert res["raw_video_url"] == "/media/clips/mock_traffic_clip.mp4"
    assert res["boxes_url"] == "/outputs/mock_traffic_clip/boxes.json"
    assert isinstance(res["tracks"], list)
    assert len(res["tracks"]) == 1

    track0 = res["tracks"][0]
    assert track0["track_id"] == 7
    assert track0["vehicle_class"] == "car"
    assert "crop_url" in track0
    assert "start_sec" in track0
    assert "end_sec" in track0


def test_select_local_fast_render_tier2(tmp_path, monkeypatch):
    """Verify POST /api/videos/select-local triggers Tier 2 fast render when history exists without web video."""
    clips_dir = tmp_path / "Final"
    outputs_dir = tmp_path / "outputs"
    clip_out_dir = outputs_dir / "tier2_clip"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    (clips_dir / "tier2_clip.mp4").write_bytes(b"\x00" * 2048)
    (clip_out_dir / "tracks_summary.json").write_text(json.dumps({"tracks": []}), encoding="utf-8")
    (clip_out_dir / "track_history.json").write_text(json.dumps({"tracks": {}}), encoding="utf-8")

    submitted = []
    monkeypatch.setattr(backend_main, "CLIPS_DIR", clips_dir)
    monkeypatch.setattr(backend_main, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(backend_main, "JOBS", {})
    monkeypatch.setattr(backend_main.STAGE1_EXECUTOR, "submit", lambda fn, *a: submitted.append((fn.__name__, a)))

    response = client.post("/api/videos/select-local", json={"filename": "tier2_clip.mp4"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["stage_step"] == 2
    assert len(submitted) == 1
    assert submitted[0][0] == "_run_fast_render_job"


def test_video_status_endpoint(tmp_path, monkeypatch):
    """Verify GET /api/videos/{job_id}/status for ready and missing jobs."""
    clips_dir, outputs_dir = _create_cached_stage1_fixture(tmp_path, "mock_status_clip")
    monkeypatch.setattr(backend_main, "CLIPS_DIR", clips_dir)
    monkeypatch.setattr(backend_main, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(backend_main, "JOBS", {})

    # Known ready job on disk
    r_ready = client.get("/api/videos/mock_status_clip/status")
    assert r_ready.status_code == 200
    ready_data = r_ready.json()
    assert ready_data["status"] == "ready"
    assert ready_data["progress"] == 100
    assert ready_data["result"]["tracked_video_url"] == "/outputs/mock_status_clip/mock_status_clip_tracked_web.mp4"

    # Missing job
    r_missing = client.get("/api/videos/non_existent_job_xyz_999/status")
    assert r_missing.status_code == 404


def test_forensic_pdf_generation_and_headers(tmp_path):
    """Verify generate_forensic_pdf creates a publication-grade PDF file with valid header."""
    mock_verification = {
        "case_id": "INV-TEST-TRACK-020",
        "job_id": "file_07",
        "track_key": "track_020",
        "filename": "file_07.mp4",
        "target_id": 20,
        "merged_track_ids": [20],
        "is_merged_track": False,
        "generated_at": "2026-11-24T08:00:00Z",
        "processing_time_sec": 4.12,
        "plate": {
            "final_plate": "HR26DK8337",
            "overall_confidence": 0.94,
            "valid_state_code": True,
            "provider_used": "groq",
            "model_used": "qwen/qwen3.8-27b",
            "reconciled_slots": [
                {
                    "slot_index": 0,
                    "slot_type": "LETTER",
                    "final_char": "H",
                    "final_char_soft_score": 0.98,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "H",
                    "pipe_a_conf": 0.97,
                    "pipe_b_char": "H",
                    "pipe_b_conf": 0.99,
                    "vlm_char": "H",
                    "vlm_conf": 0.98,
                },
                {
                    "slot_index": 1,
                    "slot_type": "LETTER",
                    "final_char": "R",
                    "final_char_soft_score": 0.97,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "R",
                    "pipe_a_conf": 0.96,
                    "pipe_b_char": "R",
                    "pipe_b_conf": 0.98,
                    "vlm_char": "R",
                    "vlm_conf": 0.97,
                },
                {
                    "slot_index": 2,
                    "slot_type": "DIGIT",
                    "final_char": "2",
                    "final_char_soft_score": 0.95,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "2",
                    "pipe_a_conf": 0.94,
                    "pipe_b_char": "2",
                    "pipe_b_conf": 0.96,
                    "vlm_char": "2",
                    "vlm_conf": 0.95,
                },
                {
                    "slot_index": 3,
                    "slot_type": "DIGIT",
                    "final_char": "6",
                    "final_char_soft_score": 0.93,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "6",
                    "pipe_a_conf": 0.92,
                    "pipe_b_char": "6",
                    "pipe_b_conf": 0.94,
                    "vlm_char": "6",
                    "vlm_conf": 0.93,
                },
                {
                    "slot_index": 4,
                    "slot_type": "LETTER",
                    "final_char": "D",
                    "final_char_soft_score": 0.88,
                    "action": "TIE_BREAK_PIPE_A",
                    "ui_tile_tone": "amber",
                    "pipe_a_char": "D",
                    "pipe_a_conf": 0.85,
                    "pipe_b_char": "0",
                    "pipe_b_conf": 0.65,
                    "vlm_char": "D",
                    "vlm_conf": 0.90,
                },
                {
                    "slot_index": 5,
                    "slot_type": "LETTER",
                    "final_char": "K",
                    "final_char_soft_score": 0.94,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "K",
                    "pipe_a_conf": 0.93,
                    "pipe_b_char": "K",
                    "pipe_b_conf": 0.95,
                    "vlm_char": "K",
                    "vlm_conf": 0.94,
                },
                {
                    "slot_index": 6,
                    "slot_type": "DIGIT",
                    "final_char": "8",
                    "final_char_soft_score": 0.91,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "8",
                    "pipe_a_conf": 0.90,
                    "pipe_b_char": "8",
                    "pipe_b_conf": 0.92,
                    "vlm_char": "8",
                    "vlm_conf": 0.91,
                },
                {
                    "slot_index": 7,
                    "slot_type": "DIGIT",
                    "final_char": "3",
                    "final_char_soft_score": 0.96,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "3",
                    "pipe_a_conf": 0.95,
                    "pipe_b_char": "3",
                    "pipe_b_conf": 0.97,
                    "vlm_char": "3",
                    "vlm_conf": 0.96,
                },
                {
                    "slot_index": 8,
                    "slot_type": "DIGIT",
                    "final_char": "3",
                    "final_char_soft_score": 0.95,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "3",
                    "pipe_a_conf": 0.94,
                    "pipe_b_char": "3",
                    "pipe_b_conf": 0.96,
                    "vlm_char": "3",
                    "vlm_conf": 0.95,
                },
                {
                    "slot_index": 9,
                    "slot_type": "DIGIT",
                    "final_char": "7",
                    "final_char_soft_score": 0.92,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                    "pipe_a_char": "7",
                    "pipe_a_conf": 0.91,
                    "pipe_b_char": "7",
                    "pipe_b_conf": 0.93,
                    "vlm_char": "7",
                    "vlm_conf": 0.92,
                },
            ],
            "ghost_predictions": [
                {
                    "candidate_plate": "HR26DK8337",
                    "combined_confidence": 0.942,
                    "rank": 1,
                    "slot_choices": ["H", "R", "2", "6", "D", "K", "8", "3", "3", "7"],
                },
                {
                    "candidate_plate": "HR260K8337",
                    "combined_confidence": 0.781,
                    "rank": 2,
                    "slot_choices": ["H", "R", "2", "6", "0", "K", "8", "3", "3", "7"],
                },
            ],
        },
        "vehicle": {
            "body_type": "Sedan",
            "body_type_conf": 0.92,
            "colour": "white",
            "colour_conf": 0.95,
            "make": "Hyundai",
            "make_conf": 0.88,
            "model": "Verna",
            "model_conf": 0.84,
            "provider_used": "groq",
            "selected_crops": ["crop_001.png", "crop_002.png", "crop_003.png"],
        },
        "registry_matches": [
            {
                "rank": 1,
                "plate_number": "HR26DK8337",
                "total_confidence": 0.96,
                "plate_score": 0.98,
                "visual_score": 0.94,
                "body_type": "car",
                "colour": "white",
                "make": "Hyundai",
                "model": "Verna",
                "owner_name": "Rajesh Kumar",
                "registered_rto": "Gurgaon (HR-26)",
                "registration_year": 2021,
                "sightings": [
                    {
                        "camera_id": "CAM_01",
                        "timestamp": "2026-11-24 07:45:12",
                        "confidence": 0.95,
                    }
                ],
            },
            {
                "rank": 2,
                "plate_number": "HR26DK8331",
                "total_confidence": 0.72,
                "plate_score": 0.74,
                "visual_score": 0.70,
                "body_type": "car",
                "colour": "silver",
                "make": "Hyundai",
                "model": "i20",
                "owner_name": "Amit Sharma",
                "registered_rto": "Gurgaon (HR-26)",
                "registration_year": 2019,
                "sightings": [],
            },
        ],
        "evidence_urls": {
            "best_vehicle_crop_url": None,
            "vehicle_contact_strip_url": None,
            "clearest_plate_crop_url": None,
            "plate_grid_unenhanced_url": None,
            "plate_grid_enhanced_url": None,
        },
        "pdf_report_url": "/api/reports/file_07/track_020/pdf",
    }

    out_pdf = tmp_path / "forensic_report_test.pdf"
    generate_forensic_pdf(mock_verification, out_pdf)

    assert out_pdf.is_file()
    pdf_bytes = out_pdf.read_bytes()
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-1.")


def test_pipeline_verify_request_validation():
    """Verify POST /api/pipeline/verify rejects requests missing both track_ids and track_id."""
    response = client.post("/api/pipeline/verify", json={"job_id": "file_07"})
    assert response.status_code == 400
    assert "either" in response.json()["detail"].lower()


def test_pipeline_verify_cache_hit_and_pdf_download(tmp_path):
    """Verify POST /api/pipeline/verify returns instant cached result and PDF endpoint serves file."""
    # Create temporary mock verification directory under outputs/
    mock_job = "test_clip_001"
    mock_track = "track_099"
    test_track_dir = OUTPUTS_DIR / mock_job / mock_track
    test_track_dir.mkdir(parents=True, exist_ok=True)

    mock_res = {
        "case_id": "INV-TEST-099",
        "job_id": mock_job,
        "track_key": mock_track,
        "filename": "test_clip_001.mp4",
        "target_id": 99,
        "merged_track_ids": [99],
        "is_merged_track": False,
        "generated_at": "2026-11-24T08:00:00Z",
        "processing_time_sec": 1.25,
        "plate": {
            "final_plate": "MH12AB1234",
            "overall_confidence": 0.95,
            "valid_state_code": True,
            "provider_used": "groq",
            "model_used": "qwen/qwen3.8-27b",
            "reconciled_slots": [
                {
                    "slot_index": 0,
                    "slot_type": "LETTER",
                    "final_char": "M",
                    "final_char_soft_score": 0.95,
                    "action": "TRIPLE_CHECK_CONFIRMED",
                    "ui_tile_tone": "green",
                }
            ],
            "ghost_predictions": [],
        },
        "vehicle": {
            "body_type": "car",
            "body_type_conf": 0.9,
            "colour": "white",
            "colour_conf": 0.9,
            "make": "Maruti",
            "make_conf": 0.8,
            "model": "Swift",
            "model_conf": 0.8,
            "provider_used": "groq",
            "selected_crops": [],
        },
        "registry_matches": [],
        "evidence_urls": {},
        "pdf_report_url": f"/api/reports/{mock_job}/{mock_track}/pdf",
    }

    ver_file = test_track_dir / "verification_result.json"
    with open(ver_file, "w", encoding="utf-8") as f:
        json.dump(mock_res, f)

    try:
        # Call verify endpoint - should return cached JSON immediately
        response = client.post(
            "/api/pipeline/verify",
            json={"job_id": mock_job, "track_ids": [99]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["case_id"] == "INV-TEST-099"
        assert data["plate"]["final_plate"] == "MH12AB1234"

        # Verify PDF endpoint generates and serves PDF
        pdf_resp = client.get(f"/api/reports/{mock_job}/{mock_track}/pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert len(pdf_resp.content) > 1000
        assert pdf_resp.content.startswith(b"%PDF-1.")
    finally:
        # Cleanup mock directory
        import shutil
        shutil.rmtree(OUTPUTS_DIR / mock_job, ignore_errors=True)


def test_pdf_resolve_image_path(tmp_path):
    """Verify _resolve_image_path resolves /outputs URLs and nested track subdirectories."""
    track_dir = tmp_path / "outputs" / "demo_clip" / "track_012"
    plate_dir = track_dir / "plate_crops_all" / "unenhanced"
    plate_dir.mkdir(parents=True, exist_ok=True)
    plate_img = plate_dir / "plate_000010.png"
    plate_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    resolved_url = _resolve_image_path(
        "/outputs/demo_clip/track_012/plate_crops_all/unenhanced/plate_000010.png",
        track_dir,
        base_dir=tmp_path,
    )
    assert resolved_url == plate_img

    resolved_rel = _resolve_image_path("plate_crops_all/unenhanced/plate_000010.png", track_dir, base_dir=tmp_path)
    assert resolved_rel == plate_img


def test_video_renderer_pass2_and_transcode(tmp_path):
    """Verify Pass 2 rendering from track_history.json produces raw and H.264 web MP4s."""
    video_path = tmp_path / "tiny_clip.mp4"
    out_dir = tmp_path / "outputs" / "tiny_clip"
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (320, 240))
    for i in range(5):
        frame = np.full((240, 320, 3), 40 + i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    summary = {"tracks": [{"id": 1, "class_name": "car", "category": "car"}]}
    history = {
        "tracks": {
            "1": {
                "id": 1,
                "class_name": "car",
                "category": "car",
                "observations": [
                    {"frame_idx": i, "bbox": [40 + i * 5, 60, 140 + i * 5, 160], "confidence": 0.9}
                    for i in range(5)
                ],
            }
        }
    }
    (out_dir / "tracks_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (out_dir / "track_history.json").write_text(json.dumps(history), encoding="utf-8")

    web_path = ensure_tracked_web_video(video_path, out_dir, "tiny_clip")
    assert web_path.is_file()
    assert web_path.name == "tiny_clip_tracked_web.mp4"
    assert web_path.stat().st_size > 1000
    assert (out_dir / "tiny_clip_tracked.mp4").is_file()
