"""Unit tests for Multi-ID vehicle tracking and 3-frame class-switch hysteresis."""

import json
from pathlib import Path
import numpy as np
import pytest

from cv_pipeline.crop_extractor import parse_target_ids, extract_vehicle_crops
from cv_pipeline.splitter import split_tracks_on_sudden_jumps
from cv_pipeline.vehicle_crop_selector import select_top_vehicle_crops


class DummyCropSelector:
    def __init__(self):
        self._best_candidates = {}

    def get_best(self, tid):
        return None


def test_parse_target_ids():
    """Verify robust parsing of various target ID representations."""
    assert parse_target_ids(26) == [26]
    assert parse_target_ids("26") == [26]
    assert parse_target_ids("26, 64") == [26, 64]
    assert parse_target_ids("26,64") == [26, 64]
    assert parse_target_ids("[26, 64]") == [26, 64]
    assert parse_target_ids([26, 64]) == [26, 64]
    assert parse_target_ids(["26", "64"]) == [26, 64]
    assert parse_target_ids("26, 64, 26") == [26, 64]

    with pytest.raises(TypeError):
        parse_target_ids(None)


def test_multi_id_crop_pooling(tmp_path):
    """Verify chronological pooling and overlap deduplication across multiple IDs."""
    # Create a tiny dummy video
    import cv2
    video_path = tmp_path / "dummy_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for _ in range(25):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    track_history = {
        "tracks": {
            "26": {
                "class_name": "motorcycle",
                "category": "two-wheeler",
                "start_frame": 2,
                "end_frame": 10,
                "observations": [
                    {"frame_idx": 2, "bbox": [100, 100, 200, 200], "confidence": 0.70, "timestamp_sec": 0.067, "timestamp": "00:00:00.067"},
                    {"frame_idx": 6, "bbox": [110, 110, 210, 210], "confidence": 0.60, "timestamp_sec": 0.200, "timestamp": "00:00:00.200"}, # lower conf on overlap
                    {"frame_idx": 10, "bbox": [120, 120, 220, 220], "confidence": 0.75, "timestamp_sec": 0.333, "timestamp": "00:00:00.333"},
                ],
            },
            "64": {
                "class_name": "motorcycle",
                "category": "two-wheeler",
                "start_frame": 6,
                "end_frame": 18,
                "observations": [
                    {"frame_idx": 6, "bbox": [112, 112, 212, 212], "confidence": 0.95, "timestamp_sec": 0.200, "timestamp": "00:00:00.200"}, # higher conf wins!
                    {"frame_idx": 14, "bbox": [130, 130, 230, 230], "confidence": 0.80, "timestamp_sec": 0.467, "timestamp": "00:00:00.467"},
                    {"frame_idx": 18, "bbox": [140, 140, 240, 240], "confidence": 0.85, "timestamp_sec": 0.600, "timestamp": "00:00:00.600"},
                ],
            },
        }
    }

    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=[26, 64],
        output_dir=tmp_path / "outputs",
    )

    assert manifest["is_merged_track"] is True
    assert manifest["merged_track_ids"] == [26, 64]
    assert manifest["track_id"] == 26
    assert manifest["total_crops_saved"] == 5  # frames 2, 6, 10, 14, 18

    # Frame 6 must have used confidence 0.95 from Track 64
    frame_6_crop = next(c for c in manifest["crops"] if c["frame_idx"] == 6)
    assert frame_6_crop["confidence"] == 0.95

    # Directory name check
    manifest_path = tmp_path / "outputs" / "dummy_video" / "track_026_064" / "crop_manifest.json"
    assert manifest_path.is_file()


def test_splitter_class_switch_hysteresis():
    """Verify 3-frame hysteresis prevents splitting on single-frame YOLO class flicker."""
    # Track with 1-frame flicker: motorcycle (f0..f3) -> car (f4) -> motorcycle (f5..f8)
    # Area remains constant to isolate class-switch trigger
    labels_flicker = [
        "motorcycle", "motorcycle", "motorcycle", "motorcycle",
        "car",  # single-frame flicker!
        "motorcycle", "motorcycle", "motorcycle", "motorcycle",
    ]
    frame_obs_flicker = {}
    for i, lbl in enumerate(labels_flicker):
        frame_obs_flicker[i] = [{
            "id": 10,
            "label": lbl,
            "area": 20000.0,
            "bbox": [100, 100, 200, 300],
            "confidence": 0.85,
        }]

    tracks_meta_10 = {
        10: {
            "first_crop": np.zeros((100, 100, 3), dtype=np.uint8),
            "last_crop": np.zeros((100, 100, 3), dtype=np.uint8),
            "head_crops": [],
            "tail_crops": [],
        }
    }
    traj_10 = {10: [(i, 150.0, 200.0) for i in range(len(labels_flicker))]}
    sel_10 = DummyCropSelector()

    meta_res, frame_res, splits = split_tracks_on_sudden_jumps(
        tracks_meta=tracks_meta_10,
        frame_observations=frame_obs_flicker,
        crop_selector=sel_10,
        video_path="dummy.mp4",
        fps=30.0,
    )

    # 1-frame flicker should NOT cause any splits!
    assert len(splits) == 0, f"Expected 0 splits for 1-frame flicker, got {len(splits)}"

    # Track with sustained 3-frame switch: motorcycle (f0..f3) -> car (f4..f8)
    labels_sustained = [
        "motorcycle", "motorcycle", "motorcycle", "motorcycle",
        "car", "car", "car", "car", "car",
    ]
    frame_obs_sustained = {}
    for i, lbl in enumerate(labels_sustained):
        frame_obs_sustained[i] = [{
            "id": 20,
            "label": lbl,
            "area": 20000.0,
            "bbox": [100, 100, 200, 300],
            "confidence": 0.85,
        }]

    tracks_meta_20 = {
        20: {
            "first_crop": np.zeros((100, 100, 3), dtype=np.uint8),
            "last_crop": np.zeros((100, 100, 3), dtype=np.uint8),
            "head_crops": [],
            "tail_crops": [],
        }
    }
    sel_20 = DummyCropSelector()

    meta_res2, frame_res2, splits2 = split_tracks_on_sudden_jumps(
        tracks_meta=tracks_meta_20,
        frame_observations=frame_obs_sustained,
        crop_selector=sel_20,
        video_path="dummy.mp4",
        fps=30.0,
    )

    # Sustained switch SHOULD split!
    assert len(splits2) == 1, f"Expected 1 split for sustained switch, got {len(splits2)}"
    assert splits2[0]["split_frame"] == 4


def test_vlm_verifier_merged_metadata_propagation():
    """Verify reconcile_vlm_with_ocr properly propagates merged metadata into final_plate_result."""
    from cv_pipeline.vlm_verifier import reconcile_vlm_with_ocr

    ocr_mock = {
        "track_id": 26,
        "target_id": 26,
        "merged_track_ids": [26, 64],
        "is_merged_track": True,
        "recovered_plate": "DL8CAF5032",
        "overall_confidence": 0.88,
        "plate_length": 10,
        "slots": [],
    }

    result = reconcile_vlm_with_ocr(ocr_result=ocr_mock, vlm_output=None)
    assert result["is_merged_track"] is True
    assert result["merged_track_ids"] == [26, 64]
    assert result["track_id"] == 26
    assert result["target_id"] == 26


def test_interpolate_track_gaps_midpoint():
    """Verify vid_stride=2 gap (frame 0 & frame 2) produces exact midpoint at frame 1 tagged 'interpolated': True."""
    from cv_pipeline.pipeline import interpolate_track_gaps

    track_obs = [
        {"frame_idx": 0, "bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.80, "id": 1, "label": "car"},
        {"frame_idx": 2, "bbox": [110.0, 120.0, 210.0, 220.0], "confidence": 0.90, "id": 1, "label": "car"},
    ]

    interpolated = interpolate_track_gaps(track_obs, max_gap_frames=6)
    assert len(interpolated) == 3

    # Frame 0
    assert interpolated[0]["frame_idx"] == 0
    assert interpolated[0]["interpolated"] is False
    assert interpolated[0]["bbox"] == [100.0, 100.0, 200.0, 200.0]

    # Frame 1 (interpolated midpoint)
    assert interpolated[1]["frame_idx"] == 1
    assert interpolated[1]["interpolated"] is True
    assert interpolated[1]["bbox"] == [105.0, 110.0, 205.0, 210.0]
    assert interpolated[1]["confidence"] == 0.85

    # Frame 2
    assert interpolated[2]["frame_idx"] == 2
    assert interpolated[2]["interpolated"] is False
    assert interpolated[2]["bbox"] == [110.0, 120.0, 210.0, 220.0]


def test_interpolate_track_gaps_large_gap_rejected():
    """Verify multi-second occlusion gaps (> max_gap_frames) are NOT interpolated."""
    from cv_pipeline.pipeline import interpolate_track_gaps

    track_obs = [
        {"frame_idx": 0, "bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.80, "id": 1},
        {"frame_idx": 20, "bbox": [150.0, 150.0, 250.0, 250.0], "confidence": 0.85, "id": 1},
    ]

    # max_gap_frames = 6 (e.g. vid_stride=2 * 3)
    interpolated = interpolate_track_gaps(track_obs, max_gap_frames=6)
    # Since gap is 20 frames (> 6), no intermediate frames should be added
    assert len(interpolated) == 2
    assert [o["frame_idx"] for o in interpolated] == [0, 20]


def test_crop_extractor_skips_interpolated_frames(tmp_path):
    """Verify crop_extractor skips 'interpolated': True frames, running Stage 2 crops only on true YOLO boxes."""
    import cv2
    video_path = tmp_path / "dummy_interp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    track_history = {
        "tracks": {
            "1": {
                "class_name": "car",
                "category": "car",
                "start_frame": 0,
                "end_frame": 2,
                "observations": [
                    {"frame_idx": 0, "bbox": [50, 50, 150, 150], "confidence": 0.85, "timestamp_sec": 0.0, "timestamp": "00:00:00.000", "interpolated": False},
                    {"frame_idx": 1, "bbox": [55, 55, 155, 155], "confidence": 0.87, "timestamp_sec": 0.033, "timestamp": "00:00:00.033", "interpolated": True}, # should be skipped!
                    {"frame_idx": 2, "bbox": [60, 60, 160, 160], "confidence": 0.90, "timestamp_sec": 0.067, "timestamp": "00:00:00.067", "interpolated": False},
                ],
            }
        }
    }

    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=1,
        output_dir=tmp_path / "outputs",
    )

    # Exactly 2 true crops should be saved (frame 0 and frame 2), skipping frame 1
    assert manifest["total_crops_saved"] == 2
    saved_frames = [c["frame_idx"] for c in manifest["crops"]]
    assert saved_frames == [0, 2]


