"""Video Rendering & H.264 Web Transcoding Engine for Invictus Vehicle Trace.

Provides:
1. Fast, reliable H.264 transcoding (yuv420p + faststart) for HTML5 browser playback.
2. Fast Pass 2 rendering directly from track_history.json & tracks_summary.json without re-running YOLO.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2

from cv_pipeline.annotator import VehicleAnnotator

logger = logging.getLogger("cv_pipeline.video_renderer")


def get_ffmpeg_executable() -> str:
    """Find a working ffmpeg binary on PATH or via imageio_ffmpeg."""
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except ImportError:
        pass

    raise RuntimeError(
        "No ffmpeg executable found on system PATH or via imageio_ffmpeg. "
        "Please install ffmpeg or run 'pip install imageio-ffmpeg'."
    )


def transcode_to_h264(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
) -> Path:
    """Transcode any video into web-playable H.264 MP4 (yuv420p, faststart, ultrafast)."""
    inp = Path(input_video_path).resolve()
    out = Path(output_video_path).resolve()

    if not inp.is_file():
        raise FileNotFoundError(f"Input video to transcode does not exist: {inp}")

    ffmpeg_exe = get_ffmpeg_executable()
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(inp),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(out),
    ]

    logger.info(f"Transcoding video to browser-playable H.264: {inp.name} -> {out.name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"FFmpeg transcoding failed:\n{res.stderr}")
        raise RuntimeError(f"FFmpeg failed with exit code {res.returncode}: {res.stderr[:400]}")

    logger.info(f"Transcode complete: {out} ({out.stat().st_size} bytes)")
    return out


def render_annotated_video_from_history(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    clip_stem: str,
    output_video_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Render smooth Pass 2 annotated video directly from track_history.json & tracks_summary.json.

    Bypasses YOLO Pass 1 inference (~5-15s instead of full detection pipeline).
    Returns (raw_tracked_path, web_tracked_path).
    """
    v_path = Path(video_path).resolve()
    out_dir = Path(output_dir).resolve()
    raw_name = output_video_name or f"{clip_stem}_tracked.mp4"
    web_name = f"{clip_stem}_tracked_web.mp4"

    raw_out = out_dir / raw_name
    web_out = out_dir / web_name

    summ_path = out_dir / "tracks_summary.json"
    hist_path = out_dir / "track_history.json"

    if not summ_path.is_file() or not hist_path.is_file():
        raise FileNotFoundError(f"Missing summary or history for {clip_stem}: {summ_path}, {hist_path}")

    with open(summ_path, "r", encoding="utf-8") as f:
        summ = json.load(f)
    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    kept_track_ids = {int(t["id"]) for t in summ.get("tracks", []) if "id" in t}
    frame_obs: Dict[int, list] = {}

    hist_tracks = hist.get("tracks", {})
    if isinstance(hist_tracks, list):
        tracks_iter = {t.get("id"): t for t in hist_tracks if "id" in t}.items()
    else:
        tracks_iter = hist_tracks.items()

    for tid_str, t_data in tracks_iter:
        try:
            tid = int(t_data.get("id", tid_str))
        except (ValueError, TypeError):
            continue

        if tid not in kept_track_ids:
            continue

        cname = t_data.get("class_name", "vehicle")
        category = t_data.get("category", "vehicle")
        observations = t_data.get("observations") or t_data.get("frame_observations") or []

        for obs in observations:
            f_idx = int(obs.get("frame_idx", 0))
            bbox = obs.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            if f_idx not in frame_obs:
                frame_obs[f_idx] = []

            frame_obs[f_idx].append({
                "id": tid,
                "label": cname,
                "category": category,
                "conf": float(obs.get("confidence", 1.0)),
                "bbox": [float(c) for c in bbox[:4]],
            })

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {v_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale to max 1280px (720p) for high-performance rendering & playback
    scale = min(1.0, 1280.0 / width) if width > 1280 else 1.0
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))
    out_w -= (out_w % 2)
    out_h -= (out_h % 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_out), fourcc, fps, (out_w, out_h))
    annotator = VehicleAnnotator(font_scale=0.6, box_thickness=2)

    logger.info(f"Rendering Pass 2 annotated video ({out_w}x{out_h} @ {fps:.1f} fps) to {raw_out.name}...")

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        valid_dets = frame_obs.get(f_idx, [])
        if scale < 1.0:
            frame_to_draw = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            scaled_dets = []
            for d in valid_dets:
                d_copy = dict(d)
                d_copy["bbox"] = [c * scale for c in d["bbox"]]
                scaled_dets.append(d_copy)
            dets_to_draw = scaled_dets
        else:
            frame_to_draw = frame
            dets_to_draw = valid_dets

        ann = annotator.annotate_frame(
            frame=frame_to_draw,
            tracks=dets_to_draw,
            frame_idx=f_idx,
            total_frames=total_frames,
            fps=fps,
            total_unique_ids=len(kept_track_ids),
        )
        writer.write(ann)

    cap.release()
    writer.release()
    logger.info(f"Raw annotated video rendered: {raw_out}")

    # Transcode raw video to browser-safe H.264
    transcode_to_h264(raw_out, web_out)

    return raw_out, web_out


def ensure_tracked_web_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    clip_stem: str,
) -> Path:
    """Ensure <clip_stem>_tracked_web.mp4 exists, transcoding or rendering if needed."""
    v_path = Path(video_path).resolve()
    out_dir = Path(output_dir).resolve()
    web_out = out_dir / f"{clip_stem}_tracked_web.mp4"
    raw_out = out_dir / f"{clip_stem}_tracked.mp4"

    if web_out.is_file() and web_out.stat().st_size > 1000:
        return web_out

    if raw_out.is_file() and raw_out.stat().st_size > 1000:
        logger.info(f"Raw tracked video exists for {clip_stem}, transcoding to web format...")
        return transcode_to_h264(raw_out, web_out)

    summ_path = out_dir / "tracks_summary.json"
    hist_path = out_dir / "track_history.json"
    if summ_path.is_file() and hist_path.is_file():
        logger.info(f"Summary and history exist for {clip_stem}, executing Pass 2 render...")
        _, web_path = render_annotated_video_from_history(v_path, out_dir, clip_stem)
        return web_path

    raise FileNotFoundError(
        f"Cannot generate web video: missing both video and track history for {clip_stem}"
    )
