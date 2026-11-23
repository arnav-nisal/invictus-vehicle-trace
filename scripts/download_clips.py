#!/usr/bin/env python3
"""Download and prepare traffic clips for vehicle tracking pipeline validation."""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import yt_dlp

CLIPS_DIR = Path("data/clips")
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def download_video(url: str, output_path: Path) -> None:
    """Download best mp4 video stream up to 1080p using yt-dlp."""
    print(f"\n[INFO] Downloading '{url}' -> '{output_path}'...")
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "outtmpl": str(output_path.with_suffix("")),  # yt-dlp adds extension
        "ffmpeg_location": FFMPEG_EXE,
        "merge_output_format": "mp4",
        "overwrites": True,
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Ensure output exists with exact name
    final_file = output_path
    if not final_file.exists():
        candidates = list(output_path.parent.glob(f"{output_path.stem}.*"))
        if candidates:
            candidates[0].rename(final_file)


def extract_clip(src_video: Path, dst_clip: Path, start_sec: float, end_sec: float) -> None:
    """Extract a sub-clip using ffmpeg stream copy or frame re-encoding."""
    print(f"\n[INFO] Extracting segment [{start_sec}s -> {end_sec}s] from '{src_video}' -> '{dst_clip}'...")
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-ss",
        str(start_sec),
        "-to",
        str(end_sec),
        "-i",
        str(src_video),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        str(dst_clip),
    ]
    subprocess.run(cmd, check=True)


def verify_clip(path: Path) -> None:
    """Verify video file properties with OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total_frames / fps if fps > 0 else 0.0
    cap.release()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [OK] {path.name}: {w}x{h} @ {fps:.1f} FPS | {total_frames} frames ({dur:.2f}s) | {size_mb:.2f} MB")


def main():
    print("=" * 80)
    print("  DOWNLOADING TEST CLIPS FOR VEHICLE TRACKING PIPELINE")
    print("=" * 80)

    # 1. Full intersection video (_5nGL47gNzo)
    full_intersection = CLIPS_DIR / "full_intersection.mp4"
    if not full_intersection.exists() or full_intersection.stat().st_size < 1000:
        download_video("https://www.youtube.com/watch?v=_5nGL47gNzo", full_intersection)
    verify_clip(full_intersection)

    # 2. Extract 15s segment: 00:00:25 to 00:00:40 -> intersection_traffic.mp4
    intersection_clip = CLIPS_DIR / "intersection_traffic.mp4"
    extract_clip(
        src_video=full_intersection,
        dst_clip=intersection_clip,
        start_sec=25.0,
        end_sec=40.0,
    )
    verify_clip(intersection_clip)

    # 3. Highway video (tQnVX3nj3Co) -> highway_traffic.mp4
    highway_clip = CLIPS_DIR / "highway_traffic.mp4"
    if not highway_clip.exists() or highway_clip.stat().st_size < 1000:
        download_video("https://www.youtube.com/watch?v=tQnVX3nj3Co", highway_clip)
    verify_clip(highway_clip)

    print("\n" + "=" * 80)
    print("  ALL TEST CLIPS DOWNLOADED AND VERIFIED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
