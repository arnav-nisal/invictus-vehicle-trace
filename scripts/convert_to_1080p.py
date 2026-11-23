#!/usr/bin/env python3
"""Convert any video to 1080p (1920x1080) @ 30 FPS using OpenCV with progress reporting."""

import argparse
import time
from pathlib import Path
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Convert video to 1080p @ 30 FPS.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/clips/simple_traffic.mp4",
        help="Path to source video file (default: data/clips/simple_traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/clips/simple_traffic_1080p.mp4",
        help="Path to output 1080p video file (default: data/clips/simple_traffic_1080p.mp4)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target frame rate (default: 30.0)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional max seconds to convert (default: None for full video)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Source video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = total_src_frames / src_fps if src_fps > 0 else 0

    target_w, target_h = 1920, 1080
    target_fps = args.target_fps

    if args.max_seconds is not None:
        target_frames = min(total_src_frames, int(target_fps * args.max_seconds))
    else:
        target_frames = total_src_frames

    print("=" * 80)
    print("  CONVERTING VIDEO TO 1080p @ 30 FPS")
    print("=" * 80)
    print(f" Source:      {input_path} ({src_w}x{src_h} @ {src_fps:.1f} FPS, {total_src_frames} frames / {src_duration:.1f}s)")
    print(f" Target:      {output_path} ({target_w}x{target_h} @ {target_fps:.1f} FPS, {target_frames} frames / {target_frames/target_fps:.1f}s)")
    print("=" * 80)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (target_w, target_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to initialize VideoWriter for: {output_path}")

    t0 = time.time()
    frame_count = 0

    while cap.isOpened() and frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"\nStream reached end at frame {frame_count}")
            break

        # Area interpolation produces crisp downsampling from 4K to 1080p
        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        writer.write(frame_resized)
        frame_count += 1

        if frame_count % 150 == 0 or frame_count == target_frames:
            elapsed = time.time() - t0
            fps_proc = frame_count / elapsed if elapsed > 0 else 0
            eta = (target_frames - frame_count) / fps_proc if fps_proc > 0 else 0
            pct = (frame_count / target_frames) * 100
            print(f"[{pct:5.1f}%] Frame {frame_count}/{target_frames} ({frame_count/target_fps:5.1f}s) | Speed: {fps_proc:5.1f} FPS | ETA: {eta:4.0f}s")

    cap.release()
    writer.release()

    total_time = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print(f" Conversion Complete in {total_time:.1f}s ({frame_count/total_time:.1f} FPS average)")
    print(f" Output Video: {output_path} ({size_mb:.2f} MB, {frame_count} frames)")
    print("=" * 80)


if __name__ == "__main__":
    main()
