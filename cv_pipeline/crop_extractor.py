"""Stage 2: Full-track vehicle crop extraction, adaptive Lanczos upscaling, and clarity enhancement workflow."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_sharpness(img: np.ndarray) -> float:
    """Computes Laplacian variance on grayscale image as a measure of visual focus/sharpness."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def upscale_crop_lanczos(
    raw_crop: np.ndarray,
    min_target_dimension: int = 480,
    max_scale_factor: float = 4.0,
) -> Tuple[np.ndarray, float]:
    """Applies adaptive Lanczos4 upscaling to a raw crop.

    If the longer side is < 480px, scales up using cv2.INTER_LANCZOS4 so the longer side
    reaches 480px, capped at max_scale_factor (4.0x). Crops already >= 480px remain at 1.0x.
    Applies zero post-processing (untouched pixel values).

    Returns:
        upscaled_crop (np.ndarray): The resized image with untouched pixel values.
        scale_factor (float): The actual scale factor applied.
    """
    if raw_crop is None or raw_crop.size == 0:
        raise ValueError("Cannot upscale empty image crop.")

    raw_h, raw_w = raw_crop.shape[:2]
    max_side = max(raw_w, raw_h)

    if max_side < min_target_dimension:
        target_scale = min(max_scale_factor, float(min_target_dimension) / float(max_side))
        new_w = max(1, int(round(raw_w * target_scale)))
        new_h = max(1, int(round(raw_h * target_scale)))
        upscaled = cv2.resize(raw_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        scale_factor = round(target_scale, 3)
    else:
        upscaled = raw_crop.copy()
        scale_factor = 1.0

    return upscaled, scale_factor


def apply_enhancement_pipeline(
    crop_img: np.ndarray,
    clahe_clip_limit: float = 1.8,
    clahe_tile_grid: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Applies edge-preserving bilateral filtering, luminance-only CLAHE, and controlled
    unsharp masking to an image.
    """
    if crop_img is None or crop_img.size == 0:
        raise ValueError("Cannot enhance empty image crop.")

    # 1. Edge-preserving bilateral filter (cleans video compression noise without smearing text)
    denoised = cv2.bilateralFilter(crop_img, d=5, sigmaColor=30, sigmaSpace=30)

    # 2. Luminance-only CLAHE (brings out shadowed plates without shifting vehicle color)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    contrast_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # 3. Controlled Unsharp Mask (sigma=1.0, blended 1.35 * img - 0.35 * blurred)
    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=1.0)
    enhanced = np.clip(
        1.35 * contrast_enhanced.astype(np.float32) - 0.35 * blurred.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return enhanced


def extract_vehicle_crops(
    video_path: Union[str, Path],
    track_history: Dict[str, Any],
    target_id: int,
    output_dir: Union[str, Path] = "outputs",
    padding_pct: float = 0.08,
    crop_video_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Extract and save all cropped images for a specified vehicle track across every frame it appears in.

    Saves crops into three lossless .png subfolders:
      - vehicle_crops_raw/: Untouched original pixels cropped with 8% safety margin padding.
      - vehicle_crops_unenhanced/: Enlarged with adaptive Lanczos4 upscaling (zero post-processing).
      - vehicle_crops_enhanced/: Enlarged with identical scaling + bilateral filter + luminance CLAHE + unsharp mask.

    Supports optional high-resolution source mapping via --crop-video.
    """
    track_video_path = Path(video_path)
    if not track_video_path.exists():
        raise FileNotFoundError(f"Tracking video file not found: {track_video_path}")

    # Determine crop source video (high-res or original tracking video)
    if crop_video_path is not None:
        actual_crop_video_path = Path(crop_video_path)
        if not actual_crop_video_path.exists():
            raise FileNotFoundError(f"Specified high-res crop video not found: {actual_crop_video_path}")
    else:
        actual_crop_video_path = track_video_path

    tracks_dict = track_history.get("tracks", {})
    target_str_id = str(target_id)
    if target_str_id not in tracks_dict:
        available_ids = sorted([int(k) for k in tracks_dict.keys()])
        raise ValueError(
            f"Target ID {target_id} not found in track history. Available IDs: {available_ids}"
        )

    track_info = tracks_dict[target_str_id]
    observations = track_info.get("observations", [])
    if not observations:
        raise ValueError(f"Track {target_id} has no recorded frame observations.")

    # Determine dedicated output folder for this track: outputs/<clip_name>/track_<ID>/
    clip_name = track_video_path.stem
    output_dir = Path(output_dir)
    if output_dir.name == clip_name:
        track_dir = output_dir / f"track_{target_id:03d}"
    else:
        track_dir = output_dir / clip_name / f"track_{target_id:03d}"

    raw_crops_dir = track_dir / "vehicle_crops_raw"
    unenhanced_crops_dir = track_dir / "vehicle_crops_unenhanced"
    enhanced_crops_dir = track_dir / "vehicle_crops_enhanced"
    raw_crops_dir.mkdir(parents=True, exist_ok=True)
    unenhanced_crops_dir.mkdir(parents=True, exist_ok=True)
    enhanced_crops_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = track_dir / "crop_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            if len(existing_manifest.get("crops", [])) == len(observations):
                logger.info(
                    f"Reusing existing {len(observations)} vehicle crops from {manifest_file}..."
                )
                return existing_manifest
        except Exception as e:
            logger.warning(f"Could not reuse existing crop manifest: {e}. Re-extracting...")

    # Get tracking video dimensions for coordinate normalization
    track_cap = cv2.VideoCapture(str(track_video_path))
    if not track_cap.isOpened():
        raise RuntimeError(f"Failed to open tracking video: {track_video_path}")
    track_w = int(track_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    track_h = int(track_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    track_fps = float(track_cap.get(cv2.CAP_PROP_FPS))
    track_cap.release()

    # Open crop video source
    crop_cap = cv2.VideoCapture(str(actual_crop_video_path))
    if not crop_cap.isOpened():
        raise RuntimeError(f"Failed to open crop video source: {actual_crop_video_path}")
    crop_w = int(crop_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(crop_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_fps = float(crop_cap.get(cv2.CAP_PROP_FPS))
    crop_total_frames = int(crop_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    is_high_res_mapping = actual_crop_video_path != track_video_path
    if is_high_res_mapping:
        logger.info(
            f"High-Resolution Source Mapping enabled: Tracking ({track_w}x{track_h}) -> Crop Source ({crop_w}x{crop_h} @ {crop_fps:.1f}fps)"
        )

    manifest_crops = []
    sharpness_scores = []
    usable_count = 0

    logger.info(
        f"Extracting {len(observations)} crops for Track {target_id} ({track_info['class_name']}) with {padding_pct*100:.0f}% padding across 3 tiers..."
    )

    curr_crop_frame_idx = -1
    curr_frame_image = None

    for idx, obs in enumerate(observations):
        track_f_idx = obs["frame_idx"]
        ts_sec = float(obs.get("timestamp_sec", track_f_idx / (track_fps if track_fps > 0 else 30.0)))

        # Determine target frame on crop video
        if is_high_res_mapping:
            target_crop_f_idx = int(round(ts_sec * crop_fps))
        else:
            target_crop_f_idx = track_f_idx

        target_crop_f_idx = max(0, min(crop_total_frames - 1, target_crop_f_idx))

        # Optimized video frame reading (avoids redundant/slow seeking)
        if target_crop_f_idx == curr_crop_frame_idx and curr_frame_image is not None:
            frame = curr_frame_image
        elif target_crop_f_idx == curr_crop_frame_idx + 1:
            ret, frame = crop_cap.read()
            if not ret or frame is None:
                continue
            curr_crop_frame_idx = target_crop_f_idx
            curr_frame_image = frame
        elif 0 < target_crop_f_idx - curr_crop_frame_idx <= 12:
            # Short forward gap: grab without full decode
            for _ in range(target_crop_f_idx - curr_crop_frame_idx - 1):
                crop_cap.grab()
            ret, frame = crop_cap.read()
            if not ret or frame is None:
                continue
            curr_crop_frame_idx = target_crop_f_idx
            curr_frame_image = frame
        else:
            crop_cap.set(cv2.CAP_PROP_POS_FRAMES, target_crop_f_idx)
            ret, frame = crop_cap.read()
            if not ret or frame is None:
                continue
            curr_crop_frame_idx = target_crop_f_idx
            curr_frame_image = frame

        # Map bounding box coordinates from tracking space to crop space
        bbox = obs["bbox"]  # [x1, y1, x2, y2]
        if is_high_res_mapping:
            norm_x1 = bbox[0] / float(track_w)
            norm_y1 = bbox[1] / float(track_h)
            norm_x2 = bbox[2] / float(track_w)
            norm_y2 = bbox[3] / float(track_h)
            bx1 = norm_x1 * float(crop_w)
            by1 = norm_y1 * float(crop_h)
            bx2 = norm_x2 * float(crop_w)
            by2 = norm_y2 * float(crop_h)
        else:
            bx1, by1, bx2, by2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        # Apply safety margin padding (default 8%)
        pad_x = bw * padding_pct
        pad_y = bh * padding_pct

        crop_x1 = max(0, int(round(bx1 - pad_x)))
        crop_y1 = max(0, int(round(by1 - pad_y)))
        crop_x2 = min(crop_w, int(round(bx2 + pad_x)))
        crop_y2 = min(crop_h, int(round(by2 + pad_y)))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            logger.warning(f"Invalid crop dimensions for frame {track_f_idx}: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}]")
            continue

        # 1. Raw untouched crop
        raw_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        raw_h, raw_w = raw_crop.shape[:2]

        # Sharpness score on raw crop
        sharpness = round(compute_sharpness(raw_crop), 2)
        sharpness_scores.append(sharpness)

        # Quality flag
        quality_flag = "usable" if (raw_w >= 40 and raw_h >= 40) else "too_small"
        if quality_flag == "usable":
            usable_count += 1

        # 2. Unenhanced crop (Adaptive Lanczos upscaling ONLY, zero post-processing)
        unenhanced_crop, scale_factor = upscale_crop_lanczos(
            raw_crop,
            min_target_dimension=480,
            max_scale_factor=4.0,
        )
        enh_h, enh_w = unenhanced_crop.shape[:2]

        # 3. Enhanced crop (Exact same dimensions + Bilateral + Luminance CLAHE + Unsharp Mask)
        enhanced_crop = apply_enhancement_pipeline(
            unenhanced_crop,
            clahe_clip_limit=1.8,
            clahe_tile_grid=(4, 4),
        )

        # File naming: lossless .png format
        crop_filename = f"frame_{track_f_idx:06d}_crop.png"
        raw_crop_path = raw_crops_dir / crop_filename
        unenhanced_crop_path = unenhanced_crops_dir / crop_filename
        enhanced_crop_path = enhanced_crops_dir / crop_filename

        cv2.imwrite(str(raw_crop_path), raw_crop)
        cv2.imwrite(str(unenhanced_crop_path), unenhanced_crop)
        cv2.imwrite(str(enhanced_crop_path), enhanced_crop)

        manifest_entry = {
            "crop_index": idx,
            "frame_idx": int(track_f_idx),
            "crop_frame_idx": int(target_crop_f_idx),
            "timestamp_sec": round(ts_sec, 3),
            "timestamp": obs.get("timestamp", ""),
            "raw_bbox": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
            "padded_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
            "padding_percent": round(padding_pct * 100.0, 1),
            "raw_width": int(raw_w),
            "raw_height": int(raw_h),
            "scale_factor": float(scale_factor),
            "enhanced_width": int(enh_w),
            "enhanced_height": int(enh_h),
            "sharpness_score": float(sharpness),
            "quality_flag": quality_flag,
            "raw_crop_path": str(raw_crop_path.as_posix()),
            "raw_path": str(raw_crop_path.as_posix()),
            "unenhanced_crop_path": str(unenhanced_crop_path.as_posix()),
            "unenhanced_path": str(unenhanced_crop_path.as_posix()),
            "enhanced_crop_path": str(enhanced_crop_path.as_posix()),
            "enhanced_path": str(enhanced_crop_path.as_posix()),
            "confidence": float(obs.get("confidence", 0.0)),
            "area": round(float(bw * bh), 1),
        }
        manifest_crops.append(manifest_entry)

    crop_cap.release()

    avg_sharpness = round(sum(sharpness_scores) / len(sharpness_scores), 2) if sharpness_scores else 0.0

    manifest_data = {
        "track_id": int(target_id),
        "class_name": track_info.get("class_name", "vehicle"),
        "category": track_info.get("category", "vehicle"),
        "start_frame": track_info.get("start_frame"),
        "end_frame": track_info.get("end_frame"),
        "start_timestamp": track_info.get("start_timestamp"),
        "end_timestamp": track_info.get("end_timestamp"),
        "total_crops_saved": len(manifest_crops),
        "usable_crops_count": usable_count,
        "avg_sharpness_score": avg_sharpness,
        "video_source": str(track_video_path),
        "crop_video_source": str(actual_crop_video_path),
        "padding_percent": round(padding_pct * 100.0, 1),
        "raw_crops_dir": str(raw_crops_dir.as_posix()),
        "unenhanced_crops_dir": str(unenhanced_crops_dir.as_posix()),
        "enhanced_crops_dir": str(enhanced_crops_dir.as_posix()),
        "original_ids": track_info.get("original_ids", [target_id]),
        "crops": manifest_crops,
    }

    manifest_path = track_dir / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Saved {len(manifest_crops)} lossless crops across 3 folders ('{raw_crops_dir}', '{unenhanced_crops_dir}', '{enhanced_crops_dir}'). Manifest: '{manifest_path}'."
    )
    return manifest_data


def display_detected_vehicles_table(summary_data: Dict[str, Any]):
    """Print a concise table of tracked vehicles to help the user select a vehicle ID."""
    tracks = summary_data.get("tracks", [])
    print("\n" + "=" * 88)
    print(f"  DETECTED VEHICLES SUMMARY ({len(tracks)} Clean Vehicles Available)")
    print("=" * 88)
    print(f" {'ID':>4} | {'Class':<12} | {'Category':<12} | {'Frames':>6} | {'Duration':>8} | {'Time Range':^25} | {'Status'}")
    print("-" * 88)

    for t in tracks:
        tid = t.get("id")
        cname = t.get("class_name", "")
        cat = t.get("category", "")
        fcnt = t.get("frame_count", 0)
        dur = f"{t.get('duration_seconds', 0.0):.2f}s"
        st = t.get("start_timestamp", "")
        et = t.get("end_timestamp", "")
        timerange = f"{st} -> {et}"
        orig = t.get("original_ids", [tid])
        status = f"Stitched {orig}" if len(orig) > 1 else "Single Track"
        print(f" {tid:>4} | {cname:<12} | {cat:<12} | {fcnt:>6} | {dur:>8} | {timerange:^25} | {status}")
    print("=" * 88 + "\n")


def display_crop_comparison_summary(manifest: Dict[str, Any]):
    """Print a clear summary comparing raw vs. unenhanced vs. enhanced crop dimensions and sharpness metrics."""
    crops = manifest.get("crops", [])
    if not crops:
        return

    print("\n" + "=" * 105)
    print(f"  STAGE 2 THREE-TIER CROP QUALITY REPORT: TRACK {manifest['track_id']} ({manifest['class_name'].upper()})")
    print("=" * 105)
    print(f"  Total Crops:        {manifest['total_crops_saved']} frames (Lossless .png)")
    print(f"  Usable Frames:      {manifest['usable_crops_count']} / {manifest['total_crops_saved']}")
    print(f"  Average Sharpness:  {manifest['avg_sharpness_score']:.1f} (Laplacian Var on Raw)")
    print(f"  Crop Source Video:  {manifest['crop_video_source']}")
    print(f"  Raw Crops Dir:      {manifest['raw_crops_dir']}")
    print(f"  Unenhanced Dir:     {manifest['unenhanced_crops_dir']}")
    print(f"  Enhanced Crops Dir: {manifest['enhanced_crops_dir']}")
    print("-" * 105)
    print(f" {'Idx':>3} | {'Frame':>6} | {'Time':^10} | {'Raw WxH':^11} | {'Scale':>6} | {'Unenh/Enh WxH':^15} | {'Sharpness':>9} | {'Quality':^9}")
    print("-" * 105)

    # Sample representative frames: first, quartiles, last, and peak sharpness
    num_crops = len(crops)
    if num_crops <= 8:
        sample_indices = list(range(num_crops))
    else:
        sample_indices = [
            0,
            num_crops // 4,
            num_crops // 2,
            3 * num_crops // 4,
            num_crops - 1,
        ]
        # Also include highest sharpness crop
        best_sharp_idx = max(range(num_crops), key=lambda i: crops[i]["sharpness_score"])
        if best_sharp_idx not in sample_indices:
            sample_indices.append(best_sharp_idx)
        sample_indices.sort()

    for idx in sample_indices:
        c = crops[idx]
        raw_dim = f"{c['raw_width']}x{c['raw_height']}"
        enh_dim = f"{c['enhanced_width']}x{c['enhanced_height']}"
        scale_str = f"{c['scale_factor']:.2f}x"
        sharp_str = f"{c['sharpness_score']:.1f}"
        print(
            f" {c['crop_index']:>3} | {c['frame_idx']:>6} | {c['timestamp']:^10} | {raw_dim:^11} | "
            f"{scale_str:>6} | {enh_dim:^15} | {sharp_str:>9} | {c['quality_flag']:^9}"
        )

    print("=" * 105 + "\n")


def interactive_vehicle_selection(
    summary_data: Dict[str, Any],
    track_history: Dict[str, Any],
    video_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    target_id: Optional[int] = None,
    padding_pct: float = 0.08,
    crop_video_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Present the annotated video path and vehicle list, prompt user for target ID (or use provided ID),
    and extract the full chronological crops and manifest across all three tiers.
    """
    video_path = Path(video_path)
    output_video_path = summary_data.get("output_video_path", "")

    print("\n" + "#" * 88)
    print("  STAGE 2: VEHICLE SELECTION & LOSSLESS THREE-TIER CROP EXTRACTION")
    print("#" * 88)
    if output_video_path:
        print(f"  Annotated Tracked Video:  {output_video_path}")
    print(f"  Source Tracking Video:    {video_path}")
    if crop_video_path:
        print(f"  High-Res Crop Video:      {crop_video_path}")
    print("#" * 88)

    display_detected_vehicles_table(summary_data)

    available_ids = set()
    for t in summary_data.get("tracks", []):
        available_ids.add(int(t["id"]))

    chosen_id = target_id

    # If target_id is not passed, prompt interactively
    if chosen_id is None:
        while True:
            try:
                user_input = input(
                    "Enter the Tracking ID spotted in the video to extract all crops (or 'q' to quit): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSelection canceled.")
                return None

            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting vehicle crop extraction.")
                return None

            if not user_input.isdigit():
                print("Please enter a valid numeric tracking ID.")
                continue

            parsed_id = int(user_input)
            if parsed_id not in available_ids:
                print(f"Track ID {parsed_id} is not in the detected list. Available IDs: {sorted(available_ids)}")
                continue

            chosen_id = parsed_id
            break

    if chosen_id not in available_ids:
        raise ValueError(
            f"Specified target ID {chosen_id} is not in detected vehicle tracks. Available IDs: {sorted(available_ids)}"
        )

    print(f"\nProcessing lossless full-track crops across 3 tiers for selected Vehicle Track ID: {chosen_id}...")
    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=chosen_id,
        output_dir=output_dir,
        padding_pct=padding_pct,
        crop_video_path=crop_video_path,
    )

    display_crop_comparison_summary(manifest)
    return manifest
