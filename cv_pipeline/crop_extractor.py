"""Stage 2: Full-track vehicle crop extraction and interactive selection workflow."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cv2

logger = logging.getLogger(__name__)


def extract_vehicle_crops(
    video_path: Union[str, Path],
    track_history: Dict[str, Any],
    target_id: int,
    output_dir: Union[str, Path] = "outputs",
    padding_pct: float = 0.05,
) -> Dict[str, Any]:
    """Extract and save all cropped images for a specified vehicle track across every frame it appears in.

    Applies a ~5% safety margin around bounding box edges so bumper-edge license plates are never clipped.
    Saves chronological crop images and a clean metadata crop manifest (crop_manifest.json).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

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
    clip_name = video_path.stem
    output_dir = Path(output_dir)
    if output_dir.name == clip_name:
        track_dir = output_dir / f"track_{target_id:03d}"
    else:
        track_dir = output_dir / clip_name / f"track_{target_id:03d}"

    crops_dir = track_dir / "vehicle_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manifest_crops = []
    logger.info(
        f"Extracting {len(observations)} crops for Track {target_id} ({track_info['class_name']}) with {padding_pct*100:.0f}% padding..."
    )

    for idx, obs in enumerate(observations):
        f_idx = obs["frame_idx"]
        if f_idx >= total_video_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"Could not read frame {f_idx} from {video_path}")
            continue

        bbox = obs["bbox"]  # [x1, y1, x2, y2]
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))

        # Calculate safety margin padding (~5%)
        pad_x = bw * padding_pct
        pad_y = bh * padding_pct

        crop_x1 = max(0, int(round(bbox[0] - pad_x)))
        crop_y1 = max(0, int(round(bbox[1] - pad_y)))
        crop_x2 = min(video_w, int(round(bbox[2] + pad_x)))
        crop_y2 = min(video_h, int(round(bbox[3] + pad_y)))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            logger.warning(f"Invalid crop dimensions for frame {f_idx}: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}]")
            continue

        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_filename = f"frame_{f_idx:06d}_crop.jpg"
        crop_file_path = crops_dir / crop_filename

        cv2.imwrite(str(crop_file_path), crop_img)

        manifest_entry = {
            "crop_index": idx,
            "frame_idx": int(f_idx),
            "timestamp_sec": float(obs.get("timestamp_sec", 0.0)),
            "timestamp": obs.get("timestamp", ""),
            "original_bbox": [float(c) for c in bbox],
            "padded_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
            "padding_percent": round(padding_pct * 100.0, 1),
            "crop_image_path": str(crop_file_path.as_posix()),
            "crop_width": int(crop_x2 - crop_x1),
            "crop_height": int(crop_y2 - crop_y1),
            "confidence": float(obs.get("confidence", 0.0)),
            "area": float(obs.get("area", bw * bh)),
        }
        manifest_crops.append(manifest_entry)

    cap.release()

    manifest_data = {
        "track_id": int(target_id),
        "class_name": track_info.get("class_name", "vehicle"),
        "category": track_info.get("category", "vehicle"),
        "start_frame": track_info.get("start_frame"),
        "end_frame": track_info.get("end_frame"),
        "start_timestamp": track_info.get("start_timestamp"),
        "end_timestamp": track_info.get("end_timestamp"),
        "total_crops_saved": len(manifest_crops),
        "video_source": str(video_path),
        "padding_percent": round(padding_pct * 100.0, 1),
        "crops_dir": str(crops_dir.as_posix()),
        "original_ids": track_info.get("original_ids", [target_id]),
        "crops": manifest_crops,
    }

    manifest_path = track_dir / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Saved {len(manifest_crops)} vehicle crops into '{crops_dir}' and manifest to '{manifest_path}'."
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


def interactive_vehicle_selection(
    summary_data: Dict[str, Any],
    track_history: Dict[str, Any],
    video_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    target_id: Optional[int] = None,
    padding_pct: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Present the annotated video path and vehicle list, prompt user for target ID (or use provided ID),

    and extract the full chronological crops and manifest.
    """
    video_path = Path(video_path)
    output_video_path = summary_data.get("output_video_path", "")

    print("\n" + "#" * 88)
    print("  STAGE 2: VEHICLE SELECTION & FULL-TRACK CROP EXTRACTION")
    print("#" * 88)
    if output_video_path:
        print(f"  Annotated Tracked Video:  {output_video_path}")
    print(f"  Source Input Video:       {video_path}")
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

    print(f"\nProcessing full-track crops for selected Vehicle Track ID: {chosen_id}...")
    manifest = extract_vehicle_crops(
        video_path=video_path,
        track_history=track_history,
        target_id=chosen_id,
        output_dir=output_dir,
        padding_pct=padding_pct,
    )

    print("\n" + "=" * 88)
    print(f"  CROP EXTRACTION COMPLETE FOR TRACK {chosen_id} ({manifest['class_name'].upper()})")
    print("=" * 88)
    print(f"  Total Crops Saved:   {manifest['total_crops_saved']} images")
    print(f"  Time Active:         {manifest['start_timestamp']} -> {manifest['end_timestamp']}")
    print(f"  Safety Margin Pad:   {manifest['padding_percent']}%")
    print(f"  Crops Directory:     {manifest['crops_dir']}")
    print(f"  Crop Manifest File:  {manifest['crops_dir']}/../crop_manifest.json")
    print("=" * 88 + "\n")

    return manifest
