"""Top vehicle crop selection and Stage 3 attribute extraction integration."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("cv_pipeline.vehicle_crop_selector")


def _compute_crop_quality_score(img: np.ndarray) -> float:
    """Compute combined visual quality score for full vehicle crop.

    Combines:
    - Laplacian variance (focus / high-frequency texture sharpness)
    - Effective resolution / area (square root of pixel area so large clear crops beat tiny distant ones)
    """
    if img is None or img.size == 0:
        return 0.0
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Area penalty for extreme aspect ratios or miniature boxes
    area = float(h * w)
    # Composite score: sqrt(area) * log1p(lap_var)
    # Rewards clear full-frame vehicles while penalizing motion-blurred or distant thumbnails
    score = (area ** 0.5) * np.log1p(max(0.0, lap_var))
    return float(score)


def select_top_vehicle_crops(track_dir: Union[str, Path], top_k: int = 3) -> List[Path]:
    """Select the top-k highest quality, temporally diverse vehicle crops for a track.

    Looks in:
      1. <track_dir>/vehicle_crops_unenhanced
      2. <track_dir>/vehicle_crops_raw
      3. <track_dir>/vehicle_crops_enhanced
      4. <track_dir>/vehicle_crops

    Returns a list of Path objects (up to top_k) sorted chronologically.
    """
    t_dir = Path(track_dir).resolve()

    search_dirs = [
        t_dir / "vehicle_crops_unenhanced",
        t_dir / "vehicle_crops_raw",
        t_dir / "vehicle_crops_enhanced",
        t_dir / "vehicle_crops",
    ]

    crops_dir: Optional[Path] = None
    for d in search_dirs:
        if d.is_dir() and any(d.glob("*.png")):
            crops_dir = d
            break
        elif d.is_dir() and any(d.glob("*.jpg")):
            crops_dir = d
            break

    if not crops_dir:
        # Fallback: check if track directory itself has any crop images
        all_imgs = [p for p in t_dir.glob("*.png") if "plate" not in p.name.lower() and "grid" not in p.name.lower()]
        if not all_imgs:
            logger.warning(f"No vehicle crops found in {t_dir}")
            return []
        img_files = all_imgs
    else:
        img_files = sorted(
            [p for p in crops_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
            key=lambda x: x.name,
        )

    if not img_files:
        return []

    # Score each crop
    scored: List[Tuple[Path, int, float]] = []
    for p in img_files:
        # Parse frame index if present
        m = re.search(r"(\d+)", p.stem)
        frame_idx = int(m.group(1)) if m else 0

        img = cv2.imread(str(p))
        if img is None:
            continue

        score = _compute_crop_quality_score(img)
        if score > 0.0:
            scored.append((p, frame_idx, score))

    if not scored:
        return [p for p in img_files[:top_k]]

    # If count is less than or equal to top_k, return sorted by frame index
    if len(scored) <= top_k:
        scored.sort(key=lambda x: x[1])
        return [x[0] for x in scored]

    # Temporal diversity: when > 6 crops exist, slice track into 3 temporal buckets
    # (early, middle, late) and select the highest quality crop in each bucket.
    scored_by_time = sorted(scored, key=lambda x: x[1])
    n = len(scored_by_time)

    if n > 6 and top_k == 3:
        b1 = scored_by_time[: n // 3]
        b2 = scored_by_time[n // 3 : 2 * (n // 3)]
        b3 = scored_by_time[2 * (n // 3) :]

        chosen = [
            max(b1, key=lambda x: x[2]),
            max(b2, key=lambda x: x[2]),
            max(b3, key=lambda x: x[2]),
        ]
        chosen.sort(key=lambda x: x[1])
        return [x[0] for x in chosen]

    # Otherwise sort descending by quality score and pick top_k, then sort by frame_idx
    best = sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]
    best.sort(key=lambda x: x[1])
    return [x[0] for x in best]


def build_vehicle_crops_contact_strip(
    crop_paths: List[Path],
    out_path: Union[str, Path],
    target_height: int = 320,
) -> Optional[Path]:
    """Generate a 1xK side-by-side contact strip of selected vehicle crops."""
    if not crop_paths:
        return None

    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    loaded_cells: List[np.ndarray] = []
    header_h = 28
    border_w = 4

    for p in crop_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = (target_height - header_h) / max(1, h)
        new_w = max(40, int(w * scale))
        resized = cv2.resize(img, (new_w, target_height - header_h), interpolation=cv2.INTER_LANCZOS4)

        # Create header
        cell = np.zeros((target_height, new_w, 3), dtype=np.uint8)
        cell[:header_h, :] = (35, 30, 30)  # Dark slate header
        cell[header_h:, :] = resized

        label = p.name
        cv2.putText(
            cell,
            label,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        loaded_cells.append(cell)

    if not loaded_cells:
        return None

    # Stitch with vertical separator borders
    sep = np.full((target_height, border_w, 3), 20, dtype=np.uint8)
    strip_parts = []
    for i, c in enumerate(loaded_cells):
        strip_parts.append(c)
        if i < len(loaded_cells) - 1:
            strip_parts.append(sep)

    strip = np.hstack(strip_parts)
    cv2.imwrite(str(out_file), strip)
    logger.info(f"Saved Stage 3 vehicle contact strip ({len(loaded_cells)} crops) to {out_file.name}")
    return out_file


def run_stage3_vehicle_attributes(
    track_dir: Union[str, Path],
    coarse_class: Optional[str] = None,
    track_id: Optional[int] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Stage 3 end-to-end execution for a vehicle track:

    1. Selects top-k sharpest & diverse vehicle crops via select_top_vehicle_crops().
    2. Builds visual 1xK contact strip (stage3_vehicle_crops_top3.png).
    3. Calls backend.opensource_extractor with all selected crops.
    4. Saves vehicle_attributes.json in track directory.
    5. Returns unified dictionary payload.
    """
    t_dir = Path(track_dir).resolve()
    t0 = time.time()

    # 1. Select top vehicle crops
    selected_crops = select_top_vehicle_crops(t_dir, top_k=top_k)

    # 2. Build 1xK contact strip
    contact_strip_path = t_dir / "stage3_vehicle_crops_top3.png"
    built_strip = build_vehicle_crops_contact_strip(selected_crops, contact_strip_path)

    # 3. Call Samarth's opensource extractor
    from backend.opensource_extractor import extract_vehicle_attributes_with_meta

    image_inputs = [str(p) for p in selected_crops] if selected_crops else None

    # Derive track_id if not explicitly provided
    if track_id is None:
        m = re.search(r"track_(\d+)", t_dir.name)
        if m:
            track_id = int(m.group(1))

    merged_ids = [track_id]
    is_merged = False
    cm_path = t_dir / "crop_manifest.json"
    if cm_path.is_file():
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                cm = json.load(f)
            if not coarse_class:
                coarse_class = cm.get("class_name") or cm.get("category")
            merged_ids = cm.get("merged_track_ids", [track_id])
            is_merged = bool(cm.get("is_merged_track", False))
        except Exception:
            pass

    attrs_meta = extract_vehicle_attributes_with_meta(
        images=image_inputs,
        coarse_class=coarse_class,
        track_id=track_id,
    )

    elapsed = round(time.time() - t0, 3)

    result_payload = {
        "schema_version": "3.0.0",
        "track_id": track_id,
        "target_id": track_id,
        "merged_track_ids": merged_ids,
        "is_merged_track": is_merged,
        "coarse_class": coarse_class or "unknown",
        "provider": attrs_meta.provider,
        "model_used": attrs_meta.model_used,
        "latency_sec": attrs_meta.latency_sec,
        "total_stage3_sec": elapsed,
        "selected_crop_paths": [str(p) for p in selected_crops],
        "selected_crop_names": [p.name for p in selected_crops],
        "contact_strip_path": str(built_strip) if built_strip else None,
        "attributes": {
            "body_type": attrs_meta.body_type,
            "body_type_conf": attrs_meta.body_type_conf,
            "colour": attrs_meta.colour,
            "colour_conf": attrs_meta.colour_conf,
            "make": attrs_meta.make,
            "make_conf": attrs_meta.make_conf,
            "model": attrs_meta.model,
            "model_conf": attrs_meta.model_conf,
        },
        "warning": attrs_meta.warning,
        "tier_errors": attrs_meta.tier_errors,
    }

    # 4. Save vehicle_attributes.json in track directory
    out_json = t_dir / "vehicle_attributes.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    logger.info(
        f"Stage 3 complete for {t_dir.name}: {attrs_meta.colour} {attrs_meta.body_type} "
        f"({attrs_meta.make} {attrs_meta.model}) via {attrs_meta.provider} in {elapsed}s"
    )
    return result_payload
