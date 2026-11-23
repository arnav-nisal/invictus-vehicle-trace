"""Post-tracking tracklet stitching module for reconnecting occluded vehicle tracks.

Implements spatiotemporal association, directional path consistency, trajectory projection,
and multi-region visual appearance (central body + full crop) matching in HSV and Lab color spaces.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from .config import (
    DEFAULT_MAX_STITCH_GAP_SEC,
    DEFAULT_MAX_STITCH_SPEED_PX_S,
    DEFAULT_MIN_HIST_CORREL,
    DEFAULT_MAX_LAB_DIST,
    DEFAULT_MAX_SIZE_RATIO,
    DEFAULT_MAX_LATERAL_ERR_PX,
)

logger = logging.getLogger(__name__)


def extract_central_body(
    crop: Optional[np.ndarray],
    y_margin: Tuple[float, float] = (0.18, 0.82),
    x_margin: Tuple[float, float] = (0.12, 0.88),
) -> Optional[np.ndarray]:
    """Extract central body region of vehicle crop to eliminate overhead bars, windshield glare, and ground shadows."""
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return crop
    y1 = int(round(h * y_margin[0]))
    y2 = int(round(h * y_margin[1]))
    x1 = int(round(w * x_margin[0]))
    x2 = int(round(w * x_margin[1]))
    if y2 > y1 and x2 > x1:
        return crop[y1:y2, x1:x2]
    return crop


def compute_patch_features(patch: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute normalized 2D HSV histogram and mean Lab color vector for an image patch."""
    if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return None, None
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab).mean(axis=(0, 1))
        return hist, lab
    except Exception as e:
        logger.debug(f"Color feature extraction failed: {e}")
        return None, None


def compute_crop_color_features(crop: Optional[np.ndarray]) -> Tuple[
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
]:
    """Compute color features for both full crop and central body crop."""
    full_feats = compute_patch_features(crop)
    center_crop = extract_central_body(crop)
    center_feats = compute_patch_features(center_crop)
    return full_feats, center_feats


def compare_patch_appearances(
    feats_a: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    feats_b: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
) -> Tuple[float, float]:
    """Compute HSV histogram correlation and Lab Euclidean distance between two feature tuples."""
    hist_a, lab_a = feats_a
    hist_b, lab_b = feats_b
    if hist_a is None or hist_b is None or lab_a is None or lab_b is None:
        return 0.0, 999.0
    try:
        correl = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        lab_dist = float(np.linalg.norm(lab_a - lab_b))
        return correl, lab_dist
    except Exception:
        return 0.0, 999.0


def compare_appearances(
    feats_a: Any,
    feats_b: Any,
) -> Tuple[float, float]:
    """Compare appearances across both full crop and central body crop, choosing the best match."""
    if feats_a is None or feats_b is None:
        return 0.0, 999.0
    if isinstance(feats_a[0], tuple) and isinstance(feats_b[0], tuple):
        full_a, center_a = feats_a
        full_b, center_b = feats_b
        correl_full, lab_full = compare_patch_appearances(full_a, full_b)
        correl_center, lab_center = compare_patch_appearances(center_a, center_b)
        return max(correl_full, correl_center), min(lab_full, lab_center)
    else:
        return compare_patch_appearances(feats_a, feats_b)


class TrackletStitcher:
    """Stitches broken tracklet segments across brief occlusions based on spatiotemporal and appearance cues."""

    def __init__(
        self,
        max_gap_seconds: float = DEFAULT_MAX_STITCH_GAP_SEC,
        max_speed_px_s: float = DEFAULT_MAX_STITCH_SPEED_PX_S,
        min_hist_correlation: float = DEFAULT_MIN_HIST_CORREL,
        max_lab_distance: float = DEFAULT_MAX_LAB_DIST,
        max_size_ratio: float = DEFAULT_MAX_SIZE_RATIO,
        max_lateral_err_px: float = DEFAULT_MAX_LATERAL_ERR_PX,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.max_speed_px_s = max_speed_px_s
        self.min_hist_correlation = min_hist_correlation
        self.max_lab_distance = max_lab_distance
        self.max_size_ratio = max_size_ratio
        self.max_lateral_err_px = max_lateral_err_px

    def stitch(
        self,
        tracks_meta: Dict[int, dict],
        crop_selector: Any,
        frame_observations: Dict[int, List[dict]],
        fps: float = 30.0,
    ) -> Tuple[Dict[int, dict], Dict[int, List[dict]], List[dict]]:
        """Identify broken tracklet pairs, merge their metadata and observations, and return updated structures."""
        logger.info("Running post-tracking tracklet stitching...")

        # Pre-compute color features (full and central body) for best crops
        track_best_features = {}
        for tid, meta in tracks_meta.items():
            candidate = crop_selector.get_best(tid)
            if candidate is not None and candidate.crop_image is not None:
                feats = compute_crop_color_features(candidate.crop_image)
                track_best_features[tid] = (feats, candidate.crop_image)
            else:
                track_best_features[tid] = (None, None)

        tids = sorted(tracks_meta.keys())
        potential_merges = []

        for i in range(len(tids)):
            tid_a = tids[i]
            meta_a = tracks_meta[tid_a]

            for j in range(len(tids)):
                if i == j:
                    continue
                tid_b = tids[j]
                meta_b = tracks_meta[tid_b]

                # Condition 1: Same vehicle category (two-wheeler, car, bus, truck)
                if meta_a["category"] != meta_b["category"]:
                    continue

                # Condition 2: Temporal gap (T_A ends shortly before T_B starts, gap < 2.5s)
                gap_frames = meta_b["start_frame"] - meta_a["end_frame"]
                gap_sec = gap_frames / fps
                if not (-0.15 <= gap_sec <= self.max_gap_seconds):
                    continue

                # Condition 3: Spatial feasibility & plausible forward movement path and speed
                last_box_a = meta_a.get("last_bbox")
                first_box_b = meta_b.get("first_bbox")

                if last_box_a is None or first_box_b is None:
                    continue

                w_a = max(1.0, float(last_box_a[2] - last_box_a[0]))
                h_a = max(1.0, float(last_box_a[3] - last_box_a[1]))
                w_b = max(1.0, float(first_box_b[2] - first_box_b[0]))
                h_b = max(1.0, float(first_box_b[3] - first_box_b[1]))

                area_a = w_a * h_a
                area_b = w_b * h_b

                # Enforce bounding-box scale consistency (within 2.1x)
                # Prevents different vehicle classes from merging (>2.7x) while accommodating perspective growth across intersection (~1.9x)
                scale_a = float(np.sqrt(area_a))
                scale_b = float(np.sqrt(area_b))
                size_ratio = max(scale_a, scale_b) / min(scale_a, scale_b)

                if size_ratio > self.max_size_ratio:
                    logger.debug(
                        f"Rejecting merge {tid_a} -> {tid_b}: size mismatch ratio {size_ratio:.2f} > {self.max_size_ratio} "
                        f"(scale {scale_a:.1f}px vs {scale_b:.1f}px)"
                    )
                    continue

                cx_a = (last_box_a[0] + last_box_a[2]) / 2.0
                cy_a = (last_box_a[1] + last_box_a[3]) / 2.0
                cx_b = (first_box_b[0] + first_box_b[2]) / 2.0
                cy_b = (first_box_b[1] + first_box_b[3]) / 2.0

                dx = cx_b - cx_a
                dy = cy_b - cy_a
                dist_px = float(np.sqrt(dx ** 2 + dy ** 2))
                eff_gap_sec = max(0.05, abs(gap_sec))
                speed_px_s = dist_px / eff_gap_sec

                # Centroid shift allowance: permits bounding box boundary expansions/recoveries across occlusions
                # (e.g., car emerging from under an overhead bar where bounding box expands by half its length in 1 frame)
                max_allowed_dist = max(60.0, max(w_a, h_a) * 0.85, self.max_speed_px_s * eff_gap_sec)

                # Speed check: vehicle cannot teleport across the scene faster than physical limit
                if dist_px > max_allowed_dist and speed_px_s > self.max_speed_px_s:
                    logger.debug(
                        f"Rejecting merge {tid_a} -> {tid_b}: excessive speed {speed_px_s:.1f}px/s, dist={dist_px:.1f}px > {max_allowed_dist:.1f}px"
                    )
                    continue

                # Forward movement / directional check
                traj_a = meta_a.get("trajectory", [])
                if len(traj_a) >= 5 and dist_px > 45.0:
                    anchor_idx = max(0, len(traj_a) - 6)
                    _, anc_x, anc_y = traj_a[anchor_idx]
                    vx_a = cx_a - anc_x
                    vy_a = cy_a - anc_y
                    v_mag = float(np.sqrt(vx_a ** 2 + vy_a ** 2))
                    if v_mag > 10.0:
                        cos_sim = float((vx_a * dx + vy_a * dy) / (v_mag * dist_px))
                        # Vehicle cannot move backward opposite to its prior motion
                        if cos_sim < -0.35:
                            logger.debug(
                                f"Rejecting merge {tid_a} -> {tid_b}: backwards movement cos_sim={cos_sim:.2f}"
                            )
                            continue

                # Trajectory projection & lateral error check (prevents cross-lane merges)
                lateral_err = 0.0
                total_proj_err = dist_px
                if len(traj_a) >= 3:
                    anchor_idx = max(0, len(traj_a) - 8)
                    f_anc, x_anc, y_anc = traj_a[anchor_idx]
                    f_last, x_last, y_last = traj_a[-1]
                    df = f_last - f_anc
                    if df > 0:
                        vx = (x_last - x_anc) / df
                        vy = (y_last - y_anc) / df
                        v_mag = float(np.sqrt(vx ** 2 + vy ** 2))
                        if v_mag > 0.3:  # vehicle is moving
                            ux = vx / v_mag
                            uy = vy / v_mag
                            nx = -uy
                            ny = ux
                            pred_x = x_last + vx * gap_frames
                            pred_y = y_last + vy * gap_frames
                            dx_proj = cx_b - pred_x
                            dy_proj = cy_b - pred_y
                            lateral_err = float(abs(dx_proj * nx + dy_proj * ny))
                            total_proj_err = float(np.sqrt(dx_proj ** 2 + dy_proj ** 2))

                            if lateral_err > self.max_lateral_err_px:
                                logger.debug(
                                    f"Rejecting merge {tid_a} -> {tid_b}: excessive lateral error {lateral_err:.1f}px > "
                                    f"{self.max_lateral_err_px}px (cross-lane drift)"
                                )
                                continue

                # Condition 4: Visual appearance & colour matching (boundary + best crops, full + central body)
                crop_end_a = meta_a.get("last_crop")
                crop_start_b = meta_b.get("first_crop")
                b_feats_a = compute_crop_color_features(crop_end_a)
                b_feats_b = compute_crop_color_features(crop_start_b)
                correl_boundary, lab_boundary = compare_appearances(b_feats_a, b_feats_b)

                best_feats_a, _ = track_best_features.get(tid_a, (None, None))
                best_feats_b, _ = track_best_features.get(tid_b, (None, None))
                correl_best, lab_best = compare_appearances(best_feats_a, best_feats_b)

                effective_correl = max(correl_boundary, correl_best)
                effective_lab = min(lab_boundary, lab_best)

                if effective_correl < self.min_hist_correlation or effective_lab > self.max_lab_distance:
                    logger.debug(
                        f"Appearance mismatch for {tid_a} -> {tid_b}: "
                        f"correl={effective_correl:.3f} (min {self.min_hist_correlation}), "
                        f"lab={effective_lab:.1f} (max {self.max_lab_distance})"
                    )
                    continue

                # Composite match score: higher is better
                # Rewards tight temporal continuity, minimal trajectory projection error, and strong visual match
                score = (
                    effective_correl * 100.0
                    - effective_lab * 0.5
                    - lateral_err * 0.25
                    - total_proj_err * 0.15
                    - max(0.0, gap_sec) * 15.0
                )

                potential_merges.append(
                    {
                        "tid_a": int(tid_a),
                        "tid_b": int(tid_b),
                        "gap_sec": float(round(gap_sec, 2)),
                        "dist_px": float(round(dist_px, 1)),
                        "speed_px_s": float(round(speed_px_s, 1)),
                        "lateral_err": float(round(lateral_err, 1)),
                        "total_proj_err": float(round(total_proj_err, 1)),
                        "hist_correl": float(round(effective_correl, 3)),
                        "lab_dist": float(round(effective_lab, 2)),
                        "score": float(round(score, 2)),
                    }
                )

        # Sort candidate merges by highest compatibility score
        potential_merges.sort(key=lambda m: m["score"], reverse=True)

        # Disjoint-set union mapping: map target ID -> master ID
        id_map: Dict[int, int] = {int(tid): int(tid) for tid in tids}
        stitched_pairs = []
        used_a = set()
        used_b = set()

        def find_root(t: int) -> int:
            t = int(t)
            while id_map[t] != t:
                id_map[t] = id_map[id_map[t]]
                t = id_map[t]
            return t

        for m in potential_merges:
            root_a = find_root(m["tid_a"])
            root_b = find_root(m["tid_b"])

            if root_a == root_b:
                continue
            if m["tid_a"] in used_a or m["tid_b"] in used_b:
                continue

            # Ensure chronological order: root is the earlier track
            meta_a_root = tracks_meta[root_a]
            meta_b_root = tracks_meta[root_b]
            if meta_a_root["start_frame"] > meta_b_root["start_frame"]:
                root_a, root_b = root_b, root_a

            id_map[root_b] = root_a
            used_a.add(m["tid_a"])
            used_b.add(m["tid_b"])
            stitched_entry = {
                "tid_a": int(m["tid_a"]),
                "tid_b": int(m["tid_b"]),
                "master_id": int(root_a),
                "gap_sec": float(m["gap_sec"]),
                "dist_px": float(m["dist_px"]),
                "speed_px_s": float(m["speed_px_s"]),
                "lateral_err": float(m.get("lateral_err", 0.0)),
                "total_proj_err": float(m.get("total_proj_err", 0.0)),
                "hist_correl": float(m["hist_correl"]),
                "lab_dist": float(m["lab_dist"]),
                "score": float(m["score"]),
            }
            stitched_pairs.append(stitched_entry)

            logger.info(
                f"STITCH SUCCESS: Track {m['tid_a']} -> Track {m['tid_b']} merged into Master ID {root_a} | "
                f"Gap: {m['gap_sec']}s, Dist: {m['dist_px']}px, LatErr: {m.get('lateral_err', 0.0)}px, "
                f"HSV Correl: {m['hist_correl']}, Lab Dist: {m['lab_dist']}"
            )

        if not stitched_pairs:
            logger.info("No tracklet merges required; all tracks intact.")
            return tracks_meta, frame_observations, []

        # Merge metadata for all tracks mapped to the same root ID
        merged_meta: Dict[int, dict] = {}
        for tid in tids:
            root_id = int(find_root(tid))
            meta = tracks_meta[tid]

            if root_id not in merged_meta:
                merged_meta[root_id] = dict(meta)
                merged_meta[root_id]["id"] = int(root_id)
                merged_meta[root_id]["original_ids"] = [int(tid)]
            else:
                curr = merged_meta[root_id]
                curr["end_frame"] = max(curr["end_frame"], meta["end_frame"])
                curr["end_timestamp_sec"] = max(curr["end_timestamp_sec"], meta["end_timestamp_sec"])
                curr["start_frame"] = min(curr["start_frame"], meta["start_frame"])
                curr["start_timestamp_sec"] = min(curr["start_timestamp_sec"], meta["start_timestamp_sec"])
                curr["frame_count"] += meta["frame_count"]
                curr["max_conf"] = max(curr["max_conf"], meta["max_conf"])
                curr["conf_sum"] += meta["conf_sum"]
                curr["max_area"] = max(curr["max_area"], meta["max_area"])
                curr["max_dim"] = max(curr["max_dim"], meta["max_dim"])
                if meta["end_frame"] >= curr["end_frame"]:
                    curr["last_bbox"] = meta.get("last_bbox", curr.get("last_bbox"))
                    curr["last_crop"] = meta.get("last_crop", curr.get("last_crop"))
                curr["original_ids"].append(int(tid))

                # Keep the sharper / better crop candidate in crop_selector
                cand_root = crop_selector.get_best(root_id)
                cand_tid = crop_selector.get_best(tid)
                if cand_tid is not None:
                    if cand_root is None or cand_tid.quality_score > cand_root.quality_score:
                        crop_selector._best_candidates[root_id] = cand_tid

        # Update frame observations to replace merged IDs with root IDs
        updated_observations: Dict[int, List[dict]] = {}
        for f_idx, dets in frame_observations.items():
            updated_dets = []
            for d in dets:
                old_id = d["id"]
                new_id = find_root(old_id)
                new_d = dict(d)
                new_d["id"] = new_id
                updated_dets.append(new_d)
            updated_observations[f_idx] = updated_dets

        return merged_meta, updated_observations, stitched_pairs
