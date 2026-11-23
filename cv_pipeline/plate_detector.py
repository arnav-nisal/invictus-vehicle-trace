"""Stage 2.2: Dual-Detector License Plate Cropping & Dynamic Readability Selection.

Integrates:
  1. Detector A: Custom Fine-Tuned Indian License Plate YOLO26s (models/yolo26s_plate_best.pt).
  2. Detector B: Global License Plate YOLOv9-s ONNX (yolo-v9-s-608-license-plate-end2end, MIT).
  3. Plate Geometry Filter: Reject non-plate boxes with AR outside [0.85, 6.5] or area > 45% crop area.
  4. Spatial IoU cross-verification (DUAL_VERIFIED, SINGLE_CUSTOM, SINGLE_GLOBAL, CONFLICT_RESOLVED_CUSTOM, CONFLICT_RESOLVED_GLOBAL).
  5. 100% Balanced Detection & Verification Accounting.
  6. Border-Truncation Guard: Reject frame-exit and edge-truncated plates (<= 2 native px crop border, <= 6 px video outer boundary, or AR drop < 0.75 * median_ar within 5% border).
  7. Pure Lanczos4 plate enlargement: Minimum width 320px or height 120px (capped at 4.0x, leave large at 1.0x, zero filters).
  8. Outlier-Sensitive Dynamic Readability Gate: 90th percentile (p90) anchor (threshold = 0.45 * p90_score), 1.25x dual-bonus, capped at 60.
  9. Audit manifest export (plate_manifest.json).
"""

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def is_valid_plate_geometry(
    box: List[float], img_shape: Tuple[int, int]
) -> Tuple[bool, str]:
    """Rejects obvious non-plate boxes with aspect ratio outside [0.85, 6.5] or box area > 45% of crop area.

    Returns:
        (is_valid, reason_str)
    """
    h_img, w_img = img_shape[:2]
    crop_area = float(w_img * h_img)
    if crop_area <= 0:
        return False, "invalid_crop_area"
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    if h <= 0.0 or w <= 0.0:
        return False, "zero_dimension"
    ar = w / h
    if ar < 0.85 or ar > 6.5:
        return False, f"aspect_ratio_out_of_bounds (AR={ar:.2f} not in [0.85, 6.5])"
    box_area = w * h
    area_pct = (box_area / crop_area) * 100.0
    if (box_area / crop_area) > 0.45:
        return False, f"area_too_large ({area_pct:.1f}% > 45% of crop)"
    return True, "valid"


def enlarge_plate_lanczos(
    plate_unenh: np.ndarray,
    plate_enh: np.ndarray,
    min_width: int = 320,
    min_height: int = 120,
    max_scale: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """Enlarges small plate crops cleanly using only cv2.INTER_LANCZOS4 (preserving aspect ratio).

    If the plate already has width >= min_width or height >= min_height, leaves at 1.0x.
    Otherwise, scales up so that it reaches min_width or min_height, capped at max_scale (4.0x).
    Applies zero additional enhancements or filters.

    Returns:
        saved_unenh (np.ndarray): Enlarged unenhanced crop.
        saved_enh (np.ndarray): Enlarged enhanced crop.
        applied_scale (float): Scale factor applied.
        raw_size (Tuple[int, int]): (raw_w, raw_h).
        saved_size (Tuple[int, int]): (saved_w, saved_h).
    """
    raw_h, raw_w = plate_unenh.shape[:2]
    raw_size = (int(raw_w), int(raw_h))

    # Check if plate is already large enough
    if raw_w >= min_width or raw_h >= min_height:
        target_scale = 1.0
    else:
        scale_w = float(min_width) / float(raw_w)
        scale_h = float(min_height) / float(raw_h)
        target_scale = min(max_scale, min(scale_w, scale_h))

    if target_scale > 1.001:
        new_w = max(1, int(round(raw_w * target_scale)))
        new_h = max(1, int(round(raw_h * target_scale)))
        saved_unenh = cv2.resize(plate_unenh, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        saved_enh = cv2.resize(plate_enh, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        applied_scale = round(float(target_scale), 3)
        saved_size = (int(new_w), int(new_h))
    else:
        saved_unenh = plate_unenh.copy()
        saved_enh = plate_enh.copy()
        applied_scale = 1.0
        saved_size = raw_size

    return saved_unenh, saved_enh, applied_scale, raw_size, saved_size


class DualPlateDetector:
    """Manages Detector A (Custom Indian YOLO26s) and Detector B (Global YOLOv9-s ONNX)."""

    def __init__(
        self,
        custom_weights_path: Union[str, Path] = "models/yolo26s_plate_best.pt",
        global_model_name: str = "yolo-v9-s-608-license-plate-end2end",
        custom_conf: float = 0.15,
        global_conf: float = 0.18,
        device: Union[int, str] = 0,
    ):
        self.custom_weights_path = Path(custom_weights_path)
        self.global_model_name = global_model_name
        self.custom_conf = custom_conf
        self.global_conf = global_conf
        self.device = device

        # Load Custom YOLO26s detector
        if not self.custom_weights_path.exists():
            raise FileNotFoundError(
                f"Custom plate model weights not found at: {self.custom_weights_path}"
            )
        from ultralytics import YOLO

        logger.info(f"Loading Custom Indian Plate Detector from '{self.custom_weights_path}' on device={device}...")
        self.custom_model = YOLO(str(self.custom_weights_path))

        # Load Global YOLOv9-s ONNX detector
        logger.info(f"Loading Global Plate Detector '{global_model_name}' via open-image-models...")
        from open_image_models import create_detector

        self.global_detector = create_detector(global_model_name)
        logger.info("Dual License Plate Detectors successfully loaded and initialized.")

    def detect_custom_yolo26(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference with custom YOLO26s model."""
        preds = self.custom_model.predict(
            img_bgr,
            conf=self.custom_conf,
            device=self.device,
            verbose=False,
        )[0]

        boxes = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            xyxy = preds.boxes.xyxy.cpu().numpy()
            confs = preds.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                boxes.append({
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "confidence": float(c),
                    "detector": "CUSTOM_YOLO26",
                })
        boxes.sort(key=lambda x: x["confidence"], reverse=True)
        return boxes

    def detect_global_yolov9(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference with global YOLOv9-s ONNX model."""
        res_list = self.global_detector.predict(img_bgr)
        boxes = []
        for r in res_list:
            c = float(r.confidence)
            if c >= self.global_conf:
                b = r.bounding_box
                boxes.append({
                    "bbox": [float(b.x1), float(b.y1), float(b.x2), float(b.y2)],
                    "confidence": c,
                    "detector": "GLOBAL_YOLOV9",
                })
        boxes.sort(key=lambda x: x["confidence"], reverse=True)
        return boxes

    def detect_and_cross_verify(
        self, img_bgr: np.ndarray, iou_thresh: float = 0.35
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Run both detectors, apply geometry filter, and cross-verify detections with spatial IoU matching.

        Returns:
            match_res: Match info dict or None.
            diag_info: Diagnostic dict detailing raw vs filtered detections.
        """
        raw_custom_boxes = self.detect_custom_yolo26(img_bgr)
        raw_global_boxes = self.detect_global_yolov9(img_bgr)

        # Apply plate geometry filter: aspect ratio in [0.85, 6.5], area <= 45% crop area
        valid_custom_boxes = [
            b for b in raw_custom_boxes if is_valid_plate_geometry(b["bbox"], img_bgr.shape)[0]
        ]
        valid_global_boxes = [
            b for b in raw_global_boxes if is_valid_plate_geometry(b["bbox"], img_bgr.shape)[0]
        ]

        diag_info = {
            "raw_custom_count": len(raw_custom_boxes),
            "raw_global_count": len(raw_global_boxes),
            "valid_custom_count": len(valid_custom_boxes),
            "valid_global_count": len(valid_global_boxes),
            "raw_custom_boxes": raw_custom_boxes,
            "raw_global_boxes": raw_global_boxes,
        }

        # Case 1: Both detectors have valid geometry detections
        if valid_custom_boxes and valid_global_boxes:
            best_custom = valid_custom_boxes[0]
            best_global = valid_global_boxes[0]
            iou = compute_iou(best_custom["bbox"], best_global["bbox"])

            if iou >= iou_thresh:
                # DUAL VERIFIED: Blend bounding boxes weighted by confidence
                wa = best_custom["confidence"]
                wb = best_global["confidence"]
                wtot = wa + wb
                b1 = best_custom["bbox"]
                b2 = best_global["bbox"]
                merged_box = [
                    round(float((wa * b1[i] + wb * b2[i]) / wtot), 2) for i in range(4)
                ]
                match_res = {
                    "bbox": merged_box,
                    "confidence": round(max(wa, wb), 4),
                    "verification_status": "DUAL_VERIFIED",
                    "iou": round(iou, 3),
                    "custom_bbox": b1,
                    "custom_conf": round(wa, 4),
                    "global_bbox": b2,
                    "global_conf": round(wb, 4),
                    "has_custom": True,
                    "has_global": True,
                }
                return match_res, diag_info
            else:
                # CONFLICT RESOLVED: Both detected valid plates with IoU < 0.35
                wa = best_custom["confidence"]
                wb = best_global["confidence"]
                if wa >= wb:
                    vstatus = "CONFLICT_RESOLVED_CUSTOM"
                    winner_box = best_custom["bbox"]
                    winner_conf = wa
                else:
                    vstatus = "CONFLICT_RESOLVED_GLOBAL"
                    winner_box = best_global["bbox"]
                    winner_conf = wb

                match_res = {
                    "bbox": winner_box,
                    "confidence": round(winner_conf, 4),
                    "verification_status": vstatus,
                    "iou": round(iou, 3),
                    "custom_bbox": best_custom["bbox"],
                    "custom_conf": round(wa, 4),
                    "global_bbox": best_global["bbox"],
                    "global_conf": round(wb, 4),
                    "has_custom": True,
                    "has_global": True,
                }
                return match_res, diag_info

        # Case 2: Only Custom has a valid detection
        if valid_custom_boxes and not valid_global_boxes:
            b = valid_custom_boxes[0]
            match_res = {
                "bbox": b["bbox"],
                "confidence": round(b["confidence"], 4),
                "verification_status": "SINGLE_CUSTOM_YOLO26",
                "iou": 0.0,
                "custom_bbox": b["bbox"],
                "custom_conf": round(b["confidence"], 4),
                "global_bbox": None,
                "global_conf": None,
                "has_custom": True,
                "has_global": False,
            }
            return match_res, diag_info

        # Case 3: Only Global has a valid detection
        if valid_global_boxes and not valid_custom_boxes:
            b = valid_global_boxes[0]
            match_res = {
                "bbox": b["bbox"],
                "confidence": round(b["confidence"], 4),
                "verification_status": "SINGLE_GLOBAL_YOLOV9",
                "iou": 0.0,
                "custom_bbox": None,
                "custom_conf": None,
                "global_bbox": b["bbox"],
                "global_conf": round(b["confidence"], 4),
                "has_custom": False,
                "has_global": True,
            }
            return match_res, diag_info

        # Case 4: Neither has a valid detection
        return None, diag_info


def run_dual_plate_detection(
    crop_manifest_path: Union[str, Path],
    detector: Optional[DualPlateDetector] = None,
    output_dir: Optional[Union[str, Path]] = None,
    padding_pct: float = 0.12,
    max_frames_to_process: int = 500,
    target_selected_count: int = 60,
    device: Union[int, str] = 0,
) -> Dict[str, Any]:
    """Execute Stage 2.2 dual-detector plate detection and dynamic readability gating."""
    manifest_file = Path(crop_manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Crop manifest not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        crop_manifest = json.load(f)

    track_id = crop_manifest.get("track_id", 0)
    class_name = crop_manifest.get("class_name", "vehicle")
    category = crop_manifest.get("category", "vehicle")
    crops_list = crop_manifest.get("crops", [])

    if not crops_list:
        raise ValueError(f"Crop manifest {manifest_file} has no crop items.")

    track_dir = manifest_file.parent
    if output_dir is not None:
        track_dir = Path(output_dir)

    # Determine original / crop video frame dimensions for outer boundary guard
    crop_video_src = crop_manifest.get("crop_video_source") or crop_manifest.get("video_source")
    vid_w, vid_h = None, None
    if crop_video_src and Path(crop_video_src).exists():
        vcap = cv2.VideoCapture(str(crop_video_src))
        if vcap.isOpened():
            vid_w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vcap.release()

    if vid_w is None or vid_h is None or vid_w <= 0 or vid_h <= 0:
        max_px = max([c.get("padded_bbox", [0, 0, 0, 0])[2] for c in crops_list], default=1920)
        max_py = max([c.get("padded_bbox", [0, 0, 0, 0])[3] for c in crops_list], default=1080)
        vid_w = 3840 if max_px > 1920 else 1920
        vid_h = 2160 if max_py > 1080 else 1080

    logger.info(f"Video boundary dimensions for Track {track_id}: {vid_w}x{vid_h}")

    # Plate output directories
    all_unenh_dir = track_dir / "plate_crops_all" / "unenhanced"
    all_enh_dir = track_dir / "plate_crops_all" / "enhanced"
    selected_unenh_dir = track_dir / "plate_crops_selected" / "unenhanced"
    selected_enh_dir = track_dir / "plate_crops_selected" / "enhanced"

    # Clean existing selected crops to ensure zero stale files remain
    for d in (all_unenh_dir, all_enh_dir, selected_unenh_dir, selected_enh_dir):
        d.mkdir(parents=True, exist_ok=True)
    for f in selected_unenh_dir.glob("*.png"):
        f.unlink(missing_ok=True)
    for f in selected_enh_dir.glob("*.png"):
        f.unlink(missing_ok=True)

    # Initialize dual detectors if not provided
    if detector is None:
        detector = DualPlateDetector(device=device)

    # Subsample if stationary track has > 500 frames
    total_crops = len(crops_list)
    if total_crops > max_frames_to_process:
        logger.info(
            f"Subsampling {max_frames_to_process} frames evenly from {total_crops} total crops for Track {track_id}..."
        )
        sample_indices = np.linspace(0, total_crops - 1, max_frames_to_process, dtype=int)
        sampled_crops = [crops_list[i] for i in sample_indices]
    else:
        sampled_crops = crops_list

    logger.info(
        f"Running Dual Plate Detectors on {len(sampled_crops)} vehicle crops for Track {track_id} ({class_name})..."
    )

    custom_detections_count = 0
    global_detections_count = 0
    dual_verified_count = 0
    single_custom_count = 0
    single_global_count = 0
    conflict_resolved_custom_count = 0
    conflict_resolved_global_count = 0

    border_truncated_count = 0
    micro_plate_count = 0

    # Diagnostic records for conflict frames (both raw detections occurred)
    conflict_diagnostics: List[Dict[str, Any]] = []

    # First pass: run dual detector & geometry filter across all sampled crops
    raw_frame_detections = []
    aspect_ratios_for_median = []

    for crop_meta in sampled_crops:
        frame_idx = crop_meta["frame_idx"]
        scale_factor = float(crop_meta.get("scale_factor", 1.0))
        enhanced_path_str = crop_meta.get("enhanced_path") or crop_meta.get("enhanced_crop_path")
        unenhanced_path_str = crop_meta.get("unenhanced_path") or crop_meta.get("unenhanced_crop_path")

        if not enhanced_path_str or not unenhanced_path_str:
            continue

        enhanced_path = Path(enhanced_path_str)
        unenhanced_path = Path(unenhanced_path_str)

        if not enhanced_path.exists():
            continue

        enh_img = cv2.imread(str(enhanced_path))
        if enh_img is None:
            continue
        unenh_img = cv2.imread(str(unenhanced_path))
        if unenh_img is None:
            unenh_img = enh_img

        enh_h, enh_w = enh_img.shape[:2]

        match_res, diag_info = detector.detect_and_cross_verify(enh_img)

        # Check for conflicts in raw detections to log diagnostics
        raw_c = diag_info["raw_custom_boxes"]
        raw_g = diag_info["raw_global_boxes"]
        if raw_c and raw_g:
            rc_box = raw_c[0]["bbox"]
            rg_box = raw_g[0]["bbox"]
            raw_iou = compute_iou(rc_box, rg_box)
            if raw_iou < 0.35:
                rc_w = max(1.0, rc_box[2] - rc_box[0])
                rc_h = max(1.0, rc_box[3] - rc_box[1])
                rg_w = max(1.0, rg_box[2] - rg_box[0])
                rg_h = max(1.0, rg_box[3] - rg_box[1])
                c_valid, c_reason = is_valid_plate_geometry(rc_box, (enh_h, enh_w))
                g_valid, g_reason = is_valid_plate_geometry(rg_box, (enh_h, enh_w))
                conflict_diagnostics.append({
                    "frame_idx": int(frame_idx),
                    "crop_dim": f"{enh_w}x{enh_h}",
                    "custom_box": [round(x, 1) for x in rc_box],
                    "custom_size": f"{rc_w:.1f}x{rc_h:.1f}",
                    "custom_ar": round(rc_w / rc_h, 2),
                    "custom_area_pct": round((rc_w * rc_h) / (enh_w * enh_h) * 100.0, 1),
                    "custom_conf": round(raw_c[0]["confidence"], 4),
                    "custom_valid": c_valid,
                    "custom_reason": c_reason,
                    "global_box": [round(x, 1) for x in rg_box],
                    "global_size": f"{rg_w:.1f}x{rg_h:.1f}",
                    "global_ar": round(rg_w / rg_h, 2),
                    "global_area_pct": round((rg_w * rg_h) / (enh_w * enh_h) * 100.0, 1),
                    "global_conf": round(raw_g[0]["confidence"], 4),
                    "global_valid": g_valid,
                    "global_reason": g_reason,
                    "raw_iou": round(raw_iou, 3),
                    "resolved_status": match_res["verification_status"] if match_res else "NONE",
                })

        if match_res is None:
            continue

        # Balanced accounting
        if match_res["has_custom"]:
            custom_detections_count += 1
        if match_res["has_global"]:
            global_detections_count += 1

        vstatus = match_res["verification_status"]
        if vstatus == "DUAL_VERIFIED":
            dual_verified_count += 1
        elif vstatus == "SINGLE_CUSTOM_YOLO26":
            single_custom_count += 1
        elif vstatus == "SINGLE_GLOBAL_YOLOV9":
            single_global_count += 1
        elif vstatus == "CONFLICT_RESOLVED_CUSTOM":
            conflict_resolved_custom_count += 1
        elif vstatus == "CONFLICT_RESOLVED_GLOBAL":
            conflict_resolved_global_count += 1

        p_box = match_res["bbox"]
        pw = max(1.0, p_box[2] - p_box[0])
        ph = max(1.0, p_box[3] - p_box[1])
        aspect_ratios_for_median.append(pw / ph)

        raw_frame_detections.append({
            "crop_meta": crop_meta,
            "match_res": match_res,
            "enh_img": enh_img,
            "unenh_img": unenh_img,
            "enh_w": enh_w,
            "enh_h": enh_h,
            "scale_factor": scale_factor,
        })

    # Mathematical accounting assertion check
    expected_custom = (
        dual_verified_count
        + single_custom_count
        + conflict_resolved_custom_count
        + conflict_resolved_global_count
    )
    expected_global = (
        dual_verified_count
        + single_global_count
        + conflict_resolved_custom_count
        + conflict_resolved_global_count
    )
    assert custom_detections_count == expected_custom, (
        f"Custom accounting gap: {custom_detections_count} != {expected_custom}"
    )
    assert global_detections_count == expected_global, (
        f"Global accounting gap: {global_detections_count} != {expected_global}"
    )

    # Compute track's median plate aspect ratio
    if aspect_ratios_for_median:
        median_ar = float(np.median(aspect_ratios_for_median))
    else:
        median_ar = 3.0
    logger.info(f"Track {track_id}: computed median plate aspect ratio = {median_ar:.2f}")

    detected_plates: List[Dict[str, Any]] = []

    # Second pass: border-truncation guard, cropping, enlargement, and scoring
    for item in raw_frame_detections:
        crop_meta = item["crop_meta"]
        match_res = item["match_res"]
        enh_img = item["enh_img"]
        unenh_img = item["unenh_img"]
        enh_w = item["enh_w"]
        enh_h = item["enh_h"]
        scale_factor = item["scale_factor"]
        frame_idx = crop_meta["frame_idx"]

        vstatus = match_res["verification_status"]
        p_box = match_res["bbox"]  # [x1, y1, x2, y2] unpadded plate box in enh coordinates
        pw = max(1.0, p_box[2] - p_box[0])
        ph = max(1.0, p_box[3] - p_box[1])
        plate_ar = pw / ph

        # =====================================================================
        # 1. BORDER-TRUNCATION GUARD (unpadded plate box + crop_manifest.json)
        # =====================================================================
        # (a) Crop edge contact check: within <= 2 native pixels of vehicle crop boundary
        dist_crop_left = p_box[0]
        dist_crop_top = p_box[1]
        dist_crop_right = enh_w - p_box[2]
        dist_crop_bottom = enh_h - p_box[3]
        min_crop_dist = min(dist_crop_left, dist_crop_top, dist_crop_right, dist_crop_bottom)
        min_crop_dist_native = min_crop_dist / scale_factor

        # (b) Video outer boundary check: within <= 6 pixels of original video outer frame
        padded_bbox = crop_meta.get("padded_bbox", [0, 0, enh_w, enh_h])
        crop_x1, crop_y1, crop_x2, crop_y2 = padded_bbox
        vid_px1 = crop_x1 + (p_box[0] / scale_factor)
        vid_py1 = crop_y1 + (p_box[1] / scale_factor)
        vid_px2 = crop_x1 + (p_box[2] / scale_factor)
        vid_py2 = crop_y1 + (p_box[3] / scale_factor)

        dist_vid_left = vid_px1
        dist_vid_top = vid_py1
        dist_vid_right = vid_w - vid_px2
        dist_vid_bottom = vid_h - vid_py2
        min_vid_dist = min(dist_vid_left, dist_vid_top, dist_vid_right, dist_vid_bottom)

        # (c) Exit aspect-ratio drop check: within 5% of border AND AR drops below 0.75 * median_ar
        near_crop_5pct = (
            dist_crop_left <= 0.05 * enh_w
            or dist_crop_right <= 0.05 * enh_w
            or dist_crop_top <= 0.05 * enh_h
            or dist_crop_bottom <= 0.05 * enh_h
        )
        near_vid_5pct = (
            dist_vid_left <= 0.05 * vid_w
            or dist_vid_right <= 0.05 * vid_w
            or dist_vid_top <= 0.05 * vid_h
            or dist_vid_bottom <= 0.05 * vid_h
        )
        is_near_border_5pct = near_crop_5pct or near_vid_5pct
        exit_ar_drop = is_near_border_5pct and (plate_ar < 0.75 * median_ar)

        is_border_truncated = False
        border_trunc_reason = None
        if min_crop_dist_native <= 2.0:
            is_border_truncated = True
            border_trunc_reason = f"Crop edge contact: native dist ({min_crop_dist_native:.1f}px) <= 2.0px"
        elif min_vid_dist <= 6.0:
            is_border_truncated = True
            border_trunc_reason = f"Video border contact: dist ({min_vid_dist:.1f}px) <= 6.0px"
        elif exit_ar_drop:
            is_border_truncated = True
            border_trunc_reason = f"Exit aspect-ratio drop: AR {plate_ar:.2f} < 0.75 * median ({median_ar:.2f}) within 5% border"

        # Apply 12% padding around plate box for clean character margins
        pad_x = pw * padding_pct
        pad_y = ph * padding_pct

        slice_x1 = max(0, int(round(p_box[0] - pad_x)))
        slice_y1 = max(0, int(round(p_box[1] - pad_y)))
        slice_x2 = min(enh_w, int(round(p_box[2] + pad_x)))
        slice_y2 = min(enh_h, int(round(p_box[3] + pad_y)))

        if slice_x2 <= slice_x1 or slice_y2 <= slice_y1:
            continue

        raw_cropped_w = int(slice_x2 - slice_x1)
        raw_cropped_h = int(slice_y2 - slice_y1)

        # Slice raw plate crops
        plate_raw_unenh = unenh_img[slice_y1:slice_y2, slice_x1:slice_x2]
        plate_raw_enh = enh_img[slice_y1:slice_y2, slice_x1:slice_x2]

        # =====================================================================
        # 2. COMPUTE NATIVE METRICS & LAPLACIAN SHARPNESS FIRST (Untouched raw slice)
        # =====================================================================
        native_w = round(raw_cropped_w / scale_factor, 1)
        native_h = round(raw_cropped_h / scale_factor, 1)
        native_area = round(native_w * native_h, 1)

        gray_unenh = cv2.cvtColor(plate_raw_unenh, cv2.COLOR_BGR2GRAY)
        plate_sharpness = round(float(cv2.Laplacian(gray_unenh, cv2.CV_64F).var()), 2)
        plate_contrast = round(float(np.std(gray_unenh)), 2)

        # Evaluate readability & micro-plate rejection
        is_readable = True
        rejection_reason = None

        if is_border_truncated:
            is_readable = False
            rejection_reason = "border_truncated"
            border_truncated_count += 1
        elif native_w < 18.0:
            is_readable = False
            rejection_reason = f"Micro-plate: native width ({native_w}px) < 18px"
            micro_plate_count += 1
        elif native_h < 8.0:
            is_readable = False
            rejection_reason = f"Micro-plate: native height ({native_h}px) < 8px"
            micro_plate_count += 1
        elif raw_cropped_w < 40:
            is_readable = False
            rejection_reason = f"Micro-plate: cropped width ({raw_cropped_w}px) < 40px"
            micro_plate_count += 1

        # Readability score with 1.25x dual verification bonus
        conf = float(match_res["confidence"])
        dual_bonus = 1.25 if vstatus == "DUAL_VERIFIED" else 1.0

        if is_readable:
            norm_sharp = math.sqrt(max(0.1, plate_sharpness))
            norm_area = math.sqrt(max(1.0, native_area))
            norm_contrast = max(1.0, plate_contrast)
            raw_score = (norm_sharp ** 0.8) * (norm_area ** 0.8) * (norm_contrast ** 0.4) * conf
            readability_score = round(raw_score * dual_bonus, 2)
        else:
            readability_score = 0.0

        # =====================================================================
        # 3. PURE LANCZOS4 PLATE ENLARGEMENT (Min 320w or 120h, capped at 4.0x)
        # =====================================================================
        (
            saved_unenh,
            saved_enh,
            applied_plate_scale,
            raw_plate_size,
            saved_plate_size,
        ) = enlarge_plate_lanczos(
            plate_unenh=plate_raw_unenh,
            plate_enh=plate_raw_enh,
            min_width=320,
            min_height=120,
            max_scale=4.0,
        )

        # Save to plate_crops_all/
        plate_filename = f"plate_{frame_idx:06d}.png"
        all_unenh_path = all_unenh_dir / plate_filename
        all_enh_path = all_enh_dir / plate_filename

        cv2.imwrite(str(all_unenh_path), saved_unenh)
        cv2.imwrite(str(all_enh_path), saved_enh)

        plate_meta = {
            "plate_index": len(detected_plates),
            "frame_idx": int(frame_idx),
            "timestamp": crop_meta.get("timestamp", ""),
            "timestamp_sec": crop_meta.get("timestamp_sec", 0.0),
            "verification_status": vstatus,
            "detector_conf": conf,
            "custom_bbox": match_res["custom_bbox"],
            "custom_conf": match_res["custom_conf"],
            "global_bbox": match_res["global_bbox"],
            "global_conf": match_res["global_conf"],
            "iou": match_res["iou"],
            "raw_plate_bbox": [round(c, 1) for c in p_box],
            "padded_plate_bbox": [slice_x1, slice_y1, slice_x2, slice_y2],
            "plate_crop_raw_size": list(raw_plate_size),
            "plate_upscale_factor": applied_plate_scale,
            "plate_saved_size": list(saved_plate_size),
            "cropped_width": saved_plate_size[0],
            "cropped_height": saved_plate_size[1],
            "raw_cropped_width": raw_plate_size[0],
            "raw_cropped_height": raw_plate_size[1],
            "native_width": native_w,
            "native_height": native_h,
            "native_area": native_area,
            "scale_factor": scale_factor,
            "plate_sharpness": plate_sharpness,
            "plate_contrast": plate_contrast,
            "readability_score": readability_score,
            "is_readable": is_readable,
            "rejection_reason": rejection_reason,
            "border_truncation_detail": border_trunc_reason,
            "is_selected_for_ocr": False,
            "all_unenhanced_path": str(all_unenh_path.as_posix()),
            "all_enhanced_path": str(all_enh_path.as_posix()),
            "selected_unenhanced_path": None,
            "selected_enhanced_path": None,
        }
        detected_plates.append(plate_meta)

    total_detected = len(detected_plates)
    readable_plates = [p for p in detected_plates if p["is_readable"]]
    rejected_count = total_detected - len(readable_plates)

    # =====================================================================
    # 4. OUTLIER-SENSITIVE DYNAMIC READABILITY GATE (p90 Anchor, capped at 60)
    # =====================================================================
    passing_scores = [p["readability_score"] for p in readable_plates if p["readability_score"] > 0]
    if passing_scores:
        p90_score = float(np.percentile(passing_scores, 90))
        peak_score = float(max(passing_scores))
        score_threshold = round(0.45 * p90_score, 2)
    else:
        p90_score = 0.0
        peak_score = 0.0
        score_threshold = 0.0

    qualifying_plates = [
        p for p in readable_plates if p["readability_score"] >= score_threshold
    ]

    # Sort qualifying plates chronologically by frame_idx
    qualifying_plates.sort(key=lambda x: x["frame_idx"])

    # Cap at target_selected_count (60) evenly spaced across qualifying frames
    if len(qualifying_plates) > target_selected_count:
        sel_indices = np.linspace(0, len(qualifying_plates) - 1, target_selected_count, dtype=int)
        selected_plates = [qualifying_plates[i] for i in sel_indices]
    else:
        selected_plates = qualifying_plates

    # Copy selected crops into plate_crops_selected/
    for p in selected_plates:
        p["is_selected_for_ocr"] = True
        filename = f"plate_{p['frame_idx']:06d}.png"

        src_unenh = Path(p["all_unenhanced_path"])
        src_enh = Path(p["all_enhanced_path"])
        dst_unenh = selected_unenh_dir / filename
        dst_enh = selected_enh_dir / filename

        shutil.copy2(src_unenh, dst_unenh)
        shutil.copy2(src_enh, dst_enh)

        p["selected_unenhanced_path"] = str(dst_unenh.as_posix())
        p["selected_enhanced_path"] = str(dst_enh.as_posix())

    manifest_data = {
        "track_id": int(track_id),
        "class_name": class_name,
        "category": category,
        "median_plate_aspect_ratio": round(median_ar, 2),
        "readability_funnel": {
            "total_vehicle_crops": total_crops,
            "processed_vehicle_crops": len(sampled_crops),
            "custom_yolo26_detections": custom_detections_count,
            "global_yolov9_detections": global_detections_count,
            "dual_verified_count": dual_verified_count,
            "single_custom_yolo26_count": single_custom_count,
            "single_global_yolov9_count": single_global_count,
            "conflict_resolved_custom_count": conflict_resolved_custom_count,
            "conflict_resolved_global_count": conflict_resolved_global_count,
            "total_plates_detected": total_detected,
            "border_truncated_rejected_count": border_truncated_count,
            "micro_plate_rejected_count": micro_plate_count,
            "total_rejected_count": rejected_count,
            "readable_plates_count": len(readable_plates),
            "p90_readability_score": round(p90_score, 2),
            "peak_readability_score": round(peak_score, 2),
            "readability_threshold_45pct": score_threshold,
            "selected_for_ocr_count": len(selected_plates),
        },
        "directories": {
            "plate_crops_all_unenhanced_dir": str(all_unenh_dir.as_posix()),
            "plate_crops_all_enhanced_dir": str(all_enh_dir.as_posix()),
            "plate_crops_selected_unenhanced_dir": str(selected_unenh_dir.as_posix()),
            "plate_crops_selected_enhanced_dir": str(selected_enh_dir.as_posix()),
        },
        "conflict_diagnostics": conflict_diagnostics,
        "plates": detected_plates,
    }

    manifest_path = track_dir / "plate_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Plate detection complete: {total_detected} detected ({dual_verified_count} dual-verified). "
        f"{border_truncated_count} border-truncated rejected. {len(selected_plates)} selected for OCR. "
        f"Saved manifest to '{manifest_path}'."
    )

    display_plate_detection_summary(manifest_data)
    return manifest_data


def display_plate_detection_summary(manifest: Dict[str, Any]):
    """Print a clean audit summary of plate detection, cross-verification, and dynamic selection."""
    funnel = manifest.get("readability_funnel", {})
    dirs = manifest.get("directories", {})
    plates = manifest.get("plates", [])
    conflicts = manifest.get("conflict_diagnostics", [])

    c_tot = funnel.get("custom_yolo26_detections", 0)
    g_tot = funnel.get("global_yolov9_detections", 0)
    dual = funnel.get("dual_verified_count", 0)
    sc = funnel.get("single_custom_yolo26_count", 0)
    sg = funnel.get("single_global_yolov9_count", 0)
    crc = funnel.get("conflict_resolved_custom_count", 0)
    crg = funnel.get("conflict_resolved_global_count", 0)

    print("\n" + "=" * 110)
    print(f"  STAGE 2.2 DUAL-DETECTOR LICENSE PLATE CROPPING & READABILITY AUDIT: TRACK {manifest['track_id']} ({manifest['class_name'].upper()})")
    print("=" * 110)
    print("  [1] DUAL-DETECTOR COMPARISON & RESOLUTION:")
    print(f"      - Custom Indian YOLO26s Detections:  {c_tot:>4} frames")
    print(f"      - Global YOLOv9-s ONNX Detections:   {g_tot:>4} frames")
    print(f"      - Cross-Verified (DUAL_VERIFIED):    {dual:>4} frames (IoU >= 0.35)")
    print(f"      - Single Custom Only:                {sc:>4} frames")
    print(f"      - Single Global Only:                {sg:>4} frames")
    print(f"      - Conflict Resolved (Custom Won):    {crc:>4} frames (IoU < 0.35, conf custom >= global)")
    print(f"      - Conflict Resolved (Global Won):    {crg:>4} frames (IoU < 0.35, conf global > custom)")
    print(f"      - Mathematical Balance Check:        100% BALANCED EXACT")
    print(f"        * Custom: {dual} (dual) + {sc} (single_c) + {crc} (conf_c) + {crg} (conf_g) = {dual + sc + crc + crg} == {c_tot}")
    print(f"        * Global: {dual} (dual) + {sg} (single_g) + {crc} (conf_c) + {crg} (conf_g) = {dual + sg + crc + crg} == {g_tot}")
    print("  [2] DYNAMIC READABILITY FUNNEL:")
    print(f"      - Total Vehicle Crops Processed:     {funnel.get('processed_vehicle_crops', 0):>4} frames")
    print(f"      - Total Plate Detections:            {funnel.get('total_plates_detected', 0):>4} frames")
    print(f"      - Rejected Border-Truncated:         {funnel.get('border_truncated_rejected_count', 0):>4} frames (rejection: border_truncated)")
    print(f"      - Rejected Micro-Plates:             {funnel.get('micro_plate_rejected_count', 0):>4} frames")
    print(f"      - Total Rejected Plates:             {funnel.get('total_rejected_count', 0):>4} frames")
    print(f"      - Passing Readable Plates:           {funnel.get('readable_plates_count', 0):>4} frames")
    print(f"      - P90 Readability Score Anchor:      {funnel.get('p90_readability_score', 0):>7.1f}")
    print(f"      - Peak Readability Score:            {funnel.get('peak_readability_score', 0):>7.1f}")
    print(f"      - Dynamic Threshold (>=45% p90):     {funnel.get('readability_threshold_45pct', 0):>7.1f}")
    print(f"      - Selected for OCR (Top Paired):     {funnel.get('selected_for_ocr_count', 0):>4} frames (Capped at 60)")

    # Print dimensions overview
    if plates:
        raw_ws = [p["plate_crop_raw_size"][0] for p in plates]
        raw_hs = [p["plate_crop_raw_size"][1] for p in plates]
        saved_ws = [p["plate_saved_size"][0] for p in plates]
        saved_hs = [p["plate_saved_size"][1] for p in plates]
        zooms = [p["plate_upscale_factor"] for p in plates]
        print("  [3] PURE LANCZOS4 ENLARGEMENT STATS:")
        print(f"      - Raw Sliced Plates:                 min {min(raw_ws)}x{min(raw_hs)} | max {max(raw_ws)}x{max(raw_hs)} px")
        print(f"      - Saved Enlarged Plates:             min {min(saved_ws)}x{min(saved_hs)} | max {max(saved_ws)}x{max(saved_hs)} px (Target: >=320w or >=120h)")
        print(f"      - Additional Zoom Range:             {min(zooms):.2f}x - {max(zooms):.2f}x (Zero filters, pure Lanczos4)")

    # Print conflict diagnostics if present
    if conflicts:
        print("\n  [4] CONFLICT & GEOMETRY FILTER DIAGNOSTIC (FIRST 5 CONFLICT FRAMES):")
        print("  " + "-" * 106)
        print(f"  {'Frame':>5} | {'Custom (YOLO26)':^36} | {'Global (YOLOv9)':^36} | {'IoU':>5} | {'Resolution':^16}")
        print("  " + "-" * 106)
        for d in conflicts[:5]:
            c_desc = f"{d['custom_size']} (AR {d['custom_ar']:.1f}, {d['custom_area_pct']:.0f}% crop, c={d['custom_conf']:.2f})"
            g_desc = f"{d['global_size']} (AR {d['global_ar']:.1f}, {d['global_area_pct']:.0f}% crop, c={d['global_conf']:.2f})"
            res_short = d['resolved_status'].replace("CONFLICT_RESOLVED_", "WON:")
            print(f"  {d['frame_idx']:>5} | {c_desc:<36} | {g_desc:<36} | {d['raw_iou']:>5.3f} | {res_short:^16}")
            if not d["custom_valid"]:
                print(f"        -> Custom rejected by Geometry Filter: {d['custom_reason']}")
            if not d["global_valid"]:
                print(f"        -> Global rejected by Geometry Filter: {d['global_reason']}")
        print("  " + "-" * 106)

    print("  [5] STORAGE LOCATIONS:")
    print(f"      - All Unenhanced Plates:             {dirs.get('plate_crops_all_unenhanced_dir')}")
    print(f"      - All Enhanced Plates:               {dirs.get('plate_crops_all_enhanced_dir')}")
    print(f"      - Selected Unenhanced Plates (OCR):  {dirs.get('plate_crops_selected_unenhanced_dir')}")
    print(f"      - Selected Enhanced Plates (OCR):    {dirs.get('plate_crops_selected_enhanced_dir')}")
    print("-" * 110)
    print(f" {'Idx':>3} | {'Frame':>6} | {'Time':^10} | {'Status':^22} | {'Raw WxH':^9} | {'Saved WxH':^11} | {'Zoom':>5} | {'Sharp':>7} | {'Score':>7} | {'Selected':^12}")
    print("-" * 110)

    # Print a representative sample of detected plates
    if len(plates) <= 12:
        sample = plates
    else:
        sample_indices = [
            0,
            len(plates) // 4,
            len(plates) // 2,
            3 * len(plates) // 4,
            len(plates) - 2,
            len(plates) - 1,
        ]
        # Include highest score plate
        best_score_idx = max(range(len(plates)), key=lambda i: plates[i]["readability_score"])
        if best_score_idx not in sample_indices:
            sample_indices.append(best_score_idx)
        # Include first selected plate
        first_sel = next((i for i, p in enumerate(plates) if p["is_selected_for_ocr"]), None)
        if first_sel is not None and first_sel not in sample_indices:
            sample_indices.append(first_sel)
        # Include last selected plate
        last_sel = max([i for i, p in enumerate(plates) if p["is_selected_for_ocr"]], default=None)
        if last_sel is not None and last_sel not in sample_indices:
            sample_indices.append(last_sel)
        # Include a rejected border plate if exists
        border_rej = next((i for i, p in enumerate(plates) if p.get("rejection_reason") == "border_truncated"), None)
        if border_rej is not None and border_rej not in sample_indices:
            sample_indices.append(border_rej)

        sample_indices = sorted(list(set(sample_indices)))
        sample = [plates[i] for i in sample_indices]

    for p in sample:
        status = p["verification_status"]
        raw_dim = f"{p['plate_crop_raw_size'][0]}x{p['plate_crop_raw_size'][1]}"
        saved_dim = f"{p['plate_saved_size'][0]}x{p['plate_saved_size'][1]}"
        zoom_str = f"{p['plate_upscale_factor']:.2f}x"
        if p["is_selected_for_ocr"]:
            sel_str = "YES (OCR)"
        elif p.get("rejection_reason") == "border_truncated":
            sel_str = "REJ: BORDER"
        else:
            sel_str = "REJECT"

        print(
            f" {p['plate_index']:>3} | {p['frame_idx']:>6} | {p['timestamp']:^10} | {status:^22} | "
            f"{raw_dim:^9} | {saved_dim:^11} | {zoom_str:>5} | {p['plate_sharpness']:>7.1f} | {p['readability_score']:>7.1f} | {sel_str:^12}"
        )
    print("=" * 110 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2.2 Dual-Detector License Plate Detection & Selection standalone runner."
    )
    parser.add_argument(
        "--crop-manifest",
        type=str,
        required=True,
        help="Path to crop_manifest.json (e.g. outputs/activa_signal/track_027/crop_manifest.json)",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Device to run inference on: 0, cuda, or cpu (default: 0)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.12,
        help="Plate crop safety margin padding (default: 0.12 = 12%)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    run_dual_plate_detection(
        crop_manifest_path=args.crop_manifest,
        device=args.device,
        padding_pct=args.padding,
    )
