"""Video frame visual annotator with clean bounding boxes, ID badges, and tracking HUD."""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class VehicleAnnotator:
    """Renders polished visual annotations for tracked vehicles."""

    def __init__(self, font_scale: float = 0.65, box_thickness: int = 2):
        self.font_scale = font_scale
        self.box_thickness = box_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a deterministic, vibrant BGR color for a given track ID."""
        if track_id in self._color_cache:
            return self._color_cache[track_id]

        # Use golden ratio hue spacing for high distinction
        hue = int((track_id * 137.508) % 360 / 2)  # OpenCV hue range is 0-179
        saturation = 220
        value = 240
        hsv_pixel = np.uint8([[[hue, saturation, value]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
        self._color_cache[track_id] = color
        return color

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[dict],
        frame_idx: int,
        total_frames: int,
        fps: float,
        total_unique_ids: int,
    ) -> np.ndarray:
        """Draw bounding boxes, ID badges, and status overlay on the frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw each tracked vehicle
        for trk in tracks:
            track_id = trk["id"]
            label = trk["label"]
            conf = trk.get("conf", 1.0)
            x1, y1, x2, y2 = [int(round(c)) for c in trk["bbox"]]

            color = self.get_color(track_id)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

            # Draw small corner accents on the box for a professional tech aesthetic
            accent_len = max(8, min(24, int((x2 - x1) * 0.15)))
            cv2.line(annotated, (x1, y1), (x1 + accent_len, y1), color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + accent_len), color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2 - accent_len, y1), color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + accent_len), color, self.box_thickness + 1, cv2.LINE_AA)

            cv2.line(annotated, (x1, y2), (x1 + accent_len, y2), color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1, y2 - accent_len), color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2 - accent_len, y2), color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - accent_len), color, self.box_thickness + 1, cv2.LINE_AA)

            # Prepare badge text: e.g. "ID #3 | motorcycle 0.88"
            badge_text = f"ID #{track_id} | {label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, self.font, self.font_scale, 2)

            # Position badge above box, or below if near top
            badge_y2 = y1 - 4 if y1 - text_h - 10 > 0 else y1 + text_h + 16
            badge_y1 = badge_y2 - text_h - 8
            badge_x1 = max(0, x1)
            badge_x2 = min(w, badge_x1 + text_w + 14)

            # Badge background (solid color matching track ID)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Subtle dark inner border
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), 1, cv2.LINE_AA)

            # Text color: dark or light based on color luminance
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (15, 15, 15) if luminance > 128 else (255, 255, 255)

            cv2.putText(
                annotated,
                badge_text,
                (badge_x1 + 7, badge_y2 - 6),
                self.font,
                self.font_scale,
                text_color,
                2,
                cv2.LINE_AA,
            )

        # Draw HUD banner overlay at top-left
        self._draw_hud(annotated, frame_idx, total_frames, fps, len(tracks), total_unique_ids)

        return annotated

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        active_count: int,
        total_unique_ids: int,
    ) -> None:
        """Draw sleek semi-transparent telemetry overlay."""
        timestamp_sec = frame_idx / fps if fps > 0 else 0
        minutes = int(timestamp_sec // 60)
        seconds = timestamp_sec % 60

        hud_w, hud_h = 360, 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (20 + hud_w, 20 + hud_h), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (20, 20), (20 + hud_w, 20 + hud_h), (80, 110, 140), 1, cv2.LINE_AA)

        # Header
        cv2.putText(
            frame,
            "INVICTUS VEHICLE TRACE — BoT-SORT + ReID",
            (32, 42),
            self.font,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        # Telemetry metrics
        time_str = f"Time: {minutes:02d}:{seconds:05.2f} ({frame_idx + 1}/{total_frames})"
        counts_str = f"Active: {active_count:2d}  |  Total Vehicles: {total_unique_ids:2d}"

        cv2.putText(frame, time_str, (32, 63), self.font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, counts_str, (32, 84), self.font, 0.45, (180, 240, 180), 1, cv2.LINE_AA)
