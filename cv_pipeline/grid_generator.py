"""Contact-sheet grid generator for visual verification of tracked vehicles."""

import math
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np


class ContactSheetGenerator:
    """Creates a high-resolution contact-sheet grid image of all tracked vehicles."""

    def __init__(
        self,
        tile_width: int = 260,
        tile_height: int = 280,
        cols: int = 5,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        self.tile_w = tile_width
        self.tile_h = tile_height
        self.cols = cols
        self.font = font

    def create_grid(
        self,
        tracks: List[Dict[str, Any]],
        output_path: str,
        title: str = "INVICTUS VEHICLE TRACE — TRACKED VEHICLES CONTACT SHEET",
        subtitle: str = "Ultralytics yolo26s (1280px) + BoT-SORT on NVIDIA RTX 3050",
    ) -> str:
        """Render a clean grid image of all kept vehicle crops with ID and timestamps."""
        if not tracks:
            # Create a blank fallback image
            blank = np.zeros((300, 600, 3), dtype=np.uint8)
            cv2.putText(blank, "No tracks kept", (50, 150), self.font, 1.0, (255, 255, 255), 2)
            cv2.imwrite(output_path, blank)
            return output_path

        n = len(tracks)
        cols = min(self.cols, n)
        rows = math.ceil(n / cols)

        header_h = 100
        padding = 12
        grid_w = cols * (self.tile_w + padding) + padding
        grid_h = header_h + rows * (self.tile_h + padding) + padding

        # Dark slate background canvas
        canvas = np.full((grid_h, grid_w, 3), (22, 26, 32), dtype=np.uint8)

        # Draw header banner
        cv2.rectangle(canvas, (0, 0), (grid_w, header_h), (14, 17, 22), -1)
        cv2.line(canvas, (0, header_h), (grid_w, header_h), (50, 65, 80), 2)

        cv2.putText(canvas, title, (24, 42), self.font, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
        sub_text = f"{subtitle}  |  Total Kept Vehicles: {n}"
        cv2.putText(canvas, sub_text, (24, 74), self.font, 0.50, (180, 200, 220), 1, cv2.LINE_AA)

        # Render each tile
        for idx, trk in enumerate(tracks):
            r = idx // cols
            c = idx % cols

            x0 = padding + c * (self.tile_w + padding)
            y0 = header_h + padding + r * (self.tile_h + padding)
            x1 = x0 + self.tile_w
            y1 = y0 + self.tile_h

            tid = trk["id"]
            label = trk["class_name"]
            frame_cnt = trk["frame_count"]
            start_ts = trk.get("start_timestamp", "")[:8]
            end_ts = trk.get("end_timestamp", "")[:8]
            dur = trk.get("duration_seconds", 0.0)

            # Check if this is the moving Activa (ID 17) or waiting Activa (ID 4)
            is_activa = (tid == 17) or (tid == 4) or ("activa" in trk.get("notes", "").lower())
            tile_border_color = (0, 215, 255) if is_activa else (50, 62, 75)
            badge_bg_color = (0, 160, 230) if is_activa else (40, 52, 65)

            # Tile background & border
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (18, 22, 28), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), tile_border_color, 2 if is_activa else 1, cv2.LINE_AA)

            # Tile header (badge)
            badge_h = 34
            cv2.rectangle(canvas, (x0, y0), (x1, y0 + badge_h), badge_bg_color, -1)
            cv2.line(canvas, (x0, y0 + badge_h), (x1, y0 + badge_h), tile_border_color, 1)

            header_text = f"ID #{tid} | {label.upper()}"
            if is_activa and tid == 17:
                header_text = f"ID #{tid} | ACTIVA (MOVING)"
            elif is_activa and tid == 4:
                header_text = f"ID #{tid} | ACTIVA (SIGNAL)"

            cv2.putText(canvas, header_text, (x0 + 8, y0 + 23), self.font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # Tile footer (timestamps)
            footer_h = 42
            footer_y0 = y1 - footer_h
            cv2.rectangle(canvas, (x0, footer_y0), (x1, y1), (14, 17, 22), -1)
            cv2.line(canvas, (x0, footer_y0), (x1, footer_y0), (45, 55, 65), 1)

            time_text = f"{start_ts} -> {end_ts}"
            dur_text = f"{dur:.2f}s ({frame_cnt} frames)"
            cv2.putText(canvas, time_text, (x0 + 8, footer_y0 + 18), self.font, 0.42, (200, 220, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, dur_text, (x0 + 8, footer_y0 + 35), self.font, 0.40, (140, 160, 180), 1, cv2.LINE_AA)

            # Crop image placement (between header and footer)
            img_y0 = y0 + badge_h + 2
            img_y1 = footer_y0 - 2
            target_w = self.tile_w - 4
            target_h = img_y1 - img_y0

            crop_img = trk.get("crop_image")
            if crop_img is None and trk.get("saved_image_path"):
                p = Path(trk["saved_image_path"])
                if p.exists():
                    crop_img = cv2.imread(str(p))

            if crop_img is not None and crop_img.size > 0:
                ih, iw = crop_img.shape[:2]
                scale = min(target_w / iw, target_h / ih)
                nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                resized = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_AREA)

                # Center the resized crop inside the image box
                off_x = x0 + 2 + (target_w - nw) // 2
                off_y = img_y0 + (target_h - nh) // 2
                canvas[off_y : off_y + nh, off_x : off_x + nw] = resized
            else:
                cv2.putText(canvas, "No Photo", (x0 + 70, y0 + 130), self.font, 0.5, (100, 100, 100), 1)

        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return output_path
