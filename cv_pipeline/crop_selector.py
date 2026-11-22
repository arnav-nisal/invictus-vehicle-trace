"""Selector for finding the clearest and sharpest cropped photo for each tracked vehicle."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class CropCandidate:
    frame_idx: int
    timestamp_sec: float
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    sharpness: float
    quality_score: float
    crop_image: np.ndarray


class BestCropSelector:
    """Maintains and selects the best cropped image for each tracked vehicle ID."""

    def __init__(self, padding_ratio: float = 0.05, min_size: int = 8):
        self.padding_ratio = padding_ratio
        self.min_size = min_size
        self._best_candidates: Dict[int, CropCandidate] = {}

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate image sharpness using variance of the Laplacian."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: list[float],
        confidence: float,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        """Evaluate a new observation of a track and store it if it exceeds previous quality."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

        # Bound coordinates
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < self.min_size or box_h < self.min_size:
            # Even if below min_size, store as candidate if none exists yet so every track has a photo
            if track_id not in self._best_candidates and box_w > 0 and box_h > 0:
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    sharpness = self.calculate_sharpness(crop)
                    self._best_candidates[track_id] = CropCandidate(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        bbox=[x1, y1, x2, y2],
                        confidence=float(confidence),
                        sharpness=round(sharpness, 2),
                        quality_score=0.1,
                        crop_image=crop,
                    )
            return

        # Add slight padding around vehicle for a cleaner photo
        pad_x = int(box_w * self.padding_ratio)
        pad_y = int(box_h * self.padding_ratio)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            return

        sharpness = self.calculate_sharpness(crop)

        # Border penalty: give slight preference to crops not clipped at image edges
        is_touching_border = (x1 <= 2) or (y1 <= 2) or (x2 >= w - 2) or (y2 >= h - 2)
        border_multiplier = 0.75 if is_touching_border else 1.0

        # Quality score considers sharpness, detection confidence, resolution, and boundary padding
        area = box_w * box_h
        size_factor = min(1.2, np.sqrt(area) / 100.0)
        quality_score = sharpness * (confidence ** 0.5) * border_multiplier * size_factor

        candidate = CropCandidate(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            bbox=[x1, y1, x2, y2],
            confidence=float(confidence),
            sharpness=round(sharpness, 2),
            quality_score=round(quality_score, 2),
            crop_image=crop,
        )

        if track_id not in self._best_candidates:
            self._best_candidates[track_id] = candidate
        else:
            if quality_score > self._best_candidates[track_id].quality_score:
                self._best_candidates[track_id] = candidate

    def get_best(self, track_id: int) -> Optional[CropCandidate]:
        """Get best crop candidate for a track ID."""
        return self._best_candidates.get(track_id)

    def get_all(self) -> Dict[int, CropCandidate]:
        """Get all stored best crop candidates."""
        return self._best_candidates
