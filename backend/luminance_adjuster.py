"""
Robust Luminance Adjuster Module
Handles lighting normalization for consistent rPPG signal extraction.
Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and
white-balance correction to stabilize luminance across frames.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class LuminanceAdjuster:
    """Adjusts for changes in lighting conditions to improve consistency."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        self._reference_luminance: Optional[float] = None
        self._ema_alpha = 0.05  # Exponential moving average for smooth tracking
        self._running_mean_luminance: Optional[float] = None

    def reset(self):
        """Reset internal state for a new session."""
        self._reference_luminance = None
        self._running_mean_luminance = None

    def adjust_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply robust luminance normalization to a BGR frame.
        
        Steps:
        1. Convert to LAB color space
        2. Apply CLAHE on the L (lightness) channel
        3. Normalize against running mean luminance
        4. Convert back to BGR
        """
        if frame is None or frame.size == 0:
            return frame

        # Convert BGR to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the L channel
        l_enhanced = self.clahe.apply(l_channel)

        # Track luminance for normalization
        current_mean = float(np.mean(l_enhanced))
        if self._reference_luminance is None:
            self._reference_luminance = current_mean
            self._running_mean_luminance = current_mean
        else:
            self._running_mean_luminance = (
                self._ema_alpha * current_mean +
                (1 - self._ema_alpha) * self._running_mean_luminance
            )

        # Normalize luminance to maintain consistency
        if self._running_mean_luminance > 0:
            scale = self._reference_luminance / self._running_mean_luminance
            scale = np.clip(scale, 0.7, 1.4)  # Clamp to avoid extreme adjustments
            l_enhanced = np.clip(l_enhanced * scale, 0, 255).astype(np.uint8)

        # Merge channels back
        lab_adjusted = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

        return result

    def get_white_balanced(self, frame: np.ndarray) -> np.ndarray:
        """Apply simple white balance using gray-world assumption."""
        if frame is None or frame.size == 0:
            return frame

        result = frame.copy().astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        if avg_b > 0:
            result[:, :, 0] *= avg_all / avg_b
        if avg_g > 0:
            result[:, :, 1] *= avg_all / avg_g
        if avg_r > 0:
            result[:, :, 2] *= avg_all / avg_r

        return np.clip(result, 0, 255).astype(np.uint8)

    def full_adjustment(self, frame: np.ndarray) -> np.ndarray:
        """Apply both white balance and CLAHE luminance normalization."""
        balanced = self.get_white_balanced(frame)
        adjusted = self.adjust_frame(balanced)
        return adjusted

    def get_luminance_stats(self) -> dict:
        """Return current luminance tracking statistics."""
        return {
            "reference_luminance": self._reference_luminance,
            "running_mean_luminance": self._running_mean_luminance,
        }
