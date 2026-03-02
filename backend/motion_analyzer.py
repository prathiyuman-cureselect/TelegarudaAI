"""
Motion Analyzer Module
Detects and quantifies motion within video frames to:
1. Flag excessive motion that would corrupt rPPG signals
2. Track facial landmark stability for recognition quality
3. Provide motion-compensated ROI tracking
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict


class MotionAnalyzer:
    """Analyzes movement within frames to enhance facial recognition accuracy."""

    # Thresholds
    MOTION_LOW = 0.5
    MOTION_MODERATE = 2.0
    MOTION_HIGH = 5.0

    def __init__(self, buffer_size: int = 30):
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_landmarks: Optional[np.ndarray] = None
        self._motion_history: deque = deque(maxlen=buffer_size)
        self._landmark_stability_history: deque = deque(maxlen=buffer_size)
        self._optical_flow_params = dict(
            pyrScale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            polyN=5,
            polySigma=1.2,
            flags=0
        )

    def reset(self):
        """Reset analyzer state for a new session."""
        self._prev_gray = None
        self._prev_landmarks = None
        self._motion_history.clear()
        self._landmark_stability_history.clear()

    def analyze_frame(self, frame: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze motion in the current frame.
        
        Returns:
            dict with keys:
                - motion_score: float (0 = no motion, higher = more motion)
                - motion_level: str ('low', 'moderate', 'high', 'excessive')
                - is_stable: bool (True if suitable for rPPG measurement)
                - landmark_stability: float (if landmarks provided)
                - motion_vector: tuple (dx, dy) average displacement
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = {
            "motion_score": 0.0,
            "motion_level": "low",
            "is_stable": True,
            "landmark_stability": 1.0,
            "motion_vector": (0.0, 0.0),
        }

        if self._prev_gray is not None:
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, **self._optical_flow_params
            )

            # Calculate motion magnitude
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = float(np.mean(mag))
            avg_dx = float(np.mean(flow[..., 0]))
            avg_dy = float(np.mean(flow[..., 1]))

            self._motion_history.append(motion_score)

            # Classify motion level
            if motion_score < self.MOTION_LOW:
                motion_level = "low"
            elif motion_score < self.MOTION_MODERATE:
                motion_level = "moderate"
            elif motion_score < self.MOTION_HIGH:
                motion_level = "high"
            else:
                motion_level = "excessive"

            result["motion_score"] = round(motion_score, 3)
            result["motion_level"] = motion_level
            result["is_stable"] = motion_score < self.MOTION_MODERATE
            result["motion_vector"] = (round(avg_dx, 2), round(avg_dy, 2))

        # Landmark stability analysis
        if face_landmarks is not None and self._prev_landmarks is not None:
            if face_landmarks.shape == self._prev_landmarks.shape:
                displacement = np.linalg.norm(
                    face_landmarks - self._prev_landmarks, axis=1
                )
                stability = 1.0 / (1.0 + float(np.mean(displacement)))
                self._landmark_stability_history.append(stability)
                result["landmark_stability"] = round(stability, 4)

        # Update previous state
        self._prev_gray = gray.copy()
        if face_landmarks is not None:
            self._prev_landmarks = face_landmarks.copy()

        return result

    def get_motion_quality(self) -> float:
        """
        Get overall motion quality score (0-1).
        1.0 = perfectly stable, 0.0 = excessive motion.
        """
        if not self._motion_history:
            return 1.0
        avg_motion = np.mean(list(self._motion_history))
        quality = max(0.0, 1.0 - (avg_motion / self.MOTION_HIGH))
        return round(quality, 3)

    def get_average_motion(self) -> float:
        """Get the average motion score over the buffer."""
        if not self._motion_history:
            return 0.0
        return round(float(np.mean(list(self._motion_history))), 3)

    def should_skip_frame(self) -> bool:
        """Check if current motion is too high for reliable rPPG measurement."""
        if not self._motion_history:
            return False
        recent = list(self._motion_history)[-5:]
        return float(np.mean(recent)) > self.MOTION_HIGH

    def get_stats(self) -> Dict:
        """Return comprehensive motion statistics."""
        return {
            "average_motion": self.get_average_motion(),
            "motion_quality": self.get_motion_quality(),
            "should_skip": self.should_skip_frame(),
            "buffer_size": len(self._motion_history),
            "landmark_stability_avg": (
                round(float(np.mean(list(self._landmark_stability_history))), 4)
                if self._landmark_stability_history else 1.0
            ),
        }
