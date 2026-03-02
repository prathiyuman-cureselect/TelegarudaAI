"""
Face Recognition Engine Module
Handles face detection, embedding extraction, and face comparison.
Uses MediaPipe for real-time face detection/mesh and
a lightweight approach for face recognition.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import base64
import json
import os
import time


class FaceRecognitionEngine:
    """Face detection, landmark extraction, and recognition engine."""

    def __init__(self, known_faces_dir: str = "./known_faces"):
        # MediaPipe Face Mesh for detailed landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4, # Lowered from 0.6
            min_tracking_confidence=0.4, # Lowered from 0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range (up to 5m) is more robust
            min_detection_confidence=0.3, # Even more lenient
        )

        # Face database
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings: Dict[str, np.ndarray] = {}
        self.known_face_names: List[str] = []
        
        # ROI indices for rPPG (forehead and cheek regions from MediaPipe landmarks)
        # Forehead landmarks
        self.forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        # Left cheek
        self.left_cheek_indices = [36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 120, 121, 128, 245]
        # Right cheek
        self.right_cheek_indices = [266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 349, 350, 357, 465]

        os.makedirs(known_faces_dir, exist_ok=True)

    def detect_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face in the frame using MediaPipe.
        Returns face bounding box, landmarks, and ROI for rPPG.
        """
        if frame is None or frame.size == 0:
            return None

        h, w, _ = frame.shape
        # Pre-process for detection only: Boost brightness and contrast
        # This helps in dim lighting without affecting the rPPG signal (which uses the original/adjusted frame)
        alpha = 1.3 # Contrast
        beta = 10   # Brightness
        detect_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)

        # Face mesh detection
        mesh_results = self.face_mesh.process(rgb_frame)

        if not mesh_results.multi_face_landmarks:
            # Fallback to simple face detection
            detect_results = self.face_detection.process(rgb_frame)
            if not detect_results.detections:
                return None
            
            # Get primary face from detection
            detection = detect_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Simple bounding box from detection
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Forehead ROI approximation (upper 1/3 of face box)
            # Use small crop for detection-only fallback
            x, y, bw, bh = max(0, x_min), max(0, y_min), min(width, w-x_min), min(height, h-y_min)
            rppg_roi = frame[y:y+bh//3, x+bw//4:x+3*bw//4] if bh > 10 else None

            return {
                "detected": True,
                "bbox": {"x": x, "y": y, "width": bw, "height": bh},
                "landmarks": None,
                "landmark_count": 0,
                "rppg_roi": rppg_roi,
                "embedding": None,
                "confidence": float(detection.score[0]),
            }

        # Face mesh detected
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        landmarks_px = []
        for lm in face_landmarks.landmark:
            landmarks_px.append([int(lm.x * w), int(lm.y * h)])
        landmarks_array = np.array(landmarks_px)

        # Get bounding box
        x_min = max(0, np.min(landmarks_array[:, 0]) - 20)
        y_min = max(0, np.min(landmarks_array[:, 1]) - 20)
        x_max = min(w, np.max(landmarks_array[:, 0]) + 20)
        y_max = min(h, np.max(landmarks_array[:, 1]) + 20)

        # Extract ROI for rPPG
        forehead_roi = self._extract_roi(frame, landmarks_array, self.forehead_indices)
        left_cheek_roi = self._extract_roi(frame, landmarks_array, self.left_cheek_indices)
        right_cheek_roi = self._extract_roi(frame, landmarks_array, self.right_cheek_indices)

        # Combine ROIs for rPPG
        rppg_roi = self._combine_rois(forehead_roi, left_cheek_roi, right_cheek_roi)

        # Compute face embedding (using landmark geometry)
        embedding = self._compute_embedding(landmarks_array, w, h)

        return {
            "detected": True,
            "bbox": {
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min),
                "height": int(y_max - y_min),
            },
            "landmarks": landmarks_array,
            "landmark_count": len(landmarks_array),
            "rppg_roi": rppg_roi,
            "embedding": embedding,
            "confidence": self._estimate_detection_confidence(landmarks_array, w, h),
        }

    def _extract_roi(self, frame: np.ndarray, landmarks: np.ndarray, indices: List[int]) -> Optional[np.ndarray]:
        """Extract a region of interest from face using landmark indices."""
        try:
            valid_indices = [i for i in indices if i < len(landmarks)]
            if len(valid_indices) < 3:
                return None

            points = landmarks[valid_indices]
            
            # Create a mask for the ROI
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # Get bounding rect
            x, y, bw, bh = cv2.boundingRect(hull)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            if bw <= 0 or bh <= 0:
                return None

            roi = frame[y:y+bh, x:x+bw].copy()
            roi_mask = mask[y:y+bh, x:x+bw]

            # Apply mask to ROI
            roi[roi_mask == 0] = 0

            return roi
        except Exception:
            return None

    def _combine_rois(self, *rois) -> Optional[np.ndarray]:
        """Combine multiple ROIs into a single ROI for rPPG analysis."""
        valid_rois = [r for r in rois if r is not None and r.size > 0]
        if not valid_rois:
            return None

        # Use the largest ROI, but average pixel values from all
        max_roi = max(valid_rois, key=lambda r: r.size)
        
        if len(valid_rois) == 1:
            return max_roi

        # Stack and average all ROIs (resized to same dimensions)
        target_h, target_w = max_roi.shape[:2]
        combined_pixels = []
        
        for roi in valid_rois:
            resized = cv2.resize(roi, (target_w, target_h))
            # Only include non-zero pixels
            non_zero_mask = np.any(resized > 0, axis=2)
            if np.any(non_zero_mask):
                combined_pixels.append(resized[non_zero_mask])

        if not combined_pixels:
            return max_roi

        return max_roi  # Return the primary ROI for rPPG analysis

    def _compute_embedding(self, landmarks: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
        """
        Compute a geometric face embedding from landmark positions.
        Normalized to be scale and translation invariant.
        """
        # Normalize landmarks to [0, 1]
        norm_landmarks = landmarks.astype(float)
        norm_landmarks[:, 0] /= frame_w
        norm_landmarks[:, 1] /= frame_h

        # Center landmarks
        center = np.mean(norm_landmarks, axis=0)
        centered = norm_landmarks - center

        # Scale to unit variance
        scale = np.std(centered) + 1e-8
        normalized = centered / scale

        # Extract key geometric features
        # Use a subset of landmarks for efficiency
        key_indices = [
            10, 152,  # Top/bottom of face
            234, 454,  # Left/right of face
            1, 4,  # Nose bridge, nose tip
            33, 263,  # Left/right eye inner corners
            61, 291,  # Left/right mouth corners
            70, 300,  # Left/right eyebrow
            13, 14,  # Upper/lower lip
        ]
        
        valid_keys = [i for i in key_indices if i < len(normalized)]
        key_points = normalized[valid_keys]

        # Compute pairwise distances as features
        n = len(key_points)
        features = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(key_points[i] - key_points[j])
                features.append(dist)

        # Also include angles between key triplets
        for i in range(0, min(n, 10), 3):
            if i + 2 < n:
                v1 = key_points[i + 1] - key_points[i]
                v2 = key_points[i + 2] - key_points[i]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                features.append(cos_angle)

        return np.array(features, dtype=np.float32)

    def _estimate_detection_confidence(self, landmarks: np.ndarray, w: int, h: int) -> float:
        """Estimate face detection confidence based on landmark distribution."""
        norm_x = landmarks[:, 0] / w
        norm_y = landmarks[:, 1] / h

        # Check if landmarks are well-distributed (not clustered)
        x_spread = np.std(norm_x)
        y_spread = np.std(norm_y)

        # A well-detected face should have reasonable spread
        spread_score = min(1.0, (x_spread + y_spread) * 5.0)

        # Check if face is within frame
        margin_score = 1.0
        if np.min(norm_x) < 0.05 or np.max(norm_x) > 0.95:
            margin_score *= 0.7
        if np.min(norm_y) < 0.05 or np.max(norm_y) > 0.95:
            margin_score *= 0.7

        confidence = spread_score * margin_score
        return round(float(np.clip(confidence, 0, 1)), 3)

    def register_face(self, name: str, embedding: np.ndarray) -> bool:
        """Register a new face in the database."""
        try:
            self.known_face_encodings[name] = embedding
            self.known_face_names.append(name)

            # Save to disk
            save_path = os.path.join(self.known_faces_dir, f"{name}.npy")
            np.save(save_path, embedding)
            return True
        except Exception:
            return False

    def identify_face(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[Dict]:
        """
        Identify a face by comparing its embedding to known faces.
        Returns the best match if above threshold.
        """
        if not self.known_face_encodings:
            return None

        best_match = None
        best_score = 0.0

        for name, known_embedding in self.known_face_encodings.items():
            # Ensure both embeddings are the same length
            min_len = min(len(embedding), len(known_embedding))
            if min_len == 0:
                continue

            e1 = embedding[:min_len]
            e2 = known_embedding[:min_len]

            # Cosine similarity
            dot = np.dot(e1, e2)
            norm1 = np.linalg.norm(e1) + 1e-8
            norm2 = np.linalg.norm(e2) + 1e-8
            similarity = dot / (norm1 * norm2)

            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_match and best_score >= threshold:
            return {
                "name": best_match,
                "confidence": round(float(best_score), 3),
            }

        return None

    def load_known_faces(self):
        """Load all known face encodings from disk."""
        if not os.path.exists(self.known_faces_dir):
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(".npy"):
                name = filename[:-4]
                filepath = os.path.join(self.known_faces_dir, filename)
                try:
                    embedding = np.load(filepath)
                    self.known_face_encodings[name] = embedding
                    if name not in self.known_face_names:
                        self.known_face_names.append(name)
                except Exception:
                    continue

    def get_face_roi_visualization(self, frame: np.ndarray, face_data: Dict) -> np.ndarray:
        """Draw face detection visualization on the frame."""
        vis = frame.copy()
        
        if face_data and face_data.get("detected"):
            bbox = face_data["bbox"]
            
            # Draw bounding box
            cv2.rectangle(
                vis,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                (0, 255, 0), 2
            )

            # Draw confidence
            conf = face_data.get("confidence", 0)
            cv2.putText(
                vis,
                f"Conf: {conf:.2f}",
                (bbox["x"], bbox["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        return vis

    def cleanup(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
        self.face_detection.close()
