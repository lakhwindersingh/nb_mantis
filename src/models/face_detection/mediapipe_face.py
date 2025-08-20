""" MediaPipe-based face detection implementation """
import cv2
import mediapipe as mp
import numpy as np
from typing import (Dict, Any, Optional)
class MediaPipeFaceDetector:
    """Face detector using MediaPipe"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect(self, frame: np.ndarray):
        """
        Detect face landmarks in frame

        Args:
            frame: Input image frame

        Returns:
            MediaPipe detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.face_mesh.process(rgb_frame)

        return results

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw face landmarks on frame"""
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_spec
                )
        return frame