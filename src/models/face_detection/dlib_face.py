""" Dlib-based face detection implementation """
import cv2
import dlib
import numpy as np
from typing import Dict, Any, Optional

class DlibFaceDetector:
    """Face detector using Dlib"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()

        # Path to shape predictor model
        predictor_path = self.config.get('predictor_path')
        if predictor_path:
            self.predictor = dlib.shape_predictor(predictor_path)
        else:
            self.predictor = None
            print("Warning: No shape predictor path provided")

    def detect(self, frame: np.ndarray):
        """
        Detect face landmarks in frame

        Args:
            frame: Input image frame

        Returns:
            List of detected face landmarks
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        landmarks_list = []

        if self.predictor:
            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks_points = []

                for i in range(68):  # 68 face landmarks
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    landmarks_points.append([x, y])

                landmarks_list.append(landmarks_points)

        return landmarks_list

    def draw_landmarks(self, frame: np.ndarray, landmarks_list: list) -> np.ndarray:
        """Draw face landmarks on frame"""
        for landmarks in landmarks_list:
            for point in landmarks:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        return frame