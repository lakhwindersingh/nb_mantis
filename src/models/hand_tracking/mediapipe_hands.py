""" MediaPipe-based hand tracking implementation """
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any, Optional

class MediaPipeHandDetector:
    """Hand detector using MediaPipe"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.get('max_num_hands', 2),
            min_detection_confidence=self.config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
        )

        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray):
        """
        Detect hand landmarks in frame

        Args:
            frame: Input image frame

        Returns:
            MediaPipe detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.hands.process(rgb_frame)

        return results

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame