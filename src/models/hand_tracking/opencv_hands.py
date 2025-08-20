""" OpenCV-based hand tracking implementation """
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class OpenCVHandDetector:
    """Simple hand detector using OpenCV"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Contour filtering parameters
        self.min_contour_area = self.config.get('min_contour_area', 5000)
        self.max_contour_area = self.config.get('max_contour_area', 50000)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hands using contour detection

        Args:
            frame: Input image frame

        Returns:
            List of detected hand regions
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_regions = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_contour_area < area < self.max_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center and basic features
                center = (x + w//2, y + h//2)

                hand_regions.append({
                    'bounding_box': (x, y, w, h),
                    'center': center,
                    'area': area,
                    'contour': contour
                })

        return hand_regions

    def draw_detections(self, frame: np.ndarray, hand_regions: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detected hand regions"""
        for region in hand_regions:
            x, y, w, h = region['bounding_box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, region['center'], 5, (255, 0, 0), -1)

        return frame