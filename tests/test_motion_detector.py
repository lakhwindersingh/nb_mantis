""" Tests for motion detector """
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.agents.motion_detector import MotionDetector
class TestMotionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MotionDetector()

    def test_empty_frame(self):
        """Test motion detection with empty frame"""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_motions(empty_frame)

        self.assertIsInstance(result, dict)
        self.assertIn('face', result)
        self.assertIn('hands', result)

    def test_none_frame(self):
        """Test motion detection with None frame"""
        result = self.detector.detect_motions(None)

        self.assertIsInstance(result, dict)
        self.assertEqual(result['face']['detected'], False)
        self.assertEqual(result['hands']['detected'], False)

    if __name__ == '__main__':
        unittest.main()