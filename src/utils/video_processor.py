""" Video processing utilities """
import cv2
import numpy as np
from typing import Optional, Tuple, Iterator
import logging

class VideoProcessor:
    """Handles video input/output operations"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def read_video_frames(self, video_path: str) -> Iterator[np.ndarray]:
        """
        Read frames from video file

        Args:
            video_path: Path to video file

        Yields:
            Video frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def write_video(self, frames: Iterator[np.ndarray],
                   output_path: str, fps: int = 30,
                   resolution: Optional[Tuple[int, int]] = None):
        """
        Write frames to video file

        Args:
            frames: Iterator of video frames
            output_path: Output video file path
            fps: Frames per second
            resolution: Output resolution (width, height)
        """
        writer = None

        try:
            for frame in frames:
                if writer is None:
                    # Initialize writer with first frame
                    if resolution:
                        frame = cv2.resize(frame, resolution)

                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                    if not writer.isOpened():
                        self.logger.error(f"Failed to create video writer: {output_path}")
                        return

                if resolution:
                    frame = cv2.resize(frame, resolution)

                writer.write(frame)

        finally:
            if writer:
                writer.release()

    def get_camera_stream(self, camera_id: int = 0) -> cv2.VideoCapture:
        """
        Get camera stream

        Args:
            camera_id: Camera device ID

        Returns:
            OpenCV VideoCapture object
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            self.logger.error(f"Failed to open camera: {camera_id}")
            return None

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        return cap