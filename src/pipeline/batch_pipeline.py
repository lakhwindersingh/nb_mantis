""" Batch video processing pipeline """
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Iterator
import logging
from tqdm import (tqdm)
from src.agents.motion_detector import MotionDetector
from src.agents.character_animator import CharacterAnimator
from src.agents.background_generator import BackgroundGenerator
from src.utils.video_processor import VideoProcessor
from src.utils.image_utils import pygame_to_numpy
from config.settings import load_custom_config, CHARACTER_CONFIG

class BatchPipeline:
    """Batch video processing pipeline"""
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = load_custom_config(config_path) if config_path else {}

        # Initialize components
        self.motion_detector = MotionDetector(self.config.get('motion_detection', {}))
        self.character_animator = CharacterAnimator(
            CHARACTER_CONFIG,
            Path(self.config.get('assets_path', 'assets'))
        )
        self.background_generator = BackgroundGenerator(self.config.get('background', {}))
        self.video_processor = VideoProcessor()

        self.logger.info("Batch pipeline initialized")

    def process_video(self, input_path: str, output_path: Optional[str] = None):
        """
        Process video file

        Args:
            input_path: Input video file path
            output_path: Output video file path
        """
        if not output_path:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_mimic.mp4")

        self.logger.info(f"Processing video: {input_path} -> {output_path}")

        # Get video properties
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Process frames
        processed_frames = self._process_frames_generator(input_path, total_frames)

        # Write output video
        output_resolution = self.config.get('output_resolution', (1920, 1080))
        self.video_processor.write_video(
            processed_frames,
            output_path,
            int(fps),
            output_resolution
        )

        self.logger.info(f"Video processing complete: {output_path}")

    def _process_frames_generator(self, input_path: str, total_frames: int) -> Iterator[np.ndarray]:
        """Generator that processes video frames"""
        frame_reader = self.video_processor.read_video_frames(input_path)

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame in frame_reader:
                try:
                    # Detect motions
                    motion_data = self.motion_detector.detect_motions(frame)

                    # Animate character
                    character_surface = self.character_animator.animate_character(motion_data)

                    # Generate background
                    background_context = self._get_background_context(motion_data)
                    background = self.background_generator.generate_background(background_context)

                    # Composite final frame
                    final_frame = self._composite_frame(background, character_surface, frame)

                    yield final_frame

                except Exception as e:
                    self.logger.error(f"Error processing frame: {e}")
                    # Yield original frame on error
                    yield frame

                pbar.update(1)

    def _composite_frame(self, background: np.ndarray,
                        character_surface, original_frame: np.ndarray) -> np.ndarray:
        """Composite final output frame"""
        # Convert character surface to numpy array
        character_array = pygame_to_numpy(character_surface)

        # Get output dimensions
        output_resolution = self.config.get('output_resolution', (1920, 1080))
        output_width, output_height = output_resolution

        # Resize background
        background_resized = cv2.resize(background, (output_width, output_height))

        # Position character
        char_height, char_width = character_array.shape[:2]
        y_offset = (output_height - char_height) // 2
        x_offset = (output_width - char_width) // 2

        # Create composite
        final_frame = background_resized.copy()

        # Simple alpha blending (assuming character has non-black pixels)
        mask = np.any(character_array != [0, 0, 0], axis=2)

        if y_offset >= 0 and x_offset >= 0:
            end_y = min(y_offset + char_height, output_height)
            end_x = min(x_offset + char_width, output_width)
            char_end_y = end_y - y_offset
            char_end_x = end_x - x_offset

            char_region = character_array[:char_end_y, :char_end_x]
            mask_region = mask[:char_end_y, :char_end_x]

            final_frame[y_offset:end_y, x_offset:end_x][mask_region] = char_region[mask_region]

        return final_frame

    def _get_background_context(self, motion_data) -> dict:
        """Get context for background generation"""
        return {
            'style': 'cartoon',
            'scene_description': 'animated cartoon background'
        }
