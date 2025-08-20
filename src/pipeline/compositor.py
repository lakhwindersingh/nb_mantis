""" Video compositor for combining character and background elements """
import cv2
import numpy as np
import pygame
from typing import Tuple, Optional, Dict, Any
import logging
from src.utils.image_utils import pygame_to_numpy


class VideoCompositor:
    """Handles compositing of character and background elements"""
    def __init__(self, output_resolution: Tuple[int, int] = (1920, 1080)):
        self.logger = logging.getLogger(__name__)
        self.output_resolution = output_resolution
        self.output_width, self.output_height = output_resolution

    def composite_frame(self, background: np.ndarray,
                       character: pygame.Surface,
                       original_frame: Optional[np.ndarray] = None,
                       blend_mode: str = 'alpha') -> np.ndarray:
        """
        Composite character onto background

        Args:
            background: Background image
            character: Character pygame surface
            original_frame: Original video frame (optional)
            blend_mode: Blending mode ('alpha', 'add', 'multiply')

        Returns:
            Composited frame
        """
        # Prepare background
        bg_resized = cv2.resize(background, self.output_resolution)

        # Convert character to numpy array
        char_array = pygame_to_numpy(character)

        # Apply blending
        if blend_mode == 'alpha':
            result = self._alpha_blend(bg_resized, char_array)
        elif blend_mode == 'add':
            result = self._additive_blend(bg_resized, char_array)
        elif blend_mode == 'multiply':
            result = self._multiply_blend(bg_resized, char_array)
        else:
            result = self._alpha_blend(bg_resized, char_array)  # Default

        return result

    def _alpha_blend(self, background: np.ndarray, character: np.ndarray) -> np.ndarray:
        """Alpha blending"""
        char_height, char_width = character.shape[:2]

        # Center character
        y_offset = (self.output_height - char_height) // 2
        x_offset = (self.output_width - char_width) // 2

        result = background.copy()

        # Handle bounds
        if y_offset < 0 or x_offset < 0:
            return result

        end_y = min(y_offset + char_height, self.output_height)
        end_x = min(x_offset + char_width, self.output_width)
        char_end_y = end_y - y_offset
        char_end_x = end_x - x_offset

        # Create alpha mask (non-black pixels)
        alpha = np.any(character[:char_end_y, :char_end_x] != [0, 0, 0], axis=2).astype(float)

        # Blend
        for c in range(3):
            bg_region = result[y_offset:end_y, x_offset:end_x, c]
            char_region = character[:char_end_y, :char_end_x, c]

            result[y_offset:end_y, x_offset:end_x, c] = (
                bg_region * (1 - alpha) + char_region * alpha
            )

        return result.astype(np.uint8)

    def _additive_blend(self, background: np.ndarray, character: np.ndarray) -> np.ndarray:
        """Additive blending"""
        char_height, char_width = character.shape[:2]
        y_offset = (self.output_height - char_height) // 2
        x_offset = (self.output_width - char_width) // 2

        result = background.copy().astype(np.float32)

        if y_offset >= 0 and x_offset >= 0:
            end_y = min(y_offset + char_height, self.output_height)
            end_x = min(x_offset + char_width, self.output_width)
            char_end_y = end_y - y_offset
            char_end_x = end_x - x_offset

            result[y_offset:end_y, x_offset:end_x] += character[:char_end_y, :char_end_x]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _multiply_blend(self, background: np.ndarray, character: np.ndarray) -> np.ndarray:
        """Multiply blending"""
        char_height, char_width = character.shape[:2]
        y_offset = (self.output_height - char_height) // 2
        x_offset = (self.output_width - char_width) // 2

        result = background.copy().astype(np.float32) / 255.0

        if y_offset >= 0 and x_offset >= 0:
            end_y = min(y_offset + char_height, self.output_height)
            end_x = min(x_offset + char_width, self.output_width)
            char_end_y = end_y - y_offset
            char_end_x = end_x - x_offset

            char_normalized = character[:char_end_y, :char_end_x].astype(np.float32) / 255.0
            result[y_offset:end_y, x_offset:end_x] *= char_normalized

        return (result * 255).astype(np.uint8)

    def add_effects(self, frame: np.ndarray, effects: Dict[str, Any]) -> np.ndarray:
        """Add visual effects to frame"""
        result = frame.copy()

        # Brightness adjustment
        if 'brightness' in effects:
            brightness = effects['brightness']
            result = cv2.convertScaleAbs(result, alpha=1.0, beta=brightness)

        # Contrast adjustment
        if 'contrast' in effects:
            contrast = effects['contrast']
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)

        # Blur effect
        if 'blur' in effects:
            blur_amount = effects['blur']
            if blur_amount > 0:
                result = cv2.GaussianBlur(result, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

        # Color tint
        if 'tint' in effects:
            tint_color = effects['tint']  # (R, G, B) tuple
            overlay = np.full_like(result, tint_color, dtype=np.uint8)
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        return result