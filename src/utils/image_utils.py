""" Image processing utilities """
import cv2
import numpy as np
from PIL import Image
import pygame
from typing import Union, Tuple

def numpy_to_pygame(array: np.ndarray) -> pygame.Surface:
    """Convert numpy array to pygame surface"""
    # Ensure RGB format
    if len(array.shape) == 3:
        if array.shape[2] == 4:
        # RGBA
            array = array[:, :, :3]
        # Drop alpha channel
        elif array.shape[2] == 1:
        # Grayscale
            array = np.repeat(array, 3, axis=2)
    # Convert to pygame surface
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))

def pygame_to_numpy(surface: pygame.Surface) -> np.ndarray:
    """Convert pygame surface to numpy array"""
    array = pygame.surfarray.array3d(surface)
    return array.swapaxes(0, 1)

def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format"""
    numpy_image = np.array(pil_image)
    if len(numpy_image.shape) == 3:
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return numpy_image

def opencv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    if len(cv_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    return Image.fromarray(cv_image)

def resize_maintain_aspect(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h))

    # Create canvas and center image
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

def blend_images(background: np.ndarray, foreground: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Blend two images with alpha blending"""
    return cv2.addWeighted(background, 1-alpha, foreground, alpha, 0)
