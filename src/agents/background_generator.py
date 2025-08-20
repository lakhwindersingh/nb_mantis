import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, Union
import logging
import sys
sys.path.insert(0, '../models')

from src.models.background_gen.diffusion_bg import DiffusionBackgroundGenerator
from src.models.background_gen.gan_bg import GANBackgroundGenerator
# from video_mimic_ai.src.models.background_gen.gan_bg import ProceduralBackgroundGenerator


class ProceduralBackgroundGenerator:
    """Simple procedural background generator as fallback"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resolution = config.get('resolution', (1920, 1080))

    def generate(self, style: str) -> np.ndarray:
        """Generate procedural background"""
        width, height = self.resolution

        if style == 'gradient':
            return self._generate_gradient_background(width, height)
        elif style == 'noise':
            return self._generate_noise_background(width, height)
        else:  # cartoon
            return self._generate_cartoon_background(width, height)

    def _generate_gradient_background(self, width: int, height: int) -> np.ndarray:
        """Generate gradient background"""
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            factor = i / height
            gradient[i, :] = [
                int(135 + factor * 120),  # Sky blue to light
                int(206 + factor * 49),
                int(235 + factor * 20)
            ]

        return gradient

    def _generate_noise_background(self, width: int, height: int) -> np.ndarray:
        """Generate noise-based background"""
        # Simple Perlin-like noise
        noise = np.random.rand(height // 10, width // 10, 3) * 255
        noise = cv2.resize(noise.astype(np.uint8), (width, height))

        # Apply color filter
        noise = cv2.applyColorMap(noise[:, :, 0], cv2.COLORMAP_VIRIDIS)
        return noise

    def _generate_cartoon_background(self, width: int, height: int) -> np.ndarray:
        """Generate simple cartoon-style background"""
        background = np.ones((height, width, 3), dtype=np.uint8) * 240

        # Add some simple shapes
        cv2.circle(background, (width // 4, height // 4), 100, (255, 255, 0), -1)  # Sun
        cv2.rectangle(background, (0, height - 200), (width, height), (0, 200, 0), -1)  # Ground

        return background



class BackgroundGenerator:
    """ Main background generation agent that creates dynamic backgrounds """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.generation_method = config.get('generation_method', 'procedural')

        # Initialize generators based on method
        self.generators = {}

        if self.generation_method in ['diffusion', 'all']:
            try:
                self.generators['diffusion'] = DiffusionBackgroundGenerator(config)
                self.logger.info("Diffusion background generator loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load diffusion generator: {e}")

        if self.generation_method in ['gan', 'all']:
            try:
                self.generators['gan'] = GANBackgroundGenerator(config)
                self.logger.info("GAN background generator loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load GAN generator: {e}")

        # Fallback to procedural generation
        self.procedural_generator = ProceduralBackgroundGenerator(config)

        # Background cache and state - ADD THESE LINES
        self.background_cache = {}
        self.current_background = None  # Initialize this attribute
        self.frames_since_update = 0    # Initialize this attribute
        self.update_frequency = config.get('update_frequency', 30)

        self.logger.info(f"Background generator initialized with method: {self.generation_method}")

    def generate_background(self, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate background image based on context

        Args:
            context: Context information for background generation

        Returns:
            Background image as numpy array
        """
        context = context or {}

        # Check if we need to update background
        if (self.current_background is None or
            self.frames_since_update >= self.update_frequency):

            self.current_background = self._generate_new_background(context)
            self.frames_since_update = 0
        else:
            self.frames_since_update += 1

        return self.current_background

    def _generate_new_background(self, context: Dict[str, Any]) -> np.ndarray:
        """Generate a new background image"""
        scene_description = context.get('scene_description')
        style = context.get('style', self.config.get('style', 'cartoon'))

        try:
            if self.generation_method == 'diffusion' and 'diffusion' in self.generators:
                return self._generate_diffusion_background(scene_description, style)
            elif self.generation_method == 'gan' and 'gan' in self.generators:
                return self._generate_gan_background(style)
            else:
                return self._generate_procedural_background(style)

        except Exception as e:
            self.logger.error(f"Error generating background: {e}")
            return self._generate_procedural_background(style)

    def _generate_diffusion_background(self, description: Optional[str],
                                     style: str) -> np.ndarray:
        """Generate background using diffusion model"""
        if not description:
            description = f"A {style} style background scene"

        prompt = f"{description}, {style} art style, high quality, detailed"
        background = self.generators['diffusion'].generate(prompt)
        return self._prepare_background_image(background)

    def _generate_gan_background(self, style: str) -> np.ndarray:
        """Generate background using GAN"""
        background = self.generators['gan'].generate(style)
        return self._prepare_background_image(background)

    def _generate_procedural_background(self, style: str) -> np.ndarray:
        """Generate procedural background"""
        background = self.procedural_generator.generate(style)
        return self._prepare_background_image(background)

    def _prepare_background_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Prepare background image for compositing"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to target resolution
        target_resolution = self.config.get('resolution', (1920, 1080))
        image = cv2.resize(image, target_resolution)

        return image

class ProceduralBackgroundGeneratorA:
    """Simple procedural background generator as fallback"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resolution = config.get('resolution', (1920, 1080))

    def generate(self, style: str) -> np.ndarray:
        """Generate procedural background"""
        width, height = self.resolution

        if style == 'gradient':
            return self._generate_gradient_background(width, height)
        elif style == 'noise':
            return self._generate_noise_background(width, height)
        else:  # cartoon
            return self._generate_cartoon_background(width, height)

    def _generate_gradient_background(self, width: int, height: int) -> np.ndarray:
        """Generate gradient background"""
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            factor = i / height
            gradient[i, :] = [
                int(135 + factor * 120),  # Sky blue to light
                int(206 + factor * 49),
                int(235 + factor * 20)
            ]

        return gradient

    def _generate_noise_background(self, width: int, height: int) -> np.ndarray:
        """Generate noise-based background"""
        # Simple Perlin-like noise
        noise = np.random.rand(height // 10, width // 10, 3) * 255
        noise = cv2.resize(noise.astype(np.uint8), (width, height))

        # Apply color filter
        noise = cv2.applyColorMap(noise[:, :, 0], cv2.COLORMAP_VIRIDIS)
        return noise

    def _generate_cartoon_background(self, width: int, height: int) -> np.ndarray:
        """Generate simple cartoon-style background"""
        background = np.ones((height, width, 3), dtype=np.uint8) * 240

        # Add some simple shapes
        cv2.circle(background, (width // 4, height // 4), 100, (255, 255, 0), -1)  # Sun
        cv2.rectangle(background, (0, height - 200), (width, height), (0, 200, 0), -1)  # Ground

        return background