""" Stable Diffusion background generation implementation """
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import logging

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False


class DiffusionBackgroundGenerator:
    """Background generator using Stable Diffusion"""
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config

        if not DIFFUSION_AVAILABLE:
            raise ImportError("diffusers library not available")

        # Initialize pipeline
        model_name = config.get('model_name', 'runwayml/stable-diffusion-v1-5')

        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                self.logger.info("Using GPU for diffusion generation")
            else:
                self.logger.info("Using CPU for diffusion generation")

        except Exception as e:
            self.logger.error(f"Failed to load diffusion model: {e}")
            raise

        # Generation parameters
        self.guidance_scale = config.get('guidance_scale', 7.5)
        self.num_inference_steps = config.get('num_inference_steps', 20)
        self.resolution = config.get('resolution', (512, 512))

    def generate(self, prompt: str, negative_prompt: str = None) -> Image.Image:
        """
        Generate background image from text prompt

        Args:
            prompt: Text description of desired background
            negative_prompt: Text description of what to avoid

        Returns:
            Generated background image
        """
        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    width=self.resolution[0],
                    height=self.resolution[1]
                ).images[0]

            return image

        except Exception as e:
            self.logger.error(f"Error generating background: {e}")
            # Return a simple fallback image
            return Image.new('RGB', self.resolution, color=(135, 206, 235))  # Sky blue