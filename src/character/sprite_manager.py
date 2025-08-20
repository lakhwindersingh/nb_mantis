""" Sprite management system for character assets """
import pygame
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
# Add PIL import for fallback loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SpriteManager:
    """Manages character sprites and animations"""

    def __init__(self, assets_path: Path):
        self.logger = logging.getLogger(__name__)
        self.assets_path = assets_path
        self.sprites = {}
        self.animations = {}

        # Initialize pygame properly
        pygame.init()

        # Set a minimal display mode if none exists
        try:
            pygame.display.get_surface()
        except pygame.error:
            # No display mode set, create a minimal one
            pygame.display.set_mode((1, 1))

        # Load sprite configurations
        self._load_sprite_configs()

    def _load_sprite_configs(self):
        """Load sprite configuration files"""
        config_path = self.assets_path / "sprites" / "config.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                self.sprite_config = json.load(f)
        else:
            # Default configuration
            self.sprite_config = {
                "character_parts": {
                    "head": "head_neutral.png",
                    "body": "body_idle.png",
                    "left_hand": "hand_rest.png",
                    "right_hand": "hand_rest.png"
                },
                "expressions": {
                    "neutral": {"eyes": "eyes_normal.png", "mouth": "mouth_closed.png"},
                    "happy": {"eyes": "eyes_happy.png", "mouth": "mouth_smile.png"},
                    "surprised": {"eyes": "eyes_wide.png", "mouth": "mouth_open.png"}
                }
            }

    def load_sprite(self, sprite_name: str) -> Optional[pygame.Surface]:
        """Load a sprite image"""
        if sprite_name in self.sprites:
            return self.sprites[sprite_name]

        sprite_path = self.assets_path / "sprites" / sprite_name

        if sprite_path.exists():
            try:
                # Load sprite without convert_alpha() initially
                sprite = pygame.image.load(str(sprite_path))

                # Only convert_alpha() if display mode is set
                try:
                    sprite = sprite.convert_alpha()
                except pygame.error:
                    # If no display mode set, just convert() for basic format conversion
                    sprite = sprite.convert()

                self.sprites[sprite_name] = sprite
                return sprite
            except Exception as e:
                self.logger.error(f"Failed to load sprite {sprite_name}: {e}")

        # Return placeholder
        return self._create_placeholder_sprite(sprite_name)

    def _create_placeholder_sprite(self, sprite_name: str) -> pygame.Surface:
        """Create a placeholder sprite"""
        try:
            # Try to create surface with alpha
            placeholder = pygame.Surface((64, 64), pygame.SRCALPHA)
            placeholder.fill((255, 0, 255, 128))  # Magenta placeholder
        except pygame.error:
            # Fallback if no display mode is set
            placeholder = pygame.Surface((64, 64))
            placeholder.fill((255, 0, 255))  # Magenta placeholder without alpha

        return placeholder

    def get_character_sprites(self, state: Dict[str, Any]) -> Dict[str, pygame.Surface]:
        """Get sprites for current character state"""
        sprites = {}

        # Load base character parts
        for part, sprite_name in self.sprite_config["character_parts"].items():
            sprites[part] = self.load_sprite(sprite_name)

        # Override with expression-specific sprites
        expression = state.get('expression', 'neutral')
        if expression in self.sprite_config["expressions"]:
            for part, sprite_name in self.sprite_config["expressions"][expression].items():
                sprites[part] = self.load_sprite(sprite_name)

        return sprites
