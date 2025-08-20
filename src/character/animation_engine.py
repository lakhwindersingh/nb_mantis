""" Animation engine for character movement and transitions """
import pygame
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import math

class AnimationEngine:
    """Handles character animation and transitions"""
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Animation parameters
        self.fps = config.get('animation_fps', 60)
        self.interpolation_speed = config.get('interpolation_smoothness', 0.8)

        # Current animation state
        self.current_frame = 0
        self.animation_time = 0.0

        # Transition handling
        self.transitions = {}

    def render_frame(self, state: Dict[str, Any], sprite_manager) -> pygame.Surface:
        """
        Render a single animation frame

        Args:
            state: Current animation state
            sprite_manager: Sprite manager instance

        Returns:
            Rendered character frame
        """
        # Create canvas
        canvas_size = self.config.get('sprite_resolution', (512, 512))
        canvas = pygame.Surface(canvas_size, pygame.SRCALPHA)

        # Get sprites for current state
        sprites = sprite_manager.get_character_sprites(state)

        # Render character parts
        self._render_character_parts(canvas, sprites, state)

        # Apply animations
        self._apply_animations(canvas, state)

        self.current_frame += 1
        self.animation_time += 1.0 / self.fps

        return canvas

    def _render_character_parts(self, canvas: pygame.Surface,
                               sprites: Dict[str, pygame.Surface],
                               state: Dict[str, Any]):
        """Render individual character parts"""
        canvas_center = (canvas.get_width() // 2, canvas.get_height() // 2)

        # Render order (back to front)
        render_order = ['body', 'left_hand', 'right_hand', 'head']

        for part in render_order:
            if part in sprites:
                sprite = sprites[part]

                # Calculate position
                position = self._get_part_position(part, canvas_center, state)

                # Apply transformations
                transformed_sprite = self._apply_transformations(sprite, part, state)

                # Blit to canvas
                rect = transformed_sprite.get_rect(center=position)
                canvas.blit(transformed_sprite, rect)

    def _get_part_position(self, part: str, center: Tuple[int, int],
                          state: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate position for character part"""
        x, y = center

        # Default offsets
        offsets = {
            'head': (0, -80),
            'body': (0, 0),
            'left_hand': (-60, -20),
            'right_hand': (60, -20)
        }

        base_offset = offsets.get(part, (0, 0))

        # Apply head rotation offset
        head_rotation = state.get('face', {}).get('head_rotation', {})
        if part == 'head':
            rotation_x = head_rotation.get('rotation_x', 0) * 10
            rotation_y = head_rotation.get('rotation_y', 0) * 10
            base_offset = (base_offset[0] + rotation_y, base_offset[1] + rotation_x)

        return (x + base_offset[0], y + base_offset[1])

    def _apply_transformations(self, sprite: pygame.Surface, part: str,
                             state: Dict[str, Any]) -> pygame.Surface:
        """Apply transformations to sprite"""
        transformed = sprite.copy()

        # Apply rotations for head
        if part == 'head':
            head_rotation = state.get('face', {}).get('head_rotation', {})
            rotation_z = head_rotation.get('rotation_z', 0) * 57.2958  # Convert to degrees

            if abs(rotation_z) > 1:  # Only rotate if significant
                transformed = pygame.transform.rotate(transformed, -rotation_z)

        # Apply scaling based on emotions
        face_state = state.get('face', {})
        if part in ['head'] and 'eye_state' in face_state:
            if face_state['eye_state'] == 'wide':
                scale_factor = 1.1
                new_size = (int(transformed.get_width() * scale_factor),
                           int(transformed.get_height() * scale_factor))
                transformed = pygame.transform.scale(transformed, new_size)

        return transformed

    def _apply_animations(self, canvas: pygame.Surface, state: Dict[str, Any]):
        """Apply animation effects"""
        # Breathing animation
        breathing_offset = math.sin(self.animation_time * 2) * 2

        # Idle sway animation
        sway_offset = math.sin(self.animation_time * 0.5) * 1

        # Apply subtle movements (could transform the entire canvas)
        # For now, this is a placeholder for more complex animations
        pass

    def interpolate_states(self, current_state: Dict[str, Any],
                          target_state: Dict[str, Any],
                          alpha: float) -> Dict[str, Any]:
        """Interpolate between two animation states"""
        interpolated = current_state.copy()

        # Interpolate numeric values
        for key in target_state:
            if key in current_state:
                if isinstance(current_state[key], (int, float)):
                    interpolated[key] = (current_state[key] * (1 - alpha) +
                                       target_state[key] * alpha)

        return interpolated

    def render_enhanced_frame(self, state: Dict[str, Any], body_parts: Dict[str, Any], sprite_manager) -> pygame.Surface:
        """
        Render an enhanced animation frame with pose and body part data

        Args:
            state: Current animation state
            body_parts: Body part positioning data
            sprite_manager: Sprite manager instance

        Returns:
            Rendered character frame
        """
        # Create canvas
        canvas_size = self.config.get('sprite_resolution', (512, 512))
        canvas = pygame.Surface(canvas_size, pygame.SRCALPHA)

        # Get sprites for current state
        sprites = sprite_manager.get_character_sprites(state)

        # Render character parts with enhanced positioning
        self._render_enhanced_character_parts(canvas, sprites, state, body_parts)

        # Apply animations
        self._apply_animations(canvas, state)

        self.current_frame += 1
        self.animation_time += 1.0 / self.fps

        return canvas

    def _render_enhanced_character_parts(self, canvas: pygame.Surface,
                                       sprites: Dict[str, pygame.Surface],
                                       state: Dict[str, Any],
                                       body_parts: Dict[str, Any]):
        """Render individual character parts with enhanced pose data"""
        canvas_center = (canvas.get_width() // 2, canvas.get_height() // 2)

        # Render order (back to front)
        render_order = ['body', 'left_hand', 'right_hand', 'head']

        for part in render_order:
            if part in sprites:
                sprite = sprites[part]

                # Calculate enhanced position using body part data
                position = self._get_enhanced_part_position(part, canvas_center, state, body_parts)

                # Apply enhanced transformations
                transformed_sprite = self._apply_enhanced_transformations(sprite, part, state, body_parts)

                # Blit to canvas
                rect = transformed_sprite.get_rect(center=position)
                canvas.blit(transformed_sprite, rect)

    def _get_enhanced_part_position(self, part: str, center: Tuple[int, int],
                                  state: Dict[str, Any], body_parts: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate enhanced position for character part using body part data"""
        x, y = center

        # Default offsets
        offsets = {
            'head': (0, -80),
            'body': (0, 0),
            'left_hand': (-60, -20),
            'right_hand': (60, -20)
        }

        base_offset = offsets.get(part, (0, 0))

        # Apply body part positioning if available
        if part in body_parts:
            part_data = body_parts[part]
            
            # Apply rotation-based position adjustments
            if 'rotation' in part_data:
                rotation_offset_x = part_data['rotation'] * 5  # Scale factor
                base_offset = (base_offset[0] + rotation_offset_x, base_offset[1])
            
            if 'tilt' in part_data:
                tilt_offset_y = part_data['tilt'] * 3  # Scale factor
                base_offset = (base_offset[0], base_offset[1] + tilt_offset_y)

        # Apply head rotation offset from face data
        head_rotation = state.get('face', {}).get('head_rotation', {})
        if part == 'head':
            rotation_x = head_rotation.get('rotation_x', 0) * 10
            rotation_y = head_rotation.get('rotation_y', 0) * 10
            base_offset = (base_offset[0] + rotation_y, base_offset[1] + rotation_x)

        return (x + base_offset[0], y + base_offset[1])

    def _apply_enhanced_transformations(self, sprite: pygame.Surface, part: str,
                                      state: Dict[str, Any], body_parts: Dict[str, Any]) -> pygame.Surface:
        """Apply enhanced transformations to sprite using body part data"""
        transformed = sprite.copy()

        # Apply rotations for head
        if part == 'head':
            head_rotation = state.get('face', {}).get('head_rotation', {})
            rotation_z = head_rotation.get('rotation_z', 0) * 57.2958  # Convert to degrees

            if abs(rotation_z) > 1:  # Only rotate if significant
                transformed = pygame.transform.rotate(transformed, -rotation_z)

        # Apply body part specific transformations
        if part in body_parts:
            part_data = body_parts[part]
            
            # Apply rotation if available
            if 'rotation' in part_data and abs(part_data['rotation']) > 1:
                transformed = pygame.transform.rotate(transformed, -part_data['rotation'])

        # Apply scaling based on emotions
        face_state = state.get('face', {})
        if part in ['head'] and 'eye_state' in face_state:
            if face_state['eye_state'] == 'wide':
                scale_factor = 1.1
                new_size = (int(transformed.get_width() * scale_factor),
                           int(transformed.get_height() * scale_factor))
                transformed = pygame.transform.scale(transformed, new_size)

        return transformed