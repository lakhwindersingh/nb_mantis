import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import pygame
import numpy as np
import xml.etree.ElementTree as ET
import re
import math
import time

from src.character.vector.systems.rendering_system import VectorRenderer
from src.character.vector.systems.bone_system import VectorBoneSystem
from src.character.vector.engines.morphing_engine import MorphingEngine


class VectorAnimationEngine:
    """Vector-based animation engine for character SVGs"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.renderer = self._initialize_vector_renderer()
        self.bone_system = VectorBoneSystem(self.config.get('skeleton', None))
        self.morphing_engine = MorphingEngine(self.config.get('morphing', {}))
        self._load_configurations()

    def _initialize_vector_renderer(self):
        renderer_config = self.config.get('renderer', {})
        return VectorRenderer(renderer_config)

    def _load_configurations(self):
        # Placeholder for configuration loading (kept minimal for reorg)
        pass

    def animate_vector_character(self, motion_data: Dict[str, Any], character_svg: str):
        # Simplified pipeline tying together bone system, morphing, and rendering
        self.bone_system.update_bone_transforms(motion_data)
        transformed_svg = self._apply_bone_deformations_to_svg(character_svg, self.bone_system.get_all_transforms())
        morphed_svg = self.morphing_engine.apply_morphing(transformed_svg)
        size = self._calculate_adaptive_render_size(motion_data)
        transforms = self._get_global_transforms(motion_data)
        surface = self.renderer.render_svg(morphed_svg, size=size, transforms=transforms)
        surface = self._apply_motion_effects(surface, motion_data)
        return surface

    def apply_vector_deformations(self, svg_element: ET.Element, bone_transforms: Dict[str, Dict[str, Any]]):
        # Stub for detailed SVG element deformation
        return svg_element

    def interpolate_svg_paths(self, path1: str, path2: str, t: float):
        # Basic interpolation fallback
        return self._fallback_path_interpolation(path1, path2, t)

    def _apply_bone_deformations_to_svg(self, svg_data: str, bone_transforms: Dict[str, Dict[str, Any]]):
        # For reorg purpose, return unchanged
        return svg_data

    def _calculate_adaptive_render_size(self, motion_data: Dict[str, Any]):
        base_size = self.config.get('base_render_size', (512, 512))
        intensity = float(motion_data.get('movement', {}).get('intensity', 0.0))
        scale = 1.0 + 0.2 * min(max(intensity, 0.0), 1.0)
        return (int(base_size[0] * scale), int(base_size[1] * scale))

    def _get_global_transforms(self, motion_data: Dict[str, Any]):
        # Compose simple transforms based on motion
        rotation = float(motion_data.get('pose', {}).get('torso_rotation', 0.0))
        tx = float(motion_data.get('movement', {}).get('dx', 0.0))
        ty = float(motion_data.get('movement', {}).get('dy', 0.0))
        return {'rotate': rotation, 'translate': (tx, ty)}

    def _apply_motion_effects(self, surface: pygame.Surface, motion_data: Dict[str, Any]):
        # Example: subtle motion blur based on intensity
        intensity = float(motion_data.get('movement', {}).get('intensity', 0.0))
        if intensity > 0.5:
            return self._apply_subtle_motion_blur(surface, intensity)
        return surface

    def _deep_copy_svg_element(self, element: ET.Element):
        return ET.fromstring(ET.tostring(element))

    def _find_applicable_transforms(self, element_id: str, element_class: str, bone_transforms: Dict[str, Dict[str, Any]]):
        transforms = []
        if element_id in bone_transforms:
            transforms.append(bone_transforms[element_id])
        if element_class in bone_transforms:
            transforms.append(bone_transforms[element_class])
        return transforms

    def _deform_svg_path(self, path_element: ET.Element, transforms: List[Dict[str, Any]]):
        return path_element

    def _deform_svg_shape(self, shape_element: ET.Element, transforms: List[Dict[str, Any]]):
        return shape_element

    def _deform_svg_group(self, group_element: ET.Element, transforms: List[Dict[str, Any]]):
        return group_element

    def _apply_transform_to_element(self, element: ET.Element, transforms: List[Dict[str, Any]]):
        return element

    def _parse_svg_path_commands(self, path_data: str):
        return []

    def _are_paths_compatible(self, commands1: List[Dict[str, Any]], commands2: List[Dict[str, Any]]):
        return False

    def _fallback_path_interpolation(self, path1: str, path2: str, t: float):
        return path1 if t < 0.5 else path2

    def _interpolate_mismatched_commands(self, cmd1: Dict[str, Any], cmd2: Dict[str, Any], t: float):
        return cmd1

    def _interpolate_command_parameters(self, cmd1: Dict[str, Any], cmd2: Dict[str, Any], t: float):
        return []

    def _commands_to_path_string(self, commands: List[Dict[str, Any]]):
        return ""

    def _extract_path_points(self, path_data: str):
        return []

    def _transform_points(self, points: List[Tuple[float, float]], transform: Dict[str, Any]):
        return points

    def _points_to_path_data(self, original_path: str, points: List[Tuple[float, float]]):
        return original_path

    def _combine_transforms(self, transforms: List[Dict[str, Any]]):
        if not transforms:
            return ""
        parts = []
        total_tx, total_ty = 0, 0
        total_rotation = 0
        total_sx, total_sy = 1.0, 1.0
        for t in transforms:
            if 'translate' in t:
                tx, ty = t['translate']
                total_tx += tx
                total_ty += ty
            if 'rotate' in t:
                total_rotation += t['rotate']
            if 'scale' in t:
                s = t['scale']
                if isinstance(s, (list, tuple)):
                    sx, sy = s
                else:
                    sx = sy = float(s)
                total_sx *= sx
                total_sy *= sy
        if total_tx or total_ty:
            parts.append(f"translate({total_tx:.2f},{total_ty:.2f})")
        if total_rotation:
            parts.append(f"rotate({total_rotation:.2f})")
        if total_sx != 1.0 or total_sy != 1.0:
            parts.append(f"scale({total_sx:.3f},{total_sy:.3f})")
        return ' '.join(parts)

    def _apply_subtle_motion_blur(self, surface: pygame.Surface, intensity: float) -> pygame.Surface:
        blurred = surface.copy()
        blur_offset = int(intensity * 2)
        for offset in range(1, blur_offset + 1):
            offset_surface = surface.copy()
            offset_surface.set_alpha(int(255 / (offset + 1)))
            blurred.blit(offset_surface, (offset, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
        return blurred

    def _apply_expression_glow(self, surface: pygame.Surface, intensity: float) -> pygame.Surface:
        glow_surface = pygame.transform.scale(surface, (int(surface.get_width() * 1.02), int(surface.get_height() * 1.02)))
        glow_color = pygame.Color(255, 240, 200, int(intensity * 30))
        glow_surface.fill(glow_color, special_flags=pygame.BLEND_ADD)
        final_surface = glow_surface.copy()
        final_surface.blit(surface, (1, 1))
        return final_surface
