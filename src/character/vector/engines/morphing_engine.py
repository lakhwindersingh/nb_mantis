"""
Advanced morphing engine for smooth character transitions
"""
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import math

class SVGPath:
    """Represents an SVG path with morphing capabilities"""
    
    def __init__(self, path_data: str):
        self.path_data = path_data
        self.commands = []
        self.points = []
        self._parse_path()
    
    def _parse_path(self):
        """Parse SVG path data into commands and points"""
        try:
            # Simple path parser for basic commands
            command_pattern = r'([MmLlHhVvCcSsQqTtAaZz])'
            number_pattern = r'([-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?)'
            
            parts = re.split(command_pattern, self.path_data)
            parts = [p.strip() for p in parts if p.strip()]
            
            current_command = None
            
            for part in parts:
                if re.match(command_pattern, part):
                    current_command = part
                elif current_command:
                    # Extract numbers
                    numbers = re.findall(number_pattern, part)
                    coords = [float(n) for n in numbers]
                    
                    self.commands.append({
                        'command': current_command,
                        'coords': coords
                    })
                    
                    # Extract point pairs for morphing
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            self.points.append([coords[i], coords[i + 1]])
                            
        except Exception as e:
            logging.warning(f"Path parsing failed: {e}")
            self.commands = []
            self.points = []
    
    def interpolate_with(self, other: 'SVGPath', t: float) -> str:
        """Interpolate with another path"""
        if len(self.points) != len(other.points):
            # Paths have different point counts, use fallback
            return self.path_data if t < 0.5 else other.path_data
        
        # Interpolate points
        interpolated_points = []
        for p1, p2 in zip(self.points, other.points):
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            interpolated_points.append([x, y])
        
        # Rebuild path with interpolated points
        return self._rebuild_path(interpolated_points)
    
    def _rebuild_path(self, points: List[List[float]]) -> str:
        """Rebuild path string from interpolated points"""
        if not points or not self.commands:
            return self.path_data
        
        path_parts = []
        point_index = 0
        
        for cmd in self.commands:
            command = cmd['command']
            original_coords = cmd['coords']
            
            path_parts.append(command)
            
            # Replace coordinates with interpolated ones
            new_coords = []
            coords_per_point = 2  # x, y
            
            for i in range(0, len(original_coords), coords_per_point):
                if point_index < len(points):
                    new_coords.extend(points[point_index])
                    point_index += 1
                else:
                    # Use original coordinates if we run out of interpolated points
                    new_coords.extend(original_coords[i:i+coords_per_point])
            
            # Format coordinates
            coord_str = ' '.join(f"{coord:.2f}" for coord in new_coords)
            path_parts.append(coord_str)
        
        return ' '.join(path_parts)

class MorphTarget:
    """Represents a morph target with associated SVG modifications"""
    
    def __init__(self, name: str, svg_data: str, weight: float = 0.0):
        self.name = name
        self.svg_data = svg_data
        self.weight = weight
        self.element_paths: Dict[str, SVGPath] = {}
        self._extract_paths()
    
    def _extract_paths(self):
        """Extract path data from SVG elements"""
        try:
            root = ET.fromstring(self.svg_data)
            
            # Find all elements with path data
            for element in root.iter():
                element_id = element.get('id')
                if element_id and element.tag.endswith('path'):
                    path_data = element.get('d')
                    if path_data:
                        self.element_paths[element_id] = SVGPath(path_data)
                        
        except Exception as e:
            logging.error(f"Failed to extract paths from morph target {self.name}: {e}")

class MorphingEngine:
    """Advanced morphing system for character expressions and poses"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Morph targets storage
        self.morph_targets: Dict[str, MorphTarget] = {}
        self.base_svg: Optional[str] = None
        
        # Animation properties
        self.transition_duration = self.config.get('transition_duration', 0.5)
        self.easing_function = self._cubic_ease_in_out
        
        # Current morph state
        self.current_weights: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
        self.transition_progress: Dict[str, float] = {}
        
        self.logger.info("Morphing engine initialized")
    
    def load_morph_targets(self, targets_config: Dict[str, Any]):
        """Load morph targets from configuration"""
        try:
            self.base_svg = targets_config.get('base_svg', '')
            
            targets = targets_config.get('targets', {})
            
            for target_name, target_data in targets.items():
                svg_data = target_data.get('svg_data', '')
                if isinstance(target_data, str):
                    # If target_data is just a file path
                    svg_path = Path(target_data)
                    if svg_path.exists():
                        svg_data = svg_path.read_text()
                
                morph_target = MorphTarget(target_name, svg_data)
                self.morph_targets[target_name] = morph_target
                self.current_weights[target_name] = 0.0
                self.target_weights[target_name] = 0.0
                self.transition_progress[target_name] = 1.0  # Fully transitioned
                
            self.logger.info(f"Loaded {len(self.morph_targets)} morph targets")
            
        except Exception as e:
            self.logger.error(f"Failed to load morph targets: {e}")
    
    def set_expression_weights(self, expression_weights: Dict[str, float]):
        """Set target weights for expressions"""
        for expression, weight in expression_weights.items():
            if expression in self.morph_targets:
                # Clamp weight between 0 and 1
                weight = max(0.0, min(1.0, weight))
                
                if self.target_weights[expression] != weight:
                    self.target_weights[expression] = weight
                    self.transition_progress[expression] = 0.0  # Start new transition
    
    def update_morphing(self, delta_time: float):
        """Update morphing transitions"""
        for target_name in self.morph_targets:
            if self.transition_progress[target_name] < 1.0:
                # Update transition progress
                progress_delta = delta_time / self.transition_duration
                self.transition_progress[target_name] = min(1.0, 
                    self.transition_progress[target_name] + progress_delta)
                
                # Apply easing
                eased_progress = self.easing_function(self.transition_progress[target_name])
                
                # Interpolate current weight
                start_weight = self.current_weights[target_name]
                target_weight = self.target_weights[target_name]
                
                self.current_weights[target_name] = (
                    start_weight + (target_weight - start_weight) * eased_progress
                )
    
    def apply_morphing(self, base_svg: str) -> str:
        """Apply current morph weights to base SVG"""
        if not self.morph_targets:
            return base_svg
        
        try:
            # Parse base SVG
            root = ET.fromstring(base_svg)
            
            # Collect all morphed paths
            morphed_paths: Dict[str, List[Tuple[SVGPath, float]]] = {}
            
            # Gather weighted paths from active morph targets
            for target_name, weight in self.current_weights.items():
                if weight <= 0.0:
                    continue
                
                morph_target = self.morph_targets[target_name]
                
                for element_id, path in morph_target.element_paths.items():
                    if element_id not in morphed_paths:
                        morphed_paths[element_id] = []
                    morphed_paths[element_id].append((path, weight))
            
            # Apply morphing to SVG elements
            for element in root.iter():
                element_id = element.get('id')
                if element_id and element_id in morphed_paths and element.tag.endswith('path'):
                    original_path = element.get('d', '')
                    morphed_path = self._blend_paths(original_path, morphed_paths[element_id])
                    element.set('d', morphed_path)
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            self.logger.error(f"Morphing application failed: {e}")
            return base_svg
    
    def _blend_paths(self, base_path: str, morph_data: List[Tuple[SVGPath, float]]) -> str:
        """Blend multiple morphed paths with weights"""
        if not morph_data:
            return base_path
        
        try:
            # Normalize weights
            total_weight = sum(weight for _, weight in morph_data)
            if total_weight <= 0:
                return base_path
            
            base_svg_path = SVGPath(base_path)
            
            # If only one morph target, simple interpolation
            if len(morph_data) == 1:
                morph_path, weight = morph_data[0]
                normalized_weight = weight / total_weight
                return base_svg_path.interpolate_with(morph_path, normalized_weight)
            
            # Multiple targets: blend sequentially
            current_path = base_svg_path
            remaining_weight = 1.0
            
            for morph_path, weight in morph_data:
                if remaining_weight <= 0:
                    break
                
                blend_factor = min(weight / remaining_weight, 1.0) if remaining_weight > 0 else 0
                blended_path_str = current_path.interpolate_with(morph_path, blend_factor)
                current_path = SVGPath(blended_path_str)
                remaining_weight -= weight
            
            return current_path.path_data
            
        except Exception as e:
            self.logger.error(f"Path blending failed: {e}")
            return base_path
    
    def _cubic_ease_in_out(self, t: float) -> float:
        """Cubic ease-in-out easing function"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            t = t - 1
            return 1 + 4 * t * t * t
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current morph weights"""
        return self.current_weights.copy()
    
    def reset_all_weights(self):
        """Reset all morph weights to zero"""
        for target_name in self.morph_targets:
            self.target_weights[target_name] = 0.0
            self.transition_progress[target_name] = 0.0
    
    def create_expression_from_motion(self, motion_data: Dict[str, Any]) -> Dict[str, float]:
        """Convert motion data to expression weights"""
        expression_weights = {}
        
        face_data = motion_data.get('face', {})
        if not face_data.get('detected'):
            return expression_weights
        
        expressions = face_data.get('expressions', {})
        
        # Map facial expressions to morph targets
        smile_intensity = expressions.get('smile_intensity', 0.0)
        if smile_intensity > 0.3:
            expression_weights['happy'] = smile_intensity
        
        eyebrow_raise = expressions.get('eyebrow_raise', 0.5)
        if eyebrow_raise > 0.7:
            expression_weights['surprised'] = (eyebrow_raise - 0.7) / 0.3
        elif eyebrow_raise < 0.3:
            expression_weights['angry'] = (0.3 - eyebrow_raise) / 0.3
        
        eye_openness = expressions.get('eye_openness', 0.3)
        mouth_openness = expressions.get('mouth_openness', 0.2)
        
        # Combine for complex expressions
        if smile_intensity < 0.2 and eye_openness < 0.2 and mouth_openness < 0.1:
            expression_weights['sad'] = 0.8
        
        return expression_weights
