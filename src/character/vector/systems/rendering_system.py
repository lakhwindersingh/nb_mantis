"""
Vector rendering system for SVG and other vector formats
"""
import pygame
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import math

try:
    import cairo
    import gi
    gi.require_version('Rsvg', '2.0')
    from gi.repository import Rsvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False

try:
    from cairosvg import svg2png
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

class VectorRenderer:
    """High-performance vector graphics renderer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.render_cache = {}
        self.max_cache_size = config.get('cache_size', 50) * 1024 * 1024  # Convert MB to bytes
        
        # Initialize rendering backend
        self.backend = self._select_backend()
        self.logger.info(f"Vector renderer initialized with backend: {self.backend}")
        
    def _select_backend(self) -> str:
        """Select the best available rendering backend"""
        if CAIRO_AVAILABLE:
            return 'cairo'
        elif CAIROSVG_AVAILABLE:
            return 'cairosvg'
        else:
            self.logger.warning("No vector rendering backend available, using fallback")
            return 'fallback'
    
    def render_svg(self, svg_data: str, size: Tuple[int, int], 
                   transforms: Optional[Dict[str, Any]] = None) -> pygame.Surface:
        """
        Render SVG data to pygame surface
        
        Args:
            svg_data: SVG content as string
            size: Target size (width, height)
            transforms: Optional transformations to apply
            
        Returns:
            Rendered pygame surface
        """
        # Generate cache key
        cache_key = self._generate_cache_key(svg_data, size, transforms)
        
        if cache_key in self.render_cache:
            return self.render_cache[cache_key]
        
        # Apply transforms to SVG data if provided
        if transforms:
            svg_data = self._apply_svg_transforms(svg_data, transforms)
        
        # Render based on backend
        if self.backend == 'cairo':
            surface = self._render_cairo(svg_data, size)
        elif self.backend == 'cairosvg':
            surface = self._render_cairosvg(svg_data, size)
        else:
            surface = self._render_fallback(svg_data, size)
        
        # Cache result
        self._cache_surface(cache_key, surface)
        
        return surface
    
    def _render_cairo(self, svg_data: str, size: Tuple[int, int]) -> pygame.Surface:
        """Render using Cairo/Rsvg (highest quality)"""
        try:
            # Create Cairo surface
            width, height = size
            cairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            ctx = cairo.Context(cairo_surface)
            
            # Load SVG
            handle = Rsvg.Handle()
            handle.set_data(svg_data.encode('utf-8'))
            
            # Scale to fit
            svg_dim = handle.get_dimensions()
            scale_x = width / svg_dim.width
            scale_y = height / svg_dim.height
            
            ctx.scale(scale_x, scale_y)
            handle.render_cairo(ctx)
            
            # Convert to numpy array
            buf = cairo_surface.get_data()
            img_array = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
            
            # Convert BGRA to RGBA and create pygame surface
            img_array = img_array[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
            surface = pygame.surfarray.make_surface(img_array[:, :, :3].swapaxes(0, 1))
            
            return surface.convert_alpha()
            
        except Exception as e:
            self.logger.error(f"Cairo rendering failed: {e}")
            return self._render_fallback(svg_data, size)
    
    def _render_cairosvg(self, svg_data: str, size: Tuple[int, int]) -> pygame.Surface:
        """Render using CairoSVG (good quality)"""
        try:
            width, height = size
            png_data = svg2png(bytestring=svg_data.encode('utf-8'), 
                              output_width=width, output_height=height)
            
            # Create pygame surface from PNG data
            import io
            from PIL import Image
            
            pil_image = Image.open(io.BytesIO(png_data))
            
            # Convert to pygame surface
            mode = pil_image.mode
            size = pil_image.size
            raw = pil_image.tobytes()
            
            surface = pygame.image.fromstring(raw, size, mode)
            return surface.convert_alpha()
            
        except Exception as e:
            self.logger.error(f"CairoSVG rendering failed: {e}")
            return self._render_fallback(svg_data, size)
    
    def _render_fallback(self, svg_data: str, size: Tuple[int, int]) -> pygame.Surface:
        """Simple fallback renderer (basic shapes only)"""
        width, height = size
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((255, 0, 255, 128))  # Magenta placeholder
        
        # Draw simple placeholder
        pygame.draw.circle(surface, (255, 255, 255), (width//2, height//2), min(width, height)//4)
        
        self.logger.warning("Using fallback renderer - install cairo or cairosvg for proper SVG support")
        return surface
    
    def _apply_svg_transforms(self, svg_data: str, transforms: Dict[str, Any]) -> str:
        """Apply transformations to SVG elements"""
        try:
            root = ET.fromstring(svg_data)
            
            # Apply transforms to specific elements
            for element_id, transform_data in transforms.items():
                elements = root.findall(f".//*[@id='{element_id}']")
                
                for element in elements:
                    transform_str = self._build_transform_string(transform_data)
                    existing_transform = element.get('transform', '')
                    
                    if existing_transform:
                        element.set('transform', f"{existing_transform} {transform_str}")
                    else:
                        element.set('transform', transform_str)
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            self.logger.error(f"Transform application failed: {e}")
            return svg_data
    
    def _build_transform_string(self, transform_data: Dict[str, Any]) -> str:
        """Build SVG transform string from transform data"""
        transforms = []
        
        if 'translate' in transform_data:
            tx, ty = transform_data['translate']
            transforms.append(f"translate({tx},{ty})")
        
        if 'rotate' in transform_data:
            angle = transform_data['rotate']
            cx = transform_data.get('rotate_center_x', 0)
            cy = transform_data.get('rotate_center_y', 0)
            transforms.append(f"rotate({angle},{cx},{cy})")
        
        if 'scale' in transform_data:
            if isinstance(transform_data['scale'], (list, tuple)):
                sx, sy = transform_data['scale']
                transforms.append(f"scale({sx},{sy})")
            else:
                s = transform_data['scale']
                transforms.append(f"scale({s})")
        
        return ' '.join(transforms)
    
    def _generate_cache_key(self, svg_data: str, size: Tuple[int, int], 
                           transforms: Optional[Dict[str, Any]]) -> str:
        """Generate unique cache key for rendered content"""
        import hashlib
        
        content = f"{svg_data}{size}{transforms}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_surface(self, key: str, surface: pygame.Surface):
        """Cache rendered surface with size management"""
        # Estimate surface memory usage (rough approximation)
        surface_size = surface.get_width() * surface.get_height() * 4  # 4 bytes per pixel
        
        # Clean cache if needed
        while (len(self.render_cache) > 0 and 
               self._estimate_cache_size() + surface_size > self.max_cache_size):
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self.render_cache))
            del self.render_cache[oldest_key]
        
        self.render_cache[key] = surface
    
    def _estimate_cache_size(self) -> int:
        """Estimate total cache size in bytes"""
        total_size = 0
        for surface in self.render_cache.values():
            total_size += surface.get_width() * surface.get_height() * 4
        return total_size
    
    def clear_cache(self):
        """Clear the rendering cache"""
        self.render_cache.clear()
        self.logger.info("Vector render cache cleared")
