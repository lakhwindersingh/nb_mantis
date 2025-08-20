""" Mathematical utilities for animations and transforms """
import numpy as np
import math
from typing import Tuple, List

def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between two values"""
    return start + (end - start) * t

def smooth_step(t: float) -> float:
    """Smooth step function for easing"""
    return t * t * (3 - 2 * t)

def ease_in_out(t: float) -> float:
    """Ease in-out function"""
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t

def rotate_point(point: Tuple[float, float], angle: float, center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
    """Rotate a point around center by angle (in radians)"""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]

    # Rotate
    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle + y * cos_angle

    # Translate back
    return (new_x + center[0], new_y + center[1])

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def normalize_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize a 2D vector"""
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
    if magnitude == 0:
        return (0, 0)
    return (vector[0] / magnitude, vector[1] / magnitude)

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average of values"""
    if len(values) < window_size:
        return values

    result = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        result.append(sum(window) / window_size)

    return result