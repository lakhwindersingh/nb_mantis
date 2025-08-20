import os
from pathlib import Path
from typing import Dict, Any
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = ASSETS_DIR / "models"

# Video settings
VIDEO_CONFIG = {
    "input_resolution": (640, 480),
    "output_resolution": (1920, 1080),
    "fps": 30,
    "codec": "mp4v",
}

# Motion detection settings
MOTION_DETECTION_CONFIG = {
    "face_detection": {
        "model_complexity": 1,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    },
    "hand_tracking": {
        "max_num_hands": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    },
}

# Character animation settings
CHARACTER_CONFIG = {
    "sprite_resolution": (512, 512),
    "animation_fps": 60,
    "interpolation_smoothness": 0.8,
    "expression_sensitivity": 1.0,
}

# Background generation settings
BACKGROUND_CONFIG = {
    "generation_method": "diffusion",  # default
    "methods": ["diffusion", "procedural", "style_transfer"],
    "resolution": (1920, 1080),
    "style": "cartoon",
    "update_frequency": 30,  # frames
}


def load_custom_config(config_path: str) -> Dict[str, Any]:
    """Load custom configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}