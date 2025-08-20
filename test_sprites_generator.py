"""Create simple test sprites for development"""
import pygame
import numpy as np
from pathlib import Path

def create_test_sprites():
    """Create basic test sprite files"""
    pygame.init()
    
    # Create sprites directory
    sprites_dir = Path("assets/characters/sprites")
    sprites_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sprites to create
    sprites = {
        "head_neutral.png": (64, 64, (255, 220, 177)),  # Skin color head
        "body_idle.png": (64, 96, (100, 100, 200)),     # Blue shirt
        "hand_rest.png": (32, 32, (255, 220, 177)),     # Skin color hand
        "eyes_normal.png": (24, 8, (0, 0, 0)),          # Black eyes
        "mouth_closed.png": (16, 4, (200, 100, 100)),   # Red mouth
        "eyes_happy.png": (24, 8, (0, 0, 0)),           # Black eyes (happy)
        "mouth_smile.png": (16, 6, (200, 100, 100)),    # Red smile
        "eyes_wide.png": (28, 12, (0, 0, 0)),           # Wide eyes
        "mouth_open.png": (20, 8, (100, 50, 50)),       # Open mouth
    }
    
    for filename, (width, height, color) in sprites.items():
        # Create surface with alpha
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        if filename.startswith("head"):
            # Draw a circle for head
            pygame.draw.circle(surface, color, (width//2, height//2), min(width, height)//2 - 2)
        elif filename.startswith("body"):
            # Draw a rectangle for body
            pygame.draw.rect(surface, color, (4, 4, width-8, height-8))
        elif filename.startswith("hand"):
            # Draw a circle for hand
            pygame.draw.circle(surface, color, (width//2, height//2), min(width, height)//2 - 2)
        elif filename.startswith("eyes"):
            if "happy" in filename:
                # Draw happy eyes (curved)
                pygame.draw.ellipse(surface, color, (2, 2, 8, 4))
                pygame.draw.ellipse(surface, color, (14, 2, 8, 4))
            elif "wide" in filename:
                # Draw wide eyes
                pygame.draw.ellipse(surface, color, (0, 2, 12, 8))
                pygame.draw.ellipse(surface, color, (16, 2, 12, 8))
            else:
                # Draw normal eyes
                pygame.draw.ellipse(surface, color, (2, 2, 8, 4))
                pygame.draw.ellipse(surface, color, (14, 2, 8, 4))
        elif filename.startswith("mouth"):
            if "smile" in filename:
                # Draw smile
                pygame.draw.arc(surface, color, (0, 0, width, height), 0, 3.14, 2)
            elif "open" in filename:
                # Draw open mouth
                pygame.draw.ellipse(surface, color, (2, 2, width-4, height-4))
            else:
                # Draw closed mouth
                pygame.draw.rect(surface, color, (2, height//2-1, width-4, 2))
        
        # Save the sprite
        sprite_path = sprites_dir / filename
        pygame.image.save(surface, str(sprite_path))
        print(f"Created sprite: {sprite_path}")
    
    # Create config.json
    config = {
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
    
    import json
    config_path = sprites_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config: {config_path}")
    print("All test sprites created successfully!")

if __name__ == "__main__":
    create_test_sprites()
