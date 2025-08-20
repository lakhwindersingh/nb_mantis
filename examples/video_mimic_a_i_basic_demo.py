""" Basic demo of Video Mimic AI functionality """
import sys
import cv2
import numpy as np
import pygame
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.agents.motion_detector import MotionDetector
from src.agents.character_animator import CharacterAnimator
from src.agents.background_generator import BackgroundGenerator
from config.settings import CHARACTER_CONFIG

def main():
    print("🎭 Video Mimic AI - Basic Demo")
    
    # Initialize pygame first
    pygame.init()
    pygame.display.set_mode((800, 600))  # Set display mode before creating sprites
    
    # Initialize components
    motion_detector = MotionDetector()
    character_animator = CharacterAnimator(CHARACTER_CONFIG, Path("../assets"))
    background_generator = BackgroundGenerator({'generation_method': 'procedural'})

    # Create a simple test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    print("🔍 Detecting motions...")
    motion_data = motion_detector.detect_motions(test_frame)
    print(f"Motion detection result: {motion_data}")

    print("🎨 Animating character...")
    character_surface = character_animator.animate_character(motion_data)
    print(f"Character animation generated: {character_surface.get_size()}")

    print("🌄 Generating background...")
    background = background_generator.generate_background()
    print(f"Background generated: {background.shape}")

    print("✅ Basic demo completed successfully!")
    
    pygame.quit()

if __name__ == "__main__":
    main()
