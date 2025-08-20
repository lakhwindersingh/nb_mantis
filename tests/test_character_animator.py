""" Tests for character animator """
import unittest
import pygame
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.agents.character_animator import CharacterAnimator
from config.settings import CHARACTER_CONFIG

class TestCharacterAnimator(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.animator = CharacterAnimator(CHARACTER_CONFIG, Path("../assets"))

    def test_animate_empty_motion(self):
        """Test animation with empty motion data"""
        motion_data = {'face': {'detected': False}, 'hands': {'detected': False}}
        surface = self.animator.animate_character(motion_data)

        self.assertIsInstance(surface, pygame.Surface)
        self.assertGreater(surface.get_width(), 0)
        self.assertGreater(surface.get_height(), 0)

if __name__ == '__main__':
    unittest.main()