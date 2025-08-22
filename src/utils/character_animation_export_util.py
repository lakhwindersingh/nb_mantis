"""
Character animation export utility
"""
import pygame
import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.character_animator import CharacterAnimator
from utils.logging_config import setup_logging
from config.settings import CHARACTER_CONFIG

def main():
    """Main entry point for character export utility"""
    parser = argparse.ArgumentParser(description="Export character animations")
    parser.add_argument("--format", choices=["gif", "png", "mp4"], default="gif",
                       help="Export format (default: gif)")
    parser.add_argument("--duration", type=int, default=5,
                       help="Animation duration in seconds (default: 5)")
    parser.add_argument("--output", type=str,
                       help="Output file path")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        print(f"🎨 Exporting character animation as {args.format}...")
        
        # Initialize character animator
        pygame.init()
        animator = CharacterAnimator(CHARACTER_CONFIG, Path("assets"))
        
        # Create sample motion data
        sample_motion = {
            'face': {
                'detected': True,
                'expressions': {
                    'smile_intensity': 0.8,
                    'eye_openness': 0.6,
                    'eyebrow_raise': 0.5
                }
            },
            'hands': {
                'detected': True,
                'left_hand': {'gesture': 'wave'},
                'right_hand': {'gesture': 'thumbs_up'}
            }
        }
        
        # Generate animation frame
        character_surface = animator.animate_character(sample_motion)
        
        # Determine output path
        if not args.output:
            output_dir = Path("output/character_exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output = output_dir / f"character_export.{args.format}"
        
        # Export based on format
        if args.format == "png":
            pygame.image.save(character_surface, str(args.output))
        else:
            logger.warning(f"Export format {args.format} not fully implemented yet")
            # Save as PNG for now
            pygame.image.save(character_surface, str(args.output).replace(f".{args.format}", ".png"))
        
        print(f"✅ Character exported to: {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    finally:
        pygame.quit()

if __name__ == "__main__":
    sys.exit(main())
