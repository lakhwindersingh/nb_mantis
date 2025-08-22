"""
Performance benchmark utility
"""
import time
import argparse
import sys
import statistics
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.motion_detector import MotionDetector
from agents.character_animator import CharacterAnimator
from agents.background_generator import BackgroundGenerator
from utils.logging_config import setup_logging
from config.settings import CHARACTER_CONFIG
import numpy as np
import cv2

class PerformanceBenchmark:
    """System performance benchmark"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    def run_benchmark(self, duration: int = 60) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        print(f"🚀 Running performance benchmark for {duration} seconds...")
        
        # Initialize components
        motion_detector = MotionDetector()
        character_animator = CharacterAnimator(CHARACTER_CONFIG, Path("assets"))
        background_generator = BackgroundGenerator({'generation_method': 'procedural'})
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Benchmark motion detection
        print("Testing motion detection performance...")
        motion_times = self._benchmark_motion_detection(motion_detector, test_frame, duration // 3)
        
        # Benchmark character animation  
        print("Testing character animation performance...")
        motion_data = motion_detector.detect_motions(test_frame)
        animation_times = self._benchmark_character_animation(character_animator, motion_data, duration // 3)
        
        # Benchmark background generation
        print("Testing background generation performance...")
        bg_times = self._benchmark_background_generation(background_generator, duration // 3)
        
        # Compile results
        self.results = {
            'motion_detection': {
                'avg_time': statistics.mean(motion_times),
                'min_time': min(motion_times),
                'max_time': max(motion_times),
                'fps': 1.0 / statistics.mean(motion_times),
                'sample_count': len(motion_times)
            },
            'character_animation': {
                'avg_time': statistics.mean(animation_times),
                'min_time': min(animation_times), 
                'max_time': max(animation_times),
                'fps': 1.0 / statistics.mean(animation_times),
                'sample_count': len(animation_times)
            },
            'background_generation': {
                'avg_time': statistics.mean(bg_times),
                'min_time': min(bg_times),
                'max_time': max(bg_times),
                'fps': 1.0 / statistics.mean(bg_times),
                'sample_count': len(bg_times)
            }
        }
        
        self._print_results()
        return self.results
    
    def _benchmark_motion_detection(self, detector, test_frame, duration) -> List[float]:
        """Benchmark motion detection performance"""
        times = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            start = time.time()
            detector.detect_motions(test_frame)
            times.append(time.time() - start)
        
        return times
    
    def _benchmark_character_animation(self, animator, motion_data, duration) -> List[float]:
        """Benchmark character animation performance"""
        times = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            start = time.time()
            animator.animate_character(motion_data)
            times.append(time.time() - start)
        
        return times
    
    def _benchmark_background_generation(self, bg_generator, duration) -> List[float]:
        """Benchmark background generation performance"""
        times = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            start = time.time()
            bg_generator.generate_background()
            times.append(time.time() - start)
        
        return times
    
    def _print_results(self):
        """Print benchmark results"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        for component, stats in self.results.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            print(f"  Average Time: {stats['avg_time']*1000:.2f}ms")
            print(f"  Min Time: {stats['min_time']*1000:.2f}ms") 
            print(f"  Max Time: {stats['max_time']*1000:.2f}ms")
            print(f"  Est. Max FPS: {stats['fps']:.1f}")
            print(f"  Samples: {stats['sample_count']}")

def main():
    """Main entry point for benchmark utility"""
    parser = argparse.ArgumentParser(description="Benchmark system performance")
    parser.add_argument("--duration", type=int, default=60,
                       help="Benchmark duration in seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose)
    
    try:
        benchmark = PerformanceBenchmark()
        benchmark.run_benchmark(duration=args.duration)
        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
