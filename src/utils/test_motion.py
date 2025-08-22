
"""
Motion detection testing utility
Provides real-time testing and debugging of motion detection capabilities
"""
import cv2
import pygame
import numpy as np
import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.motion_detector import MotionDetector
from utils.video_processor import VideoProcessor
from utils.logging_config import setup_logging
from config.settings import MOTION_DETECTION_CONFIG

class MotionTester:
    """Interactive motion detection tester"""
    
    def __init__(self, camera_id: int = 0, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.config = config or MOTION_DETECTION_CONFIG
        
        # Initialize components
        self.motion_detector = MotionDetector(self.config)
        self.video_processor = VideoProcessor()
        
        # Display settings
        self.window_width = 1200
        self.window_height = 800
        self.fps_target = 30
        
        # Testing modes
        self.show_landmarks = True
        self.show_debug_info = True
        self.record_data = False
        self.save_frames = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.processing_times = []
        self.detection_stats = {
            'total_frames': 0,
            'face_detections': 0,
            'hand_detections': 0,
            'both_detections': 0
        }
        
        # Recording
        self.recorded_data = []
        self.frame_count = 0
        
        # Initialize display
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Motion Detection Tester")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.logger.info(f"Motion tester initialized for camera {camera_id}")
    
    def run_test(self, duration: Optional[int] = None):
        """
        Run interactive motion detection test
        
        Args:
            duration: Optional test duration in seconds
        """
        print("🎯 Motion Detection Tester")
        print("Controls:")
        print("  SPACE - Toggle landmark display")
        print("  D - Toggle debug info")
        print("  R - Toggle recording")
        print("  S - Save current frame")
        print("  C - Clear statistics")
        print("  ESC/Q - Quit")
        print("=" * 50)
        
        # Get camera stream
        cap = self.video_processor.get_camera_stream(self.camera_id)
        if not cap:
            self.logger.error(f"Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties for testing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        clock = pygame.time.Clock()
        start_time = time.time()
        running = True
        
        try:
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            self.show_landmarks = not self.show_landmarks
                            print(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
                        elif event.key == pygame.K_d:
                            self.show_debug_info = not self.show_debug_info
                            print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
                        elif event.key == pygame.K_r:
                            self.record_data = not self.record_data
                            print(f"Recording: {'ON' if self.record_data else 'OFF'}")
                        elif event.key == pygame.K_s:
                            self.save_frames = True
                            print("Frame saved!")
                        elif event.key == pygame.K_c:
                            self._clear_statistics()
                            print("Statistics cleared!")
                
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    running = False
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                self._process_frame(frame)
                
                # Update display
                pygame.display.flip()
                clock.tick(self.fps_target)
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:
                    fps = self.fps_counter / (current_time - self.fps_timer)
                    pygame.display.set_caption(f"Motion Detection Tester - FPS: {fps:.1f}")
                    self.fps_counter = 0
                    self.fps_timer = current_time
        
        finally:
            cap.release()
            
        # Show final results
        self._show_final_results()
        
        # Save recorded data if any
        if self.recorded_data:
            self._save_recorded_data()
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame and update display"""
        processing_start = time.time()
        
        # Detect motions
        motion_data = self.motion_detector.detect_motions(frame)
        
        processing_time = time.time() - processing_start
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # Update statistics
        self._update_statistics(motion_data)
        
        # Record data if enabled
        if self.record_data:
            self._record_frame_data(frame, motion_data, processing_time)
        
        # Draw visualization
        self._draw_frame(frame, motion_data, processing_time)
    
    def _update_statistics(self, motion_data: Dict[str, Any]):
        """Update detection statistics"""
        self.detection_stats['total_frames'] += 1
        
        face_detected = motion_data.get('face', {}).get('detected', False)
        hands_detected = motion_data.get('hands', {}).get('detected', False)
        
        if face_detected:
            self.detection_stats['face_detections'] += 1
        if hands_detected:
            self.detection_stats['hand_detections'] += 1
        if face_detected and hands_detected:
            self.detection_stats['both_detections'] += 1
    
    def _record_frame_data(self, frame: np.ndarray, motion_data: Dict[str, Any], processing_time: float):
        """Record frame data for analysis"""
        record_entry = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'processing_time': processing_time,
            'motion_data': motion_data,
            'frame_shape': frame.shape
        }
        
        self.recorded_data.append(record_entry)
        self.frame_count += 1
        
        # Limit recorded data size
        if len(self.recorded_data) > 1000:
            self.recorded_data.pop(0)
    
    def _draw_frame(self, frame: np.ndarray, motion_data: Dict[str, Any], processing_time: float):
        """Draw frame with visualizations"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Convert frame to pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks if enabled
        if self.show_landmarks:
            frame_rgb = self._draw_landmarks(frame_rgb, motion_data)
        
        # Scale frame to fit display
        frame_height, frame_width = frame_rgb.shape[:2]
        display_width = 640
        display_height = int((display_width / frame_width) * frame_height)
        
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (display_width, display_height))
        
        # Draw frame
        self.screen.blit(frame_surface, (10, 10))
        
        # Draw debug info if enabled
        if self.show_debug_info:
            self._draw_debug_info(motion_data, processing_time, display_width + 20, 10)
        
        # Draw statistics
        self._draw_statistics(10, display_height + 20)
        
        # Draw controls
        self._draw_controls(display_width + 20, 400)
    
    def _draw_landmarks(self, frame: np.ndarray, motion_data: Dict[str, Any]) -> np.ndarray:
        """Draw detection landmarks on frame"""
        frame_with_landmarks = frame.copy()
        
        # Draw face landmarks
        face_data = motion_data.get('face', {})
        if face_data.get('detected') and 'landmarks' in face_data:
            landmarks = face_data['landmarks']
            for landmark in landmarks:
                x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame_with_landmarks, (x, y), 1, (0, 255, 0), -1)
        
        # Draw hand landmarks
        hands_data = motion_data.get('hands', {})
        if hands_data.get('detected'):
            for hand_key in ['left_hand', 'right_hand']:
                hand_data = hands_data.get(hand_key)
                if hand_data and 'landmarks' in hand_data:
                    landmarks = hand_data['landmarks']
                    color = (255, 0, 0) if hand_key == 'left_hand' else (0, 0, 255)
                    
                    for landmark in landmarks:
                        x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv2.circle(frame_with_landmarks, (x, y), 2, color, -1)
                    
                    # Draw hand connections (simplified)
                    if len(landmarks) >= 5:
                        # Draw palm outline
                        palm_points = landmarks[:5]
                        palm_coords = [(int(p[0] * frame.shape[1]), int(p[1] * frame.shape[0])) 
                                     for p in palm_points]
                        for i in range(len(palm_coords)):
                            if i < len(palm_coords) - 1:
                                cv2.line(frame_with_landmarks, palm_coords[i], 
                                       palm_coords[i + 1], color, 1)
        
        return frame_with_landmarks
    
    def _draw_debug_info(self, motion_data: Dict[str, Any], processing_time: float, x: int, y: int):
        """Draw debug information"""
        debug_lines = []
        
        # Processing performance
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        debug_lines.append(f"Processing Time: {processing_time*1000:.1f}ms")
        debug_lines.append(f"Avg Processing: {avg_processing_time*1000:.1f}ms")
        debug_lines.append(f"Est. Max FPS: {1.0/avg_processing_time:.1f}" if avg_processing_time > 0 else "Est. Max FPS: ∞")
        debug_lines.append("")
        
        # Face detection info
        face_data = motion_data.get('face', {})
        debug_lines.append("=== FACE DETECTION ===")
        debug_lines.append(f"Detected: {face_data.get('detected', False)}")
        
        if face_data.get('detected'):
            expressions = face_data.get('expressions', {})
            debug_lines.append("Expressions:")
            for expr_name, value in expressions.items():
                debug_lines.append(f"  {expr_name}: {value:.3f}")
            
            head_pose = face_data.get('head_pose', {})
            if head_pose:
                debug_lines.append("Head Pose:")
                for pose_name, value in head_pose.items():
                    debug_lines.append(f"  {pose_name}: {value:.1f}°")
        
        debug_lines.append("")
        
        # Hand detection info
        hands_data = motion_data.get('hands', {})
        debug_lines.append("=== HAND DETECTION ===")
        debug_lines.append(f"Detected: {hands_data.get('detected', False)}")
        
        if hands_data.get('detected'):
            for hand_key in ['left_hand', 'right_hand']:
                hand_data = hands_data.get(hand_key)
                if hand_data:
                    debug_lines.append(f"{hand_key.replace('_', ' ').title()}:")
                    debug_lines.append(f"  Gesture: {hand_data.get('gesture', 'unknown')}")
                    debug_lines.append(f"  Landmarks: {len(hand_data.get('landmarks', []))}")
        
        # Draw debug text
        for i, line in enumerate(debug_lines):
            color = (255, 255, 255) if not line.startswith("===") else (255, 255, 0)
            text_surface = self.small_font.render(line, True, color)
            self.screen.blit(text_surface, (x, y + i * 16))
    
    def _draw_statistics(self, x: int, y: int):
        """Draw detection statistics"""
        stats_lines = [
            "=== STATISTICS ===",
            f"Total Frames: {self.detection_stats['total_frames']}",
            f"Face Detections: {self.detection_stats['face_detections']}",
            f"Hand Detections: {self.detection_stats['hand_detections']}",
            f"Both Detected: {self.detection_stats['both_detections']}",
        ]
        
        if self.detection_stats['total_frames'] > 0:
            face_rate = (self.detection_stats['face_detections'] / self.detection_stats['total_frames']) * 100
            hand_rate = (self.detection_stats['hand_detections'] / self.detection_stats['total_frames']) * 100
            both_rate = (self.detection_stats['both_detections'] / self.detection_stats['total_frames']) * 100
            
            stats_lines.extend([
                f"Face Rate: {face_rate:.1f}%",
                f"Hand Rate: {hand_rate:.1f}%",
                f"Both Rate: {both_rate:.1f}%"
            ])
        
        for i, line in enumerate(stats_lines):
            color = (255, 255, 255) if not line.startswith("===") else (0, 255, 0)
            text_surface = self.font.render(line, True, color)
            self.screen.blit(text_surface, (x, y + i * 25))
    
    def _draw_controls(self, x: int, y: int):
        """Draw control instructions"""
        controls = [
            "=== CONTROLS ===",
            "SPACE - Toggle landmarks",
            "D - Toggle debug info", 
            "R - Toggle recording",
            "S - Save frame",
            "C - Clear statistics",
            "ESC/Q - Quit"
        ]
        
        status_lines = [
            "",
            f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}",
            f"Debug: {'ON' if self.show_debug_info else 'OFF'}",
            f"Recording: {'ON' if self.record_data else 'OFF'}",
            f"Recorded: {len(self.recorded_data)} frames"
        ]
        
        all_lines = controls + status_lines
        
        for i, line in enumerate(all_lines):
            if line.startswith("==="):
                color = (0, 255, 255)
            elif ":" in line and not line.startswith(" "):
                color = (255, 255, 0)
            else:
                color = (200, 200, 200)
                
            text_surface = self.small_font.render(line, True, color)
            self.screen.blit(text_surface, (x, y + i * 18))
    
    def _clear_statistics(self):
        """Clear all statistics"""
        self.detection_stats = {
            'total_frames': 0,
            'face_detections': 0,
            'hand_detections': 0,
            'both_detections': 0
        }
        self.processing_times.clear()
    
    def _show_final_results(self):
        """Show final test results"""
        print("\n" + "=" * 50)
        print("MOTION DETECTION TEST RESULTS")
        print("=" * 50)
        
        if self.detection_stats['total_frames'] > 0:
            face_rate = (self.detection_stats['face_detections'] / self.detection_stats['total_frames']) * 100
            hand_rate = (self.detection_stats['hand_detections'] / self.detection_stats['total_frames']) * 100
            both_rate = (self.detection_stats['both_detections'] / self.detection_stats['total_frames']) * 100
            
            print(f"Total Frames Processed: {self.detection_stats['total_frames']}")
            print(f"Face Detection Rate: {face_rate:.1f}%")
            print(f"Hand Detection Rate: {hand_rate:.1f}%")
            print(f"Both Detected Rate: {both_rate:.1f}%")
        
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            min_time = np.min(self.processing_times)
            max_time = np.max(self.processing_times)
            
            print(f"\nPerformance:")
            print(f"Average Processing Time: {avg_time*1000:.1f}ms")
            print(f"Min Processing Time: {min_time*1000:.1f}ms")
            print(f"Max Processing Time: {max_time*1000:.1f}ms")
            print(f"Estimated Max FPS: {1.0/avg_time:.1f}")
        
        print(f"\nRecorded Data Points: {len(self.recorded_data)}")
    
    def _save_recorded_data(self):
        """Save recorded motion data to file"""
        output_dir = Path("output/motion_test_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"motion_test_{timestamp}.json"
        
        # Prepare data for JSON serialization
        serializable_data = []
        for entry in self.recorded_data:
            serializable_entry = entry.copy()
            # Convert numpy arrays to lists
            motion_data = serializable_entry['motion_data']
            if 'face' in motion_data and 'landmarks' in motion_data['face']:
                motion_data['face']['landmarks'] = [
                    [float(coord) for coord in landmark] 
                    for landmark in motion_data['face']['landmarks']
                ]
            
            serializable_data.append(serializable_entry)
        
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    'test_info': {
                        'camera_id': self.camera_id,
                        'total_frames': len(serializable_data),
                        'config': self.config
                    },
                    'statistics': self.detection_stats,
                    'data': serializable_data
                }, f, indent=2)
            
            print(f"\nRecorded data saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save recorded data: {e}")

def main():
    """Main entry point for motion testing utility"""
    parser = argparse.ArgumentParser(description="Test motion detection capabilities")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device ID (default: 0)")
    parser.add_argument("--duration", type=int, 
                       help="Test duration in seconds (default: unlimited)")
    parser.add_argument("--config", type=str,
                       help="Custom configuration file path")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--save-data", action="store_true",
                       help="Enable automatic data recording")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load custom config if provided
    config = MOTION_DETECTION_CONFIG
    if args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                custom_config = yaml.safe_load(f)
                config.update(custom_config)
        except Exception as e:
            logger.warning(f"Failed to load custom config: {e}")
    
    # Create and run tester
    try:
        tester = MotionTester(camera_id=args.camera, config=config)
        
        if args.save_data:
            tester.record_data = True
            logger.info("Auto-recording enabled")
        
        tester.run_test(duration=args.duration)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    finally:
        pygame.quit()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
