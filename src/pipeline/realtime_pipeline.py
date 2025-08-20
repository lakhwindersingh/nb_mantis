""" Real-time processing pipeline """
import cv2
import pygame
import numpy as np
import threading
import queue
import time
from typing import Dict, Any, Optional
import logging
from src.agents.motion_detector import MotionDetector
from src.agents.character_animator import CharacterAnimator
from src.agents.background_generator import BackgroundGenerator
from src.utils.video_processor import VideoProcessor
from src.utils.image_utils import pygame_to_numpy, numpy_to_pygame
from config.settings import load_custom_config, VIDEO_CONFIG, CHARACTER_CONFIG

class RealtimePipeline:
    """Real-time video processing pipeline"""
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = load_custom_config(config_path) if config_path else {}

        # Initialize components
        self.motion_detector = MotionDetector(self.config.get('motion_detection', {}))
        self.character_animator = CharacterAnimator(
            CHARACTER_CONFIG,
            self.config.get('assets_path', 'assets')
        )
        self.background_generator = BackgroundGenerator(self.config.get('background', {}))
        self.video_processor = VideoProcessor()

        # Processing queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.motion_queue = queue.Queue(maxsize=5)
        self.animation_queue = queue.Queue(maxsize=5)

        # Control flags
        self.running = False
        self.paused = False

        # Performance monitoring
        self.fps_counter = 0
        self.last_fps_time = time.time()

        # Initialize pygame
        pygame.init()
        self.screen_size = VIDEO_CONFIG['output_resolution']
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Video Mimic AI - Real-time")

        self.logger.info("Real-time pipeline initialized")

    def run(self):
        """Start real-time processing"""
        self.running = True

        # Start processing threads
        threads = [
            threading.Thread(target=self._capture_thread, daemon=True),
            threading.Thread(target=self._motion_detection_thread, daemon=True),
            threading.Thread(target=self._animation_thread, daemon=True),
            threading.Thread(target=self._background_thread, daemon=True)
        ]

        for thread in threads:
            thread.start()

        # Main display loop
        self._display_loop()

        self.running = False
        self.logger.info("Real-time processing stopped")

    def _capture_thread(self):
        """Capture frames from camera"""
        cap = self.video_processor.get_camera_stream()
        if not cap:
            self.logger.error("Failed to initialize camera")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

        finally:
            cap.release()

    def _motion_detection_thread(self):
        """Process motion detection"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                motion_data = self.motion_detector.detect_motions(frame)

                if not self.motion_queue.full():
                    self.motion_queue.put((frame, motion_data))

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Motion detection error: {e}")

    def _animation_thread(self):
        """Process character animation"""
        while self.running:
            try:
                frame, motion_data = self.motion_queue.get(timeout=0.1)
                character_surface = self.character_animator.animate_character(motion_data)

                if not self.animation_queue.full():
                    self.animation_queue.put((frame, character_surface, motion_data))

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Animation error: {e}")

    def _background_thread(self):
        """Generate backgrounds (runs less frequently)"""
        last_update = 0
        update_interval = 1.0  # Update every second

        while self.running:
            current_time = time.time()
            if current_time - last_update > update_interval:
                try:
                    # Generate new background context
                    context = self._get_background_context()
                    self.background_generator.generate_background(context)
                    last_update = current_time
                except Exception as e:
                    self.logger.error(f"Background generation error: {e}")

            time.sleep(0.1)

    def _display_loop(self):
        """Main display loop"""
        clock = pygame.time.Clock()

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

            if not self.paused:
                self._render_frame()

            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS display

            # Update FPS counter
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.last_fps_time)
                pygame.display.set_caption(f"Video Mimic AI - Real-time (FPS: {fps:.1f})")
                self.fps_counter = 0
                self.last_fps_time = current_time

        pygame.quit()

    def _render_frame(self):
        """Render a single frame"""
        try:
            # Get latest processed frame
            original_frame, character_surface, motion_data = self.animation_queue.get_nowait()

            # Generate background
            background_context = self._get_background_context(motion_data)
            background = self.background_generator.generate_background(background_context)

            # Composite final frame
            final_frame = self._composite_frame(background, character_surface, original_frame)

            # Convert to pygame surface and display
            display_surface = numpy_to_pygame(final_frame)
            display_surface = pygame.transform.scale(display_surface, self.screen_size)
            self.screen.blit(display_surface, (0, 0))

        except queue.Empty:
            # No new frame available, keep displaying last frame
            pass
        except Exception as e:
            self.logger.error(f"Render error: {e}")

    def _composite_frame(self, background: np.ndarray,
                        character_surface: pygame.Surface,
                        original_frame: np.ndarray) -> np.ndarray:
        """Composite final output frame"""
        # Convert character surface to numpy array
        character_array = pygame_to_numpy(character_surface)

        # Resize background to match output resolution
        output_height, output_width = self.screen_size[1], self.screen_size[0]
        background_resized = cv2.resize(background, (output_width, output_height))

        # Position character in center
        char_height, char_width = character_array.shape[:2]
        y_offset = (output_height - char_height) // 2
        x_offset = (output_width - char_width) // 2

        # Create alpha mask for character (assuming character has transparency)
        if character_array.shape[2] == 4:  # Has alpha channel
            alpha = character_array[:, :, 3] / 255.0
            character_rgb = character_array[:, :, :3]
        else:
            # Create alpha mask from non-black pixels
            alpha = np.any(character_array != [0, 0, 0], axis=2).astype(float)
            character_rgb = character_array

        # Blend character onto background
        final_frame = background_resized.copy()

        # Ensure we don't go out of bounds
        end_y = min(y_offset + char_height, output_height)
        end_x = min(x_offset + char_width, output_width)
        char_end_y = end_y - y_offset
        char_end_x = end_x - x_offset

        if y_offset >= 0 and x_offset >= 0:
            for c in range(3):  # RGB channels
                final_frame[y_offset:end_y, x_offset:end_x, c] = (
                    background_resized[y_offset:end_y, x_offset:end_x, c] *
                    (1 - alpha[:char_end_y, :char_end_x]) +
                    character_rgb[:char_end_y, :char_end_x, c] *
                    alpha[:char_end_y, :char_end_x]
                )

        return final_frame.astype(np.uint8)

    def _get_background_context(self, motion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get context for background generation"""
        context = {
            'style': 'cartoon',
            'scene_description': 'colorful animated background'
        }

        if motion_data:
            # Adapt background based on detected emotions/actions
            face_data = motion_data.get('face', {})
            if face_data.get('detected'):
                expressions = face_data.get('expressions', {})
                if expressions.get('smile_intensity', 0) > 0.5:
                    context['scene_description'] = 'bright, cheerful cartoon background'
                elif expressions.get('eyebrow_raise', 0.5) > 0.7:
                    context['scene_description'] = 'surprised, dynamic cartoon background'

        return context