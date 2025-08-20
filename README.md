# nb_mantis

### Pull Request Process

# 🎭 Video Mimic AI Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red.svg)](https://mediapipe.dev/)

A sophisticated AI-powered system that captures real-time facial expressions and hand gestures from video input and maps them to animated 2D cartoon characters with dynamically generated backgrounds. Perfect for content creation, virtual avatars, live streaming, and interactive applications.

## 🌟 Features

### 🎪 Motion Detection
- **Real-time Facial Expression Analysis**: Tracks eyebrow movements, eye blinking, mouth expressions, and head pose
- **Advanced Hand Gesture Recognition**: Detects hand positions, finger movements, and common gestures
- **Smooth Motion Interpolation**: Reduces jitter with temporal smoothing algorithms
- **Multi-face Support**: Can handle multiple faces in frame (configurable)

### 🎨 Character Animation
- **2D Sprite-based Animation**: Modular character system with interchangeable parts
- **Expression Mapping**: Intelligent mapping from real facial expressions to cartoon equivalents
- **Gesture Translation**: Converts hand movements into character actions and poses
- **Rigging System**: Advanced bone-based animation with hierarchical transforms
- **Smooth Transitions**: Interpolated animations for natural character movement

### 🌄 Background Generation
- **AI-Powered Backgrounds**: Multiple generation methods including Stable Diffusion
- **Procedural Generation**: Fallback system with gradient, noise, and geometric patterns
- **Context-Aware Adaptation**: Backgrounds that respond to detected emotions and actions
- **Style Consistency**: Maintains visual coherence across frames
- **Performance Optimization**: Efficient caching and update strategies

### ⚡ Processing Modes
- **Real-time Processing**: Live camera input with instant character animation
- **Batch Video Processing**: Process entire video files with progress tracking
- **Flexible Pipeline**: Modular architecture supporting custom processing workflows

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Webcam (for real-time mode)
- GPU recommended for AI background generation (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/video_mimic_ai.git
   cd video_mimic_ai
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run basic demo**
   ```bash
   python examples/basic_demo.py
   ```

### Quick Demo

**Real-time Character Animation:**
```
bash python examples/realtime_demo.py
``` 
*Controls: SPACE to pause/resume, ESC to quit*

**Process a Video File:**
```
bash python examples/batch_processing.py input_video.mp4 --output animated_result.mp4
``` 

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.15, or Ubuntu 18.04+
- **RAM**: 8GB RAM
- **Python**: 3.9+
- **Storage**: 2GB free space

### Recommended Specifications
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **RAM**: 16GB RAM
- **GPU**: NVIDIA RTX 3060 or AMD RX 6600 (for AI background generation)
- **CPU**: Intel i7 or AMD Ryzen 7
- **Storage**: 5GB free space (for models and assets)

## 🛠️ Installation Guide

### Standard Installation
```
bash
# Clone repository
git clone [https://github.com/yourusername/video_mimic_ai.git](https://github.com/yourusername/video_mimic_ai.git) cd video_mimic_ai
# Setup virtual environment
python -m venv venv source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Install in development mode
pip install -e .
``` 

### Docker Installation
```
bash
# Build Docker image
docker build -t video-mimic-ai .
# Run with camera access
docker run -it --device=/dev/video0 video-mimic-ai
``` 

### GPU Support Setup

For NVIDIA GPU acceleration:
```
bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# Install additional GPU libraries
pip install onnxruntime-gpu
``` 

## 📖 Usage Examples

### Basic Usage
```
python from pipeline.realtime_pipeline import RealtimePipeline
# Initialize pipeline
pipeline = RealtimePipeline()
# Start real-time processing
pipeline.run()
``` 

### Custom Configuration
```
python from pipeline.batch_pipeline import BatchPipeline
# Custom configuration
config = { 'motion_detection': { 'face_detection': {'min_detection_confidence': 0.8}, 'hand_tracking': {'max_num_hands': 2} }, 'character': { 'expression_sensitivity': 1.2, 'animation_fps': 60 }, 'background': { 'generation_method': 'diffusion', 'style': 'anime' } }
# Process video with custom settings
pipeline = BatchPipeline(config) pipeline.process_video('input.mp4', 'output.mp4')
``` 

### API Integration
```
python from agents.motion_detector import MotionDetector from agents.character_animator import CharacterAnimator from agents.background_generator import BackgroundGenerator
# Initialize components
detector = MotionDetector() animator = CharacterAnimator(config, assets_path) bg_generator = BackgroundGenerator(config)
# Process single frame
motion_data = detector.detect_motions(frame) character = animator.animate_character(motion_data) background = bg_generator.generate_background(context)
``` 

## ⚙️ Configuration

### Main Configuration File (`config/settings.py`)
```
python
# Video settings
VIDEO_CONFIG = { "input_resolution": (640, 480), "output_resolution": (1920, 1080), "fps": 30, "codec": "mp4v" }
# Motion detection sensitivity
MOTION_DETECTION_CONFIG = { "face_detection": { "min_detection_confidence": 0.7, "min_tracking_confidence": 0.5 }, "hand_tracking": { "max_num_hands": 2, "min_detection_confidence": 0.7 } }
# Character animation parameters
CHARACTER_CONFIG = { "sprite_resolution": (512, 512), "animation_fps": 60, "interpolation_smoothness": 0.8, "expression_sensitivity": 1.0 }
# Background generation
BACKGROUND_CONFIG = { "generation_method": "diffusion", # "diffusion", "procedural", "style_transfer" "resolution": (1920, 1080), "style": "cartoon", "update_frequency": 30 }
``` 

### Environment Variables (`.env`)
```
bash
# API Keys
OPENAI_API_KEY=your_openai_key_here HUGGINGFACE_TOKEN=your_hf_token_here
# Performance settings
USE_GPU=true MAX_FPS=30 PROCESSING_THREADS=4
# Development
DEBUG=false LOG_LEVEL=INFO
``` 

## 🎨 Character Customization

### Adding Custom Characters

1. **Create sprite assets** in `assets/characters/sprites/`
2. **Update configuration** in `assets/characters/sprites/config.json`:
```
json { "character_parts": { "head": "my_character_head.png", "body": "my_character_body.png", "left_hand": "my_character_left_hand.png", "right_hand": "my_character_right_hand.png" }, "expressions": { "happy": { "eyes": "happy_eyes.png", "mouth": "smile.png" }, "surprised": { "eyes": "wide_eyes.png", "mouth": "open_mouth.png" } } }
``` 

3. **Test your character**:
```
bash python examples/character_test.py --character my_character
``` 

### Sprite Requirements
- **Format**: PNG with transparency
- **Resolution**: 512x512 recommended
- **Style**: Consistent art style across all parts
- **Naming**: Follow convention: `part_state.png`

## 🌄 Background Customization

### Diffusion Model Setup

1. **Install diffusers**:
```
bash pip install diffusers transformers accelerate
``` 

2. **Configure model** in `config/model_configs.py`:
```
python BACKGROUND_MODELS = { "stable_diffusion": { "model_name": "runwayml/stable-diffusion-v1-5", "config": { "guidance_scale": 7.5, "num_inference_steps": 20 } } }
``` 

3. **Custom prompts** in your code:
```
python context = { 'style': 'anime', 'scene_description': 'magical forest with glowing trees' } background = bg_generator.generate_background(context)
``` 

## 🔧 CLI Usage

### Main Application
```
bash
# Real-time mode
python -m src.main --mode realtime
# Batch processing
python -m src.main --mode batch --input video.mp4 --output result.mp4
# Custom configuration
python -m src.main --config custom_config.yaml --verbose
``` 

### Utility Commands
```
bash
# Test motion detection
python -m src.utils.test_motion --camera 0
# Benchmark performance
python -m src.utils.benchmark --duration 60
# Export character animation
python -m src.utils.export_character --format gif --duration 5
``` 

## 📊 Performance Optimization

### Real-time Performance Tips

1. **Reduce Resolution**: Lower input resolution for better FPS
2. **Optimize Detection**: Adjust confidence thresholds
3. **Threading**: Enable multi-threading in config
4. **GPU Acceleration**: Use CUDA for AI models
```
python
# Performance config
PERFORMANCE_CONFIG = { 'input_resolution': (320, 240), # Lower for speed 'detection_skip_frames': 2, # Process every 2nd frame 'background_update_freq': 60, # Update less frequently 'enable_gpu': True, 'thread_count': 4 }
``` 

### Memory Management
```
python
# Memory optimization
MEMORY_CONFIG = { 'frame_buffer_size': 5, 'motion_history_length': 10, 'background_cache_size': 3, 'sprite_preload': True }
``` 

## 🧪 Testing

### Run Test Suite
```
bash
# Run all tests
python -m pytest tests/
# Run specific test category
python -m pytest tests/test_motion_detector.py -v
# Run with coverage
python -m pytest --cov=src tests/
``` 

### Manual Testing
```
bash
# Test motion detection
python tests/manual_test_motion.py
# Test character animation
python tests/manual_test_character.py
# Test background generation
python tests/manual_test_background.py
``` 

### Performance Testing
```
bash
# Benchmark motion detection speed
python tests/benchmark_motion.py --duration 30
# Test memory usage
python tests/test_memory_usage.py --profile
# GPU performance test
python tests/gpu_benchmark.py
``` 

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Detected
```
bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
# Test specific camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
``` 

#### Low FPS Performance
```
python
# Reduce processing load
config = { 'input_resolution': (320, 240), 'detection_confidence': 0.8, # Higher = less sensitive but faster 'background_method': 'procedural' # Faster than AI generation }
``` 

#### Memory Issues
```
bash
# Monitor memory usage
python -m memory_profiler src/main.py
# Clear caches
python -c "from src.agents.background_generator import BackgroundGenerator; bg = BackgroundGenerator({}); bg.clear_cache()"
``` 

#### Missing Dependencies
```
bash
# Reinstall core dependencies
pip install --upgrade opencv-python mediapipe pygame numpy
# Check MediaPipe installation
python -c "import mediapipe as mp; print(mp.**version**)"
``` 

### Debug Mode

Enable debug logging:
```
bash export LOG_LEVEL=DEBUG python src/main.py --verbose
``` 

Or in code:
```
python import logging logging.basicConfig(level=logging.DEBUG)
``` 

## 📚 API Documentation

### Core Classes

#### MotionDetector
```
python class MotionDetector: def **init**(self, config: Optional[Dict[str, Any]] = None) def detect_motions(self, frame: np.ndarray) -> Dict[str, Any] def extract_face_features(self, face_results: Any) -> Dict[str, Any] def extract_hand_features(self, hand_results: Any) -> Dict[str, Any]
``` 

#### CharacterAnimator
```
python class CharacterAnimator: def **init**(self, character_config: Dict[str, Any], assets_path: Path) def animate_character(self, motion_data: Dict[str, Any]) -> pygame.Surface def map_face_to_character(self, face_data: Dict[str, Any]) -> Dict[str, Any] def map_hands_to_character(self, hand_data: Dict[str, Any]) -> Dict[str, Any]
``` 

#### BackgroundGenerator
```
python class BackgroundGenerator: def **init**(self, config: Dict[str, Any]) def generate_background(self, context: Optional[Dict[str, Any]] = None) -> np.ndarray
``` 

### Data Structures

#### Motion Data Format
```
python { 'face': { 'detected': bool, 'landmarks': List[List[float]], # (x, y, z) coordinates 'expressions': { 'eye_openness': float, # 0.0 - 1.0 'mouth_openness': float, # 0.0 - 1.0 'eyebrow_raise': float, # 0.0 - 1.0 'smile_intensity': float # 0.0 - 1.0 }, 'head_pose': { 'yaw': float, # degrees 'pitch': float, # degrees 'roll': float # degrees } }, 'hands': { 'detected': bool, 'left_hand': { 'landmarks': List[List[float]], 'gesture': str # 'neutral', 'fist', 'open', etc. }, 'right_hand': { 'landmarks': List[List[float]], 'gesture': str } }, 'timestamp': float }
``` 

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/video_mimic_ai.git
   ```
3. **Create feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Code Style

We use Black for code formatting and flake8 for linting:
```
bash
# Format code
black src/ tests/
# Check linting
flake8 src/ tests/
# Type checking
mypy src/
``` 

### Pull Request Process

1. **Update tests** for new functionality
2. **Update documentation** as needed
3. **Ensure all tests pass**:
   ```bash
   python -m pytest tests/
   ```
4. **Submit pull request** with detailed description

### Development Guidelines

- **Follow PEP 8** style guidelines
- **Write comprehensive tests** for new features
- **Document public APIs** with docstrings
- **Use type hints** where appropriate
- **Keep commits atomic** and well-described

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** team for excellent computer vision tools
- **OpenCV** community for robust image processing
- **Stable Diffusion** developers for AI background generation
- **Pygame** community for 2D graphics capabilities
- **OpenAI** for inspiring AI applications

## 📞 Support

### Getting Help

- **Documentation**: Check the `docs/` folder
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Wiki**: Community-maintained examples and tutorials

### Community

- **Discord**: [Join our Discord server](https://discord.gg/video-mimic-ai)
- **Reddit**: [r/VideoMimicAI](https://reddit.com/r/VideoMimicAI)
- **Twitter**: [@VideoMimicAI](https://twitter.com/VideoMimicAI)

## 🗺️ Roadmap

### Version 0.2.0 (Next Release)
- [ ] Multi-character support
- [ ] Voice-to-lip sync
- [ ] Mobile app version
- [ ] Cloud processing API

### Version 0.3.0
- [ ] VR/AR integration
- [ ] Real-time collaboration
- [ ] Advanced AI backgrounds
- [ ] Character marketplace

### Version 1.0.0
- [ ] Production-ready stability
- [ ] Enterprise features
- [ ] Advanced analytics
- [ ] Commercial licensing

---

**Made with ❤️ by the Video Mimic AI Team**

*Transform your expressions into animated magic!* ✨
```
