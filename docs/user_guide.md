# User Guide
## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run setup: `python setup.py install`

## Quick Start
### Real-time Processing
``` bash
python examples/realtime_demo.py
```
Controls:
- SPACE: Pause/resume
- ESC: Quit

### Batch Processing
``` bash
python examples/batch_processing.py input_video.mp4 --output result.mp4
```
## Configuration
Edit `config/settings.py` to customize:
- Video resolution and framerate
- Motion detection sensitivity
- Character animation parameters
- Background generation settings

## Character Customization
1. Add sprites to `assets/characters/sprites/`
2. Update sprite configuration in `assets/characters/sprites/config.json`
3. Modify animation mappings in character animator

## Troubleshooting
### Common Issues
- **Camera not detected**: Check camera permissions and device ID
- **Low FPS**: Reduce resolution or disable GPU acceleration
- **Character not animating**: Verify sprite files are present EOF

cat > docs/development_guide.md << 'EOF'
# Development Guide
## Architecture Overview
The system consists of three main agents:
1. **Motion Detector**: Captures facial and hand movements
2. **Character Animator**: Maps movements to 2D character
3. **Background Generator**: Creates dynamic backgrounds

## Adding New Features
### Custom Motion Detection
1. Create new detector class in `src/models/`
2. Inherit from base detector interface
3. Register in motion detector agent

### New Character Parts
1. Add sprites to assets directory
2. Update sprite manager configuration
3. Modify animation engine mappings

### Background Generators
1. Implement generator class in `src/models/background_gen/`
2. Add to background generator agent
3. Configure model parameters

## Testing
Run tests with:
``` bash
python -m pytest tests/
```
## Performance Optimization
### GPU Acceleration
- Enable CUDA for AI models
- Use GPU-accelerated OpenCV operations
- Optimize memory usage

### Threading
- Separate capture, processing, and display threads
- Use appropriate queue sizes
- Monitor CPU usage

## Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request EOF
