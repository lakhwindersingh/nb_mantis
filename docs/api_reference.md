
# API Reference
## Motion Detection
### MotionDetector
Main class for detecting facial expressions and hand gestures.
#### Methods
- `detect_motions(frame: np.ndarray) -> Dict[str, Any]`
    - Detect all motions in the given frame
    - Returns dictionary with face and hand detection results

- `extract_face_features(face_results: Any) -> Dict[str, Any]`
    - Extract facial features from MediaPipe results
    - Returns facial landmarks and expression analysis

## Character Animation
### CharacterAnimator
Handles 2D character animation based on detected motions.
#### Methods
- `animate_character(motion_data: Dict[str, Any]) -> pygame.Surface`
    - Generate character animation frame
    - Returns rendered character as pygame Surface

- `map_face_to_character(face_data: Dict[str, Any]) -> Dict[str, Any]`
    - Map facial motions to character expressions
    - Returns animation parameters for facial features

## Background Generation
### BackgroundGenerator
Generates dynamic backgrounds using AI models.
#### Methods
- `generate_background(context: Optional[Dict[str, Any]]) -> np.ndarray`
    - Generate background image based on context
    - Returns background as numpy array

## Pipeline
### RealtimePipeline
Real-time video processing pipeline.
#### Methods
- `run()`
    - Start real-time processing loop
    - Handles camera input and live display

### BatchPipeline
Batch video file processing pipeline.
#### Methods
- `process_video(input_path: str, output_path: Optional[str])`
    - Process entire video file
    - Saves result to output path