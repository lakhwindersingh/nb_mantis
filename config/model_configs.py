""" Model-specific configurations and paths """
from config.settings import MODELS_DIR
# Face detection models
FACE_MODELS = { "mediapipe": { "model_path": None,  # Uses built-in MediaPipe model
                               "config": { "model_selection": 0, "min_detection_confidence": 0.7 } },
                "dlib": { "model_path": MODELS_DIR / "face_detection" / "shape_predictor_68_face_landmarks.dat", "config": { "predictor_points": 68 } } }
# Hand tracking models
HAND_MODELS = { "mediapipe": { "model_path": None, # Uses built-in MediaPipe model
                               "config": { "static_image_mode": False, "max_num_hands": 2, "min_detection_confidence": 0.7, "min_tracking_confidence": 0.5 } } }
# Background generation models
BACKGROUND_MODELS = { "stable_diffusion": { "model_name": "runwayml/stable-diffusion-v1-5", "config": { "guidance_scale": 7.5, "num_inference_steps": 20 } }, "style_transfer": { "model_path": MODELS_DIR / "background_generation" / "style_transfer.pth", "config": { "content_weight": 1.0, "style_weight": 1000000.0 } } }