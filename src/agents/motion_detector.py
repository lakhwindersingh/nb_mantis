import cv2
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import math
from src.models.face_detection.mediapipe_face import MediaPipeFaceDetector
from src.models.hand_tracking.mediapipe_hands import MediaPipeHandDetector

# Add MediaPipe pose detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Pose detection disabled.")

class MotionDetector:
    """ Enhanced motion detection agent that combines face, hand, and pose tracking """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Initialize detectors
        self.face_detector = MediaPipeFaceDetector(
            self.config.get('face_detection', {})
        )
        self.hand_detector = MediaPipeHandDetector(
            self.config.get('hand_tracking', {})
        )
        
        # Initialize pose detector
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config.get('pose_detection', {}).get('model_complexity', 1),
                enable_segmentation=self.config.get('pose_detection', {}).get('enable_segmentation', False),
                min_detection_confidence=self.config.get('pose_detection', {}).get('min_detection_confidence', 0.5),
                min_tracking_confidence=self.config.get('pose_detection', {}).get('min_tracking_confidence', 0.5)
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose_enabled = True
        else:
            self.pose_enabled = False

        # Motion history for smoothing and movement analysis
        self.motion_history = []
        self.pose_history = []
        self.max_history_length = 10
        
        # Previous pose landmarks for movement calculation
        self.previous_pose = None
        
        # Movement thresholds
        self.movement_threshold = self.config.get('movement_detection', {}).get('threshold', 0.02)
        self.significant_movement_threshold = self.config.get('movement_detection', {}).get('significant_threshold', 0.05)

        self.logger.info(f"Motion detector initialized with pose detection: {self.pose_enabled}")

    def detect_motions(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect all motions in the given frame including pose and body movements

        Args:
            frame: Input video frame

        Returns:
            Dictionary containing detected motions
        """
        if frame is None or frame.size == 0:
            return self._get_empty_motion_data()

        # Detect face landmarks
        face_results = self.face_detector.detect(frame)
        face_features = self.extract_face_features(face_results)

        # Detect hand landmarks
        hand_results = self.hand_detector.detect(frame)
        hand_features = self.extract_hand_features(hand_results)
        
        # Detect pose landmarks
        pose_features = self.extract_pose_features(frame)
        
        # Analyze body movements
        movement_analysis = self.analyze_body_movements(pose_features)

        motion_data = {
            'face': face_features,
            'hands': hand_features,
            'pose': pose_features,
            'body_movements': movement_analysis,
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }

        # Apply smoothing
        smoothed_data = self._apply_smoothing(motion_data)

        return smoothed_data

    def extract_pose_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract pose landmarks and analyze body posture"""
        if not self.pose_enabled:
            return {'detected': False}
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose detection
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return {'detected': False}
            
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Analyze pose characteristics
            pose_analysis = self._analyze_pose(landmarks)
            
            # Store current pose for movement analysis
            self.previous_pose = landmarks
            
            return {
                'detected': True,
                'landmarks': landmarks,
                'pose_analysis': pose_analysis,
                'body_angles': self._calculate_body_angles(landmarks),
                'posture_classification': self._classify_posture(landmarks),
                'balance_score': self._calculate_balance_score(landmarks)
            }
            
        except Exception as e:
            self.logger.error(f"Error in pose detection: {e}")
            return {'detected': False}

    def _analyze_pose(self, landmarks: List[List[float]]) -> Dict[str, Any]:
        """Analyze pose characteristics and body positioning"""
        if len(landmarks) < 33:  # MediaPipe pose has 33 landmarks
            return {}
            
        try:
            # Key landmark indices for MediaPipe Pose
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14
            LEFT_WRIST = 15
            RIGHT_WRIST = 16
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            NOSE = 0
            
            # Calculate key measurements
            shoulder_width = self._calculate_distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
            hip_width = self._calculate_distance(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
            torso_length = self._calculate_distance(
                [(landmarks[LEFT_SHOULDER][0] + landmarks[RIGHT_SHOULDER][0]) / 2,
                 (landmarks[LEFT_SHOULDER][1] + landmarks[RIGHT_SHOULDER][1]) / 2],
                [(landmarks[LEFT_HIP][0] + landmarks[RIGHT_HIP][0]) / 2,
                 (landmarks[LEFT_HIP][1] + landmarks[RIGHT_HIP][1]) / 2]
            )
            
            # Arm positions
            left_arm_raised = landmarks[LEFT_WRIST][1] < landmarks[LEFT_SHOULDER][1]
            right_arm_raised = landmarks[RIGHT_WRIST][1] < landmarks[RIGHT_SHOULDER][1]
            
            # Leg positioning
            legs_apart = hip_width > shoulder_width * 1.2
            
            # Body lean
            body_lean = self._calculate_body_lean(landmarks)
            
            return {
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'torso_length': torso_length,
                'left_arm_raised': left_arm_raised,
                'right_arm_raised': right_arm_raised,
                'both_arms_raised': left_arm_raised and right_arm_raised,
                'legs_apart': legs_apart,
                'body_lean': body_lean,
                'symmetry_score': self._calculate_body_symmetry(landmarks)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pose: {e}")
            return {}

    def _calculate_body_angles(self, landmarks: List[List[float]]) -> Dict[str, float]:
        """Calculate important body joint angles"""
        try:
            angles = {}
            
            # Left arm angle (shoulder-elbow-wrist)
            if len(landmarks) >= 17:
                left_arm_angle = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                right_arm_angle = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                
                angles['left_arm_angle'] = left_arm_angle
                angles['right_arm_angle'] = right_arm_angle
            
            # Leg angles (hip-knee-ankle)
            if len(landmarks) >= 29:
                left_leg_angle = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                right_leg_angle = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                
                angles['left_leg_angle'] = left_leg_angle
                angles['right_leg_angle'] = right_leg_angle
            
            # Torso angle (relative to vertical)
            if len(landmarks) >= 25:
                shoulder_center = [(landmarks[11][0] + landmarks[12][0]) / 2,
                                 (landmarks[11][1] + landmarks[12][1]) / 2]
                hip_center = [(landmarks[23][0] + landmarks[24][0]) / 2,
                             (landmarks[23][1] + landmarks[24][1]) / 2]
                
                torso_angle = math.degrees(math.atan2(
                    shoulder_center[0] - hip_center[0],
                    hip_center[1] - shoulder_center[1]
                ))
                angles['torso_angle'] = torso_angle
            
            return angles
            
        except Exception as e:
            self.logger.error(f"Error calculating body angles: {e}")
            return {}

    def _classify_posture(self, landmarks: List[List[float]]) -> str:
        """Classify the overall body posture"""
        try:
            pose_analysis = self._analyze_pose(landmarks)
            
            # Standing detection
            if len(landmarks) >= 29:
                avg_ankle_y = (landmarks[27][1] + landmarks[28][1]) / 2
                avg_knee_y = (landmarks[25][1] + landmarks[26][1]) / 2
                avg_hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
                
                # Check if person is standing (ankles below knees below hips)
                if avg_ankle_y > avg_knee_y > avg_hip_y:
                    if pose_analysis.get('both_arms_raised', False):
                        return "standing_arms_raised"
                    elif pose_analysis.get('left_arm_raised', False) or pose_analysis.get('right_arm_raised', False):
                        return "standing_one_arm_raised"
                    elif pose_analysis.get('legs_apart', False):
                        return "standing_wide_stance"
                    else:
                        return "standing_normal"
                        
                # Check for sitting (simplified)
                elif avg_knee_y > avg_hip_y:
                    return "sitting"
                    
                # Check for lying down
                elif abs(avg_ankle_y - avg_hip_y) < 0.1:
                    return "lying_down"
            
            return "unknown_posture"
            
        except Exception as e:
            self.logger.error(f"Error classifying posture: {e}")
            return "unknown_posture"

    def _calculate_balance_score(self, landmarks: List[List[float]]) -> float:
        """Calculate a balance score based on body positioning"""
        try:
            if len(landmarks) < 29:
                return 0.5
                
            # Get center of mass approximation
            shoulder_center_x = (landmarks[11][0] + landmarks[12][0]) / 2
            hip_center_x = (landmarks[23][0] + landmarks[24][0]) / 2
            ankle_center_x = (landmarks[27][0] + landmarks[28][0]) / 2
            
            # Calculate alignment
            torso_alignment = abs(shoulder_center_x - hip_center_x)
            leg_alignment = abs(hip_center_x - ankle_center_x)
            
            # Balance score (lower values = better balance)
            balance_score = 1.0 - min(1.0, (torso_alignment + leg_alignment) * 2)
            
            return max(0.0, balance_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating balance score: {e}")
            return 0.5

    def analyze_body_movements(self, pose_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze body movements between frames"""
        if not pose_features.get('detected', False) or not self.previous_pose:
            return {
                'movement_detected': False,
                'movement_intensity': 0.0,
                'movement_direction': 'none',
                'active_body_parts': []
            }
        
        try:
            current_landmarks = pose_features['landmarks']
            movement_vectors = []
            active_parts = []
            
            # Calculate movement for key body parts
            key_points = {
                'head': [0, 1, 2, 3, 4],  # Nose and face landmarks
                'left_arm': [11, 13, 15],  # Shoulder, elbow, wrist
                'right_arm': [12, 14, 16],
                'torso': [11, 12, 23, 24],  # Shoulders and hips
                'left_leg': [23, 25, 27],  # Hip, knee, ankle
                'right_leg': [24, 26, 28]
            }
            
            for part_name, indices in key_points.items():
                part_movement = 0.0
                valid_points = 0
                
                for idx in indices:
                    if idx < len(current_landmarks) and idx < len(self.previous_pose):
                        # Calculate 2D movement (ignoring z and visibility)
                        dx = current_landmarks[idx][0] - self.previous_pose[idx][0]
                        dy = current_landmarks[idx][1] - self.previous_pose[idx][1]
                        movement = math.sqrt(dx*dx + dy*dy)
                        
                        part_movement += movement
                        valid_points += 1
                
                if valid_points > 0:
                    avg_movement = part_movement / valid_points
                    movement_vectors.append(avg_movement)
                    
                    if avg_movement > self.movement_threshold:
                        active_parts.append(part_name)
            
            # Calculate overall movement intensity
            total_movement = sum(movement_vectors)
            movement_intensity = min(1.0, total_movement / self.significant_movement_threshold)
            
            # Determine primary movement direction
            movement_direction = self._calculate_movement_direction(current_landmarks)
            
            # Store in pose history for trend analysis
            self.pose_history.append({
                'timestamp': cv2.getTickCount() / cv2.getTickFrequency(),
                'movement_intensity': movement_intensity,
                'active_parts': active_parts
            })
            
            if len(self.pose_history) > self.max_history_length:
                self.pose_history.pop(0)
            
            return {
                'movement_detected': movement_intensity > 0.1,
                'movement_intensity': movement_intensity,
                'movement_direction': movement_direction,
                'active_body_parts': active_parts,
                'movement_trend': self._analyze_movement_trend()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing body movements: {e}")
            return {
                'movement_detected': False,
                'movement_intensity': 0.0,
                'movement_direction': 'none',
                'active_body_parts': []
            }

    def _calculate_movement_direction(self, landmarks: List[List[float]]) -> str:
        """Calculate the primary direction of body movement"""
        if not self.previous_pose:
            return 'none'
            
        try:
            # Use center of mass for overall direction
            current_center = self._get_body_center(landmarks)
            previous_center = self._get_body_center(self.previous_pose)
            
            dx = current_center[0] - previous_center[0]
            dy = current_center[1] - previous_center[1]
            
            # Determine primary direction
            if abs(dx) > abs(dy):
                return 'right' if dx > 0 else 'left'
            else:
                return 'down' if dy > 0 else 'up'
                
        except Exception:
            return 'none'

    def _get_body_center(self, landmarks: List[List[float]]) -> List[float]:
        """Calculate the center of the body"""
        if len(landmarks) >= 24:
            # Use shoulders and hips as reference points
            center_x = (landmarks[11][0] + landmarks[12][0] + landmarks[23][0] + landmarks[24][0]) / 4
            center_y = (landmarks[11][1] + landmarks[12][1] + landmarks[23][1] + landmarks[24][1]) / 4
            return [center_x, center_y]
        return [0.5, 0.5]  # Default center

    def _analyze_movement_trend(self) -> str:
        """Analyze movement trend over recent frames"""
        if len(self.pose_history) < 3:
            return 'stable'
            
        recent_intensities = [frame['movement_intensity'] for frame in self.pose_history[-3:]]
        
        if all(i > 0.3 for i in recent_intensities):
            return 'highly_active'
        elif all(i > 0.1 for i in recent_intensities):
            return 'active'
        elif sum(recent_intensities) / len(recent_intensities) > 0.05:
            return 'moderate'
        else:
            return 'stable'

    # Helper methods
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        try:
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            return math.sqrt(dx*dx + dy*dy)
        except (IndexError, TypeError):
            return 0.0

    def _calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle formed by three points"""
        try:
            # Vector from point2 to point1
            v1 = [point1[0] - point2[0], point1[1] - point2[1]]
            # Vector from point2 to point3
            v2 = [point3[0] - point2[0], point3[1] - point2[1]]
            
            # Calculate angle using dot product
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
                
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
            
            return math.degrees(math.acos(cos_angle))
            
        except (ValueError, IndexError, TypeError):
            return 0.0

    def _calculate_body_lean(self, landmarks: List[List[float]]) -> Dict[str, float]:
        """Calculate body lean in different directions"""
        try:
            if len(landmarks) < 25:
                return {'forward': 0.0, 'left': 0.0}
                
            # Use shoulders and hips to determine lean
            shoulder_center = [(landmarks[11][0] + landmarks[12][0]) / 2,
                             (landmarks[11][1] + landmarks[12][1]) / 2]
            hip_center = [(landmarks[23][0] + landmarks[24][0]) / 2,
                         (landmarks[23][1] + landmarks[24][1]) / 2]
            
            # Calculate lean angles
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            
            # Left/right lean
            left_lean = dx * 100  # Scale for visibility
            
            # Forward/backward lean (simplified)
            forward_lean = 0.0  # Would need depth information for accurate calculation
            
            return {
                'forward': forward_lean,
                'left': left_lean
            }
            
        except Exception:
            return {'forward': 0.0, 'left': 0.0}

    def _calculate_body_symmetry(self, landmarks: List[List[float]]) -> float:
        """Calculate how symmetrical the body pose is"""
        try:
            if len(landmarks) < 29:
                return 0.5
                
            # Compare left and right limb positions
            symmetry_scores = []
            
            # Arm symmetry
            left_arm_height = landmarks[15][1]  # Left wrist
            right_arm_height = landmarks[16][1]  # Right wrist
            arm_symmetry = 1.0 - abs(left_arm_height - right_arm_height)
            symmetry_scores.append(max(0.0, arm_symmetry))
            
            # Leg symmetry
            left_leg_pos = landmarks[27][1]  # Left ankle
            right_leg_pos = landmarks[28][1]  # Right ankle
            leg_symmetry = 1.0 - abs(left_leg_pos - right_leg_pos)
            symmetry_scores.append(max(0.0, leg_symmetry))
            
            return sum(symmetry_scores) / len(symmetry_scores)
            
        except Exception:
            return 0.5

    def draw_pose_landmarks(self, frame: np.ndarray, pose_features: Dict[str, Any]) -> np.ndarray:
        """Draw pose landmarks on the frame"""
        if not self.pose_enabled or not pose_features.get('detected', False):
            return frame
            
        try:
            # Convert landmarks back to MediaPipe format for drawing
            landmarks = pose_features.get('landmarks', [])
            if not landmarks:
                return frame
                
            # Create a mock results object for drawing
            height, width = frame.shape[:2]
            
            for i, landmark in enumerate(landmarks):
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                visibility = landmark[3] if len(landmark) > 3 else 1.0
                
                if visibility > 0.5:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections (simplified)
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                (11, 23), (12, 24), (23, 24),  # Torso
                (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
            ]
            
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    if len(start_point) > 3 and len(end_point) > 3:
                        if start_point[3] > 0.5 and end_point[3] > 0.5:
                            start_pos = (int(start_point[0] * width), int(start_point[1] * height))
                            end_pos = (int(end_point[0] * width), int(end_point[1] * height))
                            cv2.line(frame, start_pos, end_pos, (0, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing pose landmarks: {e}")
            return frame

    def extract_face_features(self, face_results: Any) -> Dict[str, Any]:
        """Extract relevant facial features from detection results"""
        if not face_results or not face_results.multi_face_landmarks:
            return {'detected': False}

        # Take the first detected face
        face_landmarks = face_results.multi_face_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]

        return {
            'detected': True,
            'landmarks': landmarks,
            'expressions': self._analyze_expressions(landmarks),
            'head_pose': self._estimate_head_pose(landmarks)
        }

    def extract_hand_features(self, hand_results: Any) -> Dict[str, Any]:
        """Extract hand gesture features from detection results"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return {'detected': False, 'left_hand': None, 'right_hand': None}

        hands_data = {'detected': True, 'left_hand': None, 'right_hand': None}

        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            hand_label = hand_results.multi_handedness[i].classification[0].label
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            hand_data = {
                'landmarks': landmarks,
                'gesture': self._recognize_gesture(landmarks)
            }

            if hand_label == 'Left':
                hands_data['left_hand'] = hand_data
            else:
                hands_data['right_hand'] = hand_data

        return hands_data

    def _analyze_expressions(self, landmarks: list) -> Dict[str, float]:
        """Analyze facial expressions from landmarks"""
        # Simplified expression analysis
        # In a real implementation, you'd use more sophisticated methods

        # Eye aspect ratios
        left_eye_ratio = self._calculate_eye_aspect_ratio(landmarks, 'left')
        right_eye_ratio = self._calculate_eye_aspect_ratio(landmarks, 'right')

        # Mouth aspect ratio
        mouth_ratio = self._calculate_mouth_aspect_ratio(landmarks)

        # Eyebrow height
        eyebrow_height = self._calculate_eyebrow_height(landmarks)

        return {
            'eye_openness': (left_eye_ratio + right_eye_ratio) / 2,
            'mouth_openness': mouth_ratio,
            'eyebrow_raise': eyebrow_height,
            'smile_intensity': self._calculate_smile_intensity(landmarks)
        }

    def _calculate_eye_aspect_ratio(self, landmarks: list, eye: str) -> float:
        """Calculate eye aspect ratio for blink detection"""
        # Simplified EAR calculation
        # Real implementation would use proper landmark indices
        return 0.3  # Placeholder

    def _calculate_mouth_aspect_ratio(self, landmarks: list) -> float:
        """Calculate mouth aspect ratio"""
        # Simplified MAR calculation
        return 0.2  # Placeholder

    def _calculate_eyebrow_height(self, landmarks: list) -> float:
        """Calculate eyebrow height for expression detection"""
        return 0.5  # Placeholder

    def _calculate_smile_intensity(self, landmarks: list) -> float:
        """Calculate smile intensity"""
        return 0.0  # Placeholder

    def _estimate_head_pose(self, landmarks: list) -> Dict[str, float]:
        """Estimate head pose angles"""
        return {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        }

    def _recognize_gesture(self, landmarks: list) -> str:
        """Recognize hand gesture from landmarks"""
        # Simplified gesture recognition
        # Real implementation would analyze finger positions
        return "neutral"

    def _apply_smoothing(self, motion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to reduce jitter"""
        self.motion_history.append(motion_data)

        if len(self.motion_history) > self.max_history_length:
            self.motion_history.pop(0)

        # Simple moving average for numeric values
        # In practice, you'd implement more sophisticated smoothing
        return motion_data

    def _get_empty_motion_data(self) -> Dict[str, Any]:
        """Return empty motion data structure"""
        return {
            'face': {'detected': False},
            'hands': {'detected': False, 'left_hand': None, 'right_hand': None},
            'pose': {'detected': False},
            'body_movements': {
                'movement_detected': False,
                'movement_intensity': 0.0,
                'movement_direction': 'none',
                'active_body_parts': []
            },
            'timestamp': 0.0
        }