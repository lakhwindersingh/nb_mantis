import pygame
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
from src.character.sprite_manager import SpriteManager
from src.character.animation_engine import AnimationEngine
from src.character.rigging import RiggingSystem

class CharacterAnimator:
    """ Enhanced character animation agent that maps motion data including pose and body movements """

    def __init__(self, character_config: Dict[str, Any], assets_path: Path):
        self.logger = logging.getLogger(__name__)
        self.config = character_config
        self.assets_path = assets_path

        # Initialize character systems
        self.sprite_manager = SpriteManager(assets_path / "characters")
        self.animation_engine = AnimationEngine(character_config)
        self.rigging_system = RiggingSystem(character_config)

        # Enhanced animation state with pose support
        self.current_state = {
            'expression': 'neutral',
            'hand_pose': 'rest',
            'body_pose': 'idle',
            'pose_data': {},
            'movement_state': 'static',
            'body_orientation': 'front'
        }

        # Animation history for smoothing
        self.animation_history = []
        self.max_history_length = 5
        
        # Body part tracking
        self.body_parts = {
            'head': {'rotation': 0, 'tilt': 0},
            'torso': {'rotation': 0, 'lean': 0},
            'left_arm': {'shoulder_angle': 0, 'elbow_angle': 0},
            'right_arm': {'shoulder_angle': 0, 'elbow_angle': 0},
            'left_leg': {'hip_angle': 0, 'knee_angle': 0},
            'right_leg': {'hip_angle': 0, 'knee_angle': 0}
        }
        
        # Movement animation parameters
        self.movement_config = {
            'bounce_intensity': character_config.get('movement_bounce', 0.3),
            'sway_sensitivity': character_config.get('sway_sensitivity', 0.8),
            'pose_transition_speed': character_config.get('pose_transition_speed', 0.15),
            'balance_correction': character_config.get('balance_correction', True)
        }

        # Initialize pygame for rendering
        pygame.init()
        self.screen_size = character_config.get('sprite_resolution', (512, 512))

        self.logger.info("Enhanced character animator initialized with pose support")

    def animate_character(self, motion_data: Dict[str, Any]) -> pygame.Surface:
        """
        Generate enhanced character animation frame from motion data including pose

        Args:
            motion_data: Motion detection results with pose and movement data

        Returns:
            Rendered character frame as pygame Surface
        """
        # Map all motion types to character animations
        face_animation = self.map_face_to_character(motion_data.get('face', {}))
        hand_animation = self.map_hands_to_character(motion_data.get('hands', {}))
        pose_animation = self.map_pose_to_character(motion_data.get('pose', {}))
        movement_animation = self.map_movements_to_character(motion_data.get('body_movements', {}))

        # Update comprehensive animation state
        self._update_enhanced_animation_state(
            face_animation, hand_animation, pose_animation, movement_animation
        )

        # Generate animation frame with pose consideration
        character_frame = self.animation_engine.render_enhanced_frame(
            self.current_state,
            self.body_parts,
            self.sprite_manager
        )

        return character_frame

    def map_pose_to_character(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map pose detection data to character body positioning and posture

        Args:
            pose_data: Pose detection results from motion detector

        Returns:
            Character pose animation parameters
        """
        if not pose_data.get('detected', False):
            return {
                'posture': 'idle',
                'body_angles': {},
                'stance': 'neutral',
                'balance': 1.0
            }

        # Extract pose information
        posture_classification = pose_data.get('posture_classification', 'standing_normal')
        body_angles = pose_data.get('body_angles', {})
        balance_score = pose_data.get('balance_score', 1.0)
        pose_analysis = pose_data.get('pose_analysis', {})

        # Map posture to character pose
        character_posture = self._map_posture_classification(posture_classification)
        
        # Map body angles to character limb positioning
        limb_positioning = self._map_body_angles_to_limbs(body_angles)
        
        # Determine character stance and orientation
        stance_info = self._analyze_character_stance(pose_analysis, balance_score)

        return {
            'posture': character_posture,
            'limb_positioning': limb_positioning,
            'stance': stance_info['stance'],
            'balance': balance_score,
            'body_lean': self._calculate_character_lean(pose_analysis),
            'symmetry': pose_analysis.get('symmetry_score', 1.0)
        }

    def map_movements_to_character(self, movement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map body movement analysis to character motion and animation effects

        Args:
            movement_data: Body movement analysis from motion detector

        Returns:
            Character movement animation parameters
        """
        if not movement_data.get('movement_detected', False):
            return {
                'movement_type': 'static',
                'intensity': 0.0,
                'active_parts': [],
                'animation_effects': []
            }

        movement_intensity = movement_data.get('movement_intensity', 0.0)
        movement_direction = movement_data.get('movement_direction', 'none')
        active_body_parts = movement_data.get('active_body_parts', [])
        movement_trend = movement_data.get('movement_trend', 'stable')

        # Map movement characteristics to animation
        character_movement = self._classify_character_movement(
            movement_intensity, movement_direction, active_body_parts, movement_trend
        )
        
        # Generate animation effects based on movement
        animation_effects = self._generate_movement_effects(
            character_movement, movement_intensity
        )

        return {
            'movement_type': character_movement,
            'intensity': movement_intensity,
            'direction': movement_direction,
            'active_parts': active_body_parts,
            'animation_effects': animation_effects,
            'trend': movement_trend
        }

    def _map_posture_classification(self, posture: str) -> str:
        """Map detected posture to character animation state"""
        posture_mapping = {
            'standing_normal': 'idle_standing',
            'standing_arms_raised': 'celebrating',
            'standing_one_arm_raised': 'waving',
            'standing_wide_stance': 'confident_stance',
            'sitting': 'sitting_idle',
            'lying_down': 'lying',
            'unknown_posture': 'idle_standing'
        }
        
        return posture_mapping.get(posture, 'idle_standing')

    def _map_body_angles_to_limbs(self, body_angles: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Convert body joint angles to character limb positions"""
        limb_positions = {}
        
        # Map arm angles
        if 'left_arm_angle' in body_angles:
            limb_positions['left_arm'] = {
                'shoulder_rotation': self._normalize_angle(body_angles['left_arm_angle']),
                'elbow_bend': max(0, 180 - body_angles['left_arm_angle']) / 180.0
            }
            
        if 'right_arm_angle' in body_angles:
            limb_positions['right_arm'] = {
                'shoulder_rotation': self._normalize_angle(body_angles['right_arm_angle']),
                'elbow_bend': max(0, 180 - body_angles['right_arm_angle']) / 180.0
            }
        
        # Map leg angles
        if 'left_leg_angle' in body_angles:
            limb_positions['left_leg'] = {
                'hip_rotation': self._normalize_angle(body_angles['left_leg_angle']),
                'knee_bend': max(0, 180 - body_angles['left_leg_angle']) / 180.0
            }
            
        if 'right_leg_angle' in body_angles:
            limb_positions['right_leg'] = {
                'hip_rotation': self._normalize_angle(body_angles['right_leg_angle']),
                'knee_bend': max(0, 180 - body_angles['right_leg_angle']) / 180.0
            }
        
        # Map torso angle
        if 'torso_angle' in body_angles:
            limb_positions['torso'] = {
                'rotation': self._normalize_angle(body_angles['torso_angle']),
                'tilt': abs(body_angles['torso_angle']) / 90.0  # Normalize to 0-1
            }
            
        return limb_positions

    def _analyze_character_stance(self, pose_analysis: Dict[str, Any], balance_score: float) -> Dict[str, Any]:
        """Analyze pose data to determine character stance characteristics"""
        
        # Determine stability
        if balance_score > 0.8:
            stability = 'stable'
        elif balance_score > 0.5:
            stability = 'slightly_unstable'
        else:
            stability = 'unstable'
            
        # Analyze leg positioning
        legs_apart = pose_analysis.get('legs_apart', False)
        if legs_apart:
            stance_width = 'wide'
        else:
            stance_width = 'normal'
            
        # Analyze arm positions
        both_arms_raised = pose_analysis.get('both_arms_raised', False)
        left_arm_raised = pose_analysis.get('left_arm_raised', False)
        right_arm_raised = pose_analysis.get('right_arm_raised', False)
        
        if both_arms_raised:
            arm_position = 'both_raised'
        elif left_arm_raised and right_arm_raised:
            arm_position = 'asymmetric'
        elif left_arm_raised:
            arm_position = 'left_raised'
        elif right_arm_raised:
            arm_position = 'right_raised'
        else:
            arm_position = 'lowered'
            
        return {
            'stance': f"{stance_width}_{stability}",
            'stability': stability,
            'stance_width': stance_width,
            'arm_position': arm_position
        }

    def _calculate_character_lean(self, pose_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate character body lean for animation"""
        body_lean = pose_analysis.get('body_lean', {'forward': 0.0, 'left': 0.0})
        
        # Normalize and clamp lean values for character animation
        forward_lean = max(-1.0, min(1.0, body_lean['forward'] / 30.0))  # Scale down
        side_lean = max(-1.0, min(1.0, body_lean['left'] / 30.0))
        
        return {
            'forward': forward_lean,
            'side': side_lean,
            'magnitude': math.sqrt(forward_lean**2 + side_lean**2)
        }

    def _classify_character_movement(self, intensity: float, direction: str, 
                                   active_parts: List[str], trend: str) -> str:
        """Classify the type of character movement for animation"""
        
        # High intensity movements
        if intensity > 0.7:
            if 'left_arm' in active_parts and 'right_arm' in active_parts:
                return 'energetic_gesture'
            elif len(active_parts) >= 3:
                return 'full_body_movement'
            else:
                return 'vigorous_motion'
                
        # Medium intensity movements
        elif intensity > 0.3:
            if trend == 'highly_active':
                return 'continuous_motion'
            elif 'head' in active_parts:
                return 'expressive_movement'
            elif any('arm' in part for part in active_parts):
                return 'gesture_movement'
            else:
                return 'moderate_motion'
                
        # Low intensity movements
        elif intensity > 0.1:
            if trend == 'active':
                return 'subtle_animation'
            else:
                return 'slight_movement'
        else:
            return 'static'

    def _generate_movement_effects(self, movement_type: str, intensity: float) -> List[Dict[str, Any]]:
        """Generate visual effects and secondary animations based on movement"""
        effects = []
        
        # Bounce effects for energetic movements
        if movement_type in ['energetic_gesture', 'full_body_movement', 'vigorous_motion']:
            effects.append({
                'type': 'bounce',
                'intensity': intensity * self.movement_config['bounce_intensity'],
                'frequency': 2.0 + intensity * 2.0
            })
            
        # Sway effects for continuous motion
        if movement_type == 'continuous_motion':
            effects.append({
                'type': 'sway',
                'intensity': intensity * self.movement_config['sway_sensitivity'],
                'direction': 'horizontal'
            })
            
        # Breathing/idle animation adjustments
        if movement_type == 'static':
            effects.append({
                'type': 'idle_breathing',
                'intensity': 0.5,
                'frequency': 0.3
            })
        else:
            # Reduce idle breathing during movement
            effects.append({
                'type': 'idle_breathing',
                'intensity': max(0.1, 0.5 - intensity * 0.4),
                'frequency': 0.3
            })
            
        # Secondary motion effects
        if intensity > 0.5:
            effects.append({
                'type': 'motion_blur',
                'intensity': (intensity - 0.5) * 2.0,
                'duration': 3  # frames
            })
            
        return effects

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to -1.0 to 1.0 range for character animation"""
        # Assuming input angles are in degrees
        normalized = angle / 180.0  # Convert to -1 to 1 range
        return max(-1.0, min(1.0, normalized))

    def _update_enhanced_animation_state(self, face_animation: Dict[str, Any],
                                       hand_animation: Dict[str, Any],
                                       pose_animation: Dict[str, Any],
                                       movement_animation: Dict[str, Any]):
        """Update comprehensive animation state with pose and movement data"""
        
        # Smooth transition to new state
        transition_speed = self.movement_config['pose_transition_speed']
        
        # Update body parts with pose data
        if 'limb_positioning' in pose_animation:
            limb_data = pose_animation['limb_positioning']
            
            for limb_name, limb_angles in limb_data.items():
                if limb_name in self.body_parts:
                    for angle_name, target_value in limb_angles.items():
                        if angle_name in self.body_parts[limb_name]:
                            current_value = self.body_parts[limb_name][angle_name]
                            # Smooth interpolation
                            new_value = current_value + (target_value - current_value) * transition_speed
                            self.body_parts[limb_name][angle_name] = new_value

        # Update main animation state
        self.current_state.update({
            'face': face_animation,
            'hands': hand_animation,
            'pose': pose_animation,
            'movement': movement_animation,
            'body_pose': pose_animation.get('posture', 'idle'),
            'movement_state': movement_animation.get('movement_type', 'static'),
            'balance_score': pose_animation.get('balance', 1.0)
        })
        
        # Add to animation history for advanced smoothing
        self.animation_history.append({
            'timestamp': pygame.time.get_ticks(),
            'state': self.current_state.copy(),
            'body_parts': self.body_parts.copy()
        })
        
        if len(self.animation_history) > self.max_history_length:
            self.animation_history.pop(0)

    def get_character_pose_description(self) -> str:
        """Get a text description of the current character pose"""
        current_pose = self.current_state.get('pose', {})
        current_movement = self.current_state.get('movement', {})
        
        posture = current_pose.get('posture', 'idle')
        movement_type = current_movement.get('movement_type', 'static')
        
        if movement_type != 'static':
            return f"{posture} with {movement_type}"
        else:
            return posture

    def apply_physics_corrections(self) -> None:
        """Apply physics-based corrections to maintain realistic poses"""
        if not self.movement_config['balance_correction']:
            return
            
        # Balance correction based on pose
        balance_score = self.current_state.get('balance_score', 1.0)
        
        if balance_score < 0.5:  # Unstable pose
            # Add compensatory lean to torso
            if 'torso' in self.body_parts:
                current_lean = self.body_parts['torso'].get('lean', 0)
                correction = (0.5 - balance_score) * 0.3  # Gentle correction
                self.body_parts['torso']['lean'] = current_lean + correction
                
        # Ensure arm positions are physically plausible
        for arm in ['left_arm', 'right_arm']:
            if arm in self.body_parts:
                shoulder_angle = self.body_parts[arm].get('shoulder_angle', 0)
                elbow_angle = self.body_parts[arm].get('elbow_angle', 0)
                
                # Limit extreme shoulder positions
                if abs(shoulder_angle) > 0.9:  # Very raised arms
                    self.body_parts[arm]['shoulder_angle'] = 0.9 * (1 if shoulder_angle > 0 else -1)

    def create_pose_keyframes(self, target_pose: Dict[str, Any], duration: int) -> List[Dict[str, Any]]:
        """Create smooth keyframe animation between current and target pose"""
        keyframes = []
        current_state = self.current_state.copy()
        
        for frame in range(duration):
            progress = frame / (duration - 1) if duration > 1 else 1.0
            # Use easing function for smooth animation
            eased_progress = self._ease_in_out(progress)
            
            # Interpolate between current and target state
            interpolated_state = self._interpolate_poses(current_state, target_pose, eased_progress)
            keyframes.append(interpolated_state)
            
        return keyframes

    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function for natural animation transitions"""
        return t * t * (3.0 - 2.0 * t)

    def _interpolate_poses(self, pose_a: Dict[str, Any], pose_b: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Interpolate between two pose states"""
        # This would implement smooth interpolation between poses
        # For now, return a simple linear blend
        return pose_a  # Placeholder - would implement full interpolation

    # Keep existing mapping methods with enhancements
    def map_face_to_character(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced facial mapping with pose awareness"""
        if not face_data.get('detected', False):
            return {'expression': 'neutral', 'intensity': 0.0}

        expressions = face_data.get('expressions', {})
        head_pose = face_data.get('head_pose', {})

        # Enhanced mapping with head pose consideration
        animation_data = {
            'eye_state': self._map_eye_state(expressions),
            'mouth_state': self._map_mouth_state(expressions),
            'eyebrow_state': self._map_eyebrow_state(expressions),
            'head_rotation': self._map_head_pose(head_pose),
            'expression_intensity': self._calculate_expression_intensity(expressions)
        }

        return animation_data

    def map_hands_to_character(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced hand mapping with pose context"""
        if not hand_data.get('detected', False):
            return {'left_hand': 'rest', 'right_hand': 'rest'}

        animation_data = {
            'left_hand': self._map_hand_pose(hand_data.get('left_hand')),
            'right_hand': self._map_hand_pose(hand_data.get('right_hand')),
            'hand_coordination': self._analyze_hand_coordination(hand_data)
        }

        return animation_data

    def _calculate_expression_intensity(self, expressions: Dict[str, float]) -> float:
        """Calculate overall expression intensity for dynamic animation scaling"""
        eye_intensity = abs(expressions.get('eye_openness', 0.3) - 0.3) * 2
        mouth_intensity = expressions.get('mouth_openness', 0.0) + expressions.get('smile_intensity', 0.0)
        eyebrow_intensity = abs(expressions.get('eyebrow_raise', 0.5) - 0.5) * 2
        
        return min(1.0, (eye_intensity + mouth_intensity + eyebrow_intensity) / 3.0)

    def _analyze_hand_coordination(self, hand_data: Dict[str, Any]) -> str:
        """Analyze coordination between hands for better animation"""
        left_hand = hand_data.get('left_hand')
        right_hand = hand_data.get('right_hand')
        
        if not left_hand or not right_hand:
            return 'single_hand'
            
        left_gesture = left_hand.get('gesture', 'neutral')
        right_gesture = right_hand.get('gesture', 'neutral')
        
        if left_gesture == right_gesture:
            return 'synchronized'
        elif left_gesture != 'neutral' and right_gesture != 'neutral':
            return 'complementary'
        else:
            return 'independent'

    # Keep existing helper methods
    def _map_eye_state(self, expressions: Dict[str, float]) -> str:
        """Map eye expressions to character eye states"""
        eye_openness = expressions.get('eye_openness', 0.3)

        if eye_openness < 0.1:
            return 'closed'
        elif eye_openness < 0.2:
            return 'squinting'
        elif eye_openness > 0.4:
            return 'wide'
        else:
            return 'normal'

    def _map_mouth_state(self, expressions: Dict[str, float]) -> str:
        """Map mouth expressions to character mouth states"""
        mouth_openness = expressions.get('mouth_openness', 0.2)
        smile_intensity = expressions.get('smile_intensity', 0.0)

        if smile_intensity > 0.3:
            return 'smile'
        elif mouth_openness > 0.4:
            return 'open'
        elif mouth_openness > 0.2:
            return 'speaking'
        else:
            return 'closed'

    def _map_eyebrow_state(self, expressions: Dict[str, float]) -> str:
        """Map eyebrow expressions to character eyebrow states"""
        eyebrow_height = expressions.get('eyebrow_raise', 0.5)

        if eyebrow_height > 0.7:
            return 'raised'
        elif eyebrow_height < 0.3:
            return 'furrowed'
        else:
            return 'normal'

    def _map_head_pose(self, head_pose: Dict[str, float]) -> Dict[str, float]:
        """Enhanced head pose mapping"""
        return {
            'rotation_x': head_pose.get('pitch', 0.0) * 0.5,
            'rotation_y': head_pose.get('yaw', 0.0) * 0.5,
            'rotation_z': head_pose.get('roll', 0.0) * 0.3
        }

    def _map_hand_pose(self, hand_data: Optional[Dict[str, Any]]) -> str:
        """Enhanced hand gesture mapping"""
        if not hand_data:
            return 'rest'

        gesture = hand_data.get('gesture', 'neutral')

        # Enhanced gesture mapping
        gesture_mapping = {
            'neutral': 'rest',
            'fist': 'closed',
            'open': 'open',
            'point': 'pointing',
            'thumbs_up': 'thumbs_up',
            'peace': 'peace',
            'ok': 'ok_sign',
            'wave': 'waving'
        }

        return gesture_mapping.get(gesture, 'rest')

    def get_character_bounds(self) -> Tuple[int, int, int, int]:
        """Get character bounding box for compositing with pose consideration"""
        bounds = self.rigging_system.get_bounds()
        
        # Adjust bounds based on current pose
        current_pose = self.current_state.get('pose', {})
        body_lean = current_pose.get('body_lean', {})
        
        # Expand bounds if character is leaning significantly
        lean_magnitude = body_lean.get('magnitude', 0.0)
        if lean_magnitude > 0.3:
            expansion = int(lean_magnitude * 50)  # Pixels to expand
            return (bounds[0] - expansion, bounds[1] - expansion, 
                   bounds[2] + expansion, bounds[3] + expansion)
            
        return bounds

    def get_animation_metadata(self) -> Dict[str, Any]:
        """Get metadata about current animation state for debugging/analysis"""
        return {
            'current_pose': self.get_character_pose_description(),
            'body_parts_state': self.body_parts.copy(),
            'animation_effects': self.current_state.get('movement', {}).get('animation_effects', []),
            'balance_score': self.current_state.get('balance_score', 1.0),
            'movement_intensity': self.current_state.get('movement', {}).get('intensity', 0.0),
            'pose_history_length': len(self.animation_history)
        }