"""
Vector-based bone animation system
"""
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import math

class VectorBone:
    """Represents a single bone in the vector skeleton"""
    
    def __init__(self, name: str, position: Tuple[float, float], 
                 parent: Optional['VectorBone'] = None):
        self.name = name
        self.local_position = np.array(position, dtype=float)
        self.parent = parent
        self.children: List['VectorBone'] = []
        
        # Transform properties
        self.rotation = 0.0  # radians
        self.scale = np.array([1.0, 1.0])
        self.translation = np.array([0.0, 0.0])
        
        # Bind to parent
        if parent:
            parent.add_child(self)
    
    def add_child(self, child: 'VectorBone'):
        """Add child bone"""
        self.children.append(child)
        child.parent = self
    
    def get_world_position(self) -> np.ndarray:
        """Get world space position"""
        if self.parent is None:
            return self.local_position + self.translation
        
        parent_world = self.parent.get_world_position()
        parent_rotation = self.parent.get_world_rotation()
        
        # Apply parent transforms
        rotated_pos = self._rotate_point(self.local_position + self.translation, 
                                        parent_rotation)
        return parent_world + rotated_pos
    
    def get_world_rotation(self) -> float:
        """Get world space rotation"""
        if self.parent is None:
            return self.rotation
        return self.parent.get_world_rotation() + self.rotation
    
    def get_world_scale(self) -> np.ndarray:
        """Get world space scale"""
        if self.parent is None:
            return self.scale
        return self.parent.get_world_scale() * self.scale
    
    def _rotate_point(self, point: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a point around origin"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotation_matrix @ point

class VectorBoneSystem:
    """Manages the complete bone hierarchy for vector character animation"""
    
    def __init__(self, skeleton_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.bones: Dict[str, VectorBone] = {}
        self.root_bone: Optional[VectorBone] = None
        
        # Bone influence mapping for SVG elements
        self.element_influences: Dict[str, List[Tuple[str, float]]] = {}
        
        # Initialize skeleton
        if skeleton_config:
            self.load_skeleton(skeleton_config)
        else:
            self._create_default_skeleton()
        
        self.logger.info(f"Vector bone system initialized with {len(self.bones)} bones")
    
    def load_skeleton(self, config: Dict[str, Any]):
        """Load skeleton from configuration"""
        try:
            # Clear existing bones
            self.bones.clear()
            self.root_bone = None
            
            # Create bones from config
            bone_configs = config.get('bones', {})
            
            # First pass: create all bones
            for bone_name, bone_data in bone_configs.items():
                position = bone_data.get('position', [0, 0])
                bone = VectorBone(bone_name, position)
                self.bones[bone_name] = bone
                
                if bone_data.get('is_root', False):
                    self.root_bone = bone
            
            # Second pass: establish hierarchy
            for bone_name, bone_data in bone_configs.items():
                parent_name = bone_data.get('parent')
                if parent_name and parent_name in self.bones:
                    self.bones[bone_name].parent = self.bones[parent_name]
                    self.bones[parent_name].add_child(self.bones[bone_name])
            
            # Load element influences
            self.element_influences = config.get('influences', {})
            
        except Exception as e:
            self.logger.error(f"Failed to load skeleton: {e}")
            self._create_default_skeleton()
    
    def _create_default_skeleton(self):
        """Create a basic default skeleton"""
        # Create basic humanoid skeleton
        self.root_bone = VectorBone('root', (0, 0))
        spine = VectorBone('spine', (0, -50), self.root_bone)
        neck = VectorBone('neck', (0, -30), spine)
        head = VectorBone('head', (0, -25), neck)
        
        left_shoulder = VectorBone('left_shoulder', (-20, -25), spine)
        left_arm = VectorBone('left_arm', (-25, 0), left_shoulder)
        left_hand = VectorBone('left_hand', (-20, 0), left_arm)
        
        right_shoulder = VectorBone('right_shoulder', (20, -25), spine)
        right_arm = VectorBone('right_arm', (25, 0), right_shoulder)
        right_hand = VectorBone('right_hand', (20, 0), right_arm)
        
        # Store in dictionary
        self.bones = {
            'root': self.root_bone,
            'spine': spine,
            'neck': neck,
            'head': head,
            'left_shoulder': left_shoulder,
            'left_arm': left_arm,
            'left_hand': left_hand,
            'right_shoulder': right_shoulder,
            'right_arm': right_arm,
            'right_hand': right_hand
        }
        
        # Default influences
        self.element_influences = {
            'head_group': [('head', 1.0), ('neck', 0.3)],
            'left_hand_group': [('left_hand', 1.0), ('left_arm', 0.5)],
            'right_hand_group': [('right_hand', 1.0), ('right_arm', 0.5)],
            'body_group': [('spine', 1.0), ('root', 0.2)]
        }
    
    def update_bone_transforms(self, motion_data: Dict[str, Any]):
        """Update bone transforms based on motion data"""
        try:
            # Update head bone from face data
            face_data = motion_data.get('face', {})
            if face_data.get('detected') and 'head' in self.bones:
                head_pose = face_data.get('head_pose', {})
                
                self.bones['head'].rotation = math.radians(head_pose.get('roll', 0) * 0.5)
                self.bones['neck'].rotation = math.radians(head_pose.get('pitch', 0) * 0.3)
            
            # Update hand bones from hand data
            hands_data = motion_data.get('hands', {})
            if hands_data.get('detected'):
                self._update_hand_bones(hands_data)
            
            # Apply expression-based poses
            self._apply_expression_poses(motion_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update bone transforms: {e}")
    
    def _update_hand_bones(self, hands_data: Dict[str, Any]):
        """Update hand bone positions from hand tracking data"""
        for side in ['left', 'right']:
            hand_data = hands_data.get(f'{side}_hand')
            if not hand_data:
                continue
            
            hand_bone_name = f'{side}_hand'
            arm_bone_name = f'{side}_arm'
            
            if hand_bone_name in self.bones:
                # Map hand gesture to bone rotation
                gesture = hand_data.get('gesture', 'neutral')
                rotation = self._gesture_to_rotation(gesture)
                self.bones[hand_bone_name].rotation = rotation
                
                # Adjust arm position based on hand landmarks
                landmarks = hand_data.get('landmarks', [])
                if landmarks:
                    # Use wrist position (landmark 0) to adjust arm
                    wrist_pos = landmarks[0][:2] if len(landmarks) > 0 else [0, 0]
                    
                    # Map normalized coordinates to bone space
                    bone_offset = np.array(wrist_pos) * 50  # Scale factor
                    self.bones[hand_bone_name].translation = bone_offset
    
    def _gesture_to_rotation(self, gesture: str) -> float:
        """Convert hand gesture to bone rotation"""
        gesture_rotations = {
            'fist': math.radians(15),
            'open': math.radians(-10),
            'point': math.radians(5),
            'thumbs_up': math.radians(20),
            'peace': math.radians(-5),
            'neutral': 0.0
        }
        return gesture_rotations.get(gesture, 0.0)
    
    def _apply_expression_poses(self, motion_data: Dict[str, Any]):
        """Apply subtle poses based on facial expressions"""
        face_data = motion_data.get('face', {})
        if not face_data.get('detected'):
            return
        
        expressions = face_data.get('expressions', {})
        
        # Surprise: slight lean back
        if expressions.get('eyebrow_raise', 0) > 0.7:
            if 'spine' in self.bones:
                self.bones['spine'].rotation = math.radians(-5)
        
        # Happiness: slight forward lean
        elif expressions.get('smile_intensity', 0) > 0.5:
            if 'spine' in self.bones:
                self.bones['spine'].rotation = math.radians(2)
    
    def get_element_transforms(self, element_id: str) -> Dict[str, Any]:
        """Get SVG transforms for a specific element"""
        if element_id not in self.element_influences:
            return {}
        
        influences = self.element_influences[element_id]
        
        # Calculate weighted average of bone transforms
        total_weight = sum(weight for _, weight in influences)
        if total_weight == 0:
            return {}
        
        combined_translation = np.array([0.0, 0.0])
        combined_rotation = 0.0
        combined_scale = np.array([0.0, 0.0])
        
        for bone_name, weight in influences:
            if bone_name not in self.bones:
                continue
            
            bone = self.bones[bone_name]
            world_pos = bone.get_world_position()
            world_rot = bone.get_world_rotation()
            world_scale = bone.get_world_scale()
            
            normalized_weight = weight / total_weight
            
            combined_translation += world_pos * normalized_weight
            combined_rotation += world_rot * normalized_weight
            combined_scale += world_scale * normalized_weight
        
        return {
            'translate': combined_translation.tolist(),
            'rotate': math.degrees(combined_rotation),
            'scale': combined_scale.tolist()
        }
    
    def get_all_transforms(self) -> Dict[str, Dict[str, Any]]:
        """Get transforms for all influenced elements"""
        transforms = {}
        
        for element_id in self.element_influences:
            transforms[element_id] = self.get_element_transforms(element_id)
        
        return transforms
    
    def reset_bones(self):
        """Reset all bones to default pose"""
        for bone in self.bones.values():
            bone.rotation = 0.0
            bone.scale = np.array([1.0, 1.0])
            bone.translation = np.array([0.0, 0.0])
