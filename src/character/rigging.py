""" Character rigging system for advanced animations """
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

class RiggingSystem:
    """Advanced rigging system for character animation"""
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Define character rig structure
        self.bone_hierarchy = {
            'root': {
                'position': (0, 0),
                'children': ['spine']
            },
            'spine': {
                'position': (0, -50),
                'children': ['neck', 'left_shoulder', 'right_shoulder']
            },
            'neck': {
                'position': (0, -80),
                'children': ['head']
            },
            'head': {
                'position': (0, -100),
                'children': []
            },
            'left_shoulder': {
                'position': (-30, -70),
                'children': ['left_arm']
            },
            'left_arm': {
                'position': (-50, -50),
                'children': ['left_hand']
            },
            'left_hand': {
                'position': (-70, -30),
                'children': []
            },
            'right_shoulder': {
                'position': (30, -70),
                'children': ['right_arm']
            },
            'right_arm': {
                'position': (50, -50),
                'children': ['right_hand']
            },
            'right_hand': {
                'position': (70, -30),
                'children': []
            }
        }

        # Current bone transforms
        self.bone_transforms = {}
        self._initialize_transforms()

    def _initialize_transforms(self):
        """Initialize bone transforms to default positions"""
        for bone_name, bone_data in self.bone_hierarchy.items():
            self.bone_transforms[bone_name] = {
                'position': bone_data['position'],
                'rotation': 0.0,
                'scale': 1.0
            }

    def update_rig(self, motion_data: Dict[str, Any]):
        """Update rig based on motion data"""
        # Update head position and rotation
        face_data = motion_data.get('face', {})
        if face_data.get('detected'):
            head_pose = face_data.get('head_pose', {})
            self.bone_transforms['head']['rotation'] = head_pose.get('roll', 0.0)

            # Adjust neck based on head pose
            self.bone_transforms['neck']['rotation'] = head_pose.get('pitch', 0.0) * 0.5

        # Update hand positions
        hand_data = motion_data.get('hands', {})
        if hand_data.get('detected'):
            if hand_data.get('left_hand'):
                self._update_hand_rig('left_hand', hand_data['left_hand'])
            if hand_data.get('right_hand'):
                self._update_hand_rig('right_hand', hand_data['right_hand'])

    def _update_hand_rig(self, hand_name: str, hand_data: Dict[str, Any]):
        """Update hand rig based on hand data"""
        gesture = hand_data.get('gesture', 'neutral')

        # Map gestures to bone rotations
        gesture_mappings = {
            'fist': {'rotation': 45.0, 'scale': 0.9},
            'open': {'rotation': 0.0, 'scale': 1.1},
            'point': {'rotation': -15.0, 'scale': 1.0},
            'neutral': {'rotation': 0.0, 'scale': 1.0}
        }

        if gesture in gesture_mappings:
            mapping = gesture_mappings[gesture]
            self.bone_transforms[hand_name].update(mapping)

    def get_bone_world_position(self, bone_name: str) -> Tuple[float, float]:
        """Get world position of a bone"""
        position = np.array(self.bone_transforms[bone_name]['position'])

        # Traverse up the hierarchy to accumulate transforms
        parent_bone = self._get_parent_bone(bone_name)
        while parent_bone:
            parent_pos = np.array(self.bone_transforms[parent_bone]['position'])
            position += parent_pos
            parent_bone = self._get_parent_bone(parent_bone)

        return tuple(position)

    def _get_parent_bone(self, bone_name: str) -> Optional[str]:
        """Find parent bone of given bone"""
        for parent, data in self.bone_hierarchy.items():
            if bone_name in data.get('children', []):
                return parent
        return None

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get character bounding box"""
        # Calculate based on all bone positions
        all_positions = [self.get_bone_world_position(bone)
                        for bone in self.bone_hierarchy.keys()]

        if not all_positions:
            return (0, 0, 100, 100)

        xs = [pos[0] for pos in all_positions]
        ys = [pos[1] for pos in all_positions]

        min_x, max_x = min(xs) - 50, max(xs) + 50
        min_y, max_y = min(ys) - 50, max(ys) + 50

        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))