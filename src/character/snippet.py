"""
Advanced 3D Animation Engine with Facial Expression Mapping
Focused on MediaPipe Face Landmark Integration

Requirements:
pip install pyglet moderngl pyrr pygltflib numpy scipy trimesh
"""
import math
import logging
import struct
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import pyglet
from pyglet.gl import *
import moderngl
from pyrr import Matrix44, Vector3, Quaternion
from pygltflib import GLTF2
import trimesh
from scipy.spatial.distance import euclidean

class FacialBlendShapes:
    """
    Standard facial blend shape names for 3D character animation
    Based on ARKit and industry standards
    """
    
    # Eye controls
    EYE_BLINK_LEFT = "eyeBlinkLeft"
    EYE_BLINK_RIGHT = "eyeBlinkRight"
    EYE_LOOK_UP_LEFT = "eyeLookUpLeft"
    EYE_LOOK_UP_RIGHT = "eyeLookUpRight"
    EYE_LOOK_DOWN_LEFT = "eyeLookDownLeft"
    EYE_LOOK_DOWN_RIGHT = "eyeLookDownRight"
    EYE_LOOK_IN_LEFT = "eyeLookInLeft"
    EYE_LOOK_IN_RIGHT = "eyeLookInRight"
    EYE_LOOK_OUT_LEFT = "eyeLookOutLeft"
    EYE_LOOK_OUT_RIGHT = "eyeLookOutRight"
    EYE_SQUINT_LEFT = "eyeSquintLeft"
    EYE_SQUINT_RIGHT = "eyeSquintRight"
    EYE_WIDE_LEFT = "eyeWideLeft"
    EYE_WIDE_RIGHT = "eyeWideRight"
    
    # Eyebrow controls
    BROW_DOWN_LEFT = "browDownLeft"
    BROW_DOWN_RIGHT = "browDownRight"
    BROW_INNER_UP = "browInnerUp"
    BROW_OUTER_UP_LEFT = "browOuterUpLeft"
    BROW_OUTER_UP_RIGHT = "browOuterUpRight"
    
    # Mouth controls
    MOUTH_SMILE_LEFT = "mouthSmileLeft"
    MOUTH_SMILE_RIGHT = "mouthSmileRight"
    MOUTH_FROWN_LEFT = "mouthFrownLeft"
    MOUTH_FROWN_RIGHT = "mouthFrownRight"
    MOUTH_OPEN = "mouthOpen"
    MOUTH_CLOSE = "mouthClose"
    MOUTH_PUCKER = "mouthPucker"
    MOUTH_FUNNEL = "mouthFunnel"
    MOUTH_DIMPLE_LEFT = "mouthDimpleLeft"
    MOUTH_DIMPLE_RIGHT = "mouthDimpleRight"
    MOUTH_STRETCH_LEFT = "mouthStretchLeft"
    MOUTH_STRETCH_RIGHT = "mouthStretchRight"
    MOUTH_ROLL_LOWER = "mouthRollLower"
    MOUTH_ROLL_UPPER = "mouthRollUpper"
    MOUTH_PRESS_LEFT = "mouthPressLeft"
    MOUTH_PRESS_RIGHT = "mouthPressRight"
    MOUTH_UPPER_UP_LEFT = "mouthUpperUpLeft"
    MOUTH_UPPER_UP_RIGHT = "mouthUpperUpRight"
    MOUTH_LOWER_DOWN_LEFT = "mouthLowerDownLeft"
    MOUTH_LOWER_DOWN_RIGHT = "mouthLowerDownRight"
    
    # Jaw controls
    JAW_OPEN = "jawOpen"
    JAW_FORWARD = "jawForward"
    JAW_LEFT = "jawLeft"
    JAW_RIGHT = "jawRight"
    
    # Cheek controls
    CHEEK_PUFF = "cheekPuff"
    CHEEK_SQUINT_LEFT = "cheekSquintLeft"
    CHEEK_SQUINT_RIGHT = "cheekSquintRight"
    
    # Nose controls
    NOSE_SNEER_LEFT = "noseSneerLeft"
    NOSE_SNEER_RIGHT = "noseSneerRight"

class MediaPipeFacialMapper:
    """
    Maps MediaPipe face landmarks to 3D facial blend shapes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe face landmark indices (468 landmarks)
        self.landmark_indices = {
            # Eyes
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305],
            'right_eyebrow': [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283],
            
            # Mouth
            'mouth_outer': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324],
            'upper_lip': [12, 15, 16, 17, 18, 200, 199, 175, 0, 269, 270, 267, 271, 272],
            'lower_lip': [84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 15],
            
            # Nose
            'nose_tip': [1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131],
            'nose_bridge': [9, 10, 151, 337, 299, 333, 298, 301],
            
            # Face contour
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
            'chin': [18, 175, 199, 200, 16, 17, 18, 200, 199, 175],
            
            # Cheeks
            'left_cheek': [116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 206],
            'right_cheek': [345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410]
        }
        
        # Initialize baseline measurements for normalization
        self.baseline_measurements = {}
        self.previous_landmarks = None
        
    def extract_facial_features(self, landmarks: List[List[float]]) -> Dict[str, float]:
        """
        Extract detailed facial features from MediaPipe landmarks
        
        Args:
            landmarks: List of (x, y, z) landmark coordinates (normalized)
            
        Returns:
            Dictionary of blend shape weights (0.0 to 1.0)
        """
        if len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
            self.logger.warning(f"Insufficient landmarks: {len(landmarks)}")
            return {}
            
        # Convert to numpy array for easier computation
        landmarks_np = np.array(landmarks)
        
        # Initialize baseline if first frame
        if self.baseline_measurements == {}:
            self._initialize_baseline(landmarks_np)
            
        # Calculate blend shape weights
        blend_shapes = {}
        
        # Eye blend shapes
        blend_shapes.update(self._calculate_eye_blend_shapes(landmarks_np))
        
        # Eyebrow blend shapes
        blend_shapes.update(self._calculate_eyebrow_blend_shapes(landmarks_np))
        
        # Mouth blend shapes
        blend_shapes.update(self._calculate_mouth_blend_shapes(landmarks_np))
        
        # Jaw blend shapes
        blend_shapes.update(self._calculate_jaw_blend_shapes(landmarks_np))
        
        # Cheek blend shapes
        blend_shapes.update(self._calculate_cheek_blend_shapes(landmarks_np))
        
        # Nose blend shapes
        blend_shapes.update(self._calculate_nose_blend_shapes(landmarks_np))
        
        # Store for next frame comparison
        self.previous_landmarks = landmarks_np
        
        return blend_shapes
    
    def _initialize_baseline(self, landmarks: np.ndarray):
        """Initialize baseline measurements for normalization"""
        
        # Eye measurements
        left_eye_landmarks = landmarks[[33, 133, 159, 145]]  # corners and top/bottom
        right_eye_landmarks = landmarks[[362, 263, 386, 374]]
        
        self.baseline_measurements['left_eye_width'] = euclidean(
            left_eye_landmarks[0][:2], left_eye_landmarks[1][:2])
        self.baseline_measurements['left_eye_height'] = euclidean(
            left_eye_landmarks[2][:2], left_eye_landmarks[3][:2])
        
        self.baseline_measurements['right_eye_width'] = euclidean(
            right_eye_landmarks[0][:2], right_eye_landmarks[1][:2])
        self.baseline_measurements['right_eye_height'] = euclidean(
            right_eye_landmarks[2][:2], right_eye_landmarks[3][:2])
        
        # Mouth measurements
        mouth_corners = landmarks[[61, 291]]  # left and right corners
        mouth_top_bottom = landmarks[[13, 14]]  # top and bottom center
        
        self.baseline_measurements['mouth_width'] = euclidean(
            mouth_corners[0][:2], mouth_corners[1][:2])
        self.baseline_measurements['mouth_height'] = euclidean(
            mouth_top_bottom[0][:2], mouth_top_bottom[1][:2])
        
        # Eyebrow measurements
        left_brow = landmarks[[70, 63]]  # inner and outer
        right_brow = landmarks[[300, 293]]
        
        self.baseline_measurements['left_brow_height'] = left_brow[0][1]
        self.baseline_measurements['right_brow_height'] = right_brow[0][1]
        
        self.logger.info("Baseline measurements initialized")
    
    def _calculate_eye_blend_shapes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate eye-related blend shapes"""
        blend_shapes = {}
        
        try:
            # Left eye blink
            left_eye_top = landmarks[159][1]
            left_eye_bottom = landmarks[145][1]
            left_eye_openness = abs(left_eye_top - left_eye_bottom)
            left_eye_ratio = left_eye_openness / self.baseline_measurements['left_eye_height']
            
            blend_shapes[FacialBlendShapes.EYE_BLINK_LEFT] = max(0, 1.0 - left_eye_ratio * 3.0)
            
            # Right eye blink
            right_eye_top = landmarks[386][1]
            right_eye_bottom = landmarks[374][1]
            right_eye_openness = abs(right_eye_top - right_eye_bottom)
            right_eye_ratio = right_eye_openness / self.baseline_measurements['right_eye_height']
            
            blend_shapes[FacialBlendShapes.EYE_BLINK_RIGHT] = max(0, 1.0 - right_eye_ratio * 3.0)
            
            # Eye wide (opposite of blink)
            blend_shapes[FacialBlendShapes.EYE_WIDE_LEFT] = max(0, (left_eye_ratio - 1.0) * 2.0)
            blend_shapes[FacialBlendShapes.EYE_WIDE_RIGHT] = max(0, (right_eye_ratio - 1.0) * 2.0)
            
            # Eye squint (horizontal compression)
            left_eye_inner = landmarks[133][0]
            left_eye_outer = landmarks[33][0]
            left_eye_width = abs(left_eye_outer - left_eye_inner)
            left_width_ratio = left_eye_width / self.baseline_measurements['left_eye_width']
            
            blend_shapes[FacialBlendShapes.EYE_SQUINT_LEFT] = max(0, 1.0 - left_width_ratio * 1.5)
            
            right_eye_inner = landmarks[362][0]
            right_eye_outer = landmarks[263][0]
            right_eye_width = abs(right_eye_outer - right_eye_inner)
            right_width_ratio = right_eye_width / self.baseline_measurements['right_eye_width']
            
            blend_shapes[FacialBlendShapes.EYE_SQUINT_RIGHT] = max(0, 1.0 - right_width_ratio * 1.5)
            
        except (KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Error calculating eye blend shapes: {e}")
            
        return blend_shapes
    
    def _calculate_eyebrow_blend_shapes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate eyebrow-related blend shapes"""
        blend_shapes = {}
        
        try:
            # Left eyebrow raise
            left_brow_current = landmarks[70][1]  # inner brow
            left_brow_baseline = self.baseline_measurements['left_brow_height']
            left_brow_movement = left_brow_baseline - left_brow_current  # negative Y is up
            
            if left_brow_movement > 0:
                blend_shapes[FacialBlendShapes.BROW_INNER_UP] = min(1.0, left_brow_movement * 20.0)
            else:
                blend_shapes[FacialBlendShapes.BROW_DOWN_LEFT] = min(1.0, abs(left_brow_movement) * 15.0)
            
            # Right eyebrow
            right_brow_current = landmarks[300][1]
            right_brow_baseline = self.baseline_measurements['right_brow_height']
            right_brow_movement = right_brow_baseline - right_brow_current
            
            if right_brow_movement > 0:
                blend_shapes[FacialBlendShapes.BROW_OUTER_UP_RIGHT] = min(1.0, right_brow_movement * 20.0)
            else:
                blend_shapes[FacialBlendShapes.BROW_DOWN_RIGHT] = min(1.0, abs(right_brow_movement) * 15.0)
                
        except (KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Error calculating eyebrow blend shapes: {e}")
            
        return blend_shapes
    
    def _calculate_mouth_blend_shapes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate mouth-related blend shapes"""
        blend_shapes = {}
        
        try:
            # Mouth corners for smile/frown
            left_corner = landmarks[61]  # left mouth corner
            right_corner = landmarks[291]  # right mouth corner
            mouth_center = landmarks[13]  # upper lip center
            
            # Calculate smile (corners move up and out)
            left_corner_height = left_corner[1] - mouth_center[1]
            right_corner_height = right_corner[1] - mouth_center[1]
            
            # Negative values indicate upward movement (smile)
            if left_corner_height < 0:
                blend_shapes[FacialBlendShapes.MOUTH_SMILE_LEFT] = min(1.0, abs(left_corner_height) * 30.0)
            else:
                blend_shapes[FacialBlendShapes.MOUTH_FROWN_LEFT] = min(1.0, left_corner_height * 25.0)
                
            if right_corner_height < 0:
                blend_shapes[FacialBlendShapes.MOUTH_SMILE_RIGHT] = min(1.0, abs(right_corner_height) * 30.0)
            else:
                blend_shapes[FacialBlendShapes.MOUTH_FROWN_RIGHT] = min(1.0, right_corner_height * 25.0)
            
            # Mouth opening
            upper_lip = landmarks[13][1]
            lower_lip = landmarks[14][1]
            mouth_opening = abs(lower_lip - upper_lip)
            mouth_open_ratio = mouth_opening / self.baseline_measurements['mouth_height']
            
            blend_shapes[FacialBlendShapes.MOUTH_OPEN] = min(1.0, max(0, mouth_open_ratio - 1.0) * 5.0)
            
            # Mouth width for pucker/stretch
            current_mouth_width = euclidean(left_corner[:2], right_corner[:2])
            width_ratio = current_mouth_width / self.baseline_measurements['mouth_width']
            
            if width_ratio < 0.9:
                blend_shapes[FacialBlendShapes.MOUTH_PUCKER] = min(1.0, (0.9 - width_ratio) * 10.0)
            elif width_ratio > 1.1:
                stretch_value = min(1.0, (width_ratio - 1.1) * 5.0)
                blend_shapes[FacialBlendShapes.MOUTH_STRETCH_LEFT] = stretch_value
                blend_shapes[FacialBlendShapes.MOUTH_STRETCH_RIGHT] = stretch_value
                
        except (KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Error calculating mouth blend shapes: {e}")
            
        return blend_shapes
    
    def _calculate_jaw_blend_shapes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate jaw-related blend shapes"""
        blend_shapes = {}
        
        try:
            # Jaw opening (lower jaw movement)
            chin_point = landmarks[18]
            upper_lip = landmarks[13]
            jaw_opening = abs(chin_point[1] - upper_lip[1])
            
            # Normalize by face height approximation
            face_height = abs(landmarks[10][1] - landmarks[152][1])  # forehead to chin
            jaw_ratio = jaw_opening / face_height
            
            blend_shapes[FacialBlendShapes.JAW_OPEN] = min(1.0, max(0, jaw_ratio - 0.15) * 8.0)
            
            # Jaw left/right movement
            if self.previous_landmarks is not None:
                chin_movement_x = chin_point[0] - self.previous_landmarks[18][0]
                if abs(chin_movement_x) > 0.002:  # threshold for noise
                    if chin_movement_x > 0:
                        blend_shapes[FacialBlendShapes.JAW_RIGHT] = min(1.0, chin_movement_x * 50.0)
                    else:
                        blend_shapes[FacialBlendShapes.JAW_LEFT] = min(1.0, abs(chin_movement_x) * 50.0)
                        
        except (KeyError, ZeroDivisionError, IndexError) as e:
            self.logger.warning(f"Error calculating jaw blend shapes: {e}")
            
        return blend_shapes
    
    def _calculate_cheek_blend_shapes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate cheek-related blend shapes"""
        blend_shapes = {}
        
        try:
            # Cheek puff (outward movement)
            left_cheek = landmarks[116]
            right_cheek = landmarks[345]
            
            # Use Z-coordinate for depth (puffing out)
            left_cheek_depth = left_cheek[2] if len(left_cheek) > 2 else 0
            right_cheek_depth = right_cheek[2] if len(right_cheek) > 2 else 0
            
            # Positive Z indicates forward movement (puffing)
            if left_cheek_depth > 0.01:
                blend_shapes[FacialBlendShapes.CHEEK_PUFF] = min(1.0, left_cheek_depth * 20.0)
            elif right_cheek_depth > 0.01:
                blend_shapes[FacialBlendShapes.CHEEK_PUFF] = min(1.0, right_cheek_depth * 20.0)
                
        except (KeyError, IndexError) as e:
            self.logger.warning(f"Error calculating cheek blend shapes: {e}")
            
        return blend_shapes
    
    def _calculate_nose_blend_shapes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate nose-related blend shapes"""
        blend_shapes = {}
        
        try:
            # Nose sneer (nostril flare)
            left_nostril = landmarks[59]
            right_nostril = landmarks[289]
            nose_tip = landmarks[1]
            
            # Calculate nostril width
            nostril_width = euclidean(left_nostril[:2], right_nostril[:2])
            
            # Compare to baseline (approximate based on face width)
            face_width = euclidean(landmarks[127][:2], landmarks[356][:2])  # left to right face
            nostril_ratio = nostril_width / (face_width * 0.15)  # approximate normal ratio
            
            if nostril_ratio > 1.1:
                sneer_value = min(1.0, (nostril_ratio - 1.1) * 10.0)
                blend_shapes[FacialBlendShapes.NOSE_SNEER_LEFT] = sneer_value
                blend_shapes[FacialBlendShapes.NOSE_SNEER_RIGHT] = sneer_value
                
        except (KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Error calculating nose blend shapes: {e}")
            
        return blend_shapes

class AnimationEngine3D:
    """
    Advanced 3D character animation engine with detailed facial expression mapping
    """

    def __init__(self,
                 model_path: Path,
                 config: Optional[Dict[str, Any]] = None):
        
        self.logger = logging.getLogger(__name__)
        self.cfg = config or {}
        self.window_size = self.cfg.get('window_size', (512, 512))
        self.fps = self.cfg.get('animation_fps', 60)
        self.model_path = Path(model_path)
        
        # Initialize facial mapping
        self.facial_mapper = MediaPipeFacialMapper()
        
        # Facial animation state
        self.current_blend_shapes = {}
        self.target_blend_shapes = {}
        self.blend_shape_smoothing = self.cfg.get('facial_smoothing', 0.8)
        
        # Head rotation and position
        self.head_rotation = Vector3([0.0, 0.0, 0.0])
        self.head_position = Vector3([0.0, 0.0, 0.0])
        
        # -------------------- GL Setup --------------------
        self.window = pyglet.window.Window(*self.window_size,
                                           caption="3D Character - Facial Animation",
                                           visible=False)
        self.ctx = moderngl.create_context()
        
        # Create framebuffer for off-screen rendering
        self.color_texture = self.ctx.texture(self.window_size, 4)
        self.depth_texture = self.ctx.depth_texture(self.window_size)
        self.framebuffer = self.ctx.framebuffer(
            color_attachments=[self.color_texture],
            depth_attachment=self.depth_texture
        )
        
        # -------------------- Load 3D Model --------------------
        self._load_gltf(self.model_path)
        
        # -------------------- Create Shaders --------------------
        self._create_shaders()
        
        # -------------------- Camera Setup --------------------
        self.camera_pos = Vector3([0.0, 1.6, 3.0])
        self.camera_target = Vector3([0.0, 1.5, 0.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        
        self.proj_matrix = Matrix44.perspective_projection(
            45.0, self.window_size[0] / self.window_size[1], 0.1, 100.0)
        self.view_matrix = Matrix44.look_at(
            self.camera_pos, self.camera_target, self.camera_up)
        
        # Animation timing
        self.time = 0.0
        self.delta_time = 1.0 / self.fps
        
        self.logger.info("3D Animation Engine with facial mapping initialized")

    def _load_gltf(self, path: Path):
        """Load glTF model with facial blend shape support"""
        try:
            # Load using trimesh for better glTF support
            self.scene = trimesh.load(str(path))
            self.logger.info(f"Loaded glTF model: {path.name}")
            
            # Extract meshes and materials
            if hasattr(self.scene, 'geometry'):
                self.meshes = list(self.scene.geometry.values())
            else:
                self.meshes = [self.scene] if hasattr(self.scene, 'vertices') else []
            
            # Extract blend shape information if available
            self.blend_shape_targets = {}
            self._extract_blend_shapes()
            
            # Create OpenGL buffers
            self._create_mesh_buffers()
            
        except Exception as e:
            self.logger.error(f"Failed to load glTF model: {e}")
            # Create a simple fallback mesh (cube or sphere)
            self._create_fallback_mesh()
    
    def _extract_blend_shapes(self):
        """Extract blend shape targets from the loaded model"""
        # This would extract morph targets from glTF
        # For now, we'll simulate standard blend shapes
        standard_shapes = [
            FacialBlendShapes.EYE_BLINK_LEFT,
            FacialBlendShapes.EYE_BLINK_RIGHT,
            FacialBlendShapes.MOUTH_SMILE_LEFT,
            FacialBlendShapes.MOUTH_SMILE_RIGHT,
            FacialBlendShapes.MOUTH_OPEN,
            FacialBlendShapes.BROW_INNER_UP,
            FacialBlendShapes.JAW_OPEN
        ]
        
        for shape_name in standard_shapes:
            self.blend_shape_targets[shape_name] = 0.0
            
        self.logger.info(f"Initialized {len(self.blend_shape_targets)} blend shape targets")
    
    def _create_mesh_buffers(self):
        """Create OpenGL vertex buffers for the meshes"""
        self.mesh_vaos = []
        
        for mesh in self.meshes:
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                # Create vertex buffer
                vertices = mesh.vertices.astype(np.float32)
                indices = mesh.faces.flatten().astype(np.uint32)
                
                # Add normals if available
                if hasattr(mesh, 'vertex_normals'):
                    normals = mesh.vertex_normals.astype(np.float32)
                    vertex_data = np.column_stack((vertices, normals))
                else:
                    # Calculate normals
                    mesh.vertex_normals  # This computes normals
                    normals = mesh.vertex_normals.astype(np.float32)
                    vertex_data = np.column_stack((vertices, normals))
                
                # Create VAO
                vbo = self.ctx.buffer(vertex_data.tobytes())
                ibo = self.ctx.buffer(indices.tobytes())
                
                vao = self.ctx.vertex_array(
                    self.shader_program,
                    [(vbo, '3f 3f', 'in_position', 'in_normal')],
                    ibo
                )
                
                self.mesh_vaos.append({
                    'vao': vao,
                    'count': len(indices)
                })
                
        self.logger.info(f"Created {len(self.mesh_vaos)} mesh VAOs")
    
    def _create_fallback_mesh(self):
        """Create a simple fallback mesh if model loading fails"""
        # Create a simple head-shaped ellipsoid
        vertices = np.array([
            # Basic head vertices (simplified)
            [0.0, 0.0, 0.0],    # center
            [0.5, 0.0, 0.0],    # right
            [-0.5, 0.0, 0.0],   # left
            [0.0, 0.5, 0.0],    # top
            [0.0, -0.5, 0.0],   # bottom
            [0.0, 0.0, 0.5],    # front
            [0.0, 0.0, -0.5],   # back
        ], dtype=np.float32)
        
        normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        vertex_data = np.column_stack((vertices, normals))
        
        indices = np.array([
            0, 1, 3,  0, 3, 2,  0, 2, 4,  0, 4, 1,  # sides
            5, 1, 3,  5, 3, 2,  5, 2, 4,  5, 4, 1,  # front faces
        ], dtype=np.uint32)
        
        vbo = self.ctx.buffer(vertex_data.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        vao = self.ctx.vertex_array(
            self.shader_program,
            [(vbo, '3f 3f', 'in_position', 'in_normal')],
            ibo
        )
        
        self.mesh_vaos = [{'vao': vao, 'count': len(indices)}]
        self.logger.info("Created fallback mesh")
    
    def _create_shaders(self):
        """Create shader programs for rendering"""
        vertex_shader = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_normal;
        
        uniform mat4 u_mvp;
        uniform mat4 u_model;
        uniform mat4 u_normal_matrix;
        
        // Blend shape uniforms
        uniform float u_blend_shapes[32];
        
        out vec3 v_normal;
        out vec3 v_position;
        
        void main() {
            vec3 position = in_position;
            vec3 normal = in_normal;
            
            // Apply blend shapes (simplified - would normally use morph targets)
            // This is a placeholder for actual blend shape deformation
            
            gl_Position = u_mvp * vec4(position, 1.0);
            v_normal = normalize((u_normal_matrix * vec4(normal, 0.0)).xyz);
            v_position = (u_model * vec4(position, 1.0)).xyz;
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 v_normal;
        in vec3 v_position;
        
        uniform vec3 u_light_pos;
        uniform vec3 u_camera_pos;
        uniform vec3 u_color;
        
        out vec4 fragColor;
        
        void main() {
            vec3 normal = normalize(v_normal);
            vec3 light_dir = normalize(u_light_pos - v_position);
            vec3 view_dir = normalize(u_camera_pos - v_position);
            
            // Basic Phong lighting
            float ambient = 0.3;
            float diffuse = max(dot(normal, light_dir), 0.0);
            
            vec3 reflect_dir = reflect(-light_dir, normal);
            float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.5;
            
            vec3 color = u_color * (ambient + diffuse) + vec3(specular);
            fragColor = vec4(color, 1.0);
        }
        """
        
        self.shader_program = self.ctx.program(vertex_shader, fragment_shader)
        
        # Get uniform locations
        self.uniform_locations = {
            'u_mvp': self.shader_program.get('u_mvp', None),
            'u_model': self.shader_program.get('u_model', None),
            'u_normal_matrix': self.shader_program.get('u_normal_matrix', None),
            'u_light_pos': self.shader_program.get('u_light_pos', None),
            'u_camera_pos': self.shader_program.get('u_camera_pos', None),
            'u_color': self.shader_program.get('u_color', None),
        }
        
        # Blend shape uniforms
        for i in range(32):
            uniform_name = f'u_blend_shapes[{i}]'
            self.uniform_locations[f'blend_shape_{i}'] = self.shader_program.get(uniform_name, None)

    def render_frame(self,
                     state: Dict[str, Any],
                     motion_data: Dict[str, Any]) -> pyglet.image.ImageData:
        """
        Render a frame with detailed facial animation
        
        Args:
            state: Animation state dictionary
            motion_data: Motion detection data including facial landmarks
            
        Returns:
            Rendered frame as ImageData
        """
        # Update facial animation
        self._update_facial_animation(state, motion_data)
        
        # Update head pose
        self._update_head_pose(motion_data)
        
        # Update other animations
        self._update_body_animation(state, motion_data)
        
        # Render the scene
        self._render_scene()
        
        # Read pixels from framebuffer
        self.framebuffer.use()
        pixels = self.ctx.read(self.framebuffer, viewport=self.framebuffer.viewport, components=4)
        
        # Convert to pyglet ImageData
        img = pyglet.image.ImageData(
            self.window_size[0], self.window_size[1],
            'RGBA', pixels
        )
        
        # Flip vertically (OpenGL to screen coordinates)
        img = img.get_texture().get_image_data()
        
        self.time += self.delta_time
        
        return img
    
    def _update_facial_animation(self, state: Dict[str, Any], motion_data: Dict[str, Any]):
        """Update facial blend shapes based on MediaPipe landmarks"""
        
        # Extract facial landmarks from motion data
        face_data = motion_data.get('face', {})
        
        if face_data.get('detected', False) and 'landmarks' in face_data:
            landmarks = face_data['landmarks']
            
            # Extract blend shape weights from landmarks
            new_blend_shapes = self.facial_mapper.extract_facial_features(landmarks)
            
            # Smooth blend shape transitions
            for shape_name, target_weight in new_blend_shapes.items():
                if shape_name in self.current_blend_shapes:
                    current_weight = self.current_blend_shapes[shape_name]
                    smoothed_weight = (current_weight * self.blend_shape_smoothing + 
                                     target_weight * (1.0 - self.blend_shape_smoothing))
                    self.current_blend_shapes[shape_name] = smoothed_weight
                else:
                    self.current_blend_shapes[shape_name] = target_weight
        
        # Apply blend shapes to shader uniforms
        shape_index = 0
        for shape_name, weight in self.current_blend_shapes.items():
            uniform_key = f'blend_shape_{shape_index}'
            if uniform_key in self.uniform_locations and self.uniform_locations[uniform_key]:
                self.uniform_locations[uniform_key].value = weight
            shape_index += 1
            if shape_index >= 32:  # Max blend shapes in shader
                break
    
    def _update_head_pose(self, motion_data: Dict[str, Any]):
        """Update head rotation based on face pose detection"""
        
        face_data = motion_data.get('face', {})
        head_pose = face_data.get('head_pose', {})
        
        if head_pose:
            # Extract head rotation angles (in degrees)
            yaw = math.radians(head_pose.get('yaw', 0.0))
            pitch = math.radians(head_pose.get('pitch', 0.0))
            roll = math.radians(head_pose.get('roll', 0.0))
            
            # Smooth head rotation
            target_rotation = Vector3([pitch, yaw, roll])
            smoothing = 0.7
            
            self.head_rotation = (self.head_rotation * smoothing + 
                                target_rotation * (1.0 - smoothing))
    
    def _update_body_animation(self, state: Dict[str, Any], motion_data: Dict[str, Any]):
        """Update body pose and movement animations"""
        
        # Get body movement data
        movement_data = motion_data.get('body_movements', {})
        pose_data = motion_data.get('pose', {})
        
        # Update idle animations based on movement intensity
        movement_intensity = movement_data.get('intensity', 0.0)
        
        # Reduce idle breathing during active movement
        idle_intensity = max(0.1, 1.0 - movement_intensity)
        
        # Apply subtle breathing animation
        breathing = math.sin(self.time * 2.0) * 0.01 * idle_intensity
        self.head_position = Vector3([0.0, breathing, 0.0])
    
    def _render_scene(self):
        """Render the 3D scene"""
        
        # Use our framebuffer
        self.framebuffer.use()
        
        # Clear
        self.ctx.clear(0.2, 0.2, 0.3, 1.0)  # Dark blue background
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
        # Calculate model matrix with head pose
        model_matrix = Matrix44.from_translation(self.head_position)
        rotation_matrix = (Matrix44.from_x_rotation(self.head_rotation.x) @
                          Matrix44.from_y_rotation(self.head_rotation.y) @
                          Matrix44.from_z_rotation(self.head_rotation.z))
        model_matrix = model_matrix @ rotation_matrix
        
        # Calculate MVP matrix
        mvp_matrix = self.proj_matrix @ self.view_matrix @ model_matrix
        normal_matrix = rotation_matrix.T  # Simplified normal matrix
        
        # Set shader uniforms
        if self.uniform_locations['u_mvp']:
            self.uniform_locations['u_mvp'].write(mvp_matrix.astype(np.float32).tobytes())
        if self.uniform_locations['u_model']:
            self.uniform_locations['u_model'].write(model_matrix.astype(np.float32).tobytes())
        if self.uniform_locations['u_normal_matrix']:
            self.uniform_locations['u_normal_matrix'].write(normal_matrix.astype(np.float32).tobytes())
        if self.uniform_locations['u_light_pos']:
            self.uniform_locations['u_light_pos'].value = (2.0, 3.0, 2.0)
        if self.uniform_locations['u_camera_pos']:
            self.uniform_locations['u_camera_pos'].value = tuple(self.camera_pos)
        if self.uniform_locations['u_color']:
            self.uniform_locations['u_color'].value = (0.8, 0.7, 0.6)  # Skin tone
        
        # Render meshes
        for mesh_data in self.mesh_vaos:
            mesh_data['vao'].render()
    
    def get_facial_blend_shape_weights(self) -> Dict[str, float]:
        """Get current facial blend shape weights for debugging"""
        return self.current_blend_shapes.copy()
    
    def set_camera_position(self, position: Vector3, target: Vector3 = None):
        """Set camera position and target"""
        self.camera_pos = position
        if target is not None:
            self.camera_target = target
            
        self.view_matrix = Matrix44.look_at(
            self.camera_pos, self.camera_target, self.camera_up)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if hasattr(self, 'framebuffer'):
            self.framebuffer.release()
        if hasattr(self, 'color_texture'):
            self.color_texture.release()
        if hasattr(self, 'depth_texture'):
            self.depth_texture.release()
        if hasattr(self, 'shader_program'):
            self.shader_program.release()
