"""
Lightweight 3-D Animation Engine (glTF-based)

Requirements
------------
pip install pyglet moderngl pyrr pygltflib
"""
import math
from pathlib import Path
from typing import Dict, Any, Optional

import pyglet
from pyglet.gl import *
import moderngl
from pyrr import Matrix44, Vector3
from pygltflib import GLTF2

class AnimationEngine3D:
    """
    Real-time 3-D character renderer that mimics the 2-D AnimationEngine API.
    Expects a skinned glTF model with animation clips:
        - idle, walk, run, celebrate, etc.
    """

    def __init__(self,
                 model_path: Path,
                 config: Optional[Dict[str, Any]] = None):
        self.cfg = config or {}
        self.window_size = self.cfg.get('window_size', (512, 512))
        self.fps          = self.cfg.get('animation_fps', 60)
        self.model_path   = Path(model_path)

        # -------------------- GL context --------------------
        self.window = pyglet.window.Window(*self.window_size,
                                           caption="3D Character",
                                           visible=False)       # off-screen
        self.ctx    = moderngl.create_context()

        # -------------------- Load model --------------------
        self._load_gltf(self.model_path)

        # camera
        self.camera_pos = Vector3([0.0, 1.6, 3.0])
        self.proj = Matrix44.perspective_projection(
            45.0, self.window_size[0] / self.window_size[1], 0.1, 100.0)
        self.view = Matrix44.look_at(self.camera_pos,
                                     Vector3([0.0, 1.5, 0.0]),
                                     Vector3([0.0, 1.0, 0.0]))
        # time
        self.time = 0.0

    # ------------------------------------------------------------------ #
    # public API identical to 2-D version
    # ------------------------------------------------------------------ #
    def render_frame(self,
                     state: Dict[str, Any],
                     motion_data: Dict[str, Any]) -> pyglet.image.ImageData:
        """
        Renders one frame to an off-screen buffer and returns a pyglet image.
        """
        self._update_animation(state, motion_data)
        self._render_scene()

        # Read pixels into ImageData (RGBA)
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        img = buffer.get_image_data()
        return img

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _load_gltf(self, path: Path):
        # Minimal glTF loading (geometry, skins, animations).  Full loaders
        # such as pygltflib or trimesh can be swapped in later.
        self.gltf = GLTF2().load(str(path))
        # TODO: create VAOs / VBOs, shader program, skin matrices…
        # For brevity we only stub here.
        print(f"Loaded glTF: {path.name} with {len(self.gltf.meshes)} mesh(es)")

    def _update_animation(self, state: Dict[str, Any],
                          motion_data: Dict[str, Any]):
        """
        Decides which animation clip to play and sets blend weights based
        on the same 'state' dict used by the 2-D engine.

        Example mapping:
          - state['body_pose'] → clip name
          - motion_data['movement']['intensity'] → playback speed
        """
        # Pseudocode – hook up to your animation system / shaders.
        self.time += 1.0 / self.fps

    def _render_scene(self):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        # TODO: bind shader, upload MVP + skin matrices, draw
        # This stub draws nothing yet.

        # Blit context to pyglet window’s default framebuffer
        self.ctx.copy_framebuffer(self.window._context.fbo)
        self.window.dispatch_events()
