import bpy
import gpu
from .interop_bridge import ZeroCopyGLBuffer52

def draw_particles_viewport(gl_buf: ZeroCopyGLBuffer52, running: bool):
    """Zero-copy OpenGL Core Profile draw pass for Blender 5.2+ 3D Viewport."""
    if not running or gl_buf is None:
        return

    scene = bpy.context.scene
    pt_size = getattr(scene, "warp_sph_point_size", 2.5)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
    shader.uniform_float("ModelViewProjectionMatrix", mvp)
    shader.uniform_float("color", (0.15, 0.68, 1.0, 1.0))

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.point_size_set(pt_size)
    gl_buf.draw_points()
    gpu.state.depth_test_set('NONE')
