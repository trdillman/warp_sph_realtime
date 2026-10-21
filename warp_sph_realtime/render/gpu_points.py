"""Backend-neutral particle draw for transport frames (plan M3).

One six-vertex quad batch drawn instanced ``frame.count`` times; the vertex stage fetches
its position with texelFetch by gpu_InstanceIndex from the 4096-wide RGBA32F texture.
Uses only the public gpu module, so it runs on Vulkan, Metal and OpenGL. The legacy raw-GL
SSFR liquid stays OpenGL-only; other backends draw shaded spheres for SSFR mode. On OpenGL this
renderer serves the GLInterop texture session (Points mode, chosen at Start).
Sphere depth is the billboard plane (no per-fragment depth write), matching visual parity
at viewport scale; exact sphere depth is not claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

POINTS, SPHERES = 0, 1
CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
PUSH_CONSTANT_BYTES = 64 + 3 * 16 + 8   # mat4 + right/up/color + viewport = 120 <= 128

VERTEX = """
void main() {
    int i = gpu_InstanceIndex;
    vec3 p = texelFetch(positions, ivec2(i % 4096, i / 4096), 0).xyz;
    uv = corner;
    if (u_up.w < 0.5) {
        vec4 clip = u_mvp * vec4(p, 1.0);
        clip.xy += corner * u_right.w / u_viewport * clip.w;
        gl_Position = clip;
    } else {
        vec3 world = p + (u_right.xyz * corner.x + u_up.xyz * corner.y) * u_right.w;
        gl_Position = u_mvp * vec4(world, 1.0);
    }
}
"""

FRAGMENT = """
void main() {
    if (u_up.w < 0.5) {
        fragColor = u_color;
        return;
    }
    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        discard;
    }
    float nz = sqrt(1.0 - r2);
    float diffuse = 0.3 + 0.7 * max(dot(normalize(vec3(uv.x, uv.y, nz)), normalize(vec3(0.35, 0.45, 0.82))), 0.0);
    float spec = pow(max(nz, 0.0), 48.0) * 0.25;
    fragColor = vec4(u_color.rgb * diffuse + vec3(spec), u_color.a);
}
"""


@dataclass(frozen=True)
class DrawSettings:
    mode: int
    size: float           # pixels for POINTS, world radius for SPHERES
    color: tuple
    depth_test: bool
    note: str = ''


def settings_from_scene(scene, backend: str) -> DrawSettings:
    """Map the add-on display properties; SSFR has no portable equivalent off OpenGL."""
    color = tuple(float(c) for c in getattr(scene, 'warp_sph_particle_color', (0.12, 0.68, 1.0, 1.0)))
    depth = bool(getattr(scene, 'warp_sph_depth_test', True))
    mode = getattr(scene, 'warp_sph_render_mode', 'SSFR_LIQUID')
    if mode == 'POINTS':
        return DrawSettings(POINTS, float(getattr(scene, 'warp_sph_point_size', 2.5)), color, depth)
    if mode == 'SPHERES':
        note = ''
    elif backend == 'OPENGL':  # a texture session is running: SSFR needs the legacy path, chosen at Start
        note = 'SSFR liquid needs the legacy OpenGL path: stop and start with SSFR selected; drawing spheres'
    else:
        note = f'SSFR liquid is OpenGL-only; drawing spheres on {backend}'
    return DrawSettings(SPHERES, float(getattr(scene, 'warp_sph_particle_radius', 0.022)), color, depth, note)


def uniforms(view, viewport: tuple, settings: DrawSettings) -> dict:
    """Push-constant values except the MVP: world-space camera axes from the view rotation rows."""
    rows = [[float(view[r][c]) for c in range(3)] for r in range(2)]
    return {
        'u_right': (*rows[0], settings.size),
        'u_up': (*rows[1], float(settings.mode)),
        'u_color': tuple(settings.color),
        'u_viewport': (max(1.0, float(viewport[0])), max(1.0, float(viewport[1]))),
    }


class PointRenderer:
    """Draws a transport Frame in the current viewport; compiles lazily on first draw."""

    def __init__(self, gpu_module: Any = None, batch_factory: Any = None):
        self._gpu = gpu_module
        self._batch_factory = batch_factory
        self._shader: Any = None
        self._batch: Any = None

    def _build(self):
        gpu = self._gpu
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', 'corner')
        info.sampler(0, 'FLOAT_2D', 'positions')
        info.push_constant('MAT4', 'u_mvp')
        info.push_constant('VEC4', 'u_right')
        info.push_constant('VEC4', 'u_up')
        info.push_constant('VEC4', 'u_color')
        info.push_constant('VEC2', 'u_viewport')
        iface = gpu.types.GPUStageInterfaceInfo('warp_sph_points')
        iface.smooth('VEC2', 'uv')
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', 'fragColor')
        info.vertex_source(VERTEX)
        info.fragment_source(FRAGMENT)
        self._shader = gpu.shader.create_from_info(info)
        self._batch = self._batch_factory(self._shader, 'TRIS', {'corner': CORNERS})

    def draw(self, frame, settings: DrawSettings) -> bool:
        if frame is None or frame.count <= 0:
            return False
        if self._gpu is None:
            import gpu
            self._gpu = gpu
        if self._batch_factory is None:
            from gpu_extras.batch import batch_for_shader
            self._batch_factory = batch_for_shader
        if self._shader is None:
            self._build()
        gpu = self._gpu
        view = gpu.matrix.get_model_view_matrix()
        mvp = gpu.matrix.get_projection_matrix() @ view
        viewport = cast(tuple, gpu.state.viewport_get())[2:]  # stub says Optional; always (x, y, w, h)
        depth_test, depth_mask, blend = gpu.state.depth_test_get(), gpu.state.depth_mask_get(), gpu.state.blend_get()
        try:
            gpu.state.depth_test_set('LESS_EQUAL' if settings.depth_test else 'NONE')
            gpu.state.depth_mask_set(settings.depth_test)
            gpu.state.blend_set('ALPHA' if settings.color[3] < 1.0 else 'NONE')
            shader = self._shader
            shader.bind()
            shader.uniform_sampler('positions', frame.texture)
            shader.uniform_float('u_mvp', mvp)
            for name, value in uniforms(view, viewport, settings).items():
                shader.uniform_float(name, value)
            self._batch.draw_instanced(shader, instance_count=frame.count)
        finally:
            gpu.state.depth_test_set(depth_test)
            gpu.state.depth_mask_set(depth_mask)
            gpu.state.blend_set(blend)
        return True

    def free(self):
        self._shader = self._batch = None
