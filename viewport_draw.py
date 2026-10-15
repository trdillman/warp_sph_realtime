import ctypes
from functools import wraps
import bpy
import gpu

try:
    from . import interop_bridge as _ig
    from .interop_bridge import (
        GL_BLEND,
        GL_CLAMP_TO_EDGE,
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_ATTACHMENT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_COMPONENT,
        GL_DEPTH_COMPONENT32F,
        GL_DEPTH24_STENCIL8,
        GL_DEPTH_STENCIL,
        GL_UNSIGNED_INT_24_8,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_NO_ERROR,
        GL_INVALID_OPERATION,
        GL_DEPTH_TEST,
        GL_DRAW_FRAMEBUFFER,
        GL_DRAW_FRAMEBUFFER_BINDING,
        GL_FLOAT,
        GL_FRAMEBUFFER,
        GL_LEQUAL,
        GL_LINEAR,
        GL_NEAREST,
        GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_PROGRAM_POINT_SIZE,
        GL_R16F,
        GL_R32F,
        GL_READ_FRAMEBUFFER,
        GL_READ_FRAMEBUFFER_BINDING,
        GL_RED,
        GL_RGBA,
        GL_RGBA8,
        GL_SRC_ALPHA,
        GL_TEXTURE0,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLES,
        GL_UNSIGNED_BYTE,
        GL_VIEWPORT,
        ZeroCopyGLBuffer52,
        _GLInterop,
    )
    from .ssfr_shaders import (
        SSFR_COMPOSITE_FRAG,
        SSFR_DEPTH_FRAG,
        SSFR_DEPTH_VERT,
        SSFR_FULLSCREEN_VERT,
        SSFR_NARROW_RANGE_FRAG,
        SSFR_THICKNESS_FRAG,
    )
except ImportError:
    import interop_bridge as _ig
    from interop_bridge import (
        GL_BLEND,
        GL_CLAMP_TO_EDGE,
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_ATTACHMENT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_COMPONENT,
        GL_DEPTH_COMPONENT32F,
        GL_DEPTH24_STENCIL8,
        GL_DEPTH_STENCIL,
        GL_UNSIGNED_INT_24_8,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_NO_ERROR,
        GL_INVALID_OPERATION,
        GL_DEPTH_TEST,
        GL_DRAW_FRAMEBUFFER,
        GL_DRAW_FRAMEBUFFER_BINDING,
        GL_FLOAT,
        GL_FRAMEBUFFER,
        GL_LEQUAL,
        GL_LINEAR,
        GL_NEAREST,
        GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_PROGRAM_POINT_SIZE,
        GL_R16F,
        GL_R32F,
        GL_READ_FRAMEBUFFER,
        GL_READ_FRAMEBUFFER_BINDING,
        GL_RED,
        GL_RGBA,
        GL_RGBA8,
        GL_SRC_ALPHA,
        GL_TEXTURE0,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLES,
        GL_UNSIGNED_BYTE,
        GL_VIEWPORT,
        ZeroCopyGLBuffer52,
        _GLInterop,
    )
    from ssfr_shaders import (
        SSFR_COMPOSITE_FRAG,
        SSFR_DEPTH_FRAG,
        SSFR_DEPTH_VERT,
        SSFR_FULLSCREEN_VERT,
        SSFR_NARROW_RANGE_FRAG,
        SSFR_THICKNESS_FRAG,
    )

# Shaded Eye-Space Sphere Impostor Fragment Shader (Direct single-pass mode)
SPHERE_IMPOSTOR_FRAG = """#version 450 core
in vec3 v_eye_pos;
in float v_radius;

uniform mat4 ProjectionMatrix;
uniform vec4 BaseColor;

layout(location = 0) out vec4 out_color;

void main() {
    // OpenGL gl_PointCoord has origin at upper-left (Y grows downward).
    // Invert Y so coord.y and normal N.y correctly point upward in eye space.
    vec2 coord = vec2(gl_PointCoord.x * 2.0 - 1.0, 1.0 - gl_PointCoord.y * 2.0);
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }
    float nz = sqrt(1.0 - r2);
    vec3 N = vec3(coord.x, coord.y, nz);
    float z_eye = v_eye_pos.z + nz * v_radius;

    vec4 clip_pos = ProjectionMatrix * vec4(v_eye_pos.xy, z_eye, 1.0);
    gl_FragDepth = (clip_pos.z / clip_pos.w) * 0.5 + 0.5;

    vec3 L = normalize(vec3(0.45, 0.55, 0.70));
    vec3 V = normalize(-v_eye_pos);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0) * 0.75 + 0.25;
    float spec = pow(max(dot(N, H), 0.0), 48.0) * 0.45;
    out_color = vec4(BaseColor.rgb * diff + vec3(spec), BaseColor.a);
}
"""


def _mat4_to_ctypes(mat) -> ctypes.Array:
    """Converts a Blender 4x4 Matrix (row-major) to a column-major float[16] array for OpenGL."""
    vals = [float(mat[r][c]) for c in range(4) for r in range(4)]
    return (ctypes.c_float * 16)(*vals)


def _preserve_gl_state(draw):
    """Restore caller state even on shader/allocation/draw failures.

    Capture before allocation: texture creation also mutates the active unit.
    Queries are scalar GL state queries, never simulation-buffer readbacks.
    """
    @wraps(draw)
    def guarded(*args, **kwargs):
        gl = _GLInterop
        gl.ensure_initialized()
        def ints(pname, count=1):
            out = (ctypes.c_int * count)()
            gl.glGetIntegerv(pname, out)
            return tuple(out)
        def floats(pname, count):
            out = (ctypes.c_float * count)()
            gl.glGetFloatv(pname, out)
            return tuple(out)
        keys = (_ig.GL_DEPTH_WRITEMASK, _ig.GL_DEPTH_FUNC, _ig.GL_BLEND_SRC_RGB,
                _ig.GL_BLEND_DST_RGB, _ig.GL_BLEND_SRC_ALPHA, _ig.GL_BLEND_DST_ALPHA,
                _ig.GL_BLEND_EQUATION_RGB, _ig.GL_BLEND_EQUATION_ALPHA,
                _ig.GL_CURRENT_PROGRAM, _ig.GL_VERTEX_ARRAY_BINDING,
                GL_DRAW_FRAMEBUFFER_BINDING, GL_READ_FRAMEBUFFER_BINDING,
                _ig.GL_ACTIVE_TEXTURE, _ig.GL_ARRAY_BUFFER_BINDING)
        state = {k: ints(k)[0] for k in keys}
        viewport = ints(GL_VIEWPORT, 4)
        color_mask = ints(_ig.GL_COLOR_WRITEMASK, 4)
        clear_color = floats(_ig.GL_COLOR_CLEAR_VALUE, 4)
        clear_depth = floats(_ig.GL_DEPTH_CLEAR_VALUE, 1)[0]
        caps = {cap: gl.glIsEnabled(cap) for cap in
                (GL_PROGRAM_POINT_SIZE, GL_DEPTH_TEST, GL_BLEND, _ig.GL_SCISSOR_TEST)}
        textures = {}
        for unit in (GL_TEXTURE0, GL_TEXTURE0 + 1, GL_TEXTURE0 + 2, state[_ig.GL_ACTIVE_TEXTURE]):
            gl.glActiveTexture(unit)
            textures[unit] = ints(_ig.GL_TEXTURE_BINDING_2D)[0]
        gl.glActiveTexture(state[_ig.GL_ACTIVE_TEXTURE])
        try:
            gl.glDisable(_ig.GL_SCISSOR_TEST)
            gl.glColorMask(1, 1, 1, 1)
            gl.glClearDepth(1.0)
            gl.glBlendEquationSeparate(_ig.GL_FUNC_ADD, _ig.GL_FUNC_ADD)
            return draw(*args, **kwargs)
        finally:
            gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, state[GL_DRAW_FRAMEBUFFER_BINDING])
            gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, state[GL_READ_FRAMEBUFFER_BINDING])
            gl.glViewport(*viewport)
            gl.glDepthMask(state[_ig.GL_DEPTH_WRITEMASK])
            gl.glDepthFunc(state[_ig.GL_DEPTH_FUNC])
            gl.glBlendFuncSeparate(*(state[k] for k in keys[2:6]))
            gl.glBlendEquationSeparate(state[_ig.GL_BLEND_EQUATION_RGB], state[_ig.GL_BLEND_EQUATION_ALPHA])
            gl.glClearColor(*clear_color)
            gl.glClearDepth(clear_depth)
            gl.glColorMask(*color_mask)
            gl.glUseProgram(state[_ig.GL_CURRENT_PROGRAM])
            gl.glBindVertexArray(state[_ig.GL_VERTEX_ARRAY_BINDING])
            gl.glBindBuffer(_ig.GL_ARRAY_BUFFER, state[_ig.GL_ARRAY_BUFFER_BINDING])
            for unit, texture in textures.items():
                gl.glActiveTexture(unit)
                gl.glBindTexture(GL_TEXTURE_2D, texture)
            gl.glActiveTexture(state[_ig.GL_ACTIVE_TEXTURE])
            for cap, enabled in caps.items():
                (gl.glEnable if enabled else gl.glDisable)(cap)
    return guarded


class SSFRViewportRenderer:
    """Manages per-GL-context shaders, empty fullscreen VAOs, and offscreen FBOs for SSFR."""

    def __init__(self):
        self._ctx_resources = {}
        self._pending_destroy = False

    def _get_or_create_ctx_data(self, width: int, height: int, depth_format: int = GL_DEPTH24_STENCIL8) -> dict:
        gl = _GLInterop
        gl.ensure_initialized()
        ctx = gl.current_context()
        data = self._ctx_resources.get(ctx)

        if data is None:
            empty_vao = ctypes.c_uint(0)
            gl.glGenVertexArrays(1, ctypes.byref(empty_vao))

            programs = []
            try:
                for vertex, fragment in ((SSFR_DEPTH_VERT, SSFR_DEPTH_FRAG),
                                         (SSFR_DEPTH_VERT, SSFR_THICKNESS_FRAG),
                                         (SSFR_FULLSCREEN_VERT, SSFR_NARROW_RANGE_FRAG),
                                         (SSFR_FULLSCREEN_VERT, SSFR_COMPOSITE_FRAG),
                                         (SSFR_DEPTH_VERT, SPHERE_IMPOSTOR_FRAG)):
                    programs.append(gl.compile_program(vertex, fragment))
            except Exception:
                for program in programs:
                    gl.glDeleteProgram(program)
                gl.glDeleteVertexArrays(1, ctypes.byref(empty_vao))
                raise
            prog_depth, prog_thick, prog_filter, prog_comp, prog_sphere = programs

            data = {
                "empty_vao": empty_vao,
                "prog_depth": prog_depth,
                "prog_thick": prog_thick,
                "prog_filter": prog_filter,
                "prog_comp": prog_comp,
                "prog_sphere": prog_sphere,
                "width": 0,
                "height": 0,
                "half_w": 0,
                "half_h": 0,
                "fbo_depth": ctypes.c_uint(0),
                "fbo_thick": ctypes.c_uint(0),
                "fbo_ping": ctypes.c_uint(0),
                "fbo_pong": ctypes.c_uint(0),
                "fbo_scene_col": ctypes.c_uint(0),
                "tex_scene_depth": ctypes.c_uint(0),
                "tex_depth_hw": ctypes.c_uint(0),
                "tex_thick_depth_hw": ctypes.c_uint(0),
                "tex_eye_depth": ctypes.c_uint(0),
                "tex_thick": ctypes.c_uint(0),
                "tex_ping": ctypes.c_uint(0),
                "tex_pong": ctypes.c_uint(0),
                "tex_scene_col": ctypes.c_uint(0),
            }
            self._ctx_resources[ctx] = data

        if width > 0 and height > 0 and (data["width"] != width or data["height"] != height or data.get("depth_format") != depth_format):
            self._resize_fbos(data, width, height, depth_format)

        return data

    def _resize_fbos(self, data: dict, w: int, h: int, depth_format: int = GL_DEPTH24_STENCIL8):
        gl = _GLInterop
        w = max(2, int(w))
        h = max(2, int(h))
        half_w = max(1, w // 2)
        half_h = max(1, h // 2)

        # Allocation must not replace Blender's active viewport framebuffer.
        prev_draw_fbo = ctypes.c_int(0)
        prev_read_fbo = ctypes.c_int(0)
        gl.glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, ctypes.byref(prev_draw_fbo))
        gl.glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, ctypes.byref(prev_read_fbo))

        # Delete existing FBOs/textures if resizing
        for k in ("fbo_depth", "fbo_thick", "fbo_ping", "fbo_pong", "fbo_scene_col"):
            if data[k].value:
                gl.glDeleteFramebuffers(1, ctypes.byref(data[k]))
                data[k] = ctypes.c_uint(0)
        for k in ("tex_scene_depth", "tex_depth_hw", "tex_thick_depth_hw", "tex_eye_depth", "tex_thick", "tex_ping", "tex_pong", "tex_scene_col"):
            if data.get(k) and data[k].value:
                gl.glDeleteTextures(1, ctypes.byref(data[k]))
                data[k] = ctypes.c_uint(0)

        def _make_tex(
            tex_w: int,
            tex_h: int,
            internal_fmt: int,
            fmt: int,
            pixel_type: int,
            filter_mode: int,
        ) -> ctypes.c_uint:
            t = ctypes.c_uint(0)
            gl.glGenTextures(1, ctypes.byref(t))
            gl.glBindTexture(GL_TEXTURE_2D, t.value)
            gl.glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, tex_w, tex_h, 0, fmt, pixel_type, None)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            gl.glBindTexture(GL_TEXTURE_2D, 0)
            return t

        # Match the source exactly; packed D24/S8 is not universal in Blender 5.2.
        packed = depth_format in (GL_DEPTH24_STENCIL8, _ig.GL_DEPTH32F_STENCIL8)
        depth_attachment = GL_DEPTH_STENCIL_ATTACHMENT if packed else GL_DEPTH_ATTACHMENT
        depth_external = GL_DEPTH_STENCIL if packed else GL_DEPTH_COMPONENT
        depth_type = (_ig.GL_FLOAT_32_UNSIGNED_INT_24_8_REV if depth_format == _ig.GL_DEPTH32F_STENCIL8
                      else GL_UNSIGNED_INT_24_8 if packed else GL_FLOAT)
        data["tex_scene_depth"] = _make_tex(w, h, depth_format, depth_external, depth_type, GL_NEAREST)
        data["tex_depth_hw"] = _make_tex(w, h, depth_format, depth_external, depth_type, GL_NEAREST)
        data["tex_thick_depth_hw"] = _make_tex(half_w, half_h, depth_format, depth_external, depth_type, GL_NEAREST)
        data["tex_eye_depth"] = _make_tex(w, h, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST)
        # Upgrade #1: Half-resolution thickness texture (W/2 x H/2) with linear hardware upsampling
        data["tex_thick"] = _make_tex(half_w, half_h, GL_R16F, GL_RED, GL_FLOAT, GL_LINEAR)
        data["tex_ping"] = _make_tex(w, h, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST)
        data["tex_pong"] = _make_tex(w, h, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST)
        data["tex_scene_col"] = _make_tex(w, h, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR)

        # 1. Depth FBO (Color0 = eye_depth R32F + DepthStencil = hardware depth buffer)
        gl.glGenFramebuffers(1, ctypes.byref(data["fbo_depth"]))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_depth"].value)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, data["tex_eye_depth"].value, 0)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, depth_attachment, GL_TEXTURE_2D, data["tex_depth_hw"].value, 0)

        # 2. Half-Resolution Thickness FBO (Color0 = optical thickness R16F + DepthStencil for scene occlusion)
        gl.glGenFramebuffers(1, ctypes.byref(data["fbo_thick"]))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_thick"].value)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, data["tex_thick"].value, 0)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, depth_attachment, GL_TEXTURE_2D, data["tex_thick_depth_hw"].value, 0)

        # 3. Ping-Pong Narrow-Range Filter FBOs
        gl.glGenFramebuffers(1, ctypes.byref(data["fbo_ping"]))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_ping"].value)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, data["tex_ping"].value, 0)

        gl.glGenFramebuffers(1, ctypes.byref(data["fbo_pong"]))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_pong"].value)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, data["tex_pong"].value, 0)

        # 4. Scene Background Color FBO for Screen-Space Refraction Distortion
        gl.glGenFramebuffers(1, ctypes.byref(data["fbo_scene_col"]))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_scene_col"].value)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, data["tex_scene_col"].value, 0)

        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, depth_attachment, GL_TEXTURE_2D, data["tex_scene_depth"].value, 0)

        gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_draw_fbo.value)
        gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read_fbo.value)
        data["depth_format"] = depth_format
        data["width"] = w
        data["height"] = h
        data["half_w"] = half_w
        data["half_h"] = half_h

    @_preserve_gl_state
    def draw_spheres(self, gl_buf: ZeroCopyGLBuffer52, scene, vp_height: int):
        """Renders shaded 3D sphere impostors in a single pass."""
        gl = _GLInterop
        data = self._get_or_create_ctx_data(0, 0)
        prog = data["prog_sphere"]

        mv_mat = gpu.matrix.get_model_view_matrix()
        proj_mat = gpu.matrix.get_projection_matrix()
        mv_ptr = _mat4_to_ctypes(mv_mat)
        proj_ptr = _mat4_to_ctypes(proj_mat)
        col = tuple(scene.warp_sph_particle_color)
        radius = float(getattr(scene, "warp_sph_particle_radius", 0.022))

        saved = {cap: gl.glIsEnabled(cap) for cap in (GL_PROGRAM_POINT_SIZE, GL_DEPTH_TEST, GL_BLEND)}
        gl.glEnable(GL_PROGRAM_POINT_SIZE)
        if scene.warp_sph_depth_test:
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(GL_LEQUAL)
        else:
            gl.glDisable(GL_DEPTH_TEST)

        if col[3] < 1.0:
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        else:
            gl.glDisable(GL_BLEND)

        gl.glUseProgram(prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(prog, b"ModelViewMatrix"), 1, 0, mv_ptr)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(prog, b"ProjectionMatrix"), 1, 0, proj_ptr)
        gl.glUniform1f(gl.glGetUniformLocation(prog, b"ViewportHeight"), float(vp_height))
        gl.glUniform1f(gl.glGetUniformLocation(prog, b"ParticleRadius"), radius)
        gl.glUniform4f(gl.glGetUniformLocation(prog, b"BaseColor"), col[0], col[1], col[2], col[3])

        gl_buf.bind_vao()
        gl.glDrawArrays(GL_POINTS, 0, gl_buf.num_particles)
        gl_buf.unbind_vao()
        gl.glUseProgram(0)

        for cap, was_on in saved.items():
            (gl.glEnable if was_on else gl.glDisable)(cap)

    @_preserve_gl_state
    def draw_ssfr_liquid(self, gl_buf: ZeroCopyGLBuffer52, scene, vp_x: int, vp_y: int, vp_w: int, vp_h: int):
        """Executes the 5-Pass Narrow-Range Screen-Space Fluid Rendering pipeline."""
        gl = _GLInterop
        use_depth_occlusion = bool(getattr(scene, "warp_sph_depth_test", True))
        depth_format = gl.framebuffer_depth_format() if use_depth_occlusion else GL_DEPTH24_STENCIL8
        data = self._get_or_create_ctx_data(vp_w, vp_h, depth_format)

        prev_draw_fbo = ctypes.c_int(0)
        prev_read_fbo = ctypes.c_int(0)
        gl.glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, ctypes.byref(prev_draw_fbo))
        gl.glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, ctypes.byref(prev_read_fbo))

        mv_mat = gpu.matrix.get_model_view_matrix()
        proj_mat = gpu.matrix.get_projection_matrix()
        mv_ptr = _mat4_to_ctypes(mv_mat)
        proj_ptr = _mat4_to_ctypes(proj_mat)

        radius = float(getattr(scene, "warp_sph_particle_radius", 0.022))
        sigma_px = float(getattr(scene, "warp_sph_filter_sigma", 4.0))
        iterations = int(getattr(scene, "warp_sph_filter_iterations", 2))
        abs_rgb = tuple(getattr(scene, "warp_sph_absorption", (0.85, 0.22, 0.05)))
        base_col = tuple(scene.warp_sph_particle_color)
        use_depth_occlusion = bool(getattr(scene, "warp_sph_depth_test", True))

        saved = {cap: gl.glIsEnabled(cap) for cap in (GL_PROGRAM_POINT_SIZE, GL_DEPTH_TEST, GL_BLEND)}

        # -------------------------------------------------------------------
        # Pre-Pass Blit: Capture Blender's Viewport Color & Depth Buffers
        # -------------------------------------------------------------------
        has_scene_color = 0
        gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_draw_fbo.value)
        gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, data["fbo_scene_col"].value)
        gl.glBlitFramebuffer(
            vp_x, vp_y, vp_x + vp_w, vp_y + vp_h,
            0, 0, vp_w, vp_h,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST,
        )
        has_scene_color = 1

        if use_depth_occlusion:
            # Resolve MSAA at identical dimensions before any scaling or particle writes.
            # Never silently replace missing obstacle depth with a cleared buffer.
            def copy_depth(src, dst, src_rect, dst_rect):
                for _ in range(16):
                    if gl.glGetError() == GL_NO_ERROR:
                        break
                gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, src)
                gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst)
                gl.glBlitFramebuffer(*src_rect, *dst_rect, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
                error = gl.glGetError()
                if error != GL_NO_ERROR:
                    raise RuntimeError(f"SSFR scene depth transfer failed (GL 0x{error:04x})")
            full_rect = (0, 0, vp_w, vp_h)
            copy_depth(prev_draw_fbo.value, data["fbo_scene_col"].value,
                       (vp_x, vp_y, vp_x + vp_w, vp_y + vp_h), full_rect)
            copy_depth(data["fbo_scene_col"].value, data["fbo_thick"].value,
                       full_rect, (0, 0, data["half_w"], data["half_h"]))

        # -------------------------------------------------------------------
        # Pass 1: Eye-Space Sphere Depth Pass (FBO_Depth) with Early-Z Occlusion
        # -------------------------------------------------------------------
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_depth"].value)
        gl.glViewport(0, 0, vp_w, vp_h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)  # z_eye >= 0.0 is background sentinel
        gl.glDepthMask(1)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if use_depth_occlusion:
            copy_depth(data["fbo_scene_col"].value, data["fbo_depth"].value, full_rect, full_rect)
            gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_depth"].value)

        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthFunc(GL_LEQUAL)
        gl.glDisable(GL_BLEND)
        gl.glEnable(GL_PROGRAM_POINT_SIZE)

        p_depth = data["prog_depth"]
        gl.glUseProgram(p_depth)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(p_depth, b"ModelViewMatrix"), 1, 0, mv_ptr)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(p_depth, b"ProjectionMatrix"), 1, 0, proj_ptr)
        gl.glUniform1f(gl.glGetUniformLocation(p_depth, b"ViewportHeight"), float(vp_h))
        gl.glUniform1f(gl.glGetUniformLocation(p_depth, b"ParticleRadius"), radius)

        gl_buf.bind_vao()
        gl.glDrawArrays(GL_POINTS, 0, gl_buf.num_particles)

        # -------------------------------------------------------------------
        # Pass 2: Half-Resolution Optical Thickness Pass (W/2 x H/2, GL_ONE + GL_ONE)
        # with solid scene obstacle occlusion
        # -------------------------------------------------------------------
        half_w, half_h = data["half_w"], data["half_h"]
        gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_thick"].value)
        gl.glViewport(0, 0, half_w, half_h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glDepthMask(1)
        gl.glClear(GL_COLOR_BUFFER_BIT)
        if not use_depth_occlusion:
            gl.glClear(GL_DEPTH_BUFFER_BIT)

        if use_depth_occlusion:
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(GL_LEQUAL)
        else:
            gl.glDisable(GL_DEPTH_TEST)
        gl.glDepthMask(0)
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_ONE, GL_ONE)

        p_thick = data["prog_thick"]
        gl.glUseProgram(p_thick)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(p_thick, b"ModelViewMatrix"), 1, 0, mv_ptr)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(p_thick, b"ProjectionMatrix"), 1, 0, proj_ptr)
        gl.glUniform1f(gl.glGetUniformLocation(p_thick, b"ViewportHeight"), float(half_h))
        gl.glUniform1f(gl.glGetUniformLocation(p_thick, b"ParticleRadius"), radius * 1.35)

        gl.glDrawArrays(GL_POINTS, 0, gl_buf.num_particles)
        gl_buf.unbind_vao()

        # -------------------------------------------------------------------
        # Pass 3: Perspective-Scaled Narrow-Range Depth Filter (Full Resolution)
        # -------------------------------------------------------------------
        gl.glViewport(0, 0, vp_w, vp_h)
        gl.glDisable(GL_BLEND)
        gl.glDisable(GL_PROGRAM_POINT_SIZE)
        p_flt = data["prog_filter"]
        gl.glUseProgram(p_flt)
        gl.glUniform1i(gl.glGetUniformLocation(p_flt, b"DepthTex"), 0)
        gl.glUniform1f(gl.glGetUniformLocation(p_flt, b"ParticleRadius"), radius)
        gl.glUniform1f(gl.glGetUniformLocation(p_flt, b"FilterSigmaPx"), sigma_px)
        loc_step = gl.glGetUniformLocation(p_flt, b"TexelStep")

        gl.glBindVertexArray(data["empty_vao"].value)
        gl.glActiveTexture(GL_TEXTURE0)

        curr_tex = data["tex_eye_depth"].value
        for _ in range(max(1, iterations)):
            # Horizontal 1D Pass -> FBO_Ping
            gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_ping"].value)
            gl.glBindTexture(GL_TEXTURE_2D, curr_tex)
            gl.glUniform2f(loc_step, 1.0 / float(vp_w), 0.0)
            gl.glDrawArrays(GL_TRIANGLES, 0, 3)

            # Vertical 1D Pass -> FBO_Pong
            gl.glBindFramebuffer(GL_FRAMEBUFFER, data["fbo_pong"].value)
            gl.glBindTexture(GL_TEXTURE_2D, data["tex_ping"].value)
            gl.glUniform2f(loc_step, 0.0, 1.0 / float(vp_h))
            gl.glDrawArrays(GL_TRIANGLES, 0, 3)
            curr_tex = data["tex_pong"].value

        # -------------------------------------------------------------------
        # Pass 4 & 5: Normal Reconstruction, Screen-Space Refraction & Fresnel Composite
        # -------------------------------------------------------------------
        gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_draw_fbo.value)
        gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read_fbo.value)
        gl.glViewport(vp_x, vp_y, vp_w, vp_h)
        gl.glDepthMask(1)
        if use_depth_occlusion:
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(GL_LEQUAL)
        else:
            gl.glDisable(GL_DEPTH_TEST)
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        p_comp = data["prog_comp"]
        gl.glUseProgram(p_comp)
        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(GL_TEXTURE_2D, curr_tex)
        gl.glUniform1i(gl.glGetUniformLocation(p_comp, b"SmoothedDepthTex"), 0)

        gl.glActiveTexture(GL_TEXTURE0 + 1)
        gl.glBindTexture(GL_TEXTURE_2D, data["tex_thick"].value)
        gl.glUniform1i(gl.glGetUniformLocation(p_comp, b"ThicknessTex"), 1)

        gl.glActiveTexture(GL_TEXTURE0 + 2)
        gl.glBindTexture(GL_TEXTURE_2D, data["tex_scene_col"].value)
        gl.glUniform1i(gl.glGetUniformLocation(p_comp, b"SceneColorTex"), 2)

        fx = 1.0 / max(abs(float(proj_mat[0][0])), 1e-4)
        fy = 1.0 / max(abs(float(proj_mat[1][1])), 1e-4)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(p_comp, b"ProjectionMatrix"), 1, 0, proj_ptr)
        gl.glUniform2f(gl.glGetUniformLocation(p_comp, b"InvResolution"), 1.0 / float(vp_w), 1.0 / float(vp_h))
        gl.glUniform2f(gl.glGetUniformLocation(p_comp, b"FocalScale"), fx, fy)
        gl.glUniform3f(gl.glGetUniformLocation(p_comp, b"AbsorptionRGB"), abs_rgb[0], abs_rgb[1], abs_rgb[2])
        gl.glUniform3f(gl.glGetUniformLocation(p_comp, b"FluidBaseColor"), base_col[0], base_col[1], base_col[2])
        gl.glUniform1i(gl.glGetUniformLocation(p_comp, b"HasSceneColor"), has_scene_color)

        gl.glDrawArrays(GL_TRIANGLES, 0, 3)

        # Cleanup GL state back to Blender's expectation
        gl.glBindVertexArray(0)
        for unit in (2, 1, 0):
            gl.glActiveTexture(GL_TEXTURE0 + unit)
            gl.glBindTexture(GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)

        for cap, was_on in saved.items():
            (gl.glEnable if was_on else gl.glDisable)(cap)

    def destroy_context(self, ctx: int):
        """Frees all OpenGL resources belonging to a specific GL context."""
        gl = _GLInterop
        if not gl._initialized:
            return
        data = self._ctx_resources.pop(ctx, None)
        if data is not None:
            for k in ("fbo_depth", "fbo_thick", "fbo_ping", "fbo_pong", "fbo_scene_col"):
                if data[k].value:
                    gl.glDeleteFramebuffers(1, ctypes.byref(data[k]))
            for k in ("tex_scene_depth", "tex_depth_hw", "tex_thick_depth_hw", "tex_eye_depth", "tex_thick", "tex_ping", "tex_pong", "tex_scene_col"):
                if data.get(k) and data[k].value:
                    gl.glDeleteTextures(1, ctypes.byref(data[k]))
            for k in ("prog_depth", "prog_thick", "prog_filter", "prog_comp", "prog_sphere"):
                if data[k]:
                    gl.glDeleteProgram(data[k])
            if data["empty_vao"].value:
                gl.glDeleteVertexArrays(1, ctypes.byref(data["empty_vao"]))

    def destroy(self):
        """Requests teardown of all viewport renderer GL resources without leaking across contexts."""
        gl = _GLInterop
        if not gl._initialized:
            return
        self._pending_destroy = True
        ctx = gl.current_context()
        if ctx in self._ctx_resources:
            self.destroy_context(ctx)

    def process_pending_cleanup(self):
        """Processes deferred destruction of GL resources within the active viewport draw context."""
        if not self._pending_destroy:
            return
        gl = _GLInterop
        if not gl._initialized:
            return
        ctx = gl.current_context()
        if ctx in self._ctx_resources:
            self.destroy_context(ctx)
        if not self._ctx_resources:
            self._pending_destroy = False


_SSFR_RENDERER = SSFRViewportRenderer()


def cleanup_viewport_renderer():
    _SSFR_RENDERER.destroy()


def draw_particles_viewport(gl_buf: ZeroCopyGLBuffer52, running: bool):
    """Zero-copy OpenGL Core Profile draw pass for Blender 5.2+ 3D Viewport."""
    if _SSFR_RENDERER._pending_destroy:
        _SSFR_RENDERER.process_pending_cleanup()

    if not running or gl_buf is None:
        return

    scene = bpy.context.scene
    if not getattr(scene, "warp_sph_show_particles", True):
        return

    render_mode = getattr(scene, "warp_sph_render_mode", "SSFR_LIQUID")

    if render_mode == "POINTS":
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.bind()
        mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        shader.uniform_float("ModelViewProjectionMatrix", mvp)
        color = tuple(getattr(scene, "warp_sph_particle_color", (0.12, 0.68, 1.0, 1.0)))
        shader.uniform_float("color", color)
        gl_buf.draw_points(
            getattr(scene, "warp_sph_point_size", 2.5),
            depth_test=getattr(scene, "warp_sph_depth_test", True),
            blend=color[3] < 1.0,
        )
        return

    # Query current 3D Viewport pixel rectangle
    vp = (ctypes.c_int * 4)()
    _GLInterop.ensure_initialized()
    _GLInterop.glGetIntegerv(GL_VIEWPORT, vp)
    vp_x, vp_y, vp_w, vp_h = vp[0], vp[1], max(1, vp[2]), max(1, vp[3])

    if render_mode == "SPHERES":
        _SSFR_RENDERER.draw_spheres(gl_buf, scene, vp_h)
    else:
        _SSFR_RENDERER.draw_ssfr_liquid(gl_buf, scene, vp_x, vp_y, vp_w, vp_h)
