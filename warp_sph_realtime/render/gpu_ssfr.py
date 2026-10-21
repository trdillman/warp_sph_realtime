"""Backend-neutral screen-space fluid rendering for transport frames (release plan P0.13).

Ports the legacy raw-GL SSFR passes (``viewport_draw.SSFRViewportRenderer``) to the public
``gpu`` module so the liquid look runs on Vulkan as well as OpenGL:

1. depth: instanced sphere-impostor quads (positions by texelFetch on gpu_InstanceIndex)
   write eye-space depth into a full-resolution RGBA32F GPUOffScreen (0 = background);
2. thickness: the same quads at half resolution, radius x1.35, additively accumulated;
3. filter: separable narrow-range depth filter, ping-pong, ``filter_iterations`` x (H, V);
4. composite: normal reconstruction, Beer-Lambert, Schlick Fresnel, sky, specular and foam
   onto the viewport with gl_FragDepth and the scene depth test.

The gpu module exposes no viewport colour texture without CPU readback, so the legacy
refraction (sampling the scene at an offset UV) is replaced by two blend passes that apply
the same transmitted-scene and own-colour terms in place: MULTIPLY by the transmitted
scene share, then ADDITIVE_PREMULT of the liquid's own colour. The refraction offset is
therefore not reproduced. Sprite size keeps the legacy point-size formula exactly.
"""
from __future__ import annotations

from typing import Any, cast

from .gpu_points import CORNERS, SSFR, PointRenderer

FULLSCREEN = ((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0))
THICKNESS_RADIUS_SCALE = 1.35

SPRITE_VERTEX = """
void main() {
    int i = gpu_InstanceIndex;
    vec4 world = vec4(texelFetch(positions, ivec2(i % 4096, i / 4096), 0).xyz, 1.0);
    vec3 eye = vec3(dot(u_view0, world), dot(u_view1, world), dot(u_view2, world));
    float radius = u_params.x;
    bool is_ortho = abs(u_proj[3][3] - 1.0) < 0.1 || abs(u_proj[3][2]) < 1e-4;
    float proj_scale = u_proj[1][1] * u_params.y * 0.5;
    float size_px = is_ortho ? radius * 2.0 * proj_scale : radius * proj_scale / max(-eye.z, 0.1);
    size_px = clamp(size_px, 1.5, 96.0);
    vec4 clip = u_proj * vec4(eye, 1.0);
    clip.xy += corner * size_px / vec2(u_params.z, u_params.y) * clip.w;
    gl_Position = clip;
    uv = corner;
    centre = vec4(eye, radius);
}
"""

DEPTH_FRAGMENT = """
void main() {
    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        discard;
    }
    float z_eye = centre.z + sqrt(1.0 - r2) * centre.w;
    vec4 clip = u_proj * vec4(centre.xy, z_eye, 1.0);
    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
    fragColor = vec4(z_eye, 0.0, 0.0, 1.0);
}
"""

THICKNESS_FRAGMENT = """
void main() {
    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        discard;
    }
    fragColor = vec4(2.0 * sqrt(1.0 - r2) * centre.w, 0.0, 0.0, 0.0);
}
"""

FULLSCREEN_VERTEX = """
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

FILTER_FRAGMENT = """
float fetch_depth(ivec2 p, ivec2 size) {
    return texelFetch(src, clamp(p, ivec2(0), size - 1), 0).r;
}

void main() {
    ivec2 size = textureSize(src, 0);
    ivec2 px = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
    float zi = texelFetch(src, px, 0).r;
    if (zi >= -1e-4) {
        fragColor = vec4(0.0);
        return;
    }
    float radius = u_params.x;
    float sigma = u_params.y;
    ivec2 dir = ivec2(u_params.zw);
    float mu = radius;
    float delta = 10.0 * radius;
    float delta_high = mu;
    float delta_low = delta;
    float depth_scale = clamp(3.0 / max(-zi, 0.25), 0.35, 3.0);
    float sigma_eff = clamp(sigma * depth_scale, 1.0, 18.0);
    int taps = clamp(int(ceil(2.5 * sigma_eff)), 2, 24);
    float z_acc = zi;
    float w_sum = 1.0;
    for (int d = 1; d <= taps; ++d) {
        float w = exp(-0.5 * float(d * d) / max(sigma_eff * sigma_eff, 1.0));
        float za = fetch_depth(px - dir * d, size);
        float zb = fetch_depth(px + dir * d, size);
        float low_bound = zi - delta_low;
        float high_bound = zi + delta_high;
        if (za < -1e-4 && zb < -1e-4 && za >= low_bound && zb >= low_bound &&
            za <= high_bound + delta && zb <= high_bound + delta) {
            float zca = min(za, high_bound);
            float zcb = min(zb, high_bound);
            z_acc += w * (zca + zcb);
            w_sum += 2.0 * w;
            delta_high = max(delta_high, max((zca - zi) + mu, (zcb - zi) + mu));
            delta_low = max(delta_low, max((zi - zca) + delta, (zi - zcb) + delta));
        }
    }
    fragColor = vec4(z_acc / w_sum, 0.0, 0.0, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
float depth_at(ivec2 p, ivec2 size) {
    return texelFetch(depth_tex, clamp(p, ivec2(0), size - 1), 0).r;
}

float thickness_at(vec2 st) {
    ivec2 size = textureSize(thick_tex, 0);
    ivec2 hi = size - 1;
    vec2 f = st * vec2(size) - 0.5;
    ivec2 p = ivec2(floor(f));
    vec2 t = f - vec2(p);
    float a = texelFetch(thick_tex, clamp(p, ivec2(0), hi), 0).r;
    float b = texelFetch(thick_tex, clamp(p + ivec2(1, 0), ivec2(0), hi), 0).r;
    float c = texelFetch(thick_tex, clamp(p + ivec2(0, 1), ivec2(0), hi), 0).r;
    float d = texelFetch(thick_tex, clamp(p + ivec2(1, 1), ivec2(0), hi), 0).r;
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

vec3 uv_to_eye(vec2 st, float z_eye) {
    vec2 ndc = st * 2.0 - 1.0;
    bool is_ortho = abs(u_proj[3][3] - 1.0) < 0.1 || abs(u_proj[3][2]) < 1e-4;
    vec2 eye_xy = is_ortho ? ndc * u_focal.xy : ndc * u_focal.xy * (-z_eye);
    return vec3(eye_xy, z_eye);
}

void main() {
    ivec2 size = textureSize(depth_tex, 0);
    ivec2 px = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
    float zc = depth_at(px, size);
    if (zc >= -1e-4) {
        discard;
    }
    vec2 texel = 1.0 / vec2(size);
    vec2 st = (vec2(px) + 0.5) * texel;
    float thick = max(thickness_at(st), 0.01);
    vec3 P = uv_to_eye(st, zc);
    vec4 clip = u_proj * vec4(P, 1.0);
    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;

    float zl = depth_at(px - ivec2(1, 0), size);
    float zr = depth_at(px + ivec2(1, 0), size);
    float zb = depth_at(px - ivec2(0, 1), size);
    float zt = depth_at(px + ivec2(0, 1), size);
    vec2 dx = vec2(texel.x, 0.0);
    vec2 dy = vec2(0.0, texel.y);
    vec3 dPdx = (abs(zr - zc) < abs(zc - zl) && zr < -1e-4)
        ? (uv_to_eye(st + dx, zr) - P)
        : (P - uv_to_eye(st - dx, zl < -1e-4 ? zl : zc));
    vec3 dPdy = (abs(zt - zc) < abs(zc - zb) && zt < -1e-4)
        ? (uv_to_eye(st + dy, zt) - P)
        : (P - uv_to_eye(st - dy, zb < -1e-4 ? zb : zc));
    vec3 N = normalize(cross(dPdx, dPdy));
    if (N.z < 0.0) {
        N = -N;
    }

    vec3 V = normalize(-P);
    vec3 L = normalize(vec3(0.45, 0.55, 0.70));
    vec3 H = normalize(L + V);
    vec3 transmittance = exp(-u_absorb.rgb * (thick * 4.5));
    float cos_theta = clamp(dot(N, V), 0.0, 1.0);
    float r0 = 0.02037;
    float fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
    vec3 R = reflect(-V, N);
    vec3 sky = mix(vec3(0.18, 0.24, 0.32), vec3(0.82, 0.90, 0.98), clamp(R.y * 0.5 + 0.5, 0.0, 1.0));
    float spec = pow(max(dot(N, H), 0.0), 96.0);
    float foam = smoothstep(0.35, 0.95, thick * 2.2) * 0.18;

    // Legacy: 0.98 * (mix(mix(base, scene, 0.72) * T, sky, F) + spec + foam) + 0.02 * scene,
    // split into a MULTIPLY pass (scene terms) and an ADDITIVE_PREMULT pass (own terms).
    if (u_absorb.w < 0.5) {
        fragColor = vec4(vec3(0.98 * 0.72) * (1.0 - fresnel) * transmittance + vec3(0.02), 1.0);
    } else {
        vec3 own = (1.0 - fresnel) * transmittance * 0.28 * u_base.rgb + fresnel * sky + vec3(spec * 0.85 + foam);
        fragColor = vec4(0.98 * own, 0.0);
    }
}
"""


def view_rows(view) -> list:
    """The first three model-view rows: eye = (row0 . p, row1 . p, row2 . p)."""
    return [tuple(float(view[r][c]) for c in range(4)) for r in range(3)]


def filter_iterations(settings) -> int:
    return max(1, int(settings.filter_iterations))


class _Targets:
    """Offscreens for one viewport size: eye depth, half-res thickness and the filter ping-pong."""

    def __init__(self, gpu, width: int, height: int):
        self.size = (width, height)
        self.half = (max(1, width // 2), max(1, height // 2))
        made: list = []
        try:
            for w, h, fmt in ((width, height, 'RGBA32F'), (*self.half, 'RGBA16F'),
                              (width, height, 'RGBA32F'), (width, height, 'RGBA32F')):
                made.append(gpu.types.GPUOffScreen(w, h, format=fmt))
        except BaseException:
            for offscreen in made:
                offscreen.free()
            raise
        self.depth, self.thick, self.ping, self.pong = made

    def offscreens(self) -> tuple:
        return (self.depth, self.thick, self.ping, self.pong)

    def free(self):
        for offscreen in self.offscreens():
            offscreen.free()


class SSFRRenderer:
    """Draws a transport Frame as liquid in the current viewport; compiles lazily on first draw.

    One target set is kept and reallocated whenever the viewport size changes.
    """

    def __init__(self, gpu_module: Any = None, batch_factory: Any = None):
        self._gpu = gpu_module
        self._batch_factory = batch_factory
        self._shaders: dict = {}
        self._batches: dict = {}
        self._targets: _Targets | None = None

    def _shader(self, name: str, vertex_in: str, samplers: tuple, push: tuple, flat: bool,
                vertex: str, fragment: str, depth_write: bool):
        gpu = self._gpu
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', vertex_in)
        for slot, sampler in enumerate(samplers):
            info.sampler(slot, 'FLOAT_2D', sampler)
        for kind, uniform in push:
            info.push_constant(kind, uniform)
        iface = gpu.types.GPUStageInterfaceInfo(f'warp_sph_ssfr_{name}')
        iface.smooth('VEC2', 'uv')
        if flat:
            iface.flat('VEC4', 'centre')
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', 'fragColor')
        if depth_write:
            info.depth_write('ANY')
        info.vertex_source(vertex)
        info.fragment_source(fragment)
        shader = gpu.shader.create_from_info(info)
        self._shaders[name] = shader
        attrs = {vertex_in: CORNERS if vertex_in == 'corner' else FULLSCREEN}
        self._batches[name] = self._batch_factory(shader, 'TRIS', attrs)

    def _build(self):
        sprite_push = (('MAT4', 'u_proj'), ('VEC4', 'u_view0'), ('VEC4', 'u_view1'), ('VEC4', 'u_view2'),
                       ('VEC4', 'u_params'))  # 128 bytes
        self._shader('depth', 'corner', ('positions',), sprite_push, True, SPRITE_VERTEX, DEPTH_FRAGMENT, True)
        self._shader('thickness', 'corner', ('positions',), sprite_push, True, SPRITE_VERTEX,
                     THICKNESS_FRAGMENT, False)
        self._shader('filter', 'pos', ('src',), (('VEC4', 'u_params'),), False, FULLSCREEN_VERTEX,
                     FILTER_FRAGMENT, False)
        self._shader('composite', 'pos', ('depth_tex', 'thick_tex'),
                     (('MAT4', 'u_proj'), ('VEC4', 'u_absorb'), ('VEC4', 'u_base'), ('VEC4', 'u_focal')),
                     False, FULLSCREEN_VERTEX, COMPOSITE_FRAGMENT, True)

    def _ensure_targets(self, width: int, height: int) -> _Targets:
        if self._targets is not None and self._targets.size != (width, height):
            self._targets.free()
            self._targets = None
        if self._targets is None:
            self._targets = _Targets(self._gpu, width, height)
        return self._targets

    def _sprites(self, name: str, frame, proj, rows, params):
        shader = self._shaders[name]
        shader.bind()
        shader.uniform_sampler('positions', frame.texture)
        shader.uniform_float('u_proj', proj)
        for i, row in enumerate(rows):
            shader.uniform_float(f'u_view{i}', row)
        shader.uniform_float('u_params', params)
        self._batches[name].draw_instanced(shader, instance_count=frame.count)

    def draw(self, frame, settings) -> bool:
        if frame is None or frame.count <= 0:
            return False
        if self._gpu is None:
            import gpu
            self._gpu = gpu
        if self._batch_factory is None:
            from gpu_extras.batch import batch_for_shader
            self._batch_factory = batch_for_shader
        gpu = self._gpu
        viewport = tuple(int(v) for v in cast(tuple, gpu.state.viewport_get()))
        x, y, width, height = viewport
        if width <= 0 or height <= 0:
            return False
        if not self._shaders:
            self._build()
        targets = self._ensure_targets(width, height)
        proj = gpu.matrix.get_projection_matrix()
        rows = view_rows(gpu.matrix.get_model_view_matrix())
        radius = float(settings.size)
        state = gpu.state
        saved = (state.depth_test_get(), state.depth_mask_get(), state.blend_get())
        try:
            with targets.depth.bind():
                state.viewport_set(0, 0, width, height)
                state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
                state.depth_test_set('LESS_EQUAL')
                state.depth_mask_set(True)
                state.blend_set('NONE')
                self._sprites('depth', frame, proj, rows, (radius, float(height), float(width), 0.0))

            half_w, half_h = targets.half
            with targets.thick.bind():
                state.viewport_set(0, 0, half_w, half_h)
                state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0))
                state.depth_test_set('NONE')
                state.depth_mask_set(False)
                state.blend_set('ADDITIVE_PREMULT')
                self._sprites('thickness', frame, proj, rows,
                              (radius * THICKNESS_RADIUS_SCALE, float(half_h), float(half_w), 0.0))

            state.blend_set('NONE')
            shader, batch = self._shaders['filter'], self._batches['filter']
            current = targets.depth
            for _ in range(filter_iterations(settings)):
                for dst, src, direction in ((targets.ping, current, (1.0, 0.0)),
                                            (targets.pong, targets.ping, (0.0, 1.0))):
                    with dst.bind():
                        state.viewport_set(0, 0, width, height)
                        shader.bind()
                        shader.uniform_sampler('src', src.texture_color)
                        shader.uniform_float('u_params', (radius, float(settings.filter_sigma), *direction))
                        batch.draw(shader)
                current = targets.pong

            state.viewport_set(x, y, width, height)
            state.depth_test_set('LESS_EQUAL' if settings.depth_test else 'NONE')
            state.depth_mask_set(bool(settings.depth_test))
            shader, batch = self._shaders['composite'], self._batches['composite']
            focal = (1.0 / max(abs(float(proj[0][0])), 1e-4), 1.0 / max(abs(float(proj[1][1])), 1e-4), 0.0, 0.0)
            absorb = tuple(float(c) for c in settings.absorption)[:3]
            for stage, blend in ((0.0, 'MULTIPLY'), (1.0, 'ADDITIVE_PREMULT')):
                state.blend_set(blend)
                shader.bind()
                shader.uniform_sampler('depth_tex', current.texture_color)
                shader.uniform_sampler('thick_tex', targets.thick.texture_color)
                shader.uniform_float('u_proj', proj)
                shader.uniform_float('u_absorb', (*absorb, stage))
                shader.uniform_float('u_base', tuple(float(c) for c in settings.color))
                shader.uniform_float('u_focal', focal)
                batch.draw(shader)
        finally:
            state.depth_test_set(saved[0])
            state.depth_mask_set(saved[1])
            state.blend_set(saved[2])
            state.viewport_set(x, y, width, height)
        return True

    def free(self):
        targets, self._targets = self._targets, None
        self._shaders, self._batches = {}, {}
        if targets is not None:
            targets.free()


class ViewportRenderer:
    """The session renderer: SSFR mode draws liquid, other modes draw points or spheres."""

    def __init__(self, gpu_module: Any = None, batch_factory: Any = None, *, points=None, ssfr=None):
        self.points = PointRenderer(gpu_module, batch_factory) if points is None else points
        self.ssfr = SSFRRenderer(gpu_module, batch_factory) if ssfr is None else ssfr

    def draw(self, frame, settings) -> bool:
        return (self.ssfr if settings.mode == SSFR else self.points).draw(frame, settings)

    def free(self):
        try:
            self.ssfr.free()
        finally:
            self.points.free()
