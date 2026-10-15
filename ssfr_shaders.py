"""GLSL 450 Core Shaders & Mathematical Helpers for Screen-Space Fluid Rendering (SSFR).

Implements:
1. Eye-Space Sphere Impostor Depth Pass (with early depth discard outside unit circle)
2. Half-Resolution Additive Optical Thickness Accumulation Pass
3. Truong & Yuksel (i3D 2018) 1D Separable Narrow-Range Depth Filter with
   Perspective-Scaled Sigma & Joint Antipodal Acceptance
4. View-Space Normal Reconstruction, Screen-Space Background Refraction,
   Beer-Lambert Absorption, Schlick Fresnel & Hardware Depth Composite Pass
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Reference CPU Math Helpers (Used by UI Config & Automated Unit Tests)
# ---------------------------------------------------------------------------

def compute_projected_filter_sigma(
    viewport_height: int,
    world_sigma: float,
    eye_depth_abs: float,
    fov_y_rad: float = math.radians(50.0),
    max_radius_px: int = 32,
) -> int:
    """Truong & Yuksel Eq: sigma_i = ceil(H * sigma / (2 * |z_i| * tan(alpha / 2)))."""
    denom = 2.0 * max(abs(eye_depth_abs), 1e-4) * math.tan(0.5 * fov_y_rad)
    radius = int(math.ceil((viewport_height * world_sigma) / denom))
    return max(1, min(radius, max_radius_px))


def narrow_range_filter_1d_cpu(
    depths: np.ndarray,
    particle_radius: float,
    sigma_px: float = 4.0,
    mu_factor: float = 1.0,
    delta_factor: float = 10.0,
) -> np.ndarray:
    """Reference 1D Narrow-Range Filter with joint antipodal symmetric pairing and clamping."""
    n = len(depths)
    out = np.copy(depths).astype(np.float32)
    mu = mu_factor * particle_radius
    delta = delta_factor * particle_radius
    half_w = max(1, int(math.ceil(3.0 * sigma_px)))

    for i in range(n):
        zi = depths[i]
        if zi >= 0.0:  # Background sentinel (eye-space fluid depth is negative: z_i < 0)
            continue

        delta_high = mu
        delta_low = delta
        w_sum = 1.0
        z_acc = zi

        for d in range(1, half_w + 1):
            if i - d < 0 or i + d >= n:
                break
            w = math.exp(-0.5 * (d / max(sigma_px, 1e-4)) ** 2)
            za = depths[i - d]
            zb = depths[i + d]

            # Background pixels are ignored
            if za >= 0.0 or zb >= 0.0:
                continue

            low_bound = zi - delta_low
            high_bound = zi + delta_high

            # Joint antipodal pair acceptance & clamping (Truong & Yuksel 2018 Section 3):
            # Both symmetric neighbors must satisfy lower and upper neighborhood bounds
            # so grazing-angle surfaces do not warp toward or away from camera, and silhouettes
            # do not bleed across depth discontinuities.
            if za >= low_bound and zb >= low_bound and za <= high_bound + delta and zb <= high_bound + delta:
                zca = min(za, high_bound)
                zcb = min(zb, high_bound)
                z_acc += w * (zca + zcb)
                w_sum += 2.0 * w
                delta_high = max(delta_high, (zca - zi) + mu, (zcb - zi) + mu)
                delta_low = max(delta_low, (zi - zca) + delta, (zi - zcb) + delta)

        out[i] = z_acc / w_sum
    return out


def compute_beer_lambert_transmittance(
    absorption_rgb: tuple[float, float, float],
    thickness: float,
) -> tuple[float, float, float]:
    """Evaluates I(lambda)/I0(lambda) = exp(-a(lambda) * d(x,y))."""
    d = max(0.0, float(thickness))
    return (
        math.exp(-absorption_rgb[0] * d),
        math.exp(-absorption_rgb[1] * d),
        math.exp(-absorption_rgb[2] * d),
    )


def compute_schlick_fresnel(cos_theta: float, ior: float = 1.333) -> float:
    """Evaluates Schlick's Fresnel approximation F(theta) = R0 + (1 - R0)*(1 - cos_theta)^5."""
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    c = max(0.0, min(1.0, float(cos_theta)))
    return r0 + (1.0 - r0) * ((1.0 - c) ** 5)


# ---------------------------------------------------------------------------
# GLSL 450 Core Shader Sources for Blender 5.2+ OpenGL Backend
# ---------------------------------------------------------------------------

# Pass 1: Eye-Space Sphere Impostor Depth Shader
SSFR_DEPTH_VERT = """#version 450 core
layout(location = 0) in vec3 pos;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform float ViewportHeight;
uniform float ParticleRadius;

out vec3 v_eye_pos;
out float v_radius;

void main() {
    vec4 eye_pos = ModelViewMatrix * vec4(pos, 1.0);
    v_eye_pos = eye_pos.xyz;
    v_radius = ParticleRadius;
    gl_Position = ProjectionMatrix * eye_pos;

    // Support both Perspective and Orthographic camera projections
    bool is_ortho = (abs(ProjectionMatrix[3][3] - 1.0) < 0.1 || abs(ProjectionMatrix[3][2]) < 1e-4);
    float proj_scale = ProjectionMatrix[1][1] * ViewportHeight * 0.5;
    float dist = max(-eye_pos.z, 0.1);
    float point_size = is_ortho ? (ParticleRadius * 2.0 * proj_scale) : ((ParticleRadius * proj_scale) / dist);
    gl_PointSize = clamp(point_size, 1.5, 96.0);
}
"""

SSFR_DEPTH_FRAG = """#version 450 core
in vec3 v_eye_pos;
in float v_radius;

uniform mat4 ProjectionMatrix;

layout(location = 0) out float out_eye_depth;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }
    float z_offset = sqrt(1.0 - r2) * v_radius;
    float z_eye = v_eye_pos.z + z_offset; // Negative eye-space depth (z_eye < 0)

    vec4 clip_pos = ProjectionMatrix * vec4(v_eye_pos.xy, z_eye, 1.0);
    float ndc_depth = clip_pos.z / clip_pos.w;
    gl_FragDepth = ndc_depth * 0.5 + 0.5;

    out_eye_depth = z_eye;
}
"""

# Pass 2: Optical Thickness Accumulation Shader (Half-Res Additive Blending GL_ONE, GL_ONE)
SSFR_THICKNESS_FRAG = """#version 450 core
in vec3 v_eye_pos;
in float v_radius;

layout(location = 0) out float out_thickness;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }
    float dz = 2.0 * sqrt(1.0 - r2) * v_radius;
    out_thickness = dz;
}
"""

# Shared Fullscreen Triangle Vertex Shader for Post-Processing Passes (Passes 3 & 5)
SSFR_FULLSCREEN_VERT = """#version 450 core
out vec2 v_uv;

void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Pass 3: Truong & Yuksel (2018) 1D Separable Narrow-Range Depth Filter
# with Perspective Depth-Scaled Sigma and Joint Antipodal Pairing
SSFR_NARROW_RANGE_FRAG = """#version 450 core
in vec2 v_uv;

uniform sampler2D DepthTex;
uniform vec2 TexelStep;      // (1/W, 0) for horizontal pass, (0, 1/H) for vertical pass
uniform float ParticleRadius;
uniform float FilterSigmaPx;

layout(location = 0) out float out_filtered_depth;

void main() {
    float zi = texture(DepthTex, v_uv).r;
    if (zi >= -1e-4) {
        out_filtered_depth = 0.0;
        return;
    }

    float mu = 1.0 * ParticleRadius;
    float delta = 10.0 * ParticleRadius;
    float delta_high = mu;
    float delta_low = delta;

    // Perspective-scaled sigma: larger kernel close to camera, compact in distance
    float depth_scale = clamp(3.0 / max(-zi, 0.25), 0.35, 3.0);
    float sigma_eff = clamp(FilterSigmaPx * depth_scale, 1.0, 18.0);
    int radius = clamp(int(ceil(2.5 * sigma_eff)), 2, 24);

    float z_acc = zi;
    float w_sum = 1.0;

    for (int d = 1; d <= radius; ++d) {
        float w = exp(-0.5 * float(d * d) / max(sigma_eff * sigma_eff, 1.0));
        vec2 offset = TexelStep * float(d);

        float za = texture(DepthTex, v_uv - offset).r;
        float zb = texture(DepthTex, v_uv + offset).r;

        float low_bound = zi - delta_low;
        float high_bound = zi + delta_high;

        // Joint antipodal acceptance prevents surface bending on slopes and silhouettes
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

    out_filtered_depth = z_acc / w_sum;
}
"""

# Pass 4 & 5 Fused: Normal Reconstruction + Screen-Space Background Refraction +
# Beer-Lambert Absorption + Schlick Fresnel + Hardware gl_FragDepth Composite
SSFR_COMPOSITE_FRAG = """#version 450 core
in vec2 v_uv;

uniform sampler2D SmoothedDepthTex;
uniform sampler2D ThicknessTex;
uniform sampler2D SceneColorTex;
uniform mat4 ProjectionMatrix;
uniform vec2 InvResolution;
uniform vec2 FocalScale;     // (1.0 / Proj[0][0], 1.0 / Proj[1][1])
uniform vec3 AbsorptionRGB;  // Beer-Lambert wavelength absorption e.g. (0.85, 0.22, 0.05)
uniform vec3 FluidBaseColor; // Liquid scattering tint
uniform int HasSceneColor;   // 1 if background scene color blit succeeded

layout(location = 0) out vec4 out_color;

vec3 uv_to_eye(vec2 uv, float z_eye) {
    vec2 ndc = uv * 2.0 - 1.0;
    bool is_ortho = (abs(ProjectionMatrix[3][3] - 1.0) < 0.1 || abs(ProjectionMatrix[3][2]) < 1e-4);
    vec2 eye_xy = is_ortho ? (ndc * FocalScale) : (ndc * FocalScale * (-z_eye));
    return vec3(eye_xy, z_eye);
}

void main() {
    float zc = texture(SmoothedDepthTex, v_uv).r;
    if (zc >= -1e-4) {
        discard;
    }

    float thick = max(texture(ThicknessTex, v_uv).r, 0.01);
    vec3 P = uv_to_eye(v_uv, zc);

    // Output hardware depth from smoothed eye-space surface so solid scene meshes occlude liquid
    vec4 clip_pos = ProjectionMatrix * vec4(P, 1.0);
    gl_FragDepth = (clip_pos.z / clip_pos.w) * 0.5 + 0.5;

    // Finite-difference eye-space normal reconstruction with edge discontinuity selection
    vec2 dx = vec2(InvResolution.x, 0.0);
    vec2 dy = vec2(0.0, InvResolution.y);

    float zl = texture(SmoothedDepthTex, v_uv - dx).r;
    float zr = texture(SmoothedDepthTex, v_uv + dx).r;
    float zb = texture(SmoothedDepthTex, v_uv - dy).r;
    float zt = texture(SmoothedDepthTex, v_uv + dy).r;

    vec3 dPdx = (abs(zr - zc) < abs(zc - zl) && zr < -1e-4)
        ? (uv_to_eye(v_uv + dx, zr) - P)
        : (P - uv_to_eye(v_uv - dx, zl < -1e-4 ? zl : zc));

    vec3 dPdy = (abs(zt - zc) < abs(zc - zb) && zt < -1e-4)
        ? (uv_to_eye(v_uv + dy, zt) - P)
        : (P - uv_to_eye(v_uv - dy, zb < -1e-4 ? zb : zc));

    vec3 N = normalize(cross(dPdx, dPdy));
    if (N.z < 0.0) {
        N = -N;
    }

    vec3 V = normalize(-P);
    vec3 L = normalize(vec3(0.45, 0.55, 0.70));
    vec3 H = normalize(L + V);

    // Beer-Lambert Law Transmittance: T(lambda) = exp(-a(lambda) * d(x,y))
    vec3 transmittance = exp(-AbsorptionRGB * (thick * 4.5));

    // Screen-Space Background Refraction Distortion: u_refr = u + N_xy * d(x,y) * kappa
    vec2 refr_offset = N.xy * clamp(thick * 0.045, 0.002, 0.06);
    vec2 refr_uv = clamp(v_uv + refr_offset, vec2(0.001), vec2(0.999));

    vec3 bg_col = FluidBaseColor;
    if (HasSceneColor != 0) {
        vec3 scene_sample = texture(SceneColorTex, refr_uv).rgb;
        bg_col = mix(FluidBaseColor, scene_sample, 0.72);
    }
    vec3 refracted_color = bg_col * transmittance;

    // Schlick Fresnel (IOR = 1.333 -> R0 = 0.02037)
    float cos_theta = clamp(dot(N, V), 0.0, 1.0);
    float r0 = 0.02037;
    float fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

    // Sky/Studio hemisphere reflection + Blinn-Phong specular highlight
    vec3 R = reflect(-V, N);
    vec3 sky_col = mix(vec3(0.18, 0.24, 0.32), vec3(0.82, 0.90, 0.98), clamp(R.y * 0.5 + 0.5, 0.0, 1.0));
    float spec = pow(max(dot(N, H), 0.0), 96.0);

    // Subtle whitewater thickness crest highlight
    float foam_factor = smoothstep(0.35, 0.95, thick * 2.2) * 0.18;

    vec3 final_rgb = mix(refracted_color, sky_col, fresnel) + vec3(spec * 0.85) + vec3(foam_factor);
    float alpha = HasSceneColor != 0
        ? 0.98
        : clamp(1.0 - dot(transmittance, vec3(0.333)) * 0.25 + fresnel, 0.75, 0.98);

    out_color = vec4(final_rgb, alpha);
}
"""
