import math
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ssfr_shaders import (
    SSFR_COMPOSITE_FRAG,
    SSFR_DEPTH_FRAG,
    SSFR_DEPTH_VERT,
    SSFR_FULLSCREEN_VERT,
    SSFR_NARROW_RANGE_FRAG,
    SSFR_THICKNESS_FRAG,
    compute_beer_lambert_transmittance,
    compute_projected_filter_sigma,
    compute_schlick_fresnel,
    narrow_range_filter_1d_cpu,
)


class TestSSFRRenderer(unittest.TestCase):
    def test_projected_filter_sigma_scales_with_depth(self):
        """Closer fluid particles (smaller |z_i|) should produce a larger screen-space filter sigma."""
        sigma_near = compute_projected_filter_sigma(1080, world_sigma=0.04, eye_depth_abs=1.5)
        sigma_far = compute_projected_filter_sigma(1080, world_sigma=0.04, eye_depth_abs=6.0)
        self.assertGreater(sigma_near, sigma_far)
        self.assertGreaterEqual(sigma_far, 1)

    def test_narrow_range_filter_smooths_bumps_and_preserves_silhouettes(self):
        """Verifies Truong & Yuksel Narrow-Range Filter reduces high-frequency noise while keeping step edges sharp."""
        r = 0.02
        # Create a flat eye-space surface around z = -2.0 with oscillatory particle bumps,
        # followed by a large depth discontinuity jump to z = -5.0
        x = np.linspace(0, 4 * math.pi, 32)
        surface_near = -2.0 + 0.008 * np.sin(x * 3.0)
        surface_far = np.full(32, -5.0, dtype=np.float32)
        depths = np.concatenate([surface_near, surface_far]).astype(np.float32)

        filtered = narrow_range_filter_1d_cpu(depths, particle_radius=r, sigma_px=3.0)

        # 1. Variance on the bumpy near surface should decrease significantly
        var_before = float(np.var(depths[:28]))
        var_after = float(np.var(filtered[:28]))
        self.assertLess(var_after, var_before * 0.5)

        # 2. Silhouette discontinuity at index 31 -> 32 must NOT bleed across the 3.0m gap;
        # with joint antipodal pairing, the boundary pixel 31 remains uncorrupted by the -5.0 drop.
        self.assertAlmostEqual(float(filtered[31]), float(depths[31]), places=3)
        self.assertAlmostEqual(float(filtered[32]), -5.0, places=4)

    def test_joint_antipodal_pairing_prevents_slope_bending_at_edges(self):
        """Verifies that a linear inclined ramp terminating at a cliff does not bend at the edge."""
        r = 0.02
        ramp = np.linspace(-2.0, -2.3, 16, dtype=np.float32)
        cliff = np.full(16, -6.0, dtype=np.float32)
        depths = np.concatenate([ramp, cliff])

        filtered = narrow_range_filter_1d_cpu(depths, particle_radius=r, sigma_px=3.0)
        # Interior of linear ramp stays linear (zero bias from symmetric antipodal pairs)
        np.testing.assert_allclose(filtered[4:11], ramp[4:11], atol=1e-4)

    def test_beer_lambert_transmittance_decay(self):
        """Verifies exponential Beer-Lambert attenuation across RGB channels."""
        absorption = (0.85, 0.22, 0.05)  # Red absorbed fastest, blue slowest
        t_zero = compute_beer_lambert_transmittance(absorption, thickness=0.0)
        t_deep = compute_beer_lambert_transmittance(absorption, thickness=2.0)

        self.assertEqual(t_zero, (1.0, 1.0, 1.0))
        self.assertLess(t_deep[0], t_deep[1])
        self.assertLess(t_deep[1], t_deep[2])

    def test_schlick_fresnel_grazing_vs_normal_incidence(self):
        """Verifies R0 (~0.02 for water IOR=1.333) at normal incidence and 1.0 at grazing angle."""
        f_normal = compute_schlick_fresnel(cos_theta=1.0, ior=1.333)
        f_grazing = compute_schlick_fresnel(cos_theta=0.0, ior=1.333)

        self.assertAlmostEqual(f_normal, 0.02037, places=4)
        self.assertAlmostEqual(f_grazing, 1.0, places=5)

    def test_glsl_shader_sources_valid_core_450_and_refraction(self):
        """Ensures all SSFR GLSL shader sources declare #version 450 core, main(), and refraction uniforms."""
        for shader_src in (
            SSFR_DEPTH_VERT,
            SSFR_DEPTH_FRAG,
            SSFR_THICKNESS_FRAG,
            SSFR_FULLSCREEN_VERT,
            SSFR_NARROW_RANGE_FRAG,
            SSFR_COMPOSITE_FRAG,
        ):
            self.assertIn("#version 450 core", shader_src)
            self.assertIn("void main()", shader_src)

        self.assertIn("SceneColorTex", SSFR_COMPOSITE_FRAG)
        self.assertIn("gl_FragDepth", SSFR_COMPOSITE_FRAG)
        self.assertIn("depth_scale", SSFR_NARROW_RANGE_FRAG)


if __name__ == "__main__":
    unittest.main()
