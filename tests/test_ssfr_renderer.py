"""SSFR math behavior and framebuffer-binding regression tests.

``viewport_draw`` is loaded under a unique module name while minimal Blender
stubs are scoped to that exec with ``patch.dict``; no fake modules are left in
``sys.modules``. The GL fake models only independent framebuffer bindings,
fresh names, and bound-object deletion; unrelated GL calls are harmless mocks.
"""

import ctypes
import importlib.util
import math
import os
import sys
import unittest
import unittest.mock
from types import ModuleType

import numpy as np

import warp_sph_realtime.render as render_pkg
from warp_sph_realtime.render.ssfr_shaders import (
    compute_beer_lambert_transmittance,
    compute_projected_filter_sigma,
    compute_schlick_fresnel,
    narrow_range_filter_1d_cpu,
)

_RENDER_DIR = os.path.dirname(render_pkg.__file__)


def _load_viewport_draw():
    ib_name = "warp_sph_realtime.render.interop_bridge"
    ib_spec = importlib.util.spec_from_file_location(ib_name, os.path.join(_RENDER_DIR, "interop_bridge.py"))
    ib_module = importlib.util.module_from_spec(ib_spec)
    spec = importlib.util.spec_from_file_location(
        "warp_sph_realtime.render._viewport_draw_under_test", os.path.join(_RENDER_DIR, "viewport_draw.py")
    )
    module = importlib.util.module_from_spec(spec)
    with unittest.mock.patch.dict(
        sys.modules,
        {"bpy": ModuleType("bpy"), "gpu": ModuleType("gpu"), ib_name: ib_module},
    ):
        ib_spec.loader.exec_module(ib_module)
        spec.loader.exec_module(module)
    return module


_VIEWPORT_DRAW = _load_viewport_draw()


class _StatefulGL:
    """Minimal fake for independent GL_DRAW/READ framebuffer state."""

    GL_FRAMEBUFFER = 0x8D40
    GL_DRAW_FRAMEBUFFER = 0x8CA9
    GL_READ_FRAMEBUFFER = 0x8CA8
    GL_DRAW_FRAMEBUFFER_BINDING = 0x8CA6
    GL_READ_FRAMEBUFFER_BINDING = 0x8CAA

    def __init__(self):
        self._next = 1
        self.draw_binding = self.read_binding = 0
        self.framebuffers = set()
        self.ensure_initialized = unittest.mock.Mock()
        self.current_context = unittest.mock.Mock(return_value="ssfr-test-context")
        self.glGenVertexArrays = unittest.mock.Mock(side_effect=self._gen)
        self.glGenTextures = unittest.mock.Mock(side_effect=self._gen)
        self.compile_program = unittest.mock.Mock(side_effect=self._fresh)
        self.glBindTexture = unittest.mock.Mock()
        self.glTexImage2D = unittest.mock.Mock()
        self.glTexParameteri = unittest.mock.Mock()
        self.glFramebufferTexture2D = unittest.mock.Mock()
        self.glDeleteTextures = unittest.mock.Mock()
        self.glGenFramebuffers = unittest.mock.Mock(side_effect=self._gen_framebuffers)
        self.glDeleteFramebuffers = unittest.mock.Mock(side_effect=self._delete_framebuffers)

    def _fresh(self, *_args):
        value = self._next
        self._next += 1
        return value

    def _gen(self, count, output):
        output = ctypes.cast(output, ctypes.POINTER(ctypes.c_uint))
        for i in range(count):
            output[i] = self._fresh()

    def _gen_framebuffers(self, count, output):
        output = ctypes.cast(output, ctypes.POINTER(ctypes.c_uint))
        for i in range(count):
            name = self._fresh()
            self.framebuffers.add(name)
            output[i] = name

    def _delete_framebuffers(self, count, framebuffers):
        framebuffers = ctypes.cast(framebuffers, ctypes.POINTER(ctypes.c_uint))
        for i in range(count):
            name = framebuffers[i]
            self.framebuffers.discard(name)
            if self.draw_binding == name:
                self.draw_binding = 0
            if self.read_binding == name:
                self.read_binding = 0

    def glBindFramebuffer(self, target, framebuffer):
        if framebuffer and framebuffer not in self.framebuffers:
            raise AssertionError(f"unknown framebuffer {framebuffer}")
        if target in (self.GL_FRAMEBUFFER, self.GL_DRAW_FRAMEBUFFER):
            self.draw_binding = framebuffer
        if target in (self.GL_FRAMEBUFFER, self.GL_READ_FRAMEBUFFER):
            self.read_binding = framebuffer

    def glGetIntegerv(self, pname, output):
        value = self.draw_binding if pname == self.GL_DRAW_FRAMEBUFFER_BINDING else self.read_binding
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = value


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


class SSFRFramebufferBindingRegression(unittest.TestCase):
    def setUp(self):
        self.gl = _StatefulGL()
        patcher = unittest.mock.patch.object(_VIEWPORT_DRAW, "_GLInterop", self.gl)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.renderer = _VIEWPORT_DRAW.SSFRViewportRenderer()

    def _external_framebuffers(self):
        values = (ctypes.c_uint * 2)()
        self.gl.glGenFramebuffers(2, values)
        return values[0], values[1]

    def test_allocation_preserves_distinct_draw_and_read_bindings(self):
        draw_fb, read_fb = self._external_framebuffers()
        self.gl.glBindFramebuffer(self.gl.GL_DRAW_FRAMEBUFFER, draw_fb)
        self.gl.glBindFramebuffer(self.gl.GL_READ_FRAMEBUFFER, read_fb)
        self.assertNotEqual(draw_fb, read_fb)

        self.renderer._get_or_create_ctx_data(64, 48)

        self.assertEqual(self.gl.draw_binding, draw_fb, "draw binding changed during allocation")
        self.assertEqual(self.gl.read_binding, read_fb, "read binding changed during allocation")

    def test_resize_restores_bindings_rebound_between_resizes(self):
        self.renderer._get_or_create_ctx_data(64, 48)
        draw_fb, read_fb = self._external_framebuffers()
        self.gl.glBindFramebuffer(self.gl.GL_DRAW_FRAMEBUFFER, draw_fb)
        self.gl.glBindFramebuffer(self.gl.GL_READ_FRAMEBUFFER, read_fb)

        data = self.renderer._get_or_create_ctx_data(32, 16)

        self.assertEqual((data["width"], data["height"]), (32, 16))
        self.assertEqual(self.gl.draw_binding, draw_fb, "draw binding changed during resize")
        self.assertEqual(self.gl.read_binding, read_fb, "read binding changed during resize")

    def test_allocation_on_default_framebuffer_stays_on_default(self):
        data = self.renderer._get_or_create_ctx_data(64, 48)
        allocated = {data[key].value for key in ("fbo_depth", "fbo_thick", "fbo_ping", "fbo_pong", "fbo_scene_col")}

        self.assertEqual(self.gl.draw_binding, 0, "default draw binding changed")
        self.assertEqual(self.gl.read_binding, 0, "default read binding changed")
        self.assertNotIn(0, allocated)


if __name__ == "__main__":
    unittest.main()
