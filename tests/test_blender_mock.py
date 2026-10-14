import os
import sys
import tomllib
import types
import unittest
import numpy as np
import warp as wp

# Mock Blender `bpy` and `gpu` modules if running outside of Blender's bundled Python
if "bpy" not in sys.modules:
    class MockScene:
        pass

    bpy_mock = types.ModuleType("bpy")
    bpy_mock.types = types.SimpleNamespace(
        Operator=object,
        Panel=object,
        Scene=MockScene,
        SpaceView3D=types.SimpleNamespace(
            draw_handler_add=lambda *a, **k: None,
            draw_handler_remove=lambda *a, **k: None,
        ),
    )
    bpy_props = types.ModuleType("bpy.props")
    for p in ("IntProperty", "FloatProperty", "BoolProperty", "EnumProperty", "StringProperty"):
        setattr(bpy_props, p, lambda **kw: kw)
    bpy_mock.props = bpy_props
    sys.modules["bpy"] = bpy_mock
    sys.modules["bpy.props"] = bpy_props

    _registered_classes = []
    bpy_mock.utils = types.SimpleNamespace(
        register_class=lambda cls: _registered_classes.append(cls),
        unregister_class=lambda cls: _registered_classes.remove(cls) if cls in _registered_classes else None,
    )

if "gpu" not in sys.modules:
    gpu_mock = types.ModuleType("gpu")
    gpu_mock.platform = types.SimpleNamespace(backend_type_get=lambda: "OPENGL")
    sys.modules["gpu"] = gpu_mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import __init__ as extension_module
from sim_core import RealtimeWCSPHSolver
from scene_config import REGIME_PRESETS, apply_regime_preset
from demo_scenes import launch_demo_scene_init
from interop_bridge import ZeroCopyGLBuffer52


class TestBlenderIntegrationAndDemos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_blender_manifest_toml_schema(self):
        """Validates that blender_manifest.toml adheres to Blender 5.2+ schema 1.0.0."""
        manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "blender_manifest.toml"))
        with open(manifest_path, "rb") as f:
            data = tomllib.load(f)

        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["id"], "warp_sph_realtime")
        self.assertEqual(data["blender_version_min"], "5.2.0")
        self.assertIn("windows-x64", data["platforms"])
        self.assertIn("linux-x64", data["platforms"])

    def test_extension_register_and_unregister(self):
        """Verifies full extension lifecycle: register creates UI panel & scene props, unregister cleans up."""
        extension_module.register()
        expected_props = [
            "warp_sph_demo_scene",
            "warp_sph_particle_count",
            "warp_sph_substeps",
            "warp_sph_point_size",
            "warp_sph_use_cuda_graph",
        ]
        for prop in expected_props:
            self.assertTrue(hasattr(bpy_mock.types.Scene, prop), f"Missing scene property {prop}")

        extension_module.unregister()
        for prop in expected_props:
            self.assertFalse(hasattr(bpy_mock.types.Scene, prop), f"Property {prop} not cleaned up after unregister")

    def test_interop_bridge_interface(self):
        """Verifies that ZeroCopyGLBuffer52 class and OpenGL constants are properly exposed."""
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "map_to_warp"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "unmap"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "draw_points"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "destroy"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_regime_presets_application(self):
        """Verifies that WATER, MILK, and SHOCKWAVE presets update solver physical parameters."""
        solver = RealtimeWCSPHSolver(num_particles=1024, h=0.04)

        for key in ("WATER", "MILK", "SHOCKWAVE"):
            apply_regime_preset(solver, key)
            self.assertEqual(solver.rest_density, REGIME_PRESETS[key]["rest_density"])
            self.assertEqual(solver.stiffness, REGIME_PRESETS[key]["stiffness"])
            self.assertEqual(solver.viscosity, REGIME_PRESETS[key]["viscosity"])
            self.assertEqual(solver.dt, REGIME_PRESETS[key]["dt"])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_demo_scene_1_double_dam_pillars_init(self):
        """Verifies Demo 1 initializes opposing velocity columns and enables SDF obstacles."""
        solver = RealtimeWCSPHSolver(num_particles=4096, h=0.04)
        launch_demo_scene_init(solver, "DOUBLE_DAM_PILLARS")

        self.assertEqual(solver.enable_obstacles, 1)
        vel_np = solver.vel.numpy()
        pos_np = solver.pos_sim.numpy()

        # First half moves +X (+1.2 m/s), second half moves -X (-1.2 m/s)
        self.assertAlmostEqual(vel_np[0, 0], 1.2, places=4)
        self.assertAlmostEqual(vel_np[-1, 0], -1.2, places=4)
        self.assertFalse(np.isnan(pos_np).any())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_demo_scene_2_block_flop_splash_init(self):
        """Verifies Demo 2 initializes a stationary floor puddle and a falling upper block."""
        solver = RealtimeWCSPHSolver(num_particles=4096, h=0.04)
        launch_demo_scene_init(solver, "BLOCK_FLOP_SPLASH")

        self.assertEqual(solver.enable_obstacles, 0)
        vel_np = solver.vel.numpy()
        pos_np = solver.pos_sim.numpy()

        # Puddle particles (index 0) start at rest near floor; drop block (index -1) falls at -3.5 m/s
        self.assertAlmostEqual(vel_np[0, 2], 0.0, places=4)
        self.assertAlmostEqual(vel_np[-1, 2], -3.5, places=4)
        self.assertGreater(pos_np[-1, 2], 2.0)


if __name__ == "__main__":
    unittest.main()
