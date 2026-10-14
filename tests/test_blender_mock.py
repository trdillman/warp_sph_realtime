import os
import sys
import tomllib
import types
import unittest
import numpy as np
import warp as wp

# Mock Blender `bpy` and `gpu` modules if running outside of Blender's bundled Python
if "bpy" not in sys.modules:
    bpy_mock = types.ModuleType("bpy")
    bpy_mock.types = types.SimpleNamespace(Operator=object, Panel=object, Scene=object)
    bpy_mock.props = types.SimpleNamespace(
        IntProperty=lambda **kw: kw,
        FloatProperty=lambda **kw: kw,
        BoolProperty=lambda **kw: kw,
        EnumProperty=lambda **kw: kw,
        StringProperty=lambda **kw: kw,
    )
    sys.modules["bpy"] = bpy_mock

if "gpu" not in sys.modules:
    gpu_mock = types.ModuleType("gpu")
    gpu_mock.platform = types.SimpleNamespace(backend_type_get=lambda: "OPENGL")
    sys.modules["gpu"] = gpu_mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim_core import RealtimeWCSPHSolver
from scene_config import REGIME_PRESETS, apply_regime_preset
from demo_scenes import launch_demo_scene_init


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
