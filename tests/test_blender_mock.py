import os
import sys
import tomllib
import types
import unittest

import numpy as np
import warp as wp


def _running_inside_blender() -> bool:
    # Type-stub packages (e.g. fake-bpy-module) are importable but have no runtime `bpy.app`.
    try:
        import bpy
        return hasattr(bpy, "app") and hasattr(bpy.app, "version")
    except ImportError:
        return False


# Mock Blender `bpy` and `gpu` modules if running outside of Blender's bundled Python
if not _running_inside_blender():
    class MockScene:
        pass

    class MockObject:
        pass

    bpy_mock = types.ModuleType("bpy")
    bpy_mock.types = types.SimpleNamespace(
        Operator=object,
        Panel=object,
        AddonPreferences=object,
        PropertyGroup=object,
        Scene=MockScene,
        Object=MockObject,
        SpaceView3D=types.SimpleNamespace(
            draw_handler_add=lambda *a, **k: None,
            draw_handler_remove=lambda *a, **k: None,
        ),
    )
    bpy_props = types.ModuleType("bpy.props")
    for p in ("IntProperty", "FloatProperty", "FloatVectorProperty", "BoolProperty", "EnumProperty", "StringProperty",
              "PointerProperty"):
        setattr(bpy_props, p, lambda **kw: kw)
    bpy_mock.props = bpy_props
    sys.modules["bpy"] = bpy_mock
    sys.modules["bpy.props"] = bpy_props

    _registered_classes = []
    bpy_mock.utils = types.SimpleNamespace(
        register_class=lambda cls: _registered_classes.append(cls),
        unregister_class=lambda cls: _registered_classes.remove(cls) if cls in _registered_classes else None,
    )

    gpu_mock = types.ModuleType("gpu")
    gpu_mock.platform = types.SimpleNamespace(backend_type_get=lambda: "OPENGL")
    sys.modules["gpu"] = gpu_mock

import warp_sph_realtime as extension_module
from warp_sph_realtime.render.interop_bridge import ZeroCopyGLBuffer52, wrap_device_vec3
from warp_sph_realtime.scene.demo_scenes import launch_demo_scene_init
from warp_sph_realtime.scene.scene_config import REGIME_PRESETS, apply_regime_preset
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver


class TestBlenderIntegrationAndDemos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_blender_manifest_toml_schema(self):
        """Validates that blender_manifest.toml adheres to Blender 5.2+ schema 1.0.0."""
        manifest_path = os.path.join(os.path.dirname(extension_module.__file__), "blender_manifest.toml")
        with open(manifest_path, "rb") as f:
            data = tomllib.load(f)

        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["id"], "warp_sph_realtime")
        self.assertEqual(data["blender_version_min"], "5.2.0")
        self.assertIn("windows-x64", data["platforms"])
        self.assertIn("linux-x64", data["platforms"])

    def test_extension_register_and_unregister(self):
        """Verifies full extension lifecycle: register creates UI panel & scene props, unregister cleans up."""
        from warp_sph_realtime import props
        extension_module.register()
        expected_props = [
            "demo_scene",
            "particle_count",
            "substeps",
            "point_size",
            "use_cuda_graph",
            "time_step_scale",
            "stiffness",
            "show_particles",
            "render_mode",
            "particle_radius",
            "filter_iterations",
            "filter_sigma",
            "absorption",
            "particle_color",
            "depth_test",
            "scene_source",
            "particle_spacing",
        ]
        self.assertIs(bpy_mock.types.Scene.warp_sph["type"], props.WARPSPH_SceneSettings)
        settings = props.WARPSPH_SceneSettings.__annotations__
        cache_props = ["cache_dir", "bake_start", "bake_end", "bake_steps"]  # P2.2: new, nothing to migrate
        self.assertEqual(sorted(settings), sorted(expected_props + cache_props))
        self.assertEqual(sorted(props.LEGACY_NAMES), sorted(expected_props))  # every old flat setting migrates
        self.assertFalse([n for n in vars(bpy_mock.types.Scene) if n.startswith("warp_sph_")])
        self.assertTrue(hasattr(bpy_mock.types.Object, "warp_sph"))
        registered = {cls.__name__ for cls in _registered_classes}
        for op in ("WM_OT_warp_sph_pause", "WM_OT_warp_sph_step_frame", "WARPSPH_ObjectSettings",
                   "WARPSPH_SceneSettings", "OBJECT_OT_warp_sph_set_role", "OBJECT_OT_warp_sph_add_fluid_scene",
                   "OBJECT_PT_warp_sph", "VIEW3D_PT_warp_sph_advanced", "VIEW3D_PT_warp_sph_cache",
                   "WM_OT_warp_sph_bake", "WM_OT_warp_sph_bake_cancel", "WM_OT_warp_sph_cache_invalidate",
                   "WM_OT_warp_sph_cache_clear", "WM_OT_warp_sph_cache_playback"):
            self.assertIn(op, registered)
        # Small-range values need fine click steps (Blender step is in 1/100 units)
        radius = settings["particle_radius"]
        self.assertLessEqual(radius["step"] / 100.0, 0.0005)
        self.assertGreaterEqual(radius["precision"], 4)
        # Blender float steps must be 1..100 hundredths: stiffness is in thousands, one click = 1,000
        stiffness = settings["stiffness"]
        self.assertLessEqual(stiffness["step"], 100)
        self.assertEqual((stiffness["min"], stiffness["default"], stiffness["max"]), (10.0, 90.0, 400.0))
        self.assertEqual(props.stiffness_pa(types.SimpleNamespace(stiffness=90.0)), 90000.0)

        extension_module.unregister()
        registered = {cls.__name__ for cls in _registered_classes}
        self.assertNotIn("WM_OT_warp_sph_pause", registered)
        self.assertNotIn("WARPSPH_ObjectSettings", registered)
        self.assertFalse(hasattr(bpy_mock.types.Object, "warp_sph"))
        self.assertNotIn("WARPSPH_SceneSettings", registered)
        self.assertFalse(hasattr(bpy_mock.types.Scene, "warp_sph"), "scene settings not cleaned up after unregister")

    def test_interop_bridge_interface(self):
        """Verifies that ZeroCopyGLBuffer52 class and OpenGL methods are properly exposed."""
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "map_to_warp"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "unmap"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "draw_points"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "bind_vao"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "unbind_vao"))
        self.assertTrue(hasattr(ZeroCopyGLBuffer52, "destroy"))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_wrap_device_vec3_aliases_external_memory(self):
        """The mapped-VBO wrapper must alias the raw CUDA pointer (zero-copy), not copy it."""
        owner = wp.zeros(256, dtype=wp.vec3, device="cuda:0")
        view = wrap_device_vec3(owner.ptr, 256)
        self.assertEqual(view.shape, (256,))
        self.assertEqual(view.ptr, owner.ptr)
        view.fill_(wp.vec3(1.0, 2.0, 3.0))
        np.testing.assert_allclose(owner.numpy()[-1], [1.0, 2.0, 3.0])

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

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_demo_scenes_stay_bounded_through_impact(self):
        """Both demos must survive their impact phase without blowing up or leaving the domain."""
        for scene in ("DOUBLE_DAM_PILLARS", "BLOCK_FLOP_SPLASH"):
            # Instabilities only appear at the shipped resolution (1M particles), not at small counts.
            solver = RealtimeWCSPHSolver(num_particles=1_000_000, h=0.04, substeps=1)
            launch_demo_scene_init(solver, scene)
            solver.rebuild_cuda_graph()
            out = wp.zeros_like(solver.pos_sim)
            for _ in range(int(0.8 / solver.dt)):
                solver.step(out)
            pos = solver.pos_sim.numpy()
            speed = np.linalg.norm(solver.vel.numpy(), axis=1)
            with self.subTest(scene=scene):
                self.assertTrue(np.isfinite(pos).all() and np.isfinite(speed).all())
                self.assertLess(speed.max(), 60.0)
                self.assertTrue((pos >= [-2.5, -2.5, 0.0]).all() and (pos <= [2.5, 2.5, 5.0]).all())


if __name__ == "__main__":
    unittest.main()
