"""Physics presets (PR #15): one spill baked with Water, Viscous and Stylised, side by side.

Water rests on a shelf with an open front and spills onto the floor of the tank. The demo applies each preset
with ``wm.warp_sph_physics_preset`` (it sets Stiffness, Time Step, Viscosity and Sub-steps together) and bakes the
same scene into its own cache folder, one bake after another. It then shows each cache with **Show Cache** and
**Add Mesh Preset** and renders the same frames, composed into a labelled comparison: Water spreads fast and
thin, Viscous oozes off the shelf, Stylised is softer and bouncier.

Run: ``blender --python examples/showcase/presets.py``. Afterwards switch **Cache Folder** between the three
``cache_<preset>`` folders to compare them in the viewport. With the runner it also renders the comparison and quits.
"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _showcase as sc  # noqa: E402

PRESETS = ("WATER", "VISCOUS", "STYLISED")
PRESET_NAMES = {"WATER": "Water", "VISCOUS": "Viscous", "STYLISED": "Stylised"}
SPACING = 0.015         # m
BAKE_END = 72
FPS = 24
COMPARE_FRAMES = (30, 60)
GRAVITY = 9.81


def build():
    sc.ensure_addon()
    sc.clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 1.0
    scene.use_gravity, scene.gravity = True, (0.0, 0.0, -GRAVITY)
    scene.render.fps, scene.render.fps_base = FPS, 1.0

    sc.add_box("Tank", (1.6, 0.5, 0.8), (0.0, 0.0, 0.4), "DOMAIN")
    sc.add_box("Shelf", (0.65, 0.6, 0.3), (-0.525, 0.0, 0.15), "COLLIDER", color=(0.45, 0.42, 0.4))
    sc.add_box("Water", (0.5, 0.5, 0.3), (-0.55, 0.0, 0.45), "FLUID")
    sc.add_box("Floor", (3.0, 2.0, 0.02), (0.0, 0.0, -0.011), color=(0.55, 0.55, 0.58))
    sc.add_box("Back Wall", (3.0, 0.02, 1.2), (0.0, 0.26, 0.6), color=(0.5, 0.52, 0.56))

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    settings.particle_radius = 0.6 * SPACING
    settings.render_mode = "SSFR_LIQUID"
    settings.bake_start, settings.bake_end, settings.bake_steps = 1, BAKE_END, 0
    scene.frame_start, scene.frame_end = 1, BAKE_END
    scene.frame_set(1)


def cache_folder(preset: str) -> str:
    return str(sc.work_dir("presets") / f"cache_{preset.lower()}")


def main():
    scene = bpy.context.scene
    settings = scene.warp_sph
    out = sc.out_dir()
    sc.frame_view(target=(0.0, 0.0, 0.2), eye=(0.6, -2.0, 1.0))
    sim = sc.session()
    disk_api = sc.module(".cache.disk")
    stats = {}
    for preset in PRESETS:
        settings.cache_dir = cache_folder(preset)
        with sc.override():
            bpy.ops.wm.warp_sph_physics_preset(preset=preset)
            bpy.ops.wm.warp_sph_cache_clear()
        physics = {"stiffness": settings.stiffness, "viscosity": settings.viscosity,
                   "time_step_scale": settings.time_step_scale, "substeps": settings.substeps}
        scene.frame_set(1)
        with sc.timed(f"bake_{preset.lower()}"):
            with sc.override():
                bpy.ops.wm.warp_sph_bake(resume=False)
            if not sc.check(f"{preset.lower()}_bake_started", sim.bake is not None,
                            sim.last_error or sim.bake_result or None):
                continue
            yield from sc.wait_until(lambda: sim.bake is None, 900, f"{preset} bake")
        disk = disk_api.DiskCache(settings.cache_dir)
        span = disk.cached_range()
        sc.check(f"{preset.lower()}_bake_complete", span == (1, BAKE_END), {"range": span, "result": sim.bake_result})
        frame = disk.read_frame(COMPARE_FRAMES[0])
        stats[preset] = {**physics, "particles": int(frame.count),
                         "mean_speed_m_s": round(float(np.linalg.norm(frame.velocity, axis=1).mean()), 3),
                         "front_x": round(float(np.percentile(frame.position[:, 0], 99)), 3),
                         "transport": sc.transport(sim)["label"]}
        with sc.override():
            bpy.ops.wm.warp_sph_stop()
        yield from sc.wait_until(lambda: not sim.running, 30, f"{preset} stop")
    sc.SUMMARY["presets"] = stats
    sc.SUMMARY["particles"] = max((s["particles"] for s in stats.values()), default=0)
    sc.SUMMARY["transport"] = next((s["transport"] for s in stats.values()), None)
    if len(stats) == len(PRESETS):
        sc.check("presets_differ", len({(s["viscosity"], s["stiffness"], s["substeps"]) for s in stats.values()}) == 3,
                 {p: (s["viscosity"], s["stiffness"], s["substeps"]) for p, s in stats.items()})
        sc.check("viscous_is_slower", stats["VISCOUS"]["mean_speed_m_s"] < stats["WATER"]["mean_speed_m_s"],
                 {p: s["mean_speed_m_s"] for p, s in stats.items()})

    if out is None:  # interactive: show the last bake
        with sc.override():
            bpy.ops.wm.warp_sph_cache_playback(show=True)
        return
    sc.use_workbench((800, 450))
    sc.set_world((0.06, 0.065, 0.075))
    sc.add_camera((0.6, -2.0, 1.0), (0.0, 0.0, 0.18), lens=35.0)
    cells = []
    with sc.timed("render"):
        for preset in PRESETS:
            settings.cache_dir = cache_folder(preset)
            scene.frame_set(COMPARE_FRAMES[0])
            with sc.override():
                bpy.ops.wm.warp_sph_cache_playback(show=True)
                bpy.ops.wm.warp_sph_cache_mesh(show=True)
            mesh = sc.module(".cache.blender_output").find_mesh_object(scene)
            if mesh is not None:
                mesh.color = (0.2, 0.5, 0.95, 1.0)
            paths = sc.render_frames(COMPARE_FRAMES, out / preset.lower())
            for path, frame in zip(paths, COMPARE_FRAMES, strict=True):
                cells.append((path, f"{PRESET_NAMES[preset]}, frame {frame}"))
    # One row per compared frame, one column per preset.
    ordered = [cells[p * len(COMPARE_FRAMES) + f] for f in range(len(COMPARE_FRAMES)) for p in range(len(PRESETS))]
    grid = sc.compose_grid(ordered, out / "presets_comparison.png", columns=len(PRESETS), cell_w=560,
                           title="Physics presets: the same spill baked with Water, Viscous and Stylised")
    sc.add_media(grid)
    sc.SUMMARY["key_frame"] = str(grid)
    sc.check("rendered", grid.is_file() and all(p.is_file() for p, _ in cells), len(cells))


build()
sc.run("presets", main)
