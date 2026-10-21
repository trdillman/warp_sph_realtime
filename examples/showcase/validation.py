"""Validation (PR #11): a dam-break column next to the Martin & Moyce (1952) measured surge front.

A column of water 2 a high and a wide (a = 0.5 m) collapses along a long tank, as in the add-on's headless
validation (``sim/validation.py``, ``docs/VALIDATION.md``) but at a visual resolution and through the normal Custom
scene path: Domain and Fluid boxes, the Water preset, then **Bake**. After the bake a red marker is keyframed at the
reference front for each frame (``MARTIN_MOYCE_T`` / ``MARTIN_MOYCE_Z``, imported from ``sim/validation.py``), and the
simulated front of every cached frame is measured with the same ``front_position`` and ``compare_front``.

The slab is 16 particles thick (quasi-2D, like the headless test). Expect the front within a few tenths of ``a``
of the experiment; ``docs/VALIDATION.md`` lists the errors at several resolutions.

Run: ``blender --python examples/showcase/validation.py``. Scrub the timeline after the bake. With the runner it
also renders the frames (Workbench) with the numbers written on them and quits.
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _showcase as sc  # noqa: E402

SPACING = 0.0125            # m
CELLS = 40                  # particles across the column: a = CELLS * SPACING = 0.5 m
ROWS = 16                   # particles through the slab (y)
A = CELLS * SPACING
DEPTH = (ROWS + 1.5) * SPACING   # tank depth: ROWS lattice rows fit at one spacing from both walls
FPS = 60
BAKE_END = 34               # T = t sqrt(2 g / a) reaches the last reference point (3.33) by frame 33
MAX_MEAN_ABS_ERROR = 0.5    # loose gate on mean |Z_sim - Z_ref| at the reference times (in units of a)
GRAVITY = 9.81
VIDEO_FPS = 12


def build():
    sc.ensure_addon()
    sc.clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 1.0
    scene.use_gravity, scene.gravity = True, (0.0, 0.0, -GRAVITY)
    scene.render.fps, scene.render.fps_base = FPS, 1.0
    # The fill keeps particle centres at least one spacing from every wall: with the tank's back wall at x = 0 the
    # lattice starts at 1.5 spacings, so the box reaches one spacing further to hold CELLS x 2 CELLS particles and
    # the water's back face sits at x = SPACING.
    sc.add_box("Tank", (5 * A + SPACING, DEPTH, 4 * A), ((5 * A + SPACING) / 2, DEPTH / 2, 2 * A), "DOMAIN")
    sc.add_box("Water Column", ((CELLS + 1) * SPACING, DEPTH, (2 * CELLS + 1) * SPACING),
               ((CELLS + 1) * SPACING / 2, DEPTH / 2, (2 * CELLS + 1) * SPACING / 2), "FLUID")
    sc.add_box("Floor", (6 * A, 1.2, 0.01), (2.5 * A, DEPTH / 2, -0.006), color=(0.32, 0.33, 0.36))
    sc.add_box("Back Wall", (0.01, 1.2, 1.2), (-0.006, DEPTH / 2, 0.6), color=(0.32, 0.33, 0.36))
    sc.add_box("Reference Front", (0.012, DEPTH + 0.1, 0.4), (SPACING + A, DEPTH / 2, 0.2), color=(0.9, 0.08, 0.08))

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    settings.particle_radius = 0.6 * SPACING
    settings.render_mode = "SPHERES"
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")
    settings.bake_start, settings.bake_end, settings.bake_steps = 1, BAKE_END, 0
    settings.cache_dir = str(sc.work_dir("validation") / "cache")
    scene.frame_start, scene.frame_end = 1, BAKE_END
    scene.frame_set(1)


def measure(disk, validation, frames, seconds_per_frame):
    """(T, Z) of the simulated front in every cached frame, in the reference's non-dimensional units."""
    t_scale = math.sqrt(2.0 * GRAVITY / A)
    shift = np.array([SPACING, 0.0, SPACING], np.float32)
    T, Z = [], []
    for frame in frames:
        # front_position expects the headless lattice: the water's back face at x = 0 and its lowest layer at half
        # a spacing above z = 0. This scene's lattice sits one spacing further from the walls on both axes.
        positions = disk.read_frame(frame).position - shift
        T.append((frame - frames[0]) * seconds_per_frame * t_scale)
        Z.append(validation.front_position(positions, SPACING, ROWS) / A)
    return np.array(T), np.array(Z)


def main():
    scene = bpy.context.scene
    out = sc.out_dir()
    sc.frame_view(target=(1.3, 0.1, 0.45), eye=(1.3, -3.6, 1.4))
    sim = sc.session()
    with sc.override():
        bpy.ops.wm.warp_sph_cache_clear()
    with sc.timed("bake"):
        with sc.override():
            bpy.ops.wm.warp_sph_bake(resume=False)
        if not sc.check("bake_started", sim.bake is not None, sim.last_error or sim.bake_result or None):
            return
        yield from sc.wait_until(lambda: sim.bake is None, 900, "bake")
    disk = sc.module(".cache.disk").DiskCache(sc.module(".cache.blender_adapter").cache_dir(scene))
    span = disk.cached_range()
    sc.check("bake_complete", span == (1, BAKE_END), {"range": span, "result": sim.bake_result})
    sc.SUMMARY["particles"] = sim.solver.num_particles if sim.solver is not None else None
    sc.SUMMARY["transport"] = sc.transport(sim)["label"]
    with sc.override():
        bpy.ops.wm.warp_sph_stop()

    validation = sc.module(".sim.validation")
    seconds = sc.module(".cache.age").seconds_per_frame(disk.manifest(), FPS)
    frames = list(range(span[0], span[1] + 1))
    T, Z = measure(disk, validation, frames, seconds)
    reference = [sc.reference_front(t, validation.MARTIN_MOYCE_T, validation.MARTIN_MOYCE_Z) for t in T]
    comparison = validation.compare_front(T, Z)
    sc.SUMMARY["front"] = {"T": T.round(3).tolist(), "Z_simulated": Z.round(3).tolist(),
                           "Z_reference": np.round(reference, 3).tolist(), "seconds_per_frame": seconds}
    sc.SUMMARY["comparison"] = {k: comparison[k] for k in ("points", "mean_error", "mean_abs_error", "max_abs_error")}
    sc.check("front_matches_reference", comparison["mean_abs_error"] < MAX_MEAN_ABS_ERROR,
             {"mean_abs_error": round(comparison["mean_abs_error"], 3), "gate": MAX_MEAN_ABS_ERROR,
              "points": comparison["points"]})

    marker = bpy.data.objects[sc.PREFIX + "Reference Front"]
    for frame, z_ref in zip(frames, reference, strict=True):
        marker.location.x = SPACING + A * z_ref
        marker.keyframe_insert("location", index=0, frame=frame)
    with sc.override():
        bpy.ops.wm.warp_sph_cache_playback(show=True)
    cloud = sc.module(".cache.blender_adapter").find_playback_object(scene)
    sc.check("show_cache", cloud is not None, None)
    if cloud is not None:
        cloud.color = (0.15, 0.45, 0.95, 1.0)
    scene.frame_set(1)
    if out is None:
        return

    sc.use_workbench((960, 540))
    sc.set_world((0.06, 0.065, 0.075))
    sc.add_camera((1.3, -3.6, 1.4), (1.3, 0.1, 0.4), lens=32.0)
    with sc.timed("render"):
        paths = sc.render_frames(frames, out / "frames")
    for path, t, z_sim, z_ref in zip(paths, T, Z, reference, strict=True):
        sc.label_image(path, [f"Dam break vs Martin & Moyce (1952), a = {A:g} m",
                              f"T = {t:.2f}   simulated front Z = {z_sim:.2f}   reference (red) Z = {z_ref:.2f}"])
    sc.SUMMARY["frames"] = [str(p) for p in paths]
    key = int(np.argmin(np.abs(T - 2.0)))
    sc.SUMMARY["key_frame"] = str(paths[key])
    sc.SUMMARY["sequences"] = [str(out / "frames")]
    sc.SUMMARY["video_fps"] = VIDEO_FPS
    sc.check("rendered", all(p.is_file() for p in paths), len(paths))


build()
sc.run("validation", main)
