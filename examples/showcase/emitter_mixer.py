"""Emitters and moving colliders (PR #13): a stream pours into a tank stirred by a rotating paddle.

A centimetre scene (Scene > Units: Unit Scale 0.01, Length Centimeters), so every length below is in centimetres:
an 80 cm tank with a 15 cm pool, an emitter plane pouring a 150 cm/s stream from 70 cm up for the first 100
frames, and a keyframed collider paddle turning two full turns in 150 frames (about 4.8 degrees per frame;
WarpSPH follows animated location and rotation, each frame's rotation by its shortest arc, so keep it under
half a turn per frame; animated scale or shear stops the run). Gravity is Blender's own (Scene > Gravity, m/s^2)
and the emitter speed and the geometry are converted with the unit scale. After the bake, **Add Mesh Preset**
surfaces the cache with a radius and voxel size taken from the bake, in centimetres.

Run: ``blender --python examples/showcase/emitter_mixer.py``. With the runner it also renders every other frame
(Workbench) and quits.
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _showcase as sc  # noqa: E402
from _metrics import paddle_influence, same_initial_state  # noqa: E402

UNIT_SCALE = 0.01           # metres per scene unit: lengths below are centimetres
SPACING = 2.0               # cm
TANK = 80.0                 # cm, a cube from z = 0
POOL_DEPTH = 15.0           # cm
EMIT_FRAMES = (1, 100)
EMIT_SPEED = 150.0          # cm/s
BAKE_END = 150
TURNS = 2.0                 # paddle turns over the bake
FPS = 30
MAX_PARTICLES = 60_000
GRAVITY = 9.81              # m/s^2, whatever the unit scale
VIDEO_FPS = 15              # every other frame of 30 fps
CONTROL_FRAMES = (11, 21, 31)  # early pool response, before the emitter fills the tank


def build():
    sc.ensure_addon()
    sc.clear_scene()
    scene = bpy.context.scene
    units = scene.unit_settings
    units.system, units.scale_length, units.length_unit = "METRIC", UNIT_SCALE, "CENTIMETERS"
    scene.use_gravity, scene.gravity = True, (0.0, 0.0, -GRAVITY)
    scene.render.fps, scene.render.fps_base = FPS, 1.0

    sc.add_box("Tank", (TANK, TANK, TANK), (0.0, 0.0, TANK / 2), "DOMAIN")
    sc.add_box("Pool", (TANK, TANK, POOL_DEPTH), (0.0, 0.0, POOL_DEPTH / 2), "FLUID")
    sc.add_box("Floor", (TANK + 6, TANK + 6, 1.0), (0.0, 0.0, -0.55), color=(0.32, 0.33, 0.36))
    for name, size, location in (("Wall +Y", (TANK + 6, 1.0, 30.0), (0.0, TANK / 2 + 0.55, 15.0)),
                                 ("Wall -X", (1.0, TANK + 6, 30.0), (-TANK / 2 - 0.55, 0.0, 15.0))):
        sc.add_box(name, size, location, color=(0.32, 0.33, 0.36))
    paddle = sc.add_box("Paddle", (50.0, 6.0, 25.0), (0.0, 0.0, 16.0), "COLLIDER", color=(0.95, 0.55, 0.1))
    paddle.rotation_euler = (0.0, 0.0, 0.0)
    paddle.keyframe_insert("rotation_euler", index=2, frame=1)
    paddle.rotation_euler.z = TURNS * 2.0 * math.pi
    paddle.keyframe_insert("rotation_euler", index=2, frame=BAKE_END)
    _linear_keys(paddle)

    emitter = sc.add_plane("Emitter", (16.0, 16.0), (20.0, 0.0, 70.0), (math.pi, 0.0, 0.0), "EMITTER")
    emit = emitter.warp_sph
    emit.emit_velocity_mode = "NORMAL"
    emit.emit_speed = EMIT_SPEED
    emit.emit_rate = 0.0  # a full stream
    emit.emit_frame_start, emit.emit_frame_end = EMIT_FRAMES

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    settings.max_particles = MAX_PARTICLES
    # Points: Particle Radius (Spheres, SSFR, the cache point cloud) is not converted by the unit scale and
    # is capped at 0.1, too small for centimetres; the rendered surface below is the mesh preset.
    settings.render_mode = "POINTS"
    settings.point_size = 3.0
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")
    settings.bake_start, settings.bake_end, settings.bake_steps = 1, BAKE_END, 0
    settings.cache_dir = str(sc.work_dir("emitter_mixer") / "cache")
    scene.frame_start, scene.frame_end = 1, BAKE_END
    scene.frame_set(1)


def _linear_keys(obj) -> None:
    """Constant angular speed: linear interpolation on every keyframe of the object (layered actions too)."""
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return
    curves = list(getattr(action, "fcurves", ()) or ())
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                curves.extend(bag.fcurves)
    for curve in curves:
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"


def main():
    scene = bpy.context.scene
    out = sc.out_dir()
    sc.frame_view(target=(0.0, 0.0, 20.0), eye=(95.0, -120.0, 105.0))
    sim = sc.session()
    # Paired negative control through the same public bake workflow, with only
    # paddle animation disabled. Separate caches preserve the demonstration.
    settings = scene.warp_sph
    paddle = scene.objects[sc.PREFIX + "Paddle"]
    action, slot = paddle.animation_data.action, paddle.animation_data.action_slot
    animated_cache, bake_end = settings.cache_dir, settings.bake_end
    disk_type = sc.module(".cache.disk").DiskCache
    control_dir = sc.work_dir("emitter_mixer") / "stationary_paddle_cache"
    try:
        paddle.animation_data.action = None
        paddle.rotation_euler = (0.0, 0.0, 0.0)
        scene.frame_set(1)
        settings.cache_dir, settings.bake_end = str(control_dir), max(CONTROL_FRAMES)
        with sc.override():
            bpy.ops.wm.warp_sph_cache_clear()
            bpy.ops.wm.warp_sph_bake(resume=False)
        if not sc.check("stationary_control_started", sim.bake is not None, sim.last_error or None):
            return
        with sc.timed("stationary_control_bake"):
            yield from sc.wait_until(lambda: sim.bake is None, 900, "stationary paddle control")
        control = disk_type(control_dir)
        if not sc.check("stationary_control_complete", control.cached_range() == (1, max(CONTROL_FRAMES)),
                        {"range": control.cached_range(), "result": sim.bake_result}):
            return
    finally:
        with sc.override():
            bpy.ops.wm.warp_sph_stop()
        paddle.animation_data.action = action
        paddle.animation_data.action_slot = slot
        settings.cache_dir, settings.bake_end = animated_cache, bake_end
        scene.frame_set(1)
    yield from sc.wait_until(lambda: not sim.running and sim.solver is None, 30, "stationary paddle teardown")
    initial_matrix = np.array(paddle.matrix_world).copy()
    scene.frame_set(CONTROL_FRAMES[1])
    changed_matrix = np.array(paddle.matrix_world).copy()
    scene.frame_set(1)
    sc.check("paddle_animation_evaluated", not np.allclose(initial_matrix, changed_matrix, atol=1e-6),
             {"frame_1": initial_matrix.tolist(), "frame_21": changed_matrix.tolist()})
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
    signature = (disk.manifest() or {}).get("signature", {})
    sc.check("bake_complete", span == (1, BAKE_END), {"range": span, "result": sim.bake_result})
    sc.check("unit_scale_recorded", math.isclose(signature.get("unit_scale", 1.0), UNIT_SCALE, rel_tol=1e-6),
             signature.get("unit_scale"))
    first, last = len(disk.read_frame(1).id), len(disk.read_frame(BAKE_END).id)
    sc.check("emitter_added_particles", last > first, {"first_frame": first, "last_frame": last})
    sc.SUMMARY["particles"] = last
    sc.SUMMARY["particles_first_frame"] = first
    sc.SUMMARY["transport"] = sc.transport(sim)["label"]
    with sc.override():
        bpy.ops.wm.warp_sph_stop()

    yield from sc.wait_until(lambda: not sim.running and sim.solver is None, 30, "animated paddle teardown")

    initial_match = same_initial_state(control.read_frame(1), disk.read_frame(1))
    sc.check("paired_initial_state", initial_match)
    samples = {str(frame): paddle_influence(control.read_frame(frame), disk.read_frame(frame))
               for frame in CONTROL_FRAMES}
    # The stationary cache against itself must reject stirring even though
    # gravity/emission produce large speeds. This exercises the actual measured data.
    negatives = {str(frame): paddle_influence(control.read_frame(frame), control.read_frame(frame))
                 for frame in CONTROL_FRAMES}
    sc.check("stationary_negative_control", all(not m["ok"] for m in negatives.values()), negatives)
    sc.check("paddle_stirs", initial_match and sum(m["ok"] for m in samples.values()) >= 2, samples)
    sc.SUMMARY["paddle_influence"] = {"samples": samples, "negative_controls": negatives,
                                     "stationary_cache": str(control_dir), "animated_cache": animated_cache}

    with sc.override():
        bpy.ops.wm.warp_sph_cache_mesh(show=True)
    mesh = sc.module(".cache.blender_output").find_mesh_object(scene)
    radius = sc.module(".cache.blender_output").input_socket(mesh, "Radius") if mesh is not None else None
    sc.check("mesh_preset", mesh is not None and radius is not None and radius.value > SPACING,
             {"radius_cm": round(radius.value, 3) if radius is not None else None})
    if mesh is not None:
        mesh.color = (0.2, 0.5, 0.95, 1.0)
    scene.frame_set(1)
    if out is None:
        return

    sc.use_workbench((960, 540))
    sc.set_world((0.06, 0.065, 0.075))
    sc.add_camera((95.0, -120.0, 105.0), (0.0, 0.0, 18.0), lens=35.0, clip_end=5000.0)
    frames = list(range(1, BAKE_END + 1, 2))
    with sc.timed("render"):
        paths = sc.render_frames(frames, out / "frames")
    for path, frame in zip(paths, frames, strict=True):
        sc.label_image(path, [f"Emitter + rotating collider paddle, centimetre scene (Unit Scale {UNIT_SCALE:g})",
                              f"frame {frame} / {BAKE_END}"])
    sc.SUMMARY["frames"] = [str(p) for p in paths]
    # Frame 39: the stream, the paddle and the stirred pool are all in view (later the pool covers the paddle).
    sc.SUMMARY["key_frame"] = str(paths[min(len(paths) - 1, 19)])
    sc.SUMMARY["sequences"] = [str(out / "frames")]
    sc.SUMMARY["video_fps"] = VIDEO_FPS
    sc.check("rendered", all(p.is_file() for p in paths), len(paths))


sc.run("emitter_mixer", main, build=build)
