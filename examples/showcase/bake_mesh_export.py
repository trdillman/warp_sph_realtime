"""Bake, mesh and export (PR #12 + #14): a splash baked to disk, surfaced, rendered in Cycles and exported to USD.

A block of water drops into a shallow pool while a side emitter adds a short stream. **Bake** writes every frame
to the cache folder; **Show Cache** adds the point cloud that follows the timeline (with velocity, id and age
attributes) and **Add Mesh Preset** surfaces it with Geometry Nodes. The mesh gets a clear water material
(Principled BSDF, full transmission, IOR 1.33) through the preset's Material input, and a second still colours
the point cloud by particle age (seconds since emission: the stream is younger than the pool). **Export Meshed
Sequence** writes the meshed fluid as USD, and the demo opens the file again to count the exported frames.

Run: ``blender --python examples/showcase/bake_mesh_export.py``. With the runner it also renders every other frame
in Cycles (16 samples, denoised), the age still and the USD file, and quits.
"""
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _showcase as sc  # noqa: E402

SPACING = 0.02          # m
BAKE_END = 48
FPS = 24
EMIT_FRAMES = (1, 30)
MAX_PARTICLES = 40_000
GRAVITY = 9.81
VIDEO_FPS = 12


def build():
    sc.ensure_addon()
    sc.clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 1.0
    scene.use_gravity, scene.gravity = True, (0.0, 0.0, -GRAVITY)
    scene.render.fps, scene.render.fps_base = FPS, 1.0

    sc.add_box("Domain", (1.2, 0.8, 1.0), (0.0, 0.0, 0.5), "DOMAIN")
    sc.add_box("Pool", (1.2, 0.8, 0.2), (0.0, 0.0, 0.1), "FLUID")
    sc.add_box("Falling Block", (0.25, 0.25, 0.25), (0.15, 0.0, 0.62), "FLUID")
    emitter = sc.add_plane("Side Emitter", (0.1, 0.1), (-0.52, 0.0, 0.55), (0.0, math.radians(60.0), 0.0),
                           "EMITTER")
    emit = emitter.warp_sph
    emit.emit_velocity_mode = "NORMAL"
    emit.emit_speed = 1.5
    emit.emit_rate = 0.0
    emit.emit_frame_start, emit.emit_frame_end = EMIT_FRAMES
    sc.add_box("Floor", (4.0, 4.0, 0.02), (0.0, 0.0, -0.011), color=(0.55, 0.55, 0.58))
    sc.add_box("Back Wall", (4.0, 0.02, 2.0), (0.0, 0.42, 1.0), color=(0.5, 0.52, 0.56))

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    settings.max_particles = MAX_PARTICLES
    settings.particle_radius = 0.6 * SPACING   # the cache point cloud's radius (the age still)
    settings.render_mode = "SSFR_LIQUID"
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")
    settings.bake_start, settings.bake_end, settings.bake_steps = 1, BAKE_END, 0
    settings.cache_dir = str(sc.work_dir("bake_mesh_export") / "cache")
    scene.frame_start, scene.frame_end = 1, BAKE_END
    scene.frame_set(1)


def usd_frames(path) -> dict:
    """Time samples of the exported mesh's points, read back with Blender's bundled USD library."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    samples = [len(m.GetPointsAttr().GetTimeSamples()) for m in meshes]
    return {"meshes": len(meshes), "points_time_samples": max(samples, default=0),
            "start": stage.GetStartTimeCode(), "end": stage.GetEndTimeCode()}


def main():
    scene = bpy.context.scene
    out = sc.out_dir()
    sc.frame_view(target=(0.0, 0.0, 0.25), eye=(1.1, -1.9, 1.1))
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
    sc.SUMMARY["particles"] = len(disk.read_frame(BAKE_END).id)
    sc.SUMMARY["particles_first_frame"] = len(disk.read_frame(1).id)
    sc.SUMMARY["transport"] = sc.transport(sim)["label"]
    with sc.override():
        bpy.ops.wm.warp_sph_stop()

    adapter, blender_output = sc.module(".cache.blender_adapter"), sc.module(".cache.blender_output")
    with sc.override():
        bpy.ops.wm.warp_sph_cache_playback(show=True)
        bpy.ops.wm.warp_sph_cache_mesh(show=True)
    cloud, mesh = adapter.find_playback_object(scene), blender_output.find_mesh_object(scene)
    sc.check("show_cache_and_mesh", cloud is not None and mesh is not None, None)
    if cloud is None or mesh is None:
        return
    water = sc.water_material()
    sc.set_mesh_input(mesh, "Material", water)
    mesh.color = water.diffuse_color

    usd = sc.work_dir("bake_mesh_export") / "fluid.usdc"
    with sc.timed("export_usd"), sc.override():
        result = bpy.ops.wm.warp_sph_cache_export("EXEC_DEFAULT", filepath=str(usd), format="USD")
    exported = usd_frames(usd) if usd.is_file() else {}
    sc.SUMMARY["usd"] = {"path": str(usd), **exported}
    sc.check("usd_export", result == {"FINISHED"} and exported.get("points_time_samples") == BAKE_END,
             {"result": sorted(result), **exported, "expected_frames": BAKE_END})
    if out is None:
        scene.frame_set(1)
        return
    sc.add_media(usd)

    device = sc.use_cycles((960, 540), samples=16)
    sc.SUMMARY["cycles_device"] = device
    sc.set_world((0.62, 0.68, 0.78), strength=0.8)
    sc.add_sun(4.0, (40.0, 0.0, 35.0))
    sc.add_camera((1.1, -1.9, 1.1), (0.0, 0.0, 0.22), lens=35.0)
    frames = list(range(1, BAKE_END + 1, 2))
    with sc.timed("render"):
        paths = sc.render_frames(frames, out / "frames")
    for path, frame in zip(paths, frames, strict=True):
        sc.label_image(path, ["Baked splash, meshed (Add Mesh Preset), water material, Cycles",
                              f"frame {frame} / {BAKE_END}"])

    # Age still: the point cloud coloured by seconds since emission, at the last frame.
    seconds = sc.module(".cache.age").seconds_per_frame(disk.manifest(), FPS)
    max_age = seconds * (BAKE_END - 1)
    scene.frame_set(BAKE_END)
    mesh.hide_render, cloud.hide_render = True, False
    cloud.data.materials.clear()
    cloud.data.materials.append(sc.age_material(max_age))
    with sc.timed("render_age"):
        age = sc.render_still(out / "age.png")
    sc.label_image(age, [f"Point cloud coloured by particle age, frame {BAKE_END}",
                         f"yellow = just emitted, blue = {max_age:.1f} s old"])
    sc.add_media(age)
    mesh.hide_render, cloud.hide_render = False, True

    sc.SUMMARY["frames"] = [str(p) for p in paths]
    sc.SUMMARY["key_frame"] = str(paths[min(len(paths) - 1, 9)])
    sc.SUMMARY["sequences"] = [str(out / "frames")]
    sc.SUMMARY["video_fps"] = VIDEO_FPS
    sc.check("rendered", all(p.is_file() for p in [*paths, age]), len(paths) + 1)


build()
sc.run("bake_mesh_export", main)
