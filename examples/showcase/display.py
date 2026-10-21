"""Viewport display (PR #6): the three render modes and the transport that carries particles to the viewport.

Loads the built-in Double Dam demo, then for Points, Spheres and SSFR Liquid: sets **Render Mode**, starts the
simulation, lets it run, pauses it and shows an overlay with the mode, the transport and the GPU backend. The
transport is chosen at Start, so each mode gets its own Stop + Start.

* Vulkan: Vulkan interop, a GPU copy with no CPU readback, for every mode.
* OpenGL: Points use the OpenGL texture interop; Spheres and SSFR Liquid use the legacy VBO interop. Both are GPU
  copies with no CPU readback.
* If an interop cannot be used the add-on falls back to Readback (a CPU copy); the overlay says so.

Run: ``blender --python examples/showcase/display.py`` (add ``--gpu-backend opengl`` before ``--python`` for the
OpenGL routes). With the runner (``scripts/render_showcase.py``) it also screenshots each mode and quits.
"""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _showcase as sc  # noqa: E402

DEMO = "DOUBLE_DAM_PILLARS"
PARTICLES = 300_000
RENDER_MODES = ("POINTS", "SPHERES", "SSFR_LIQUID")
MODE_NAMES = {"POINTS": "Points", "SPHERES": "Spheres", "SSFR_LIQUID": "SSFR Liquid"}
WARMUP_STEPS = 150          # let the dams collapse a little before the screenshot


def build():
    sc.ensure_addon()
    sc.clear_scene()
    scene = bpy.context.scene
    settings = scene.warp_sph
    settings.scene_source = "DEMO"
    settings.demo_scene = DEMO
    settings.particle_count = PARTICLES
    settings.point_size = 3.0
    settings.show_particles = True
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")


def expected_label(backend: str, mode: str) -> str:
    labels = sc.module(".render.session")
    if backend == "OPENGL":
        return labels.GL_LABEL if mode == "POINTS" else sc.GL_VBO_LABEL
    return labels.VULKAN_LABEL


def main():
    scene = bpy.context.scene
    settings = scene.warp_sph
    out = sc.out_dir()
    sc.frame_view(target=(0.0, 0.0, 0.6), eye=(6.5, -8.0, 5.0))
    results, shots = {}, []
    for mode in RENDER_MODES:
        settings.render_mode = mode
        with sc.timed(f"start_{mode.lower()}"), sc.override():
            bpy.ops.wm.warp_sph_load_demo(demo_id=DEMO)   # sets the demo scene and starts the simulation
        sim = sc.session()
        if not sc.check(f"{mode.lower()}_started", sim.running, sim.last_error or None):
            continue
        yield from sc.wait_until(lambda s=sim: s.step_count >= WARMUP_STEPS, 180, f"{mode} warm-up")
        step_ms, particles = sim.frame_time_ms, sim.solver.num_particles
        info = sc.transport(sim)
        with sc.override():
            bpy.ops.wm.warp_sph_pause()
        lines = [f"WarpSPH Render Mode: {MODE_NAMES[mode]}",
                 f"Transport: {info['label']}",
                 f"GPU backend: {info['backend']}",
                 f"{particles:,} particles, {step_ms:.1f} ms per simulation step"]
        if info["readback"]:
            lines.append("Readback fallback: particles go through a CPU copy" +
                         (f" ({info['reason']})" if info["reason"] else ""))
        sc.set_overlay(lines)
        yield from sc.pause(0.8)
        if out is not None:
            shots.append((sc.screenshot_area(out / f"display_{mode.lower()}.png"),
                          f"{MODE_NAMES[mode]}: {info['label']}"))
        expected = expected_label(info["backend"], mode)
        results[mode] = {**info, "particles": particles, "step_ms": round(step_ms, 2), "expected": expected}
        sc.check(f"{mode.lower()}_transport_labelled", info["label"] != "none",
                 {"label": info["label"], "readback": info["readback"], "matches_route": info["label"] == expected})
        with sc.override():
            bpy.ops.wm.warp_sph_stop()
        yield from sc.wait_until(lambda s=sim: not s.running and s.solver is None, 30, f"{mode} stop")
    sc.clear_overlay()
    if out is None:  # interactive: leave the last mode running
        with sc.override():
            bpy.ops.wm.warp_sph_load_demo(demo_id=DEMO)
    sc.SUMMARY["modes"] = results
    sc.SUMMARY["particles"] = max((r["particles"] for r in results.values()), default=0)
    sc.SUMMARY["transport"] = {mode: r["label"] for mode, r in results.items()}
    sc.SUMMARY["backend"] = sc.module(".sim_session").gpu_backend()
    if out is not None:
        sheet = sc.compose_grid(shots, out / "display_modes.png", columns=3, cell_w=640,
                                title=f"WarpSPH viewport display on {sc.SUMMARY['backend']}")
        sc.add_media(sheet)
        sc.SUMMARY["key_frame"] = str(sheet)
        sc.check("screenshots", len(shots) == len(RENDER_MODES) and all(p.is_file() for p, _ in shots),
                 [str(p) for p, _ in shots])


build()
sc.run("display", main)
