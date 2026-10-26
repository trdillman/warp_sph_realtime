"""P2.9 Phase 1 runtime gates, run inside Blender 5.2 with a window (docs/INTERACTIVE_SIM_SPEC.md).

Launched by ``scripts/run_p29_runtime_gates.py`` (isolated profile, ``--enable-event-simulate``), once per GPU
backend and particle count. Builds a Custom scene (domain, a fluid block, 32 static colliders, one keyframed
collider), starts the simulation and measures, in order:

* throughput: steps/s untouched, while 32 objects are moved continuously, and while gravity is dragged;
* change handling: UI-thread time in the change tracker per tick while 32 objects are edited;
* gravity: the next step's acceleration follows a live gravity edit with exactly one graph recapture, and eager
  and graph steps agree;
* R2 / R3: simulation ticks continue during a G (grab) modal transform driven by simulated events, while the
  keyframed collider makes the session call ``frame_set``; the transform keeps tracking the mouse;
* R1 and undo / redo: a transform, a role change and a delete are undone and redone mid-run; the run keeps going
  and every object keeps its ``session_uid``;
* R4: Reset compared with Stop + Start on the same scene (3 of each, wall time of the operator and time to the
  first new step).

Writes ``summary.json`` (checks, measurements) and screenshots into SHOWCASE_OUT, then quits.
"""
import math
import os
import statistics
import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "showcase"))
import _showcase as sc  # noqa: E402  # pyright: ignore[reportMissingImports]

PARTICLES = int(os.environ.get("P29_PARTICLES", "100000"))
SPACING = (1.0 / PARTICLES) ** (1.0 / 3.0)  # the fluid block is 1 m^3
N_STATIC = 32
MEASURE_S = 5.0
EDIT_INTERVAL_S = 0.02
RESET_REPEATS = 3
M = {}  # measurements


def build():
    sc.ensure_addon()
    sc.clear_scene()
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.frame_start, scene.frame_end, scene.frame_current = 1, 250, 1
    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    settings.max_particles = max(settings.max_particles, int(PARTICLES * 1.2))
    settings.render_mode = "POINTS"
    settings.show_particles = True
    settings.point_size = 2.0
    sc.add_box("Domain", (2.4, 2.4, 2.0), (0.0, 0.0, 1.0), role="DOMAIN")
    sc.add_box("Fluid", (1.0, 1.0, 1.0), (-0.62, 0.0, 0.55), role="FLUID")
    for i in range(N_STATIC):
        x = 0.25 + 0.22 * (i % 4)
        y = -0.95 + (1.9 / 7) * (i // 4)
        sc.add_box(f"Static{i:02d}", (0.1, 0.1, 0.1), (x, y, 0.06), role="COLLIDER", color=(0.8, 0.5, 0.2))
    paddle = sc.add_box("Paddle", (0.08, 0.8, 0.4), (0.0, -0.6, 0.25), role="COLLIDER", color=(0.3, 0.6, 0.9))
    paddle.keyframe_insert("location", frame=1)
    paddle.location.y = 0.6
    paddle.keyframe_insert("location", frame=120)
    paddle.location.y = -0.6
    scene.frame_set(1)
    bpy.ops.ed.undo_push(message="P29 gates scene")


def obj(name):
    return bpy.data.objects[sc.PREFIX + name]


def sim():
    return sc.session()


def steps_per_second(seconds, during=None):
    """Steps/s over ``seconds``; ``during()`` is called every EDIT_INTERVAL_S."""
    s = sim()
    n0, t0 = s.step_count, time.perf_counter()
    next_edit = t0
    while time.perf_counter() - t0 < seconds:
        if during is not None and time.perf_counter() >= next_edit:
            during(time.perf_counter() - t0)
            next_edit += EDIT_INTERVAL_S
        yield 0.005
    return (sim().step_count - n0) / (time.perf_counter() - t0)


class TrackerTimer:
    """Wraps the running tracker's refresh / note_updates (both UI-thread change handling) and times them."""

    def __init__(self, tracker):
        self.tracker, self.ticks, self.notes = tracker, [], 0.0
        self._refresh, self._note = tracker.refresh, tracker.note_updates

        def refresh(scene):
            t = time.perf_counter()
            try:
                return self._refresh(scene)
            finally:
                self.ticks.append((time.perf_counter() - t) * 1000.0 + self.notes)
                self.notes = 0.0

        def note_updates(updates=()):
            t = time.perf_counter()
            try:
                return self._note(updates)
            finally:
                self.notes += (time.perf_counter() - t) * 1000.0

        tracker.refresh, tracker.note_updates = refresh, note_updates

    def remove(self):
        del self.tracker.refresh, self.tracker.note_updates

    def stats(self):
        t = sorted(self.ticks)
        if not t:
            return {"ticks": 0}
        return {"ticks": len(t), "mean_ms": round(statistics.fmean(t), 4),
                "p95_ms": round(t[int(0.95 * (len(t) - 1))], 4), "max_ms": round(t[-1], 4)}


def move_statics(elapsed):
    for i in range(N_STATIC):
        o = obj(f"Static{i:02d}")
        o.location.z = 0.06 + 0.03 * math.sin(6.0 * elapsed + i)


def start_run():
    with sc.override():
        bpy.ops.wm.warp_sph_start()
    s = sim()
    yield from sc.wait_until(lambda: s.running and s.step_count >= 1, 120, "first step")


# -- stages ----------------------------------------------------------------------------------------------------

def _select_only(name):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    target = obj(name)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    return target


def grab_drag(name, seconds, confirm=False, sample=None):
    """G-grab ``name`` with simulated events, sweeping the mouse at 60 Hz for ``seconds``; right-click cancels
    (the scene is unchanged) unless ``confirm``. ``sample()`` is called after every mouse move."""
    window, _, _ = sc.view3d()
    _select_only(name)
    cx, cy = _region_xy(0.5, 0.5)
    window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=cx, y=cy)
    yield 0.1
    window.event_simulate(type="G", value="PRESS", x=cx, y=cy)
    window.event_simulate(type="G", value="RELEASE", x=cx, y=cy)
    yield 0.1
    t0, i, x = time.perf_counter(), 0, cx
    while time.perf_counter() - t0 < seconds:
        i += 1
        x = cx + int(120 * math.sin(i / 20.0)) if not confirm else cx + 6 * i
        window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=cy)
        if sample is not None:
            sample()
        yield 1.0 / 60.0
    button = "LEFTMOUSE" if confirm else "RIGHTMOUSE"
    window.event_simulate(type=button, value="PRESS", x=x, y=cy)
    window.event_simulate(type=button, value="RELEASE", x=x, y=cy)
    yield 0.2


def _measure(gen_fn):
    """Steps/s while the generator ``gen_fn()`` runs."""
    s = sim()
    n0, t0 = s.step_count, time.perf_counter()
    yield from gen_fn()
    return (sim().step_count - n0) / (time.perf_counter() - t0)


def _count_recaptures(solver, calls):
    original = solver.rebuild_cuda_graph

    def counted():
        t = time.perf_counter()
        try:
            return original()
        finally:
            calls.append((time.perf_counter() - t) * 1000.0)

    solver.rebuild_cuda_graph = counted


def stage_throughput():
    s = sim()
    yield from sc.wait_until(lambda: s.step_count >= 60, 120, "warm-up")
    M["particles"] = s.solver.num_particles
    M["transport"] = sc.transport(s)
    idle = yield from steps_per_second(MEASURE_S)
    drag = yield from _measure(lambda: grab_drag("Static05", MEASURE_S))
    idle2 = yield from steps_per_second(MEASURE_S)
    calls: list = []
    _count_recaptures(s.solver, calls)
    try:
        gravity = yield from steps_per_second(
            MEASURE_S, during=lambda e: setattr(bpy.context.scene, "gravity", (2.0 * math.sin(3 * e), 0.0, -9.81)))
    finally:
        del s.solver.rebuild_cuda_graph
    bpy.context.scene.gravity = (0.0, 0.0, -9.81)
    yield from sc.pause(0.3)
    base = 0.5 * (idle + idle2)
    M["throughput"] = {"idle_sps": round(idle, 1), "idle_after_sps": round(idle2, 1),
                       "grab_drag_sps": round(drag, 1), "grab_drag_ratio": round(drag / base, 3),
                       "gravity_drag_sps": round(gravity, 1), "gravity_drag_ratio": round(gravity / base, 3),
                       "gravity_recaptures": len(calls),
                       "recapture_ms_mean": round(statistics.fmean(calls), 3) if calls else None}
    sc.check("throughput_grab_drag_ge_90pct", drag >= 0.9 * base, M["throughput"])
    sc.check("throughput_gravity_drag_ge_90pct", gravity >= 0.9 * base, M["throughput"])
    # Change handling with 32 live objects moved continuously (every EDIT_INTERVAL_S).
    timer = TrackerTimer(s.tracker)
    try:
        edit = yield from steps_per_second(MEASURE_S, during=move_statics)
    finally:
        timer.remove()
    out = sc.out_dir()
    if out is not None:
        sc.screenshot_area(out / "editing_32_objects.png")
    change = timer.stats()
    change["sps_while_moving_32"] = round(edit, 1)
    change["ratio_while_moving_32"] = round(edit / base, 3)
    M["change_handling"] = change
    sc.check("change_handling_le_1ms_per_tick_32_objects",
             change.get("ticks", 0) > 0 and change["mean_ms"] <= 1.0 and change["p95_ms"] <= 1.0, change)
    reasons = s.reset_reasons()
    sc.check("moved_statics_pending_reset", bool(reasons), reasons[:3])


def _host_state(solver):
    return (solver.pos_sim.numpy().copy(), solver.vel.numpy().copy(), solver.pid.numpy().copy(), solver.sim_time)


def _restore(solver, state):
    pos, vel, pid, t = state
    solver.pos_sim.assign(pos)
    solver.vel.assign(vel)
    solver.pid.assign(pid)
    solver.sim_time = t


def _step_vel(solver, state, use_graph, out):
    _restore(solver, state)
    solver.step(out, use_graph=use_graph)
    sc.module(".sim_session").runtime.wp.synchronize_device(solver.device)
    return solver.vel.numpy().copy(), solver.pid.numpy().copy()


def _by_id(vel, pid):
    import numpy as np
    order = np.argsort(pid)
    return vel[order]


def stage_gravity():
    import numpy as np
    s = sim()
    wp = sc.module(".sim_session").runtime.wp
    calls = {"n": 0}
    original = s.solver.rebuild_cuda_graph

    def counted():
        calls["n"] += 1
        return original()

    s.solver.rebuild_cuda_graph = counted
    g_old, g_new = (0.0, 0.0, -9.81), (6.0, 0.0, -9.81)
    try:
        bpy.context.scene.gravity = g_new
        yield from sc.wait_until(lambda: "gravity" not in s.tracker.pending
                                 and tuple(round(v, 4) for v in s.solver.gravity) == g_new, 10, "live gravity")
        yield from sc.pause(0.5)  # later ticks must not recapture again
    finally:
        del s.solver.rebuild_cuda_graph
    sc.check("gravity_one_recapture", calls["n"] == 1, {"recaptures": calls["n"]})
    with sc.override():
        bpy.ops.wm.warp_sph_pause()
    yield from sc.pause(0.2)
    solver = s.solver
    out = wp.empty(solver.num_particles, dtype=wp.vec3, device=solver.device)
    state = _host_state(solver)
    v0 = _by_id(state[1], state[2])
    solver.gravity = wp.vec3(*g_old)
    v_old = _by_id(*_step_vel(solver, state, False, out))
    solver.gravity = wp.vec3(*g_new)
    v_new = _by_id(*_step_vel(solver, state, False, out))
    v_graph = _by_id(*_step_vel(solver, state, True, out))
    _restore(solver, state)
    expected = solver.substeps * solver.dt * (g_new[0] - g_old[0])
    measured = float(np.mean(v_new[:, 0] - v_old[:, 0]))
    diff = float(np.max(np.abs(v_graph - v_new)))
    scale = float(np.max(np.abs(v_new - v0))) or 1.0
    M["gravity"] = {"expected_mean_dvx": expected, "measured_mean_dvx": measured, "eager_vs_graph_max_abs": diff,
                    "step_dv_max": scale, "substeps": solver.substeps, "dt": solver.dt}
    sc.check("gravity_next_step_acceleration", abs(measured - expected) <= 0.05 * abs(expected), M["gravity"])
    sc.check("gravity_eager_graph_equivalent", diff <= 1e-4 * max(scale, 1.0), M["gravity"])
    bpy.context.scene.gravity = g_old
    with sc.override():
        bpy.ops.wm.warp_sph_pause()  # resume
    yield from sc.wait_until(lambda: "gravity" not in s.tracker.pending, 10, "gravity back")


def _region_xy(fx, fy):
    _, _, region = sc.view3d()
    return int(region.x + fx * region.width), int(region.y + fy * region.height)


def stage_modal_transform():
    """R2 / R3: G-grab a static collider with simulated events while the run (with a keyframed collider) steps."""
    window, _, _ = sc.view3d()
    s = sim()
    x0 = obj("Static00").location.x
    xs, frames, ops = [], set(), []

    def sample():
        xs.append(obj("Static00").location.x)
        frames.add(bpy.context.scene.frame_current)
        ops.append(any("translate" in op.bl_idname.lower() for op in window.modal_operators))

    n0, t0 = s.step_count, time.perf_counter()
    yield from grab_drag("Static00", 2.0, confirm=True, sample=sample)
    dt = time.perf_counter() - t0
    during = s.step_count - n0
    out = sc.out_dir()
    if out is not None:
        sc.screenshot_area(out / "after_grab.png")
    x1 = obj("Static00").location.x
    tail = xs[1:]  # the first sample is taken before the transform has processed a move
    monotonic = all(b >= a - 1e-6 for a, b in zip(tail, tail[1:], strict=False))
    M["modal_transform"] = {"samples": len(ops), "samples_in_translate": sum(ops), "steps_during": during,
                            "seconds": round(dt, 2), "sps_during": round(during / dt, 1),
                            "frame_changes": len(frames), "x_before": x0, "x_after": x1,
                            "x_track_monotonic": monotonic}
    grabbed = bool(ops) and sum(ops) >= len(ops) - 1
    sc.check("r2_transform_modal_active", grabbed, M["modal_transform"])
    sc.check("r2_ticks_continue_during_transform", grabbed and during >= 10, M["modal_transform"])
    sc.check("r3_frame_set_during_transform_keeps_tracking",
             grabbed and len(frames) > 1 and monotonic and x1 > x0 + 1e-3, M["modal_transform"])


def _undo(redo=False):
    with sc.override():
        (bpy.ops.ed.redo if redo else bpy.ops.ed.undo)()


def _uids():
    api = sc.module(".live_edits")
    return {o.name: api.uid(o) for o in bpy.context.scene.objects}


def _alive_after(label, s, steps=5, timeout=10):
    n0 = s.step_count
    try:
        yield from sc.wait_until(lambda: s.running and s.step_count >= n0 + steps, timeout, label)
        return True
    except TimeoutError:
        return False


def stage_undo():
    s = sim()
    live_edits = sc.module(".live_edits")
    uids0 = _uids()
    results = {}
    # transform
    bpy.ops.ed.undo_push(message="before move")
    obj("Static01").location.x += 0.2
    bpy.ops.ed.undo_push(message="move Static01")
    moved = obj("Static01").location.x
    yield 0.2
    _undo()
    yield 0.2
    back = obj("Static01").location.x
    ok_undo = yield from _alive_after("undo transform", s)
    _undo(redo=True)
    yield 0.2
    ok_redo = yield from _alive_after("redo transform", s)
    results["transform"] = {"moved": moved, "after_undo": back, "after_redo": obj("Static01").location.x,
                            "running_after_undo": ok_undo, "running_after_redo": ok_redo,
                            "pending": list(s.tracker.pending)[:4]}
    sc.check("undo_transform_keeps_run", ok_undo and ok_redo and abs(back - (moved - 0.2)) < 1e-5
             and abs(obj("Static01").location.x - moved) < 1e-5, results["transform"])
    # role change
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj("Static02").select_set(True)
    bpy.context.view_layer.objects.active = obj("Static02")
    with sc.override():
        bpy.ops.object.warp_sph_set_role("EXEC_DEFAULT", True, role="NONE")  # undo=True, as from the UI
    yield 0.2
    role_pending = [r for r in s.reset_reasons() if "Static02" in r]
    _undo()
    yield 0.2
    role_back = obj("Static02").warp_sph.role
    static01_after_role_undo = obj("Static01").location.x
    ok_undo = yield from _alive_after("undo role", s)
    _undo(redo=True)
    yield 0.2
    ok_redo = yield from _alive_after("redo role", s)
    results["role"] = {"pending_after_change": role_pending, "role_after_undo": role_back,
                       "role_after_redo": obj("Static02").warp_sph.role, "running": [ok_undo, ok_redo],
                       "static01_x_after_role_undo": static01_after_role_undo}
    sc.check("undo_role_change_keeps_run", ok_undo and ok_redo and role_back == "COLLIDER"
             and obj("Static02").warp_sph.role == "NONE" and bool(role_pending), results["role"])
    # delete the keyframed collider mid-run
    paddle_uid = live_edits.uid(obj("Paddle"))
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj("Paddle").select_set(True)
    bpy.context.view_layer.objects.active = obj("Paddle")
    with sc.override():
        bpy.ops.object.delete("EXEC_DEFAULT", True)
    yield 0.2
    ok_deleted = yield from _alive_after("after delete", s)
    held = 0 in s._held
    _undo()
    yield 0.3
    ok_undo = yield from _alive_after("undo delete", s)
    restored = sc.PREFIX + "Paddle" in bpy.data.objects
    resolved = bool(s._colliders) and s._colliders[0] is not None and live_edits.uid(s._colliders[0]) == paddle_uid
    _undo(redo=True)
    yield 0.3
    ok_redo = yield from _alive_after("redo delete", s)
    results["delete"] = {"running_after_delete": ok_deleted, "pose_held_after_delete": held,
                         "restored_by_undo": restored, "resolved_by_uid_after_undo": resolved,
                         "running": [ok_undo, ok_redo]}
    sc.check("undo_delete_keeps_run", ok_deleted and held and ok_undo and restored and resolved and ok_redo,
             results["delete"])
    _undo()  # bring the paddle back for the Reset stage
    yield 0.3
    uids1 = {k: v for k, v in _uids().items()}
    same = {k: uids0[k] == uids1.get(k) for k in uids0}
    results["uids_compared"] = len(same)
    results["uids_changed"] = [k for k, v in same.items() if not v]
    results["scene_is_current"] = s.scene == bpy.context.scene
    M["undo"] = results
    sc.check("r1_session_uid_stable_across_undo_redo", all(same.values()) and results["scene_is_current"],
             {"compared": len(same), "changed": results["uids_changed"]})


def stage_reset():
    s = sim()
    resets, restarts = [], []
    for i in range(RESET_REPEATS):
        obj("Static03").location.z = 0.06 + 0.01 * (i + 1)  # an edit that needs Reset
        yield 0.2
        t0 = time.perf_counter()
        with sc.override():
            bpy.ops.wm.warp_sph_reset()
        t_op = time.perf_counter() - t0
        n_at = s.step_count
        yield from sc.wait_until(lambda n=n_at: s.step_count > n, 60, "step after reset")
        resets.append({"op_s": round(t_op, 3), "first_step_s": round(time.perf_counter() - t0, 3),
                       "last_reset_ms": round(s.last_reset_ms, 1), "pending_after": s.reset_reasons()[:2]})
        yield from sc.pause(0.5)
    for _ in range(RESET_REPEATS):
        t0 = time.perf_counter()
        with sc.override():
            bpy.ops.wm.warp_sph_stop()
            bpy.ops.wm.warp_sph_start()
        t_op = time.perf_counter() - t0
        yield from sc.wait_until(lambda: s.running and s.step_count >= 1, 60, "step after start")
        restarts.append({"op_s": round(t_op, 3), "first_step_s": round(time.perf_counter() - t0, 3)})
        yield from sc.pause(0.5)
    med = lambda rows, k: statistics.median(r[k] for r in rows)  # noqa: E731
    M["reset"] = {"particles": s.solver.num_particles, "reset": resets, "stop_start": restarts,
                  "reset_median_s": med(resets, "first_step_s"), "stop_start_median_s": med(restarts, "first_step_s"),
                  "reset_op_median_s": med(resets, "op_s"), "stop_start_op_median_s": med(restarts, "op_s")}
    sc.check("reset_applies_pending_edits", all(not r["pending_after"] for r in resets), resets)
    faster = M["reset"]["reset_median_s"] < M["reset"]["stop_start_median_s"]
    if PARTICLES >= 1_000_000:
        sc.check("r4_reset_1m_le_2s", M["reset"]["reset_median_s"] <= 2.0, M["reset"])
        sc.check("r4_reset_faster_than_stop_start", faster, M["reset"])
    else:
        sc.check("reset_faster_than_stop_start", faster, M["reset"])


def main():
    sc.frame_view(target=(0.0, 0.0, 0.5), eye=(3.5, -4.5, 3.0))
    yield from start_run()
    s = sim()
    if not sc.check("started", s.running, s.last_error or None):
        return
    for name, stage in (("throughput", stage_throughput), ("gravity", stage_gravity),
                        ("modal_transform", stage_modal_transform), ("undo", stage_undo), ("reset", stage_reset)):
        with sc.timed(name):
            try:
                yield from stage()
            except Exception as exc:  # one stage failing must not hide the others
                import traceback
                traceback.print_exc()
                sc.check(f"{name}_stage_completed", False, f"{type(exc).__name__}: {exc}")
        if not sim().running:
            sc.check(f"running_after_{name}", False, sim().last_error)
            yield from start_run()
    sc.SUMMARY["particles"] = M.get("particles")
    sc.SUMMARY["transport"] = M.get("transport")
    sc.SUMMARY["backend"] = sc.module(".sim_session").gpu_backend()
    sc.SUMMARY["measurements"] = M
    with sc.override():
        bpy.ops.wm.warp_sph_stop()


sc.run("p29_runtime_gates", main, build=build)
