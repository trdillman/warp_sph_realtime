"""P2.1 scene seam on the GPU: emitters and the particle budget, moving colliders, demos via colliders."""

import importlib
import math
import sys
import types

import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh, box_signed_distance
from tests.test_scene_seam_cpu import plane, rot_z
from warp_sph_realtime.scene.description import ColliderMotion, EmitterData, SceneDescription
from warp_sph_realtime.scene.fill import SceneFillError
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required for the WCSPH solver")

SPACING = 0.04
TALL = ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))
TANK = ((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0))


def run(solver, steps, feed=None):
    out = wp.zeros(solver.num_particles, dtype=wp.vec3, device="cuda:0")
    for _ in range(steps):
        if feed is not None:
            for frame in solver.collider_frames_needed():
                solver.set_collider_keyframe(frame, [feed(frame)])
        solver.step(out)
    return out.numpy()


def live(solver):
    n = solver.active_count
    return solver.pos_sim.numpy()[:n], solver.vel.numpy()[:n]


# -- emitters ---------------------------------------------------------------------------------------

def tap_scene(rate=300.0, budget=200, colliders=(), tap=None, **kwargs):
    tap = tap or plane((-0.1, -0.1), (0.1, 0.1), 1.5)
    return SceneDescription(*TALL, (0.0, 0.0, -9.81), (), tuple(colliders),
                            emitters=(EmitterData(tap, rate=rate, **kwargs),), particle_budget=budget)


def test_emitter_count_follows_the_rate_and_stops_at_the_budget():
    solver = RealtimeWCSPHSolver.from_scene(tap_scene(), SPACING)
    assert solver.num_particles == 200 and solver.active_count == 0      # the pool is the budget
    assert solver.emitter_rates == {"tap": pytest.approx(300.0)}
    capacity_arrays = (solver.pos_sim.ptr, solver.vel.ptr, solver.pos_sorted.ptr)

    def expected():  # emission runs at the start of each substep, so the last one saw sim_time - dt
        return min(200, math.floor(300.0 * (solver.sim_time - solver.dt) + 1e-9))

    for _ in range(8):  # ~0.8 s: 300/s reaches the 200 budget at ~0.67 s
        run(solver, 40)
        assert solver.active_count == expected() > 0
        assert not solver.take_exhaustion_notice() or solver.active_count == 200
    assert solver.sim_time * 300.0 > 200 and solver.active_count == 200
    assert solver.emission_exhausted
    assert not solver.take_exhaustion_notice()                       # the notice fires once (taken above)
    run(solver, 10)
    assert solver.active_count == 200
    assert (solver.pos_sim.ptr, solver.vel.ptr, solver.pos_sorted.ptr) == capacity_arrays  # never reallocated
    pos, vel = live(solver)
    assert np.isfinite(pos).all() and (pos[:, 2] < 1.5).all()        # the stream fell from the tap


def test_exhaustion_notice_is_reported_once():
    solver = RealtimeWCSPHSolver.from_scene(tap_scene(rate=0.0, budget=50), SPACING)
    run(solver, 30)
    assert solver.active_count == 50 and solver.take_exhaustion_notice() and not solver.take_exhaustion_notice()


def test_emission_follows_the_frame_range():
    desc = tap_scene(rate=300.0, budget=10_000, frame_start=3.0, frame_end=9.0)  # fps 24: t 1/12 .. 1/3 s
    solver = RealtimeWCSPHSolver.from_scene(desc, SPACING)
    while solver.sim_time < 1.0 / 12.0 - 2 * solver.dt * solver.substeps:
        run(solver, 1)
        assert solver.active_count == 0
    while solver.sim_time < 0.5:
        run(solver, 1)
    assert solver.active_count == math.floor(300.0 * 0.25 + 1e-9)    # 6 frames at 300/s, then off


def test_emitted_particles_start_clear_of_walls_and_colliders():
    block = ((0.0, -0.5, 1.4), (0.6, 0.5, 1.6))                       # the tap runs into this box
    tap = plane((-0.99, -0.3), (0.5, 0.3), 1.5)                       # and up to the -x wall
    desc = tap_scene(rate=0.0, budget=20_000, colliders=(box_mesh(*block, "block"),), tap=tap)
    solver = RealtimeWCSPHSolver.from_scene(desc, SPACING)
    spacing = 0.5 * solver.h
    n = 0
    for k in range(1, 6):  # emit directly at a few clock values (white box: placement before any forces)
        solver.sim_time = k * solver.dt
        solver._emit()
        new = solver.pos_sim.numpy()[n:solver.active_count]
        n = solver.active_count
        assert len(new) > 0
        assert (box_signed_distance(new, *block) >= spacing - 1e-5).all()
        assert (new[:, 0] >= -1.0 + spacing - 1e-5).all()
    with pytest.raises(SceneFillError, match="move it into open space"):
        RealtimeWCSPHSolver.from_scene(
            tap_scene(colliders=(box_mesh(*block, "block"),), tap=plane((0.1, -0.1), (0.3, 0.1), 1.5)), SPACING)


def test_emitter_rate_above_full_inflow_is_capped():
    solver = RealtimeWCSPHSolver.from_scene(tap_scene(rate=1e7, budget=10_000), SPACING)
    natural = RealtimeWCSPHSolver.from_scene(tap_scene(rate=0.0, budget=10_000), SPACING).emitter_rates["tap"]
    assert solver.emitter_rates["tap"] == pytest.approx(natural)
    nozzles = len(solver.nozzle_pos)
    assert natural == pytest.approx(nozzles * 2.0 / (0.5 * solver.h), rel=1e-5)  # speed 2 m/s, one per spacing


# -- moving colliders -------------------------------------------------------------------------------

PADDLE = ((-0.8, -0.5, 0.02), (-0.7, 0.5, 0.6))
PADDLE_PIVOT = (-0.75, 0.0, 0.31)


def pool(colliders, motion=()):
    return SceneDescription(*TANK, (0.0, 0.0, -9.81), (box_mesh((-0.6, -0.9, 0.02), (0.9, 0.9, 0.22), "pool"),),
                            tuple(colliders), collider_motion=tuple(motion))


def translated(origin, offset):
    m = np.eye(4)
    m[:3, 3] = np.add(origin, offset)
    return m


def test_moving_paddle_pushes_the_fluid_and_never_lets_it_in():
    speed = 1.0  # m/s along +x
    results = {}
    for moving in (False, True):
        motion = (ColliderMotion(0, "paddle", np.eye(3), PADDLE_PIVOT),) if moving else ()
        solver = RealtimeWCSPHSolver.from_scene(pool((box_mesh(*PADDLE, "paddle"),), motion), SPACING)
        feed = (lambda f: translated(PADDLE_PIVOT, (speed * (f - 1.0) / 24.0, 0.0, 0.0))) if moving else None
        worst = np.inf
        for _ in range(12):
            run(solver, 10, feed)
            shift = np.array([speed * solver.sim_time if moving else 0.0, 0.0, 0.0])
            pos, _ = live(solver)
            worst = min(worst, float(box_signed_distance(pos, PADDLE[0] + shift, PADDLE[1] + shift).min()))
        pos, vel = live(solver)
        assert np.isfinite(pos).all() and np.isfinite(vel).all()
        results[moving] = (solver.mass * vel[:, 0].sum(), worst, solver.sim_time)
    (px_still, worst_still, _), (px_moving, worst_moving, t) = results[False], results[True]
    assert t > 0.25
    # The pool settles and sloshes either way; the moving paddle adds a large +x push on top of that.
    assert px_moving - px_still > 30.0 and px_moving > 3.0 * abs(px_still), results
    assert worst_moving >= -0.25 * SPACING and worst_still >= -1e-4, results


def test_rotating_stirrer_spins_the_fluid():
    stirrer = ((-0.05, -0.4, 0.02), (0.05, 0.4, 0.5))
    pivot = (0.0, 0.0, 0.26)
    omega = 4.0  # rad/s about +z
    results = {}
    for moving in (False, True):
        motion = (ColliderMotion(0, "stirrer", np.eye(3), pivot),) if moving else ()
        desc = SceneDescription(*TANK, (0.0, 0.0, -9.81), (box_mesh((-0.9, -0.9, 0.02), (0.9, 0.9, 0.22), "pool"),),
                                (box_mesh(*stirrer, "stirrer"),), collider_motion=motion)
        solver = RealtimeWCSPHSolver.from_scene(desc, SPACING)

        def feed(f):
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = rot_z(omega * (f - 1.0) / 24.0), pivot
            return m

        run(solver, 100, feed if moving else None)
        pos, vel = live(solver)
        assert np.isfinite(pos).all()
        results[moving] = solver.mass * float((pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]).sum())
    assert results[True] > 20.0 * max(abs(results[False]), 0.01), results


def test_missing_keyframes_hold_the_collider_still():
    motion = (ColliderMotion(0, "paddle", np.eye(3), PADDLE_PIVOT),)
    solver = RealtimeWCSPHSolver.from_scene(pool((box_mesh(*PADDLE, "paddle"),), motion), SPACING)
    assert solver.collider_frames_needed() == [2]                    # frame 1 is the start pose
    run(solver, 20)                                                   # nobody feeds frames
    np.testing.assert_allclose(solver.collider_xform.numpy()[0][:3], PADDLE_PIVOT, atol=1e-6)
    np.testing.assert_allclose(solver.collider_lin.numpy()[0], 0.0)
    with pytest.raises(Exception, match="changes scale"):
        solver.set_collider_keyframe(3, [np.diag([2.0, 2.0, 2.0, 1.0])])


# -- demos through the collider path ----------------------------------------------------------------

def demo_api():
    if "bpy" not in sys.modules:
        try:
            importlib.import_module("bpy")
        except ImportError:  # the demo module imports bpy only for its viewport helpers
            sys.modules["bpy"] = types.ModuleType("bpy")
    return importlib.import_module("warp_sph_realtime.scene.demo_scenes")


def demo_penetration(pos):
    d = np.linalg.norm(pos - [0.0, 0.0, 0.55], axis=1) - 0.55
    worst = float(min(d.min(), 0.0))
    below = pos[:, 2] < 2.2
    for cx, cy in ((-0.85, -0.55), (-0.85, 0.55), (0.85, -0.55), (0.85, 0.55)):
        worst = min(worst, float((np.hypot(pos[below, 0] - cx, pos[below, 1] - cy) - 0.18).min()))
    return -worst


@pytest.mark.parametrize("scene", ["DOUBLE_DAM_PILLARS", "BLOCK_FLOP_SPLASH"])
def test_demo_matches_the_legacy_obstacle_path(scene):
    demos = demo_api()
    n = 100_000
    final = {}
    for legacy in (True, False):
        solver = RealtimeWCSPHSolver(num_particles=n, h=0.04, substeps=2, enable_obstacles=0)
        demos.launch_demo_scene_init(solver, scene)
        assert solver.scene_mode and solver.active_count == n and solver.enable_obstacles == 0
        if legacy:  # the pre-P2.1 route: analytic sphere and pillars in the legacy kernel
            solver.scene_mode = False
            solver.enable_obstacles = 1 if scene == "DOUBLE_DAM_PILLARS" else 0
        solver.rebuild_cuda_graph()
        worst = 0.0
        pos = np.empty((0, 3))
        for _ in range(6):
            pos = run(solver, 20)
            if scene == "DOUBLE_DAM_PILLARS":
                worst = max(worst, demo_penetration(pos))
        assert np.isfinite(pos).all() and len(pos) == n
        final[legacy] = (pos.mean(axis=0), worst, 0.5 * solver.h)
    (com_old, worst_old, spacing), (com_new, worst_new, _) = final[True], final[False]
    # The mesh path's soft contact layer differs slightly from the analytic one (about 1 mm at 1M
    # particles, evidence/p2.1-scene-seam); at this coarser 100k spacing the bound is a quarter spacing.
    assert np.linalg.norm(com_new - com_old) < 0.25 * spacing, (com_old, com_new)
    assert worst_new <= max(worst_old, 0.0) + 0.25 * spacing, (worst_old, worst_new)
