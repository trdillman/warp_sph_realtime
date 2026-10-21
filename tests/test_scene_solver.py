import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh, box_signed_distance
from warp_sph_realtime.scene.description import SceneDescription
from warp_sph_realtime.scene.fill import SceneFillError
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required for the WCSPH solver")

SPACING = 0.04
PLATE = ((-0.5, -0.5, 0.3), (0.5, 0.5, 0.5))


def column_scene(colliders=(box_mesh(*PLATE, name="plate"),)):
    """A 0.24 x 0.24 x 0.32 fluid column hanging ~0.4 above a plate, in a 2 x 2 x 2 domain."""
    return SceneDescription(
        (-1.0, -1.0, 0.0), (1.0, 1.0, 2.0), (0.0, 0.0, -9.81),
        (box_mesh((-0.12, -0.12, 0.92), (0.12, 0.12, 1.24), "column"),), tuple(colliders))


def run(solver, steps, use_graph):
    out = wp.zeros(solver.num_particles, dtype=wp.vec3, device="cuda:0")
    for _ in range(steps):
        solver.step(out, use_graph=use_graph)
    return out.numpy()


def test_from_scene_mirrors_demo_conventions():
    solver = RealtimeWCSPHSolver.from_scene(column_scene(), SPACING)
    assert solver.num_particles == 6 * 6 * 8
    assert solver.h == pytest.approx(2 * SPACING) and solver.enable_obstacles == 0
    assert solver.collider_radius == pytest.approx(solver.h)
    np.testing.assert_array_equal(solver.vel.numpy(), 0.0)
    assert tuple(solver.domain_max) == (1.0, 1.0, 2.0)


def test_column_never_penetrates_collider_and_graph_matches_eager():
    results = {}
    for use_graph in (False, True):
        solver = RealtimeWCSPHSolver.from_scene(column_scene(), SPACING)
        # 300 steps cover free fall (~0.29 s), impact, splash and settling.
        lowest = np.inf  # smallest signed distance to the plate over the whole run
        out = wp.zeros(solver.num_particles, dtype=wp.vec3, device="cuda:0")
        for _ in range(300):
            solver.step(out, use_graph=use_graph)
            lowest = min(lowest, float(box_signed_distance(out.numpy(), *PLATE).min()))
        pos = out.numpy()
        assert np.isfinite(pos).all() and np.isfinite(solver.vel.numpy()).all()
        assert lowest < 0.5 * solver.h, "column never reached the collider; the test did not exercise contact"
        assert lowest >= -1e-4, f"particle penetrated the collider by {-lowest:.5f} m"
        assert (pos >= [-1, -1, 0] - np.float32(1e-5)).all() and (pos <= [1, 1, 2] + np.float32(1e-5)).all()
        assert pos[:, 2].mean() < 0.9, "column did not fall"
        results[use_graph] = pos
    np.testing.assert_array_equal(results[True], results[False])


def test_hard_projection_pushes_an_embedded_particle_out():
    solver = RealtimeWCSPHSolver.from_scene(column_scene(), SPACING)
    pts = np.tile(np.array([[0.3, 0.2, 0.9]], dtype=np.float32), (solver.num_particles, 1))
    pts[0] = (0.0, 0.0, 0.45)  # inside the plate, nearest face is the top (0.05 < query radius h = 0.08)
    pts[1] = (0.48, 0.0, 0.4)  # inside, nearest face is the +x side
    pts[2:] += 0.1 * np.arange(solver.num_particles - 2)[:, None] * np.array([1, 0, 0], dtype=np.float32)
    solver.pos_sim.assign(pts)
    pos = run(solver, 1, use_graph=False)
    # The solver reorders particles every step, so check the whole set rather than indices 0 and 1.
    assert (box_signed_distance(pos, *PLATE) >= 0.0).all()


def test_no_colliders_behaves_as_free_fluid_with_user_gravity():
    desc = column_scene(colliders=())
    solver = RealtimeWCSPHSolver.from_scene(desc, SPACING)
    assert solver.colliders is not None and solver.colliders.count == 0
    z0 = solver.pos_sim.numpy()[:, 2].mean()
    pos = run(solver, 20, use_graph=True)
    assert np.isfinite(pos).all() and pos[:, 2].mean() < z0

    sideways = SceneDescription(desc.domain_min, desc.domain_max, (3.0, 0.0, 0.0), desc.fluid_volumes, ())
    solver = RealtimeWCSPHSolver.from_scene(sideways, SPACING)
    x0, z0 = solver.pos_sim.numpy().mean(axis=0)[[0, 2]]
    pos = run(solver, 20, use_graph=True)
    assert pos[:, 0].mean() > x0 + 1e-4 and abs(pos[:, 2].mean() - z0) < 1e-3


def test_from_scene_enforces_particle_limit():
    with pytest.raises(SceneFillError, match="above the 100 limit"):
        RealtimeWCSPHSolver.from_scene(column_scene(), SPACING, max_particles=100)
