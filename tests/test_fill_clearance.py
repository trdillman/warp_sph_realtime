"""The initial fill keeps one spacing from domain walls and colliders (the solver's soft boundary layer).

A lattice started closer (the cell-centred half spacing) sits inside the layer, where the wall force is
large, and is expelled violently: the validation probe measured 46 m/s and a mean height 24 % too high.
Headless on Warp's CPU device, so CI covers it.
"""

import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh, box_signed_distance
from warp_sph_realtime.scene import fill as fill_module
from warp_sph_realtime.scene.demo_description import DOMAIN_MAX, DOMAIN_MIN, dam_layout, flop_layout, wall_inset
from warp_sph_realtime.scene.description import SceneDescription
from warp_sph_realtime.scene.fill import fill_particles
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

SPACING = 0.05
STEPS = 100


def tank_scene(fluid_height=0.4):
    """Fluid filling the lower part of a closed 0.5 x 0.5 x 1 tank, touching the floor and all four walls."""
    return SceneDescription((0.0, 0.0, 0.0), (0.5, 0.5, 1.0), (0.0, 0.0, -9.81),
                            (box_mesh((0.0, 0.0, 0.0), (0.5, 0.5, fluid_height), "tank"),), ())


def test_fill_keeps_one_spacing_from_every_wall():
    pts = fill_particles(tank_scene(), SPACING, device="cpu")
    assert (pts.min(axis=0) >= SPACING - 1e-6).all()
    assert (pts.max(axis=0)[:2] <= 0.5 - SPACING + 1e-6).all()
    # The lattice is otherwise unchanged: 1.5 .. 8.5 spacings across, from 1.5 spacings above the floor.
    assert len(pts) == 8 * 8 * 7


def test_fill_keeps_one_spacing_from_colliders_and_clearance_can_be_disabled():
    plate = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.2))
    desc = SceneDescription((0.0, 0.0, 0.0), (0.5, 0.5, 1.0), (0.0, 0.0, -9.81),
                            (box_mesh((0.0, 0.0, 0.0), (0.5, 0.5, 0.4), "tank"),),
                            (box_mesh(*plate, name="plate"),))
    pts = fill_particles(desc, SPACING, device="cpu")
    assert (box_signed_distance(pts, *plate) >= SPACING - 1e-6).all()
    raw = fill_particles(desc, SPACING, device="cpu", clearance=0.0)
    assert len(raw) > len(pts) and (box_signed_distance(raw, *plate) >= 0.0).all()


def rest_run(device="cpu"):
    solver = RealtimeWCSPHSolver.from_scene(tank_scene(), SPACING, device=device)
    out = wp.zeros(solver.num_particles, dtype=wp.vec3, device=device)
    start = float(solver.pos_sim.numpy()[:, 2].mean())
    peak = 0.0
    for _ in range(STEPS):
        solver.step(out, use_graph=False)
        peak = max(peak, float(np.linalg.norm(solver.vel.numpy(), axis=1).max()))
    pos = solver.pos_sim.numpy()
    assert np.isfinite(pos).all()
    return peak, float(pos[:, 2].mean()) / start - 1.0


def test_resting_box_has_no_wall_blast():
    peak, rise = rest_run()
    # Measured at this setup (CPU): 2.0 m/s peak and a 35 % lower mean height: the column settles into the
    # half spacing of slack that the cell-centred lattice leaves at the walls (fluid at a lower wall starts
    # 1.5 spacings from it, the force-free contact distance being 1). Before the fix: 58 m/s and +179 %.
    assert peak < 4.0, f"peak speed {peak:.1f} m/s: the fill is expelled from the walls"
    assert rise < 0.02, f"mean height rose {rise:.1%}"


def test_the_regression_test_detects_the_old_half_spacing_fill(monkeypatch):
    original = fill_module.fill_particles
    monkeypatch.setattr(
        fill_module, "fill_particles", lambda d, s, device=None: original(d, s, device=device, clearance=0.5 * s))
    peak, rise = rest_run()
    assert peak > 10.0 and rise > 0.5


@pytest.mark.parametrize("n", [20_000, 100_000, 1_000_000])
def test_demo_blocks_start_outside_the_wall_layer(n):
    from warp_sph_realtime.scene.demo_description import demo_spacing

    s = demo_spacing(n)
    edge, z0, nx, ny = dam_layout(s)
    assert DOMAIN_MAX[0] - edge >= s and z0 - DOMAIN_MIN[2] >= s
    assert DOMAIN_MAX[1] - (-edge + (ny - 1) * s) >= s
    edge, z0, nx = flop_layout(s)
    assert DOMAIN_MAX[0] - (-edge + (nx - 1) * s) >= s and z0 - DOMAIN_MIN[2] >= s
    # The one-million particle demos keep their authored starting positions.
    if n == 1_000_000:
        assert dam_layout(s)[:2] == pytest.approx((2.35, 0.08)) and flop_layout(s)[:2] == pytest.approx((2.4, 0.05))
    assert wall_inset(0.0, s) == pytest.approx(1.25 * s)
