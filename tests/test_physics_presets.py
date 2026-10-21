"""P2.5 physics controls: presets stay stable, viscosity takes effect and is graph-safe.

The CPU tests (coarse spacing, eager launches) run in CI; the CUDA tests use the default 0.02 m spacing and the
CUDA graph the add-on uses. Stability here means: no NaN, nothing leaves the domain, the total mechanical energy
never grows, and speeds stay well below the sound speed. The fastest few particles (a splash droplet leaving a
jet) are judged apart from the bulk: the 99.9th percentile must stay under half the sound speed, the single
fastest under three times it.
"""

import types

import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.physics_presets import PRESET_ITEMS, PRESETS, active_preset, apply_preset, preset_settings
from warp_sph_realtime.scene.description import SceneDescription
from warp_sph_realtime.sim.sim_core import VISCOSITY_RELAXATION_LIMIT, RealtimeWCSPHSolver

needs_cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
GRAVITY = 9.81


def drop_scene():
    """A shallow pool with a block dropped into it from 0.8 m above the pool (impact about 4 m/s), in a 1 x 1 x 2 m tank."""
    return SceneDescription(
        (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), (0.0, 0.0, -GRAVITY),
        (box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 0.2), "pool"), box_mesh((0.3, 0.3, 1.0), (0.7, 0.7, 1.4), "block")), ())


def build(settings, spacing, device, desc=None):
    return RealtimeWCSPHSolver.from_scene(
        desc or drop_scene(), spacing, substeps=settings.substeps, stiffness=settings.stiffness * 1000.0,
        cfl_factor=settings.time_step_scale, viscosity=settings.viscosity, device=device)


def run(solver, seconds, use_graph):
    """Step for `seconds` of simulated time.

    Returns (peak speed, max energy growth, finite, contained, final KE, peak 99.9th-percentile speed)."""
    out = wp.zeros(solver.num_particles, dtype=wp.vec3, device=solver.device)
    steps = int(seconds / (solver.substeps * solver.dt))
    peak, bulk_peak, growth, e0 = 0.0, 0.0, 0.0, None
    for k in range(steps):
        solver.step(out, use_graph=use_graph)
        if k % 10 == 0 or k == steps - 1:
            pos, vel = solver.pos_sim.numpy().astype(np.float64), solver.vel.numpy().astype(np.float64)
            if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
                return peak, growth, False, False, float("nan"), bulk_peak
            speed = np.linalg.norm(vel, axis=1)
            peak = max(peak, float(speed.max()))
            bulk_peak = max(bulk_peak, float(np.percentile(speed, 99.9)))
            energy = 0.5 * solver.mass * float((speed**2).sum()) + solver.mass * GRAVITY * float(pos[:, 2].sum())
            e0 = energy if e0 is None else e0
            growth = max(growth, (energy - e0) / abs(e0))
    pos, vel = solver.pos_sim.numpy().astype(np.float64), solver.vel.numpy().astype(np.float64)
    inside = bool((pos >= -1e-5).all() and (pos[:, :2] <= 1.0 + 1e-5).all() and (pos[:, 2] <= 2.0 + 1e-5).all())
    return peak, growth, True, inside, 0.5 * solver.mass * float((vel**2).sum()), bulk_peak


def check_stable(solver, result):
    peak, growth, finite, inside, _, bulk_peak = result
    sound_speed = (7.0 * solver.stiffness / solver.rest_density) ** 0.5
    assert finite, "NaN or Inf"
    assert inside, "particles left the domain"
    assert growth < 0.01, f"energy grew by {growth:.1%}"
    assert bulk_peak < 0.5 * sound_speed, f"99.9th-percentile speed {bulk_peak:.1f} m/s is above half the sound speed"
    assert peak < 3.0 * sound_speed, f"peak speed {peak:.1f} m/s against a sound speed of {sound_speed:.1f}"


# -- the preset data ---------------------------------------------------------------------------------

def test_presets_match_their_menu_items_and_start_from_the_defaults():
    assert [key for key, _, _ in PRESET_ITEMS] == list(PRESETS)
    water = PRESETS["WATER"]
    solver = RealtimeWCSPHSolver(num_particles=1, h=0.04, enable_obstacles=0, device="cpu")
    # Water is the tuned default: choosing it changes nothing about a fresh run.
    assert (water["stiffness"] * 1000.0, water["time_step_scale"], water["viscosity"], water["substeps"]) == \
        (solver.stiffness, solver.cfl_factor, solver.viscosity, solver.substeps)
    assert PRESETS["VISCOUS"]["viscosity"] > 10 * water["viscosity"]
    assert PRESETS["STYLISED"]["stiffness"] < water["stiffness"]
    # dt = cfl * h / c and c ~ sqrt(stiffness): the softer preset takes larger steps at the same CFL.
    step_size = {k: v["time_step_scale"] / v["stiffness"] ** 0.5 for k, v in PRESETS.items()}
    assert step_size["STYLISED"] > 1.15 * step_size["WATER"] and PRESETS["STYLISED"]["substeps"] > water["substeps"]


def test_apply_and_detect_a_preset():
    settings = types.SimpleNamespace(stiffness=0.0, time_step_scale=0.0, viscosity=0.0, substeps=0, point_size=2.0)
    assert active_preset(settings) is None
    for key in PRESETS:
        apply_preset(settings, key)
        assert active_preset(settings) == key
        assert settings.point_size == 2.0  # only physics is touched
        settings.viscosity = np.float32(settings.viscosity)  # stored as float32 in Blender
        settings.stiffness = float(np.float32(settings.stiffness))
        assert active_preset(settings) == key
        settings.viscosity += 0.5  # any edit leaves the preset
        assert active_preset(settings) is None
    assert preset_settings("VISCOUS").viscosity == PRESETS["VISCOUS"]["viscosity"]


# -- stability ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(PRESETS))
def test_preset_is_stable_on_the_cpu(key):
    solver = build(preset_settings(key), 0.05, "cpu")
    assert solver.num_particles > 1000
    check_stable(solver, run(solver, 0.6, use_graph=False))


@needs_cuda
@pytest.mark.parametrize("key", list(PRESETS))
@pytest.mark.parametrize("spacing", [0.02, 0.01])
def test_preset_is_stable_on_cuda_with_the_graph(key, spacing):
    solver = build(preset_settings(key), spacing, "cuda:0")
    assert solver.cuda_graph is not None
    check_stable(solver, run(solver, 1.0, use_graph=True))


# -- viscosity ---------------------------------------------------------------------------------------

def test_viscosity_damps_motion_on_the_cpu():
    thin, thick = preset_settings("WATER"), preset_settings("VISCOUS")
    results = {}
    for name, settings in (("thin", thin), ("thick", thick)):
        solver = build(settings, 0.05, "cpu")
        results[name] = run(solver, 0.6, use_graph=False)
        check_stable(solver, results[name])
    assert results["thick"][4] < 0.5 * results["thin"][4], "kinetic energy should be damped"
    assert results["thick"][5] <= results["thin"][5], "the fast bulk should not speed up"


def test_effective_viscosity_is_capped_where_the_explicit_step_would_diverge():
    solver = RealtimeWCSPHSolver(num_particles=1, h=0.04, enable_obstacles=0, device="cpu")
    solver.viscosity = 0.15
    assert solver.effective_viscosity == 0.15  # the default is never touched
    solver.viscosity = 1.0e6
    limit = VISCOSITY_RELAXATION_LIMIT * solver.mass / solver.dt
    assert solver.effective_viscosity == pytest.approx(limit) and 0.0 < limit < 100.0
    solver.viscosity = 0.0
    assert solver.effective_viscosity == 0.0
    # Even the largest setting (20) stays finite at a fine spacing, where the cap binds hardest.
    fine = build(types.SimpleNamespace(stiffness=90.0, time_step_scale=0.4, viscosity=20.0, substeps=2), 0.01, "cpu",
                 SceneDescription((0, 0, 0), (0.2, 0.2, 0.4), (0, 0, -GRAVITY),
                                  (box_mesh((0, 0, 0), (0.2, 0.2, 0.1), "pool"),), ()))
    assert fine.effective_viscosity < 20.0
    out = wp.zeros(fine.num_particles, dtype=wp.vec3, device="cpu")
    for _ in range(30):
        fine.step(out, use_graph=False)
    assert np.isfinite(fine.pos_sim.numpy()).all() and np.isfinite(fine.vel.numpy()).all()


@needs_cuda
def test_viscosity_change_needs_a_graph_rebuild_and_then_matches_eager():
    """The captured graph holds the viscosity by value: it keeps the old one until it is rebuilt."""
    settings = preset_settings("WATER")

    def advance(solver, steps, use_graph):
        out = wp.zeros(solver.num_particles, dtype=wp.vec3, device="cuda:0")
        for _ in range(steps):
            solver.step(out, use_graph=use_graph)
        return solver.pos_sim.numpy(), solver.vel.numpy()

    stale = build(settings, 0.03, "cuda:0")
    stale.viscosity = 8.0  # changed without rebuilding: the graph is unaware
    fresh = build(settings, 0.03, "cuda:0")
    fresh.viscosity = 8.0
    fresh.rebuild_cuda_graph()
    eager = build(settings, 0.03, "cuda:0")
    eager.viscosity = 8.0
    original = build(settings, 0.03, "cuda:0")

    p_stale, _ = advance(stale, 80, True)
    p_fresh, _ = advance(fresh, 80, True)
    p_eager, _ = advance(eager, 80, False)
    p_original, _ = advance(original, 80, True)
    np.testing.assert_array_equal(p_stale, p_original)  # still the old viscosity
    np.testing.assert_array_equal(p_fresh, p_eager)  # the rebuilt graph agrees with an eager run
    assert np.abs(p_fresh - p_original).max() > 1e-4  # and the change is visible
