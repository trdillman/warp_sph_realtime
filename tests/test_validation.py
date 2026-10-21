"""Physics validation: dam-break front vs Martin & Moyce (1952), conservation, energy and rest density.

CI has no GPU, so a coarse variant runs on Warp's CPU device. The same checks at higher resolution run only
where CUDA exists. Tolerances are regression guards set above the numbers measured in docs/VALIDATION.md; they
are not claims of accuracy beyond those measured deviations.
"""

import numpy as np
import pytest
import warp as wp

from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver
from warp_sph_realtime.sim.validation import (
    MARTIN_MOYCE_T,
    MARTIN_MOYCE_Z,
    DamBreakConfig,
    compare_front,
    conservation_report,
    lattice_block,
    rest_density_report,
    run_dam_break,
    run_rest,
)

needs_cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")

# Measured and bounded in docs/VALIDATION.md.
MAX_ENERGY_GROWTH = 0.01  # fraction of the initial energy (measured: none)
BULK_DENSITY_ERROR = 0.01  # mean |rho/rho0 - 1| away from boundaries (measured <= 0.33 %)
ALL_DENSITY_ERROR = 0.02  # mean over every particle including the kernel-deficient surface (measured <= 0.92 %)


@pytest.fixture(scope="module")
def cpu_dam_break():
    return run_dam_break(DamBreakConfig(cells_across=12, thickness_cells=4, device="cpu"))


@pytest.fixture(scope="module")
def cpu_rest():
    return run_rest(cells=10, seconds=0.5, device="cpu")


def check_conservation(run):
    report = conservation_report(run)
    assert report["particles_final"] == report["particles_initial"]
    assert report["finite"]
    assert report["max_distance_outside_domain"] <= 1e-6
    assert report["max_energy_growth_fraction"] <= MAX_ENERGY_GROWTH
    assert report["final_energy_fraction"] < 1.0, "a dissipative WCSPH run must end below its initial energy"
    return report


def test_reference_data_is_well_formed():
    assert MARTIN_MOYCE_T.shape == MARTIN_MOYCE_Z.shape == (15,)
    assert (np.diff(MARTIN_MOYCE_T) > 0).all() and (np.diff(MARTIN_MOYCE_Z) > 0).all()
    assert MARTIN_MOYCE_Z[0] > 1.0  # the column has not moved at Z = 1
    # From T = 1.19 on, Z rises in even steps of 2/9 (guards against a transcription typo).
    assert np.allclose(np.diff(MARTIN_MOYCE_Z)[2:], 2.0 / 9.0, atol=0.012)


def test_compare_front_reports_signed_error_and_ignores_out_of_range_points():
    T = np.linspace(0.0, 2.0, 21)
    Z = 1.0 + 1.5 * T
    ref_T, ref_Z = np.array([0.5, 1.0, 3.0]), np.array([1.5, 2.5, 5.0])
    result = compare_front(T, Z, ref_T, ref_Z)
    assert result["points"] == 2  # T = 3 is beyond the run
    assert result["error"] == pytest.approx([1.75 - 1.5, 2.5 - 2.5])
    assert result["mean_error"] > 0 and result["max_abs_error"] == pytest.approx(0.25)


def test_lattice_block_is_cell_centred():
    pts = lattice_block(2, 3, 4, 0.1)
    assert pts.shape == (24, 3) and pts.dtype == np.float32
    assert pts.min() == pytest.approx(0.05) and pts.max(axis=0) == pytest.approx([0.15, 0.25, 0.35])


def test_solver_cpu_device_runs_eagerly_without_a_graph():
    spacing = 0.05
    pts = lattice_block(3, 3, 3, spacing)
    solver = RealtimeWCSPHSolver(num_particles=len(pts), h=2 * spacing, enable_obstacles=0, device="cpu")
    assert solver.cuda_graph is None and solver.pos_sim.device == wp.get_device("cpu")
    solver.pos_sim.assign(pts)
    out = wp.zeros(len(pts), dtype=wp.vec3, device="cpu")
    solver.step(out)
    assert np.isfinite(out.numpy()).all()


def test_dam_break_conservation_and_energy_cpu(cpu_dam_break):
    report = check_conservation(cpu_dam_break)
    assert report["particles_final"] == cpu_dam_break.num_particles == 12 * 4 * 24


def test_dam_break_front_matches_martin_moyce_cpu(cpu_dam_break):
    """n = 12 particles across the column. Measured: mean error -0.10 (lagging), max |error| 6.3 % of Z."""
    result = compare_front(cpu_dam_break.T, cpu_dam_break.Z)
    assert result["points"] == len(MARTIN_MOYCE_T)
    assert cpu_dam_break.Z[0] == pytest.approx(1.0, abs=1e-6)
    assert result["max_abs_relative_error"] <= 0.10
    assert result["rms_error"] <= 0.20
    assert cpu_dam_break.Z[-1] > 3.5, "the front barely advanced"


def test_rest_settling_cpu(cpu_rest):
    check_conservation(cpu_rest)
    report = rest_density_report(cpu_rest)
    assert report["bulk_particles"] >= 300
    assert report["mean_abs_density_error_bulk"] <= BULK_DENSITY_ERROR
    assert report["mean_abs_density_error_all"] <= ALL_DENSITY_ERROR
    assert cpu_rest.kinetic[-1] < 0.01 * cpu_rest.energy[0], "the block did not come to rest"


@needs_cuda
def test_dam_break_front_matches_martin_moyce_gpu():
    """n = 24 particles across the column. Measured: mean error +0.01, rms 0.053, max |error| 6.3 % of Z."""
    run = run_dam_break(DamBreakConfig(cells_across=24, thickness_cells=4, device="cuda:0"))
    check_conservation(run)
    result = compare_front(run.T, run.Z)
    assert result["points"] == len(MARTIN_MOYCE_T)
    assert result["max_abs_relative_error"] <= 0.10
    assert result["rms_error"] <= 0.08


@needs_cuda
def test_cpu_and_gpu_follow_the_same_front(cpu_dam_break):
    gpu = run_dam_break(DamBreakConfig(cells_across=12, thickness_cells=4, device="cuda:0"))
    assert len(gpu.Z) == len(cpu_dam_break.Z)
    assert np.abs(gpu.Z - cpu_dam_break.Z).max() <= 0.02


@needs_cuda
def test_rest_settling_gpu():
    run = run_rest(cells=16, seconds=1.0, device="cuda:0")
    check_conservation(run)
    report = rest_density_report(run)
    assert report["mean_abs_density_error_bulk"] <= BULK_DENSITY_ERROR
    assert report["mean_abs_density_error_all"] <= ALL_DENSITY_ERROR
    assert run.kinetic[-1] < 0.01 * run.energy[0]
