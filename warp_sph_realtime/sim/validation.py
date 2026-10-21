"""Headless physics validation: dam-break front vs Martin & Moyce, conservation, energy, rest density.

Runs the real `RealtimeWCSPHSolver` on any Warp device (no bpy), so CI can use the CPU device and a GPU
machine can run it at higher resolution. Method, reference provenance and measured numbers are in
docs/VALIDATION.md. Used by `tests/test_validation.py` and `scripts/validate_dambreak.py`.
"""

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from .sim_core import RealtimeWCSPHSolver, particle_density

# The legacy force kernel the validation drives has gravity built in (it does not read solver.gravity).
GRAVITY = 9.81

# Martin & Moyce (1952), surge front of a water column with height = 2 x width (their n^2 = 2):
#   T = t * sqrt(2 g / a)  (a = initial column width; here n*sqrt(g/a) with n^2 = 2),
#   Z = x / a              (x = front distance from the column's back wall; Z = 1 at T = 0).
# Source: J. C. Martin and W. J. Moyce, "Part IV. An experimental study of the collapse of liquid
# columns on a rigid horizontal plane", Phil. Trans. R. Soc. Lond. A 244 (1952) 312-324,
# doi:10.1098/rsta.1952.0006. The paper itself was not reachable when this was written (publisher
# paywall). These 15 (T, Z) pairs are the values the open-source Ocellaris CFD code embeds in
# scripts/plot_dam_break.py with the comment "Data from Martin and Moyce (1952), Table 2 and 6". They
# were cross-checked against two independent digitisations of the paper's Figure 3 (PySPH,
# pysph/examples/db_exp_data.py, columns of a = 1.125 in and a = 2.25 in); both agree with these pairs to
# |dZ| <= 0.11 (see docs/VALIDATION.md). Z = 1 at T = 0 is the initial geometry, not a measurement.
MARTIN_MOYCE_T = np.array(
    [0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.20, 2.32, 2.51, 2.65, 2.83, 2.98, 3.11, 3.33])
MARTIN_MOYCE_Z = np.array(
    [1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67, 3.89, 4.11])


@dataclass(frozen=True)
class DamBreakConfig:
    """A quasi-2D dam break: a slab of fluid released against a free-slip tank (n^2 = 2: height = 2 a).

    Particles sit on the solver's cubic lattice (h = 2 * spacing). The fluid occupies [0, a] x [0, W] x
    [0, 2a]; the domain walls lie half a spacing outside it, i.e. one spacing from the outermost particle
    centres. The solver's soft wall layer is `spacing` wide, so a lattice started inside it (the
    `fill_particles` convention of centres at half a spacing) is expelled violently.
    """

    cells_across: int = 12  # particles across the column width a
    thickness_cells: int = 4  # particles across the slab (the y direction)
    column_width: float = 0.25  # a [m]
    length_ratio: float = 5.0  # tank length / a; > the reference's last Z so the end wall is never reached
    end_T: float = float(MARTIN_MOYCE_T[-1])
    sample_every: int = 2  # solver step() calls between measurements
    device: str = "cuda:0"

    @property
    def spacing(self) -> float:
        return self.column_width / self.cells_across

    @property
    def num_particles(self) -> int:
        return self.cells_across * self.thickness_cells * 2 * self.cells_across


@dataclass
class Run:
    """Time series and end state of one simulation."""

    num_particles: int
    spacing: float
    mass: float
    time: np.ndarray  # simulated seconds at each sample
    kinetic: np.ndarray
    potential: np.ndarray  # m g z with z measured from the fluid's floor plane (z = 0)
    positions: np.ndarray  # final (N, 3)
    velocities: np.ndarray
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    wall_seconds: float
    steps: int  # solver step() calls

    @property
    def energy(self) -> np.ndarray:
        return self.kinetic + self.potential


@dataclass
class DamBreakRun(Run):
    T: np.ndarray  # non-dimensional time t * sqrt(2 g / a)
    Z: np.ndarray  # non-dimensional front position x / a
    column_width: float


def lattice_block(nx: int, ny: int, nz: int, spacing: float) -> np.ndarray:
    """(nx*ny*nz, 3) float32 cell-centred lattice with its lower corner at the origin."""
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return ((np.stack([i, j, k], axis=-1).reshape(-1, 3) + 0.5) * spacing).astype(np.float32)


def make_solver(points: np.ndarray, spacing: float, domain_min, domain_max, device: str) -> RealtimeWCSPHSolver:
    """A solver with default physics (stiffness, viscosity, CFL) holding `points` at rest in a closed box."""
    solver = RealtimeWCSPHSolver(
        num_particles=len(points), h=2.0 * spacing, substeps=2, enable_obstacles=0, device=device)
    solver.set_particle_spacing(spacing)
    solver.domain_min = wp.vec3(*domain_min)
    solver.domain_max = wp.vec3(*domain_max)
    solver.rebuild_cuda_graph()  # the captured launch snapshots h, mass, dt and the domain
    solver.pos_sim.assign(points)
    solver.vel.zero_()
    return solver


def kinetic_potential(solver: RealtimeWCSPHSolver) -> tuple[float, float]:
    pos, vel = solver.pos_sim.numpy().astype(np.float64), solver.vel.numpy().astype(np.float64)
    kinetic = 0.5 * solver.mass * float((vel * vel).sum())
    potential = solver.mass * GRAVITY * float(pos[:, 2].sum())
    return kinetic, potential


def front_position(positions: np.ndarray, spacing: float, rows: int) -> float:
    """x of the leading edge of the fluid along the floor.

    Mean of the `rows` foremost particle centres within the two lowest lattice layers (one per slab row,
    so a single stray particle cannot set the front), plus half a spacing to reach the fluid's edge.
    """
    low = positions[positions[:, 2] < 2.0 * spacing]
    if len(low) == 0:
        return float("nan")
    leading = np.sort(low[:, 0])[-rows:]
    return float(leading.mean()) + 0.5 * spacing


def _advance(solver: RealtimeWCSPHSolver, seconds: float, sample_every: int, on_sample) -> int:
    """Steps for `seconds` of simulated time; calls on_sample(t) at t = 0 and after every sample_every steps."""
    scratch = wp.zeros(solver.num_particles, dtype=wp.vec3, device=solver.device)
    per_step = solver.substeps * solver.dt
    steps = int(np.ceil(seconds / per_step))
    on_sample(0.0)
    for k in range(1, steps + 1):
        solver.step(scratch)
        if k % sample_every == 0 or k == steps:
            on_sample(k * per_step)
    return steps


def run_dam_break(config: DamBreakConfig | None = None) -> DamBreakRun:
    """Release the column at t = 0 and record the front Z(T) plus kinetic and potential energy."""
    config = config or DamBreakConfig()
    n, w, spacing, a = config.cells_across, config.thickness_cells, config.spacing, config.column_width
    points = lattice_block(n, w, 2 * n, spacing)
    domain_min = (-0.5 * spacing,) * 3
    domain_max = (config.length_ratio * a, w * spacing + 0.5 * spacing, 4.0 * a)
    solver = make_solver(points, spacing, domain_min, domain_max, config.device)

    t_scale = float(np.sqrt(2.0 * GRAVITY / a))
    times, kinetic, potential, fronts = [], [], [], []

    def sample(t: float):
        k, p = kinetic_potential(solver)
        times.append(t)
        kinetic.append(k)
        potential.append(p)
        fronts.append(front_position(solver.pos_sim.numpy(), spacing, w) / a)

    start = time.perf_counter()
    steps = _advance(solver, config.end_T / t_scale, config.sample_every, sample)
    wall = time.perf_counter() - start
    t = np.array(times)
    return DamBreakRun(
        num_particles=len(points), spacing=spacing, mass=solver.mass, time=t, kinetic=np.array(kinetic),
        potential=np.array(potential), positions=solver.pos_sim.numpy(), velocities=solver.vel.numpy(),
        domain_min=domain_min, domain_max=domain_max, wall_seconds=wall, steps=steps,
        T=t * t_scale, Z=np.array(fronts), column_width=a)


def compare_front(T: np.ndarray, Z: np.ndarray, ref_T=MARTIN_MOYCE_T, ref_Z=MARTIN_MOYCE_Z) -> dict:
    """Errors of the simulated front against the reference at the reference times.

    Error = Z_sim(T_ref) - Z_ref, positive when the simulated front is ahead of the experiment. Z_sim is
    interpolated linearly; only reference times inside the simulated range are used.
    """
    inside = (ref_T >= T[0]) & (ref_T <= T[-1])
    rt, rz = np.asarray(ref_T)[inside], np.asarray(ref_Z)[inside]
    err = np.interp(rt, T, Z) - rz
    # Excess travel measured from the Z = 1 starting line, so early points do not look better than they are.
    rel = err / (rz - 1.0)
    return {
        "points": int(inside.sum()),
        "T": rt.tolist(),
        "Z_reference": rz.tolist(),
        "Z_simulated": (rz + err).tolist(),
        "error": err.tolist(),
        "mean_error": float(err.mean()),
        "mean_abs_error": float(np.abs(err).mean()),
        "rms_error": float(np.sqrt((err**2).mean())),
        "max_abs_error": float(np.abs(err).max()),
        "max_abs_relative_error": float(np.abs(err / rz).max()),
        "mean_abs_excess_relative_error": float(np.abs(rel).mean()),
    }


@wp.kernel
def _k_summation_density(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    density: wp.array[float],
    mass: float,
    h: float,
):
    tid = int(wp.tid())
    density[tid] = particle_density(grid, pos, pos[tid], mass, h)


def summation_density(solver: RealtimeWCSPHSolver, positions: np.ndarray) -> np.ndarray:
    """Unclamped SPH density sum at `positions`, with the solver's kernel, mass and h.

    The solver's own `density_sorted` is clamped to >= rest density before it feeds the EOS, so it hides
    kernel deficiency and cannot measure a density error.
    """
    pos = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=solver.device)
    rho = wp.zeros(len(positions), dtype=float, device=solver.device)
    grid = wp.HashGrid(128, 128, 128, device=solver.device)
    grid.build(points=pos, radius=solver.h)
    wp.launch(_k_summation_density, dim=len(positions), inputs=[grid.id, pos, rho, solver.mass, solver.h],
              device=solver.device)
    return rho.numpy()


@dataclass
class RestRun(Run):
    density_ratio: np.ndarray  # final rho / rho0 per particle
    bulk: np.ndarray  # mask: at least h (full kernel support) from every wall and from the free surface


def run_rest(cells: int = 10, spacing: float = 0.025, seconds: float = 0.5, sample_every: int = 5,
             device: str = "cuda:0") -> RestRun:
    """A cube of fluid released from the lattice into a closed tank that fits it; it settles under gravity.

    The tank walls are one spacing from the outermost centres on the sides and floor; the top is open
    (free surface) well above the fluid.
    """
    points = lattice_block(cells, cells, cells, spacing)
    side = cells * spacing + 0.5 * spacing
    domain_min = (-0.5 * spacing,) * 3
    domain_max = (side, side, 4.0 * cells * spacing)
    solver = make_solver(points, spacing, domain_min, domain_max, device)
    times, kinetic, potential = [], [], []

    def sample(t: float):
        k, p = kinetic_potential(solver)
        times.append(t)
        kinetic.append(k)
        potential.append(p)

    start = time.perf_counter()
    steps = _advance(solver, seconds, sample_every, sample)
    wall = time.perf_counter() - start
    pos = solver.pos_sim.numpy()
    rho = summation_density(solver, pos) / solver.rest_density
    lo, hi = np.asarray(domain_min), np.asarray(domain_max)
    margin = solver.h
    bulk = ((pos[:, :2] > lo[:2] + margin) & (pos[:, :2] < hi[:2] - margin)).all(axis=1)
    bulk &= (pos[:, 2] > lo[2] + margin) & (pos[:, 2] < pos[:, 2].max() - margin)
    return RestRun(
        num_particles=len(points), spacing=spacing, mass=solver.mass, time=np.array(times),
        kinetic=np.array(kinetic), potential=np.array(potential), positions=pos,
        velocities=solver.vel.numpy(), domain_min=domain_min, domain_max=domain_max, wall_seconds=wall,
        steps=steps, density_ratio=rho, bulk=bulk)


def conservation_report(run: Run) -> dict:
    """Particle count, finiteness and domain containment of the final state, plus the energy history."""
    pos, vel = run.positions.astype(np.float64), run.velocities.astype(np.float64)
    lo, hi = np.asarray(run.domain_min), np.asarray(run.domain_max)
    outside = np.maximum(np.maximum(lo - pos, pos - hi), 0.0)
    energy = run.energy
    return {
        "particles_initial": run.num_particles,
        "particles_final": int(len(pos)),
        "finite": bool(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(energy).all()),
        "max_distance_outside_domain": float(outside.max()),
        "energy_initial": float(energy[0]),
        "energy_final": float(energy[-1]),
        "max_energy_growth_fraction": float((energy.max() - energy[0]) / energy[0]),
        "final_energy_fraction": float(energy[-1] / energy[0]),
        "final_kinetic_fraction": float(run.kinetic[-1] / energy[0]),
    }


def rest_density_report(run: RestRun) -> dict:
    dev = np.abs(run.density_ratio - 1.0)
    bulk = run.bulk
    return {
        "mean_abs_density_error_all": float(dev.mean()),
        "max_abs_density_error_all": float(dev.max()),
        "bulk_particles": int(bulk.sum()),
        "mean_abs_density_error_bulk": float(dev[bulk].mean()) if bulk.any() else float("nan"),
        "max_abs_density_error_bulk": float(dev[bulk].max()) if bulk.any() else float("nan"),
        "mean_density_ratio_all": float(run.density_ratio.mean()),
    }
