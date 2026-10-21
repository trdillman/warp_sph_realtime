"""Offline inverse-control experiment using the unchanged production WCSPH kernels.

Each step owns its states AND hash grids: rebuilding a grid before backward would
silently change the neighbor queries and permutation used by the recorded tape.
Host readback is deliberate here; this module is never called by the modal timer.
"""

import time

import numpy as np
import warp as wp

from ..sim.sim_core import (
    RealtimeWCSPHSolver,
    k_compute_density_coalesced,
    k_reorder_soa,
    k_wcsph_forces_and_copyback,
)


@wp.kernel
def initial_velocity(controls: wp.array[float], velocity: wp.array[wp.vec3]):
    velocity[wp.tid()] = wp.vec3(controls[0], controls[1], 0.0)


@wp.kernel
def receiving_loss(positions: wp.array[wp.vec3], loss: wp.array[float], count: int):
    p = positions[wp.tid()]
    dx = p[0] + 0.8
    dy = p[1]
    # Distance outside the target volume is C1 after squaring; include height
    # so a visually aligned airborne particle is not mistaken for delivery.
    ex = wp.max(wp.abs(dx) - 0.22, 0.0)
    ey = wp.max(wp.abs(dy) - 0.22, 0.0)
    ez = wp.max(p[2] - 0.18, 0.0)
    wp.atomic_add(loss, 0, (ex * ex + ey * ey + ez * ez + 0.05 * (dx * dx + dy * dy)) / float(count))


@wp.kernel
def target_loss(positions: wp.array[wp.vec3], loss: wp.array[float], count: int):
    p = positions[wp.tid()]
    dx = p[0] - 0.30
    dy = p[1] - 0.20
    wp.atomic_add(loss, 0, (dx * dx + dy * dy) / float(count))


class PourExperiment:
    """Fixed 64-particle slug; controls are initial horizontal velocity in m/s."""

    def __init__(self, steps: int = 192, fixture: str = "floor"):
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 512:
            raise ValueError("steps must be an integer in [1, 512]")
        if fixture not in ("floor", "deflected", "reachable"):
            raise ValueError("fixture must be floor, deflected or reachable")
        if wp.__version__ != "1.17.0":
            raise RuntimeError("This experiment requires warp-lang==1.17.0")
        if not wp.is_cuda_available():
            raise RuntimeError("CUDA is required; no CPU substitute")
        self.steps = steps
        self.fixture = fixture
        self.device = "cuda:0"
        self.count = 64
        self.initial = np.array(
            [[-0.35 + i * 0.06, -0.09 + j * 0.06, 0.12 + k * 0.06]
             for i in range(4) for j in range(4) for k in range(4)], dtype=np.float32
        )
        if fixture in ("deflected", "reachable"):
            self.initial += np.array([-0.50, 0.0, 0.88], dtype=np.float32)
        if fixture == "reachable":
            self.initial[:, 2] -= 0.40
        self.solver = RealtimeWCSPHSolver(
            self.count, h=0.12, substeps=1,
            enable_obstacles=2 if fixture == "reachable" else int(fixture == "deflected"))
        self.solver.set_particle_spacing(0.06)
        if fixture == "reachable":
            self.solver.cfl_factor = 0.2
            self.solver.update_time_step()
        # Small hash tables change bucket order, not the WCSPH equations.
        # Use the same dimensions in the production parity/replay lane.
        self.solver.grid = wp.HashGrid(16, 16, 16, device=self.device)
        self.solver.rebuild_cuda_graph()
        self.positions = [self._vectors() for _ in range(steps + 1)]
        self.velocities = [self._vectors() for _ in range(steps + 1)]
        self.sorted_positions = [self._vectors() for _ in range(steps)]
        self.sorted_velocities = [self._vectors() for _ in range(steps)]
        self.densities = [wp.zeros(self.count, dtype=float, device=self.device, requires_grad=True)
                          for _ in range(steps)]
        self.order_grids = [wp.HashGrid(16, 16, 16, device=self.device) for _ in range(steps)]
        self.neighbor_grids = [wp.HashGrid(16, 16, 16, device=self.device) for _ in range(steps)]
        self.controls = wp.zeros(2, dtype=float, device=self.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)

    def _vectors(self):
        return wp.zeros(self.count, dtype=wp.vec3, device=self.device, requires_grad=True)

    @staticmethod
    def validate_controls(controls):
        values = np.asarray(controls, dtype=np.float32)
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(np.abs(values) > 2):
            raise ValueError("controls must contain two finite velocities in [-2, 2]")
        return values

    def _forward(self):
        s = self.solver
        wp.launch(initial_velocity, self.count, [self.controls, self.velocities[0]], device=self.device)
        for t in range(self.steps):
            order = self.order_grids[t]
            neighbors = self.neighbor_grids[t]
            p, v = self.sorted_positions[t], self.sorted_velocities[t]
            density = self.densities[t]
            order.build(self.positions[t], s.h)
            wp.launch(k_reorder_soa, self.count,
                      [order.id, self.positions[t], self.velocities[t], p, v], device=self.device)
            neighbors.build(p, s.h)
            wp.launch(k_compute_density_coalesced, self.count,
                      [neighbors.id, p, density, s.mass, s.h, s.rest_density, s.enable_obstacles], device=self.device)
            wp.launch(k_wcsph_forces_and_copyback, self.count,
                      [neighbors.id, p, v, density, self.positions[t + 1], self.velocities[t + 1],
                       s.mass, s.h, s.rest_density, s.stiffness, s.viscosity, s.dt,
                       s.domain_min, s.domain_max, s.enable_obstacles], device=self.device)
        objective = receiving_loss if self.fixture == "reachable" else target_loss
        wp.launch(objective, self.count, [self.positions[-1], self.loss, self.count], device=self.device)

    def evaluate(self, controls, gradient=False):
        values = self.validate_controls(controls)
        wp.synchronize_device(self.device)
        start = time.perf_counter()
        self.controls.assign(values)
        self.positions[0].assign(self.initial)
        self.loss.zero_()
        tape = wp.Tape()
        if gradient:
            with tape:
                self._forward()
            tape.backward(self.loss)
            derivative = self.controls.grad.numpy().copy()
        else:
            self._forward()
            derivative = None
        positions = self.positions[-1].numpy().copy()
        velocities = self.velocities[-1].numpy().copy()
        loss = float(self.loss.numpy()[0])
        if gradient:
            tape.zero()
        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - start
        if not (np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(loss)):
            raise FloatingPointError("Non-finite forward rollout")
        if derivative is not None and not np.isfinite(derivative).all():
            raise FloatingPointError("Non-finite backward gradient")
        if np.any(positions < np.array([-2.5, -2.5, 0]) - 1e-6) or np.any(
                positions > np.array([2.5, 2.5, 5]) + 1e-6):
            raise FloatingPointError("Domain violation")
        captured = (np.abs(positions[:, 0] - 0.30) <= 0.18) & (
            np.abs(positions[:, 1] - 0.20) <= 0.18) & (positions[:, 2] <= 0.18)
        if self.fixture == "reachable":
            captured = (np.abs(positions[:, 0] + 0.8) <= 0.22) & (
                np.abs(positions[:, 1]) <= 0.22) & (positions[:, 2] <= 0.18)
        return {"loss": loss, "gradient": derivative, "coverage": float(captured.mean()),
                "positions": positions, "velocities": velocities, "seconds": elapsed}

    def check_gradient(self, controls, epsilon=0.002):
        values = self.validate_controls(controls)
        analytic = self.evaluate(values, gradient=True)["gradient"]
        numerical = []
        for i in range(2):
            delta = np.zeros(2, dtype=np.float32)
            delta[i] = epsilon
            plus = self.evaluate(values + delta)["loss"]
            minus = self.evaluate(values - delta)["loss"]
            numerical.append((plus - minus) / (2 * epsilon))
        # Near-zero components use an absolute tolerance, others 5% relative.
        passed = bool(np.allclose(analytic, numerical, rtol=0.05, atol=0.002))
        return {"controls": values.tolist(), "analytic": analytic.tolist(), "finite_difference": numerical,
                "epsilon": epsilon, "passed": passed, "rtol": 0.05, "atol": 0.002}

    def production_rollout(self, controls, use_graph=False):
        values = self.validate_controls(controls)
        s = self.solver
        s.pos_sim.assign(self.initial)
        velocities = np.tile(np.array([values[0], values[1], 0], dtype=np.float32), (self.count, 1))
        s.vel.assign(velocities)
        output = wp.empty_like(s.pos_sim)
        for _ in range(self.steps):
            s.step(output, use_graph=use_graph)
        return s.pos_sim.numpy().copy(), s.vel.numpy().copy()

    def trajectory_diagnostics(self):
        """Offline validation only; inspect every retained forward state."""
        p = np.stack([state.numpy() for state in self.positions])
        sphere_distance = np.linalg.norm(p - [0.0, 0.0, 0.55], axis=2) - 0.55
        distances = [sphere_distance]
        for x in (-0.85, 0.85):
            for y in (-0.55, 0.55):
                distances.append(np.where(p[:, :, 2] < 2.2,
                                         np.linalg.norm(p[:, :, :2] - [x, y], axis=2) - 0.18, 10.0))
        minimum = np.minimum.reduce(distances)
        return {"max_solid_penetration": float(max(0.0, -minimum.min())),
                "contact_particle_steps": int((minimum < 0.06).sum()),
                "floor_contact_particle_steps": int((p[:, :, 2] < 0.06).sum()),
                "hard_floor_clamp_particle_steps": int((p[:, :, 2] <= 1e-7).sum()),
                "max_z": float(p[:, :, 2].max()),
                "finite_all_states": bool(np.isfinite(p).all())}
