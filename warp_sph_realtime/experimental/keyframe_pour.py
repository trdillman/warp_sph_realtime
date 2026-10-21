"""Offline four-burst emitter: real WCSPH, eight emitter-position controls.

Particles inherit the emitter path's finite-difference velocity at release.
Unreleased particles are absent from all grids. Nothing steers released fluid.
This deliberately small experimental emitter is not a continuous-flow feature.
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
def append_burst(old_p: wp.array[wp.vec3], old_v: wp.array[wp.vec3], old_id: wp.array[int],
                 p: wp.array[wp.vec3], v: wp.array[wp.vec3], ids: wp.array[int],
                 controls: wp.array[float], old_count: int, key: int, interval: float):
    i = wp.tid()
    if i < old_count:
        p[i] = old_p[i]
        v[i] = old_v[i]
        ids[i] = old_id[i]
    else:
        j = i - old_count
        p[i] = wp.vec3(-1.45 + controls[2 * key] + (float(j // 4) - 1.5) * 0.06,
                       controls[2 * key + 1] + (float(j % 4) - 1.5) * 0.06, 0.50)
        left = wp.max(key - 1, 0)
        right = wp.min(key + 1, 3)
        duration = float(right - left) * interval
        v[i] = wp.vec3((controls[2 * right] - controls[2 * left]) / duration,
                       (controls[2 * right + 1] - controls[2 * left + 1]) / duration, -2.0)
        ids[i] = i


@wp.kernel
def reorder_ids(grid: wp.uint64, ids: wp.array[int], output: wp.array[int]):
    i = wp.tid()
    output[i] = ids[wp.hash_grid_point_id(grid, i)]


@wp.func
def segment_hits_spill(a: wp.vec3, b: wp.vec3):
    lower = wp.vec3(-1.22, -0.40, 0.30)
    upper = wp.vec3(-0.68, 0.40, 0.40)
    enter = float(0.0)
    leave = float(1.0)
    hit = int(1)
    for axis in range(3):
        delta = b[axis] - a[axis]
        if wp.abs(delta) < 1e-10:
            if a[axis] < lower[axis] or a[axis] > upper[axis]:
                hit = 0
        else:
            t0 = (lower[axis] - a[axis]) / delta
            t1 = (upper[axis] - a[axis]) / delta
            enter = wp.max(enter, wp.min(t0, t1))
            leave = wp.min(leave, wp.max(t0, t1))
    if enter > leave:
        hit = 0
    return hit


@wp.kernel
def spill_objective(previous: wp.array[wp.vec3], p: wp.array[wp.vec3], ids: wp.array[int], spilled: wp.array[int],
                    loss: wp.array[float], scale: float):
    i = wp.tid()
    x = p[i]
    # A dry display surface adjoining the right side of the receiving region.
    # C1 squared penetration supplies a useful gradient inside its soft margin.
    px = wp.max(0.30 - wp.abs(x[0] + 0.95), 0.0)
    py = wp.max(0.45 - wp.abs(x[1]), 0.0)
    pz = wp.max(0.09 - wp.abs(x[2] - 0.35), 0.0)
    wp.atomic_add(loss, 0, scale * px * px * py * py * pz * pz)
    if segment_hits_spill(previous[i], x) == 1:
        wp.atomic_max(spilled, ids[i], 1)


@wp.kernel
def mark_crossings(p: wp.array[wp.vec3], q: wp.array[wp.vec3], ids: wp.array[int],
                   crossed: wp.array[int], mask: wp.array[int]):
    i = wp.tid()
    mask[i] = 0
    if crossed[ids[i]] == 0 and p[i][2] > 0.35 and q[i][2] <= 0.35:
        mask[i] = 1
        crossed[ids[i]] = 1


@wp.kernel
def crossing_objective(p: wp.array[wp.vec3], q: wp.array[wp.vec3], ids: wp.array[int],
                       mask: wp.array[int], received: wp.array[int], loss: wp.array[float]):
    i = wp.tid()
    if mask[i] == 1:
        alpha = (p[i][2] - 0.35) / (p[i][2] - q[i][2])
        x = p[i] + alpha * (q[i] - p[i])
        dx = x[0] + 1.45
        dy = x[1]
        ex = wp.max(wp.abs(dx) - 0.22, 0.0)
        ey = wp.max(wp.abs(dy) - 0.22, 0.0)
        wp.atomic_add(loss, 0, (ex * ex + ey * ey + 0.02 * (dx * dx + dy * dy)) / 64.0)
        if wp.abs(dx) <= 0.22 and wp.abs(dy) <= 0.22:
            received[ids[i]] = 1


@wp.kernel
def final_objective(p: wp.array[wp.vec3], ids: wp.array[int], crossed: wp.array[int], loss: wp.array[float]):
    i = wp.tid()
    x = p[i]
    dx = x[0] + 1.45
    dy = x[1]
    ex = wp.max(wp.abs(dx) - 0.22, 0.0)
    ey = wp.max(wp.abs(dy) - 0.22, 0.0)
    ez = wp.max(x[2] - 0.35, 0.0)
    if crossed[ids[i]] == 0:
        wp.atomic_add(loss, 0, (ex * ex + ey * ey + ez * ez + 0.02 * (dx * dx + dy * dy)) / 64.0)


def outcome_rank(result):
    """Hard artist constraints decide feasibility before the smooth score."""
    violation = max(0.90 - result["coverage"], 0.0) + max(result["spill"] - 0.05, 0.0)
    return (violation > 0, violation, result["loss"])


def segments_hit_box(a, b, lower, upper):
    """Independent NumPy slab intersection, including crossings between samples."""
    delta = b - a
    parallel = np.abs(delta) < 1e-10
    safe_delta = np.where(parallel, 1.0, delta)
    first = (np.asarray(lower) - a) / safe_delta
    last = (np.asarray(upper) - a) / safe_delta
    enter = np.maximum(0.0, np.where(parallel, -np.inf, np.minimum(first, last)).max(axis=1))
    leave = np.minimum(1.0, np.where(parallel, np.inf, np.maximum(first, last)).min(axis=1))
    parallel_outside = (parallel & ((a < lower) | (a > upper))).any(axis=1)
    return (enter <= leave) & ~parallel_outside


class KeyframePour:
    steps = 640
    release_steps = (0, 96, 192, 288)
    dimensions = 8
    bound = 0.30

    def __init__(self):
        if wp.__version__ != "1.17.0" or not wp.is_cuda_available():
            raise RuntimeError("Requires warp-lang==1.17.0 and CUDA; no fallback")
        self.device = "cuda:0"
        self.solver = RealtimeWCSPHSolver(64, h=0.12, substeps=1, enable_obstacles=2)
        self.solver.set_particle_spacing(0.06)
        self.solver.cfl_factor = 0.2
        self.solver.update_time_step()
        self.interval = 96 * self.solver.dt
        self.controls = wp.zeros(8, dtype=float, device=self.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)
        self.spilled = wp.zeros(64, dtype=int, device=self.device)
        self.crossed = wp.zeros(64, dtype=int, device=self.device)
        self.received = wp.zeros(64, dtype=int, device=self.device)
        self.empty_p = self.vectors(1)
        self.empty_ids = wp.zeros(1, dtype=int, device=self.device)
        self.states = []
        for t in range(self.steps):
            n = 16 * (1 + sum(t >= r for r in self.release_steps[1:]))
            self.states.append({
                "n": n, "input_p": self.vectors(n), "input_v": self.vectors(n),
                "input_id": wp.zeros(n, dtype=int, device=self.device),
                "p": self.vectors(n), "v": self.vectors(n), "out_p": self.vectors(n),
                "out_v": self.vectors(n), "id": wp.zeros(n, dtype=int, device=self.device),
                "crossing_mask": wp.zeros(n, dtype=int, device=self.device),
                "rho": wp.zeros(n, dtype=float, device=self.device, requires_grad=True),
                "order": wp.HashGrid(16, 16, 16, device=self.device),
                "neighbors": wp.HashGrid(16, 16, 16, device=self.device),
            })

    def vectors(self, count):
        return wp.zeros(count, dtype=wp.vec3, device=self.device, requires_grad=True)

    @classmethod
    def validate_controls(cls, controls):
        values = np.asarray(controls, dtype=np.float32)
        if values.shape != (8,) or not np.isfinite(values).all() or np.any(np.abs(values) > cls.bound + 1e-7):
            raise ValueError("Eight finite keyframe position offsets in [-0.30, 0.30] required")
        return values

    def forward(self, graph=False):
        s = self.solver
        previous_p, previous_v, previous_ids = self.empty_p, self.empty_p, self.empty_ids
        for t, state in enumerate(self.states):
            n = state["n"]
            if t in self.release_steps:
                key = self.release_steps.index(t)
                wp.launch(append_burst, n, [previous_p, previous_v, previous_ids,
                          state["input_p"], state["input_v"], state["input_id"],
                          self.controls, key * 16, key, self.interval], device=self.device)
                previous_p, previous_v, previous_ids = state["input_p"], state["input_v"], state["input_id"]
            state["order"].build(previous_p, s.h)
            wp.launch(k_reorder_soa, n, [state["order"].id, previous_p, previous_v, state["p"], state["v"]],
                      device=self.device)
            wp.launch(reorder_ids, n, [state["order"].id, previous_ids, state["id"]], device=self.device)
            state["neighbors"].build(state["p"], s.h)
            arguments_density = [state["neighbors"].id, state["p"], state["rho"], s.mass, s.h,
                                 s.rest_density, s.enable_obstacles]
            arguments_force = [state["neighbors"].id, state["p"], state["v"], state["rho"],
                               state["out_p"], state["out_v"], s.mass, s.h, s.rest_density,
                               s.stiffness, s.viscosity, s.dt, s.domain_min, s.domain_max, s.enable_obstacles]
            if graph:
                if "graph" not in state:
                    with wp.ScopedCapture(device=self.device) as capture:
                        wp.launch(k_compute_density_coalesced, n, arguments_density, device=self.device)
                        wp.launch(k_wcsph_forces_and_copyback, n, arguments_force, device=self.device)
                    state["graph"] = capture.graph
                wp.capture_launch(state["graph"])
            else:
                wp.launch(k_compute_density_coalesced, n, arguments_density, device=self.device)
                wp.launch(k_wcsph_forces_and_copyback, n, arguments_force, device=self.device)
            wp.launch(spill_objective, n, [state["p"], state["out_p"], state["id"], self.spilled, self.loss,
                      200.0 / (64.0 * self.steps)], device=self.device)
            wp.launch(mark_crossings, n, [state["p"], state["out_p"], state["id"], self.crossed,
                      state["crossing_mask"]], device=self.device)
            wp.launch(crossing_objective, n, [state["p"], state["out_p"], state["id"],
                      state["crossing_mask"], self.received, self.loss], device=self.device)
            previous_p, previous_v, previous_ids = state["out_p"], state["out_v"], state["id"]
        wp.launch(final_objective, 64, [previous_p, previous_ids, self.crossed, self.loss], device=self.device)

    def evaluate(self, controls, gradient=False, graph=False):
        values = self.validate_controls(controls)
        if gradient and graph:
            raise ValueError("Graph is forward verification only")
        wp.synchronize_device(self.device)
        began = time.perf_counter()
        self.controls.assign(values)
        self.loss.zero_()
        self.spilled.zero_()
        self.crossed.zero_()
        self.received.zero_()
        tape = wp.Tape()
        if gradient:
            with tape:
                self.forward()
            tape.backward(self.loss)
            derivative = self.controls.grad.numpy().copy()
        else:
            self.forward(graph)
            derivative = None
        p = self.states[-1]["out_p"].numpy().copy()
        v = self.states[-1]["out_v"].numpy().copy()
        loss = float(self.loss.numpy()[0])
        spill = float(self.spilled.numpy().mean())
        coverage = float(self.received.numpy().mean())
        if gradient:
            tape.zero()
        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - began
        if not all(np.isfinite(x).all() for x in (p, v, loss)):
            raise FloatingPointError("Non-finite rollout")
        if derivative is not None and not np.isfinite(derivative).all():
            raise FloatingPointError("Non-finite gradient")
        return {"loss": loss, "coverage": coverage, "spill": spill,
                "success": bool(coverage >= 0.90 and spill <= 0.05),
                "gradient": derivative, "positions": p, "velocities": v, "seconds": elapsed}

    def check_gradient(self, controls, epsilon=0.0005):
        values = self.validate_controls(controls)
        ad = self.evaluate(values, gradient=True)["gradient"]
        fd = []
        for axis in range(8):
            delta = np.zeros(8, dtype=np.float32)
            delta[axis] = epsilon
            fd.append((self.evaluate(values + delta)["loss"] - self.evaluate(values - delta)["loss"]) / (2 * epsilon))
        return {"controls": values.tolist(), "analytic": ad.tolist(), "finite_difference": fd,
                "epsilon": epsilon, "rtol": 0.08, "atol": 0.0005,
                "passed": bool(np.allclose(ad, fd, rtol=0.08, atol=0.0005))}

    def trajectory(self, stride=8):
        frames = []
        minimum = np.full(3, np.inf)
        maximum = np.full(3, -np.inf)
        finite = True
        hard_floor = 0
        penetration = 0.0
        spilled = np.zeros(64, dtype=bool)
        crossed = np.zeros(64, dtype=bool)
        received = np.zeros(64, dtype=bool)
        for t, state in enumerate(self.states):
            p = state["out_p"].numpy()
            previous = state["p"].numpy()
            ids = state["id"].numpy()
            velocity = state["out_v"].numpy()
            finite = finite and bool(np.isfinite(p).all() and np.isfinite(previous).all()
                                     and np.isfinite(velocity).all() and np.isfinite(state["v"].numpy()).all())
            visit = segments_hit_box(previous, p, [-1.22, -0.40, 0.30], [-0.68, 0.40, 0.40])
            spilled[ids[visit]] = True
            crossing = (~crossed[ids]) & (previous[:, 2] > 0.35) & (p[:, 2] <= 0.35)
            if crossing.any():
                a, b = previous[crossing], p[crossing]
                alpha = (a[:, 2] - 0.35) / (a[:, 2] - b[:, 2])
                landing = a + alpha[:, None] * (b - a)
                good = (np.abs(landing[:, 0] + 1.45) <= 0.22) & (np.abs(landing[:, 1]) <= 0.22)
                crossed[ids[crossing]] = True
                received[ids[crossing][good]] = True
            minimum = np.minimum(minimum, np.minimum(p.min(axis=0), previous.min(axis=0)))
            maximum = np.maximum(maximum, np.maximum(p.max(axis=0), previous.max(axis=0)))
            hard_floor += int((p[:, 2] <= 1e-7).sum())
            both = np.concatenate([previous, p])
            sphere = np.linalg.norm(both - [0, 0, 0.55], axis=1) - 0.55
            distances = [sphere]
            for x in (-0.85, 0.85):
                for y in (-0.55, 0.55):
                    distances.append(np.linalg.norm(both[:, :2] - [x, y], axis=1) - 0.18)
            penetration = max(penetration, float(max(0, -np.minimum.reduce(distances).min())))
            if t % stride == 0 or t == self.steps - 1:
                frames.append({"step": t + 1, "positions": p.tolist()})
        return {"finite": finite, "minimum": minimum.tolist(), "maximum": maximum.tolist(),
                "hard_floor_clamps": hard_floor, "max_solid_penetration": penetration,
                "recomputed_spill": float(spilled.mean()), "recomputed_coverage": float(received.mean()),
                "crossed_fraction": float(crossed.mean()), "frames": frames}
