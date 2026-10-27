import logging
import unittest

import numpy as np
import warp as wp

from warp_sph_realtime.sim.sim_core import (
    RealtimeWCSPHSolver,
    cubic_spline_grad,
    cubic_spline_kernel,
    wall_accel,
)


@wp.kernel
def k_eval_kernel_samples(
    radii: wp.array[float],
    h: float,
    out_w: wp.array[float],
    out_grad_x: wp.array[float],
):
    i = int(wp.tid())
    r = radii[i]
    out_w[i] = cubic_spline_kernel(r, h)
    g = cubic_spline_grad(wp.vec3(r, 0.0, 0.0), r, h)
    out_grad_x[i] = g[0]


@wp.kernel
def k_wall_damping(d: wp.array[float], v_into: float, r0: float, c: float, dt: float, out: wp.array[float]):
    i = int(wp.tid())
    out[i] = wall_accel(d[i], v_into, r0, c, dt) - wall_accel(d[i], 0.0, r0, c, dt)


def wall_damping(depths, v_into, r0, c, dt):
    d = wp.array(np.asarray(depths, dtype=np.float32), dtype=float, device="cpu")
    out = wp.zeros_like(d)
    wp.launch(k_wall_damping, dim=len(depths), inputs=[d, v_into, r0, c, dt, out], device="cpu")
    return out.numpy().astype(np.float64)


def test_wall_damping_never_reverses_the_approach_velocity():
    """The wall's approach damping removes omega * overlap * dt of the speed per step; omega * dt = 2 * cfl."""
    h, c, v = 0.04, 25.0, 3.0
    r0 = 0.5 * h
    depths = np.linspace(-0.5 * r0, r0, 13)  # overlap 1 (at and past the wall) down to 0
    overlap = 1.0 - np.maximum(depths, 0.0) / r0
    for cfl in (0.1, 0.4, 0.5, 0.75, 1.0):
        dt = cfl * h / c
        damping = wall_damping(depths, v, r0, c, dt)
        uncapped = c / r0 * v * overlap
        # Never more than stopping the approach in one step (a larger value flips the velocity's sign).
        assert (damping * dt <= v * (1.0 + 1e-5)).all(), cfl
        if cfl <= 0.5:  # the cap never binds: the original law, unchanged
            np.testing.assert_allclose(damping, uncapped, rtol=1e-5, atol=1e-3)
        else:
            np.testing.assert_allclose(damping, np.minimum(uncapped, v / dt), rtol=1e-5, atol=1e-3)
            assert (damping < uncapped - 1.0).any()


def test_viscosity_cap_is_reported_once(caplog):
    solver = RealtimeWCSPHSolver(num_particles=8, h=0.04, enable_obstacles=0, device="cpu")
    with caplog.at_level(logging.WARNING, logger="warp_sph_realtime.sim.sim_core"):
        solver.rebuild_cuda_graph()
        assert not solver.viscosity_limited and not caplog.records  # the default is never capped
        solver.viscosity = 1e6
        solver.rebuild_cuda_graph()
        solver.rebuild_cuda_graph()
    assert solver.viscosity_limited and solver.effective_viscosity < 1e6
    assert len(caplog.records) == 1 and "limited to" in caplog.records[0].getMessage()


class TestSimCorePhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def eval_kernel(self, r_np, h):
        radii = wp.array(np.asarray(r_np, dtype=np.float32), dtype=float, device=self.device)
        out_w = wp.zeros_like(radii)
        out_gx = wp.zeros_like(radii)
        wp.launch(k_eval_kernel_samples, dim=len(r_np), inputs=[radii, h, out_w, out_gx], device=self.device)
        return out_w.numpy().astype(np.float64), out_gx.numpy().astype(np.float64)

    def test_cubic_spline_kernel_values(self):
        """Normalisation, gradient = dW/dr, continuity at q = 1/2 and compact support, by value."""
        h = 0.04
        # Unit volume integral: 4 pi int_0^h r^2 W(r) dr = 1 (trapezoid on 20001 samples).
        r = np.linspace(0.0, h, 20001)
        w, gx = self.eval_kernel(r, h)
        self.assertAlmostEqual(float(np.trapezoid(4.0 * np.pi * r * r * w, r)), 1.0, delta=1e-4)
        # Closed form at the centre: W(0) = 8 / (pi h^3).
        self.assertAlmostEqual(w[0] / (8.0 / (np.pi * h ** 3)), 1.0, delta=1e-5)

        # The gradient along +x equals dW/dr: compare with a central difference of W itself.
        eps = 1e-4 * h
        r_mid = np.linspace(0.02 * h, 0.98 * h, 97)
        w_plus, _ = self.eval_kernel(r_mid + eps, h)
        w_minus, _ = self.eval_kernel(r_mid - eps, h)
        _, g_mid = self.eval_kernel(r_mid, h)
        fd = (w_plus - w_minus) / (2.0 * eps)
        scale = np.abs(fd).max()
        np.testing.assert_allclose(g_mid, fd, atol=2e-3 * scale)

        # C1 at the branch point q = 1/2: both pieces meet in value and slope.
        w_b, g_b = self.eval_kernel([0.5 * h * (1 - 1e-5), 0.5 * h * (1 + 1e-5)], h)
        self.assertAlmostEqual(w_b[0] / w_b[1], 1.0, delta=1e-4)
        self.assertAlmostEqual(g_b[0] / g_b[1], 1.0, delta=1e-4)

        # Compact support: zero at and beyond h, and vanishing continuously just inside it.
        w_edge, g_edge = self.eval_kernel([h, 1.0001 * h, 2.0 * h, 0.999 * h], h)
        self.assertEqual(list(w_edge[:3]), [0.0, 0.0, 0.0])
        self.assertEqual(list(g_edge[:3]), [0.0, 0.0, 0.0])
        self.assertGreater(w_edge[3], 0.0)
        self.assertLess(w_edge[3], 1e-8 * w[0])
        self.assertLess(abs(g_edge[3]), 1e-5 * scale)

    def test_cubic_spline_kernel_properties(self):
        """Verifies non-negativity, compact support (r >= h -> 0), and monotonic decay."""
        h = 0.04
        r_np = np.array([0.0, 0.25 * h, 0.5 * h, 0.75 * h, h, 1.2 * h], dtype=np.float32)
        w_vals, gx_vals = self.eval_kernel(r_np, h)

        # Compact support outside h
        self.assertEqual(w_vals[4], 0.0)
        self.assertEqual(w_vals[5], 0.0)
        self.assertEqual(gx_vals[4], 0.0)

        # Strictly positive inside support and monotonically decreasing
        self.assertGreater(w_vals[0], w_vals[1])
        self.assertGreater(w_vals[1], w_vals[2])
        self.assertGreater(w_vals[2], w_vals[3])
        self.assertGreater(w_vals[3], 0.0)

        # Gradient is negative for repulsive radial distance (0 < r < h) and zero at r=0
        self.assertEqual(gx_vals[0], 0.0)
        self.assertLess(gx_vals[1], 0.0)
        self.assertLess(gx_vals[2], 0.0)

    def test_solver_contains_particles_thrown_at_every_wall(self):
        """Clusters fired at all six walls reach them and never leave the domain, also at the largest time step.

        Runs on the CPU (eager) and, where CUDA exists, through the captured graph.
        """
        lo, hi = np.array([-0.3, -0.3, 0.0]), np.array([0.3, 0.3, 0.6])
        centre = 0.5 * (lo + hi)
        cube = 0.02 * np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"), -1).reshape(-1, 3)
        pts, vel = [], []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                direction = np.zeros(3)
                direction[axis] = sign
                pts.append(centre + 0.1 * direction + cube)
                vel.append(np.tile(15.0 * direction, (len(cube), 1)))
        pts = np.concatenate(pts).astype(np.float32)
        vel = np.concatenate(vel).astype(np.float32)
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            for cfl in (0.4, 1.0):
                with self.subTest(device=device, cfl=cfl):
                    solver = RealtimeWCSPHSolver(num_particles=len(pts), h=0.04, substeps=2, enable_obstacles=0,
                                                 device=device)
                    solver.domain_min, solver.domain_max = wp.vec3(*lo), wp.vec3(*hi)
                    solver.cfl_factor = cfl
                    solver.update_time_step()
                    solver.rebuild_cuda_graph()
                    solver.pos_sim.assign(pts)
                    solver.vel.assign(vel)
                    out = wp.zeros(len(pts), dtype=wp.vec3, device=device)
                    closest_low, closest_high = np.full(3, np.inf), np.full(3, np.inf)
                    for _ in range(60):
                        solver.step(out, use_graph=True)
                        p = out.numpy().astype(np.float64)
                        self.assertTrue(np.isfinite(p).all() and np.isfinite(solver.vel.numpy()).all())
                        self.assertTrue((p >= lo - 1e-6).all() and (p <= hi + 1e-6).all(),
                                        "a particle left the domain")
                        closest_low = np.minimum(closest_low, (p - lo).min(axis=0))
                        closest_high = np.minimum(closest_high, (hi - p).min(axis=0))
                    # Every wall was actually hit: some particle entered its soft layer (r0 = h / 2).
                    self.assertTrue((closest_low < 0.5 * solver.h).all(), f"walls not reached: {closest_low}")
                    self.assertTrue((closest_high < 0.5 * solver.h).all(), f"walls not reached: {closest_high}")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required for CUDA Graph equivalence test")
    def test_cuda_graph_vs_eager_equivalence(self):
        """Verifies that Warp 1.16+ CUDA Graph execution matches eager kernel launches exactly."""
        num_p = 2048
        s_graph = RealtimeWCSPHSolver(num_particles=num_p, h=0.04, substeps=2, enable_obstacles=1)
        s_eager = RealtimeWCSPHSolver(num_particles=num_p, h=0.04, substeps=2, enable_obstacles=1)

        rng = np.random.default_rng(42)
        init_pos = rng.uniform(low=[-1.0, -1.0, 0.8], high=[1.0, 1.0, 2.0], size=(num_p, 3)).astype(np.float32)

        s_graph.pos_sim.assign(init_pos)
        s_eager.pos_sim.assign(init_pos)

        vbo_graph = wp.zeros(num_p, dtype=wp.vec3, device="cuda:0")
        vbo_eager = wp.zeros(num_p, dtype=wp.vec3, device="cuda:0")

        s_graph.step(vbo_graph, use_graph=True)
        s_eager.step(vbo_eager, use_graph=False)

        np.testing.assert_allclose(
            vbo_graph.numpy(),
            vbo_eager.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CUDA Graph step diverged from Eager step!",
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_vram_telemetry_calculation(self):
        solver = RealtimeWCSPHSolver(num_particles=1_000_000, h=0.04)
        vram_mb = solver.get_vram_usage_mb()
        self.assertAlmostEqual(vram_mb, (1_000_000 * 84) / (1024.0 * 1024.0), places=2)


if __name__ == "__main__":
    unittest.main()
