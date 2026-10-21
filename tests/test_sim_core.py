import unittest

import numpy as np
import warp as wp

from warp_sph_realtime.sim.sim_core import (
    RealtimeWCSPHSolver,
    cubic_spline_grad,
    cubic_spline_kernel,
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


class TestSimCorePhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_cubic_spline_kernel_properties(self):
        """Verifies non-negativity, compact support (r >= h -> 0), and monotonic decay."""
        h = 0.04
        r_np = np.array([0.0, 0.25 * h, 0.5 * h, 0.75 * h, h, 1.2 * h], dtype=np.float32)
        radii = wp.array(r_np, dtype=float, device=self.device)
        out_w = wp.zeros_like(radii)
        out_gx = wp.zeros_like(radii)

        wp.launch(k_eval_kernel_samples, dim=len(r_np), inputs=[radii, h, out_w, out_gx], device=self.device)
        w_vals = out_w.numpy()
        gx_vals = out_gx.numpy()

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

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required for RealtimeWCSPHSolver CUDA Graph tests")
    def test_solver_step_bounds_and_nan_free(self):
        """Verifies that stepping the WCSPH solver keeps all particles finite and inside domain bounds."""
        num_p = 4096
        solver = RealtimeWCSPHSolver(num_particles=num_p, h=0.04, substeps=2, enable_obstacles=1)

        # Initialize a compact fluid block above the central sphere collider
        side = 16
        idx = np.arange(num_p)
        ix = idx % side
        iy = (idx // side) % side
        iz = idx // (side * side)
        pts = np.stack([
            -0.2 + ix * 0.022,
            -0.2 + iy * 0.022,
            1.2 + iz * 0.022,
        ], axis=-1).astype(np.float32)

        solver.pos_sim.assign(pts)
        vbo_mock = wp.zeros(num_p, dtype=wp.vec3, device="cuda:0")

        for _ in range(5):
            solver.step(vbo_mock, use_graph=True)

        out_pts = vbo_mock.numpy()
        self.assertFalse(np.isnan(out_pts).any(), "Particle positions contain NaNs!")
        self.assertFalse(np.isinf(out_pts).any(), "Particle positions contain Infs!")

        # Check domain boundary enforcement
        self.assertTrue(np.all(out_pts[:, 0] >= -2.5 - 1e-5))
        self.assertTrue(np.all(out_pts[:, 0] <=  2.5 + 1e-5))
        self.assertTrue(np.all(out_pts[:, 1] >= -2.5 - 1e-5))
        self.assertTrue(np.all(out_pts[:, 1] <=  2.5 + 1e-5))
        self.assertTrue(np.all(out_pts[:, 2] >=  0.0 - 1e-5))

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
