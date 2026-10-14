import os
import sys
import tempfile
import unittest
import numpy as np
import warp as wp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ipbf_solver import k_ipbf_predict, k_ipbf_vbd_iteration
from collider_sdf import SDFColliderManager
from whitewater import k_classify_and_advect_whitewater
from particle_upsample import k_flag_surface_particles_for_split
from neighbor_rt import NeighborSearchBackend
from mesher_io import export_particles_xyz


class TestPhysicsSubmodules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_ipbf_prediction_and_vbd_iteration(self):
        """Tests variational prediction and local 3x3 Hessian/Jacobi position correction."""
        num_p = 64
        pos = wp.zeros(num_p, dtype=wp.vec3, device=self.device)
        vel = wp.zeros(num_p, dtype=wp.vec3, device=self.device)
        pos_pred = wp.zeros(num_p, dtype=wp.vec3, device=self.device)
        pos_corr = wp.zeros(num_p, dtype=wp.vec3, device=self.device)

        # Place particles in a tight cluster so density > rest_density
        side = 4
        idx = np.arange(num_p)
        pts = np.stack([
            (idx % side) * 0.01,
            ((idx // side) % side) * 0.01,
            (idx // (side * side)) * 0.01 + 1.0,
        ], axis=-1).astype(np.float32)
        pos.assign(pts)

        dt = 0.01
        wp.launch(k_ipbf_predict, dim=num_p, inputs=[pos, vel, pos_pred, dt], device=self.device)
        pred_np = pos_pred.numpy()
        # Gravity should lower Z by 0.5 * 9.81 * dt^2
        self.assertTrue(np.all(pred_np[:, 2] < pts[:, 2]))

        if wp.is_cuda_available():
            grid = NeighborSearchBackend(grid_res=32, device="cuda:0")
            grid.build(pos_pred, radius=0.04)
            wp.launch(
                k_ipbf_vbd_iteration,
                dim=num_p,
                inputs=[grid.id, pos_pred, pos_corr, 0.008, 0.04, 1000.0, 0.5],
                device="cuda:0",
            )
            corr_np = pos_corr.numpy()
            self.assertFalse(np.isnan(corr_np).any())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required for SDFColliderManager")
    def test_sdf_sphere_collider_pushes_particles_out(self):
        """Verifies that particles inside an analytical SDF sphere are projected to its surface."""
        mgr = SDFColliderManager()
        mgr.active = True
        mgr.sphere_center = wp.vec3(0.0, 0.0, 0.5)
        mgr.sphere_radius = 0.6

        # Place particle inside sphere at (0.2, 0.0, 0.5) moving inward
        pos = wp.array([[0.2, 0.0, 0.5]], dtype=wp.vec3, device="cuda:0")
        vel = wp.array([[-1.0, 0.0, 0.0]], dtype=wp.vec3, device="cuda:0")

        mgr.apply(pos, vel, num_particles=1)
        out_p = pos.numpy()[0]
        out_v = vel.numpy()[0]

        self.assertAlmostEqual(out_p[0], 0.6, places=5)
        self.assertGreater(out_v[0], 0.0)  # Velocity reflected outward

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required for HashGrid whitewater test")
    def test_whitewater_neighbor_count_classification(self):
        """Verifies Spray (n_f <= 5), Foam (5 < n_f < 15), and Bubble (n_f >= 15) classification."""
        # Create 20 fluid particles clustered at origin (0,0,0), plus 10 clustered at (1,0,0)
        f_pts = np.zeros((30, 3), dtype=np.float32)
        f_pts[20:, 0] = 1.0  # 10 particles at X=1.0

        fluid_pos = wp.array(f_pts, dtype=wp.vec3, device="cuda:0")
        fluid_vel = wp.zeros(30, dtype=wp.vec3, device="cuda:0")

        grid = wp.HashGrid(32, 32, 32, device="cuda:0")
        grid.build(fluid_pos, radius=0.05)

        # 3 secondary test particles:
        # 0: isolated in air at (5,5,5) -> n_f = 0 -> Spray (type 0)
        # 1: at (1,0,0) -> n_f = 10 -> Foam (type 1)
        # 2: at (0,0,0) -> n_f = 20 -> Bubble (type 2)
        ww_pos = wp.array([[5.0, 5.0, 5.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3, device="cuda:0")
        ww_vel = wp.zeros(3, dtype=wp.vec3, device="cuda:0")
        ww_life = wp.array([1.0, 1.0, 1.0], dtype=float, device="cuda:0")
        ww_type = wp.zeros(3, dtype=int, device="cuda:0")

        wp.launch(
            k_classify_and_advect_whitewater,
            dim=3,
            inputs=[grid.id, fluid_pos, fluid_vel, ww_pos, ww_vel, ww_life, ww_type, 0.05, 0.01],
            device="cuda:0",
        )

        types = ww_type.numpy()
        self.assertEqual(types[0], 0, "Expected Spray (type 0) when n_f <= 5")
        self.assertEqual(types[1], 1, "Expected Foam (type 1) when 5 < n_f < 15")
        self.assertEqual(types[2], 2, "Expected Bubble (type 2) when n_f >= 15")

    def test_surface_particle_upsample_flags(self):
        """Verifies free-surface density threshold flagging (< 0.85 * rest_density)."""
        densities = wp.array([600.0, 849.0, 851.0, 1000.0], dtype=float, device=self.device)
        flags = wp.zeros(4, dtype=int, device=self.device)

        wp.launch(
            k_flag_surface_particles_for_split,
            dim=4,
            inputs=[densities, flags, 1000.0],
            device=self.device,
        )
        np.testing.assert_array_equal(flags.numpy(), np.array([1, 1, 0, 0]))

    def test_mesher_xyz_export(self):
        pts = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=wp.vec3, device=self.device)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            export_particles_xyz(pts, tmp_path)
            loaded = np.loadtxt(tmp_path)
            np.testing.assert_allclose(loaded, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), atol=1e-4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
