"""Real-solver recording (P1.6, P2.2): pinned async readback through the VRAM ring, stable ids, replay, resume.

Needs CUDA; skipped otherwise.
"""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import DiskCache
from warp_sph_realtime.cache.frame_store import FrameStore
from warp_sph_realtime.cache.gpu import SolverSource, VramRing
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

SPACING = 0.02
SIDE = 14  # 2744 particles: a block that slumps under gravity
STEPS = 3
SIGNATURE = {"particles": SIDE**3, "steps_per_frame": STEPS}


def block_positions() -> np.ndarray:
    axis = np.arange(SIDE, dtype=np.float32) * SPACING
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([x.ravel() - 0.14, y.ravel() - 0.14, z.ravel() + 0.5], axis=1).astype(np.float32)


def make_solver() -> RealtimeWCSPHSolver:
    solver = RealtimeWCSPHSolver(num_particles=SIDE**3, h=2.0 * SPACING, substeps=STEPS, enable_obstacles=0)
    solver.pos_sim.assign(block_positions())
    solver.vel.zero_()
    return solver


def new_job(disk: DiskCache, solver, first: int, last: int, *, resume: bool, slots: int = 3, max_slice: int = 4):
    plan = bake.plan_bake(disk, SIGNATURE, first, last, resume=resume, particles=solver.num_particles)
    store = FrameStore(disk, VramRing(wp, solver.num_particles, slots))
    return bake.BakeJob(store, SolverSource(wp, solver, STEPS), plan, max_slice=max_slice)


def run(job: bake.BakeJob, max_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + max_seconds
    while job.active and time.monotonic() < deadline:
        job.run_chunk(0.004)
        time.sleep(0.001)
    job.store.close()
    assert not job.active, f"bake stuck in {job.state}"


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestRealSolverRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_particle_ids_follow_particles_through_the_spatial_sort(self):
        solver = make_solver()
        out = wp.zeros_like(solver.pos_sim)
        previous = {int(i): p for i, p in zip(solver.pid.numpy(), solver.pos_sim.numpy(), strict=True)}
        ids = solver.pid.numpy()
        for _ in range(4):
            solver.step(out)
            wp.synchronize_device("cuda:0")
            ids, pos = solver.pid.numpy(), solver.pos_sim.numpy()
            self.assertEqual(sorted(ids.tolist()), list(range(SIDE**3)))  # still a permutation
            moved = np.array([np.linalg.norm(p - previous[int(i)]) for i, p in zip(ids, pos, strict=True)])
            self.assertLess(moved.max(), 0.05)  # a particle moves little in a step; a random order would not
            previous = {int(i): p for i, p in zip(ids, pos, strict=True)}
        self.assertFalse(np.array_equal(ids, np.arange(SIDE**3)))  # the array order really is not the identity

    def test_recorded_frames_equal_the_solver_and_rereads_are_bit_identical(self):
        disk = DiskCache(self.root / "a")
        job = new_job(disk, make_solver(), 1, 8, resume=False)
        run(job)
        self.assertEqual(job.state, bake.DONE)
        self.assertEqual(disk.cached_range(), (1, 8))

        reference = make_solver()  # identical start, stepped by hand: frame 1 is the initial state
        out = wp.zeros_like(reference.pos_sim)
        reference.substeps = STEPS
        for frame in range(1, 9):
            if frame > 1:
                reference.step(out)
            wp.synchronize_device("cuda:0")
            read = disk.read_frame(frame)
            self.assertEqual(read.position.tobytes(), reference.pos_sim.numpy().tobytes(), f"position {frame}")
            self.assertEqual(read.velocity.tobytes(), reference.vel.numpy().tobytes(), f"velocity {frame}")
            self.assertEqual(read.id.tobytes(), reference.pid.numpy().tobytes(), f"id {frame}")
            again = disk.read_frame(frame)
            self.assertEqual(read.position.tobytes(), again.position.tobytes())
        self.assertGreater(np.abs(disk.read_frame(8).position - disk.read_frame(1).position).max(), 1e-4)  # it moved

    def test_cancel_then_resume_is_bit_identical_to_an_uninterrupted_bake(self):
        whole = DiskCache(self.root / "whole")
        run(new_job(whole, make_solver(), 1, 10, resume=False))

        part = DiskCache(self.root / "part")
        job = new_job(part, make_solver(), 1, 10, resume=False)
        while job.captured < 4:
            job.run_chunk(0.004)
            time.sleep(0.001)
        job.cancel()
        run(job)
        self.assertEqual(job.state, bake.CANCELLED)
        cached = part.cached_range()
        assert cached is not None
        self.assertEqual(cached, (1, job.captured))
        self.assertLess(cached[1], 10)
        self.assertEqual(list(part.directory.glob("*.tmp")), [])

        resumed = new_job(part, make_solver(), 1, 10, resume=True)  # a fresh solver: state comes from the cache
        self.assertEqual(resumed.plan.next_frame, cached[1] + 1)
        run(resumed)
        self.assertEqual(resumed.state, bake.DONE)
        for frame in range(1, 11):
            a, b = whole.read_frame(frame), part.read_frame(frame)
            for name in ("position", "velocity", "id"):
                self.assertEqual(getattr(a, name).tobytes(), getattr(b, name).tobytes(), f"{name} of frame {frame}")

    def test_invalidate_from_frame_then_resume_reproduces_the_tail(self):
        disk = DiskCache(self.root / "inv")
        run(new_job(disk, make_solver(), 1, 8, resume=False))
        expected = {f: disk.read_frame(f).position.tobytes() for f in range(1, 9)}
        disk.invalidate_from(5)
        self.assertEqual(disk.cached_range(), (1, 4))
        run(new_job(disk, make_solver(), 1, 8, resume=True))
        self.assertEqual(disk.cached_range(), (1, 8))
        for frame in range(1, 9):
            self.assertEqual(disk.read_frame(frame).position.tobytes(), expected[frame], f"frame {frame}")

    def test_advancing_a_frame_in_single_step_slices_gives_identical_frames(self):
        whole, sliced = DiskCache(self.root / "whole"), DiskCache(self.root / "sliced")
        run(new_job(whole, make_solver(), 1, 5, resume=False, max_slice=STEPS))
        run(new_job(sliced, make_solver(), 1, 5, resume=False, max_slice=1))
        for frame in range(1, 6):
            for name in ("position", "velocity", "id"):
                self.assertEqual(getattr(whole.read_frame(frame), name).tobytes(),
                                 getattr(sliced.read_frame(frame), name).tobytes(), f"{name} of frame {frame}")

    def test_every_frame_is_recorded_in_scene_units_and_resume_converts_back(self):
        # Unit Scale 0.01 (a scene modelled in centimetres): the solver works in metres, the cache in scene units,
        # the initial frame included; a resume loads scene units back into metres.
        unit = 0.01

        def scaled_solver():
            solver = make_solver()
            solver.unit_scale = unit
            return solver

        disk = DiskCache(self.root / "cm")
        run(new_job(disk, scaled_solver(), 1, 6, resume=False))
        reference = make_solver()  # the same run in metres, stepped by hand
        out = wp.zeros_like(reference.pos_sim)
        reference.substeps = STEPS
        for frame in range(1, 7):
            if frame > 1:
                reference.step(out)
            wp.synchronize_device("cuda:0")
            np.testing.assert_allclose(disk.read_frame(frame).position, reference.pos_sim.numpy() / unit,
                                       rtol=1e-5, atol=1e-4, err_msg=f"frame {frame}")

        def by_id(frame):  # the scale round trip is not bit-exact, so the spatial sort may order ties differently
            read = disk.read_frame(frame)
            return read.position[np.argsort(read.id)]

        whole = {f: by_id(f) for f in range(1, 7)}
        disk.invalidate_from(4)
        run(new_job(disk, scaled_solver(), 1, 6, resume=True))
        self.assertEqual(disk.cached_range(), (1, 6))
        for frame in range(4, 7):  # metres loaded as centimetres would be 100x off
            np.testing.assert_allclose(by_id(frame), whole[frame], atol=1e-2, err_msg=f"resumed frame {frame}")

    def test_a_single_slot_ring_still_records_every_frame(self):
        disk = DiskCache(self.root / "one")
        run(new_job(disk, make_solver(), 1, 6, resume=False, slots=1))
        self.assertEqual(disk.cached_range(), (1, 6))


if __name__ == "__main__":
    unittest.main()
