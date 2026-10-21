"""Output logic (P2.3) without Blender: particle age from the cache, meshing defaults and export planning."""
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from warp_sph_realtime.cache import output
from warp_sph_realtime.cache.age import MAX_ID, EmissionIndex, seconds_per_frame
from warp_sph_realtime.cache.disk import CacheError, DiskCache


class CountingDisk(DiskCache):
    """DiskCache that records which frames' ids were read."""

    def __init__(self, directory):
        super().__init__(directory)
        self.reads: list[int] = []

    def read_ids(self, frame):
        self.reads.append(frame)
        return super().read_ids(frame)


def write_emitter_cache(disk, first=10, frames=8, initial=5, per_frame=3, seed=0):
    """Frames ``first``..: ``initial`` particles at the start, ``per_frame`` more each frame, ids 0..n-1, in a
    different array order every frame (the solver reorders every step). Returns {frame: count}."""
    disk.begin(first, {"particles": initial + per_frame * frames, "steps_per_frame": 4, "dt": 0.0005, "fps": 24.0})
    rng = np.random.default_rng(seed)
    counts = {}
    for k in range(frames):
        n = initial + per_frame * k
        ids = rng.permutation(n).astype(np.int32)
        disk.write_frame(first + k, rng.random((n, 3), dtype=np.float32), np.zeros((n, 3), np.float32), ids)
        counts[first + k] = n
    return counts


class SecondsPerFrameTest(unittest.TestCase):
    def test_signature_steps_times_dt(self):
        manifest = {"signature": {"steps_per_frame": 4, "dt": 0.0005, "fps": 24.0}}
        self.assertAlmostEqual(seconds_per_frame(manifest), 0.002)

    def test_fallbacks(self):
        self.assertAlmostEqual(seconds_per_frame({"signature": {"fps": 30.0}}), 1 / 30)
        self.assertAlmostEqual(seconds_per_frame({"signature": {"particles": 5}}, fallback_fps=25.0), 1 / 25)
        self.assertAlmostEqual(seconds_per_frame(None), 1 / 24)
        for bad in (0, -1.0, float("nan"), float("inf"), True, "4", None):
            manifest = {"signature": {"steps_per_frame": bad, "dt": 0.001, "fps": bad}}
            self.assertAlmostEqual(seconds_per_frame(manifest, fallback_fps=0.0), 1 / 24, msg=repr(bad))


class EmissionIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.disk = CountingDisk(Path(self._tmp.name) / "cache")
        self.counts = write_emitter_cache(self.disk)  # frames 10..17, 5 + 3 * k particles
        self.index = EmissionIndex(self.disk)

    def ages(self, frame, per_frame=0.5):
        self.index.sync(10, frame)
        ids = self.disk.read_frame(frame).id
        return ids, self.index.ages(ids, frame, per_frame)

    def test_age_is_frames_since_first_seen(self):
        for frame in (10, 13, 17):
            ids, ages = self.ages(frame)
            self.assertEqual(ages.dtype, np.float32)
            for pid, age in zip(ids.tolist(), ages.tolist(), strict=True):
                emitted = 10 if pid < 5 else 10 + (pid - 5) // 3 + 1  # id 5.. appear 3 per frame from frame 11
                self.assertAlmostEqual(age, (frame - emitted) * 0.5, msg=f"frame {frame} id {pid}")
        _, last = self.ages(17)
        self.assertEqual(float(last.max()), 3.5)         # the initial particles, 7 frames old
        self.assertEqual(float(last.min()), 0.0)         # emitted on this very frame

    def test_scan_is_incremental_and_scrubbing_does_not_rescan(self):
        self.ages(12)
        self.assertEqual(self.disk.reads, [10, 11, 12])
        self.ages(13)
        self.assertEqual(self.disk.reads[3:], [13])      # only the new frame
        before = len(self.disk.reads)
        for frame in (11, 17, 10, 17, 12):
            self.ages(frame)
        self.assertEqual(self.disk.reads[before:], [14, 15, 16, 17])  # forward once, backward and repeats are free
        self.assertEqual(self.index.scanned_to, 17)

    def test_backward_ages_equal_forward_ages(self):
        forward = {f: self.ages(f) for f in range(10, 18)}
        for frame in range(17, 9, -1):
            ids, ages = self.ages(frame)
            self.assertEqual(ids.tolist(), forward[frame][0].tolist())
            self.assertEqual(ages.tolist(), forward[frame][1].tolist())
        fresh = EmissionIndex(self.disk)
        fresh.sync(10, 17)                                # a jump straight to the end gives the same answer
        ids = self.disk.read_frame(14).id
        self.assertEqual(fresh.ages(ids, 14, 0.5).tolist(), forward[14][1].tolist())

    def test_unknown_ids_have_age_zero(self):
        self.index.sync(10, 10)
        ages = self.index.ages(np.array([0, 4, 5, 99, -3], dtype=np.int32), 10, 1.0)
        self.assertEqual(ages.tolist(), [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.index.emission_frames(np.array([4, 5, 99], dtype=np.int32)).tolist(), [10, -1, -1])

    def test_fixed_count_cache_ages_from_the_first_frame(self):
        disk = CountingDisk(Path(self._tmp.name) / "fixed")
        disk.begin(1, {"particles": 4})
        for f in range(1, 5):
            disk.write_frame(f, np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32), np.arange(4, dtype=np.int32))
        index = EmissionIndex(disk)
        index.sync(1, 4)
        self.assertEqual(index.ages(np.arange(4, dtype=np.int32), 4, 0.1).tolist(), [np.float32(0.3)] * 4)

    def test_rebaked_frame_rolls_the_scan_back(self):
        self.ages(17)
        before = len(self.disk.reads)
        # Re-bake from frame 14: a different emission history (nothing is emitted after frame 13)
        self.disk.invalidate_from(14)
        n = self.counts[13]
        for frame in range(14, 18):
            self.disk.write_frame(frame, np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32),
                                  np.arange(n, dtype=np.int32))
            os.utime(self.disk.path(frame), ns=(time.time_ns() + 10**9 * frame,) * 2)  # distinct, newer stamps
        ids, ages = self.ages(17)
        self.assertEqual(float(ages[ids == n - 1][0]), (17 - 13) * 0.5)   # newest id first seen on 13, not 17
        self.assertEqual(self.disk.reads[before:], [14, 15, 16, 17])      # rescanned from the first changed frame
        self.index.sync(10, 12)                                           # earlier frames are untouched
        self.assertEqual(self.disk.reads[before + 4:], [])

    def test_changed_start_or_cleared_cache_resets(self):
        self.ages(14)
        self.disk.clear()
        write_emitter_cache(self.disk, first=20, frames=3, initial=2, per_frame=2, seed=1)
        self.index.sync(20, 22)
        self.assertEqual(self.index.scanned_to, 22)
        ids = self.disk.read_frame(22).id
        self.assertEqual(self.index.ages(ids, 22, 1.0).max(), 2.0)

    def test_missing_frame_and_bad_ids_raise_cache_error(self):
        with self.assertRaises(CacheError):
            self.index.sync(10, 25)                      # frame 18 does not exist
        disk = CountingDisk(Path(self._tmp.name) / "bad")
        disk.begin(1, {})
        disk.write_frame(1, np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32), np.array([0, -1], np.int32))
        disk.write_frame(2, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32), np.array([MAX_ID], np.int32))
        with self.assertRaises(CacheError):
            EmissionIndex(disk).sync(1, 1)
        with self.assertRaises(CacheError):
            EmissionIndex(disk).sync(2, 2)

    def test_read_ids_errors(self):
        with self.assertRaises(CacheError):
            self.disk.read_ids(99)
        self.disk.path(11).write_bytes(b"not a zip")
        with self.assertRaises(CacheError):
            self.disk.read_ids(11)

    def test_cache_format_unchanged(self):
        with np.load(self.disk.path(10)) as data:
            self.assertEqual(sorted(data.files), ["id", "position", "velocity"])


class OutputPlanningTest(unittest.TestCase):
    def test_particle_spacing_prefers_the_bake_signature(self):
        self.assertAlmostEqual(output.particle_spacing({"signature": {"h": 0.04}}, 0.03, 0.02), 0.02)
        self.assertAlmostEqual(output.particle_spacing({"signature": {}}, 0.03, 0.02), 0.03)
        self.assertAlmostEqual(output.particle_spacing(None, 0.0, 0.025), 0.025)
        self.assertAlmostEqual(output.particle_spacing(None, 0.0, 0.0), 0.02)

    def test_particle_spacing_is_in_scene_units(self):
        # h is in metres: a centimetre scene (Unit Scale 0.01) with 2.5 cm spacing has h = 0.05 m
        cm = {"signature": {"h": 0.05, "unit_scale": 0.01}}
        self.assertAlmostEqual(output.particle_spacing(cm, 2.5, 0.02), 2.5)
        self.assertAlmostEqual(output.meshing_defaults(output.particle_spacing(cm, 2.5)).radius, 3.75)

    def test_meshing_defaults_scale_with_spacing(self):
        small, large = output.meshing_defaults(0.01), output.meshing_defaults(0.04)
        self.assertAlmostEqual(large.radius / small.radius, 4.0)
        self.assertAlmostEqual(large.voxel_size / small.voxel_size, 4.0)
        self.assertEqual((small.threshold, small.smooth_iterations), (large.threshold, large.smooth_iterations))
        self.assertGreater(small.radius, small.voxel_size)        # the splat spans several voxels
        self.assertGreaterEqual(small.radius, 0.75 * 0.01)        # and overlaps its neighbours
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                output.meshing_defaults(bad)

    def test_export_path_extensions(self):
        self.assertEqual(output.export_path("a/shot", "USD"), Path("a/shot.usdc"))
        self.assertEqual(output.export_path("a/shot.usda", "USD"), Path("a/shot.usda"))
        self.assertEqual(output.export_path("a/shot.usdc", "ALEMBIC"), Path("a/shot.abc"))
        self.assertEqual(output.export_path("a/shot.ABC", "ALEMBIC"), Path("a/shot.ABC"))
        with self.assertRaises(ValueError):
            output.export_path("a/shot", "OBJ")

    def test_export_frames_only_baked(self):
        self.assertEqual(output.export_frames((5, 20)), (5, 20))
        self.assertEqual(output.export_frames((5, 20), (1, 10)), (5, 10))
        self.assertEqual(output.export_frames((5, 20), (18, 99)), (18, 20))
        with self.assertRaises(ValueError):
            output.export_frames(None)
        with self.assertRaises(ValueError):
            output.export_frames((5, 20), (30, 40))


if __name__ == "__main__":
    unittest.main()
