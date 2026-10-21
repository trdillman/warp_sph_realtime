"""Disk frame cache (P1.6, P2.2): round trip, atomic writes, size estimate, invalidate, plan (fresh / resume)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache import disk as cache_disk
from warp_sph_realtime.cache.disk import CacheError, CacheMismatchError, DiskCache

SIGNATURE = {"particles": 64, "steps_per_frame": 2, "dt": 0.001}


def make_frame(frame: int, count: int = 64):
    rng = np.random.default_rng(frame + 1000)
    return (rng.standard_normal((count, 3)).astype(np.float32),
            rng.standard_normal((count, 3)).astype(np.float32),
            rng.permutation(count).astype(np.int32))


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.disk = DiskCache(self.root / "cache")

    def manifest(self) -> dict:
        manifest = self.disk.manifest()
        assert manifest is not None
        return manifest

    @staticmethod
    def restore_of(plan) -> cache_disk.Frame:
        assert plan.restore is not None
        return plan.restore

    def fill(self, first: int, last: int, count: int = 64):
        for frame in range(first, last + 1):
            self.disk.write_frame(frame, *make_frame(frame, count))


class TestRoundTrip(CacheTestCase):
    def test_reread_is_bit_identical(self):
        self.disk.begin(1, SIGNATURE)
        position, velocity, ids = make_frame(7)
        self.disk.write_frame(7, position, velocity, ids)
        first, second = self.disk.read_frame(7), self.disk.read_frame(7)
        for written, read in ((position, first.position), (velocity, first.velocity), (ids, first.id)):
            self.assertEqual(written.dtype, read.dtype)
            self.assertEqual(written.tobytes(), read.tobytes())
        self.assertEqual(first.position.tobytes(), second.position.tobytes())
        self.assertEqual(first.count, 64)
        self.assertEqual(first.frame, 7)

    def test_special_float_values_survive(self):
        position = np.array([[np.nan, np.inf, -0.0], [1e-45, -3.4e38, 0.1]], dtype=np.float32)
        ids = np.array([5, 2**31 - 1], dtype=np.int32)
        self.disk.write_frame(1, position, position, ids)
        read = self.disk.read_frame(1)
        self.assertEqual(read.position.tobytes(), position.tobytes())
        self.assertEqual(read.id.tolist(), ids.tolist())

    def test_write_is_atomic_and_leaves_no_temporary(self):
        self.fill(1, 3)
        self.assertEqual([p.name for p in self.disk.directory.glob("*.tmp")], [])
        self.assertEqual(self.disk.frames(), [1, 2, 3])

    def test_rejects_mismatched_arrays_and_writes_nothing(self):
        position, velocity, ids = make_frame(1)
        with self.assertRaises(CacheError):
            self.disk.write_frame(1, position[:-1], velocity, ids)
        self.assertEqual(self.disk.frames(), [])

    def test_missing_and_corrupt_frames_raise_cache_error(self):
        with self.assertRaises(CacheError):
            self.disk.read_frame(1)
        self.fill(1, 1)
        self.disk.path(1).write_bytes(self.disk.path(1).read_bytes()[:200])  # truncated
        with self.assertRaises(CacheError):
            self.disk.read_frame(1)

    def test_negative_frame_numbers(self):
        self.disk.begin(-2, SIGNATURE)
        self.fill(-2, 1)
        self.assertEqual(self.disk.frames(), [-2, -1, 0, 1])
        self.assertEqual(self.disk.cached_range(), (-2, 1))
        self.assertEqual(self.disk.read_frame(-1).frame, -1)


class TestSizeEstimate(CacheTestCase):
    def test_estimate_matches_files_on_disk_exactly(self):
        for count in (1, 1000, 100_000):
            disk = DiskCache(self.root / f"n{count}")
            for frame in range(1, 4):
                disk.write_frame(frame, *make_frame(frame, count))
            self.assertEqual(disk.size_bytes(), cache_disk.estimate_bytes(count, 3), f"{count} particles")

    def test_estimate_scales_with_frames_and_clamps(self):
        self.assertEqual(cache_disk.estimate_bytes(1_000_000, 120), 120 * cache_disk.frame_bytes(1_000_000))
        self.assertGreater(cache_disk.estimate_bytes(1_000_000, 120), 3.3e9)
        self.assertEqual(cache_disk.estimate_bytes(100, 0), 0)
        self.assertEqual(cache_disk.estimate_bytes(100, -5), 0)

    def test_free_bytes_of_a_directory_that_does_not_exist_yet(self):
        free = cache_disk.free_bytes(self.root / "a" / "b" / "c")
        self.assertIsNotNone(free)
        self.assertGreater(free, 0)


class TestDefaultDirectory(unittest.TestCase):
    def test_next_to_the_saved_blend(self):
        path = cache_disk.default_cache_dir(str(Path("C:/work/shots/pour.blend")), "Scene")
        self.assertEqual(path, Path("C:/work/shots") / "pour_warp_sph_cache" / "Scene")

    def test_scene_names_are_made_safe_and_stay_separate(self):
        a = cache_disk.default_cache_dir("C:/w/f.blend", "Scene A/1")
        b = cache_disk.default_cache_dir("C:/w/f.blend", "Scene B")
        self.assertEqual(a.name, "Scene_A_1")
        self.assertNotEqual(a, b)

    def test_unsaved_file_uses_the_temp_folder(self):
        path = cache_disk.default_cache_dir("", "Scene")
        self.assertEqual(path.parent.parent.name, "warp_sph_cache")
        self.assertTrue(str(path).startswith(tempfile.gettempdir()))

    def test_configured_folder_wins_and_blank_means_default(self):
        self.assertEqual(cache_disk.resolve_cache_dir("D:/caches/x", "C:/w/f.blend", "Scene"), Path("D:/caches/x"))
        self.assertEqual(cache_disk.resolve_cache_dir("  ", "C:/w/f.blend", "Scene"),
                         cache_disk.default_cache_dir("C:/w/f.blend", "Scene"))


class TestInvalidate(CacheTestCase):
    def test_invalidate_from_frame_removes_that_frame_onward(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 10)
        self.assertEqual(self.disk.invalidate_from(6), 5)
        self.assertEqual(self.disk.frames(), [1, 2, 3, 4, 5])
        self.assertEqual(self.disk.cached_range(), (1, 5))
        self.assertEqual(self.disk.read_frame(5).position.tobytes(), make_frame(5)[0].tobytes())  # kept intact

    def test_invalidate_before_the_start_empties_the_cache(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 4)
        self.assertEqual(self.disk.invalidate_from(0), 4)
        self.assertEqual(self.disk.frames(), [])
        self.assertIsNone(self.disk.cached_range())

    def test_invalidate_past_the_end_changes_nothing(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 4)
        self.assertEqual(self.disk.invalidate_from(99), 0)
        self.assertEqual(self.disk.cached_range(), (1, 4))

    def test_clear_removes_frames_temporaries_and_manifest(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 3)
        (self.disk.directory / "frame_000004.npz.tmp").write_bytes(b"half")
        self.assertEqual(self.disk.clear(), 3)
        self.assertEqual(sorted(p.name for p in self.disk.directory.iterdir()), [])
        self.assertIsNone(self.disk.manifest())

    def test_cached_range_stops_at_a_gap(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 3)
        self.fill(5, 6)
        self.assertEqual(self.disk.cached_range(), (1, 3))


class TestManifest(CacheTestCase):
    def test_begin_creates_then_accepts_the_same_signature(self):
        self.disk.begin(1, SIGNATURE)
        self.disk.begin(1, dict(SIGNATURE))
        self.assertEqual(self.manifest()["start"], 1)

    def test_a_different_signature_or_start_is_refused(self):
        self.disk.begin(1, SIGNATURE)
        with self.assertRaises(CacheMismatchError):
            self.disk.begin(1, {**SIGNATURE, "particles": 65})
        with self.assertRaises(CacheMismatchError):
            self.disk.begin(2, SIGNATURE)

    def test_foreign_manifest_is_an_error_not_a_crash(self):
        self.disk.directory.mkdir(parents=True)
        self.disk.manifest_path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
        with self.assertRaises(CacheError):
            self.disk.manifest()
        self.disk.manifest_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(CacheError):
            self.disk.manifest()

    def test_begin_removes_leftovers_of_an_interrupted_write(self):
        self.disk.directory.mkdir(parents=True)
        (self.disk.directory / "frame_000002.npz.tmp").write_bytes(b"half")
        self.disk.begin(1, SIGNATURE)
        self.assertEqual(list(self.disk.directory.glob("*.tmp")), [])


class TestStepsPerFrame(unittest.TestCase):
    def test_explicit_setting_wins(self):
        self.assertEqual(bake.steps_per_frame(5, 0.001, 24.0), 5)

    def test_real_time_uses_the_frame_rate(self):
        self.assertEqual(bake.steps_per_frame(0, 0.001, 25.0), 40)    # 40 ms per frame / 1 ms
        self.assertEqual(bake.steps_per_frame(0, 0.0007, 24.0), 60)   # ceil(59.5)
        self.assertEqual(bake.steps_per_frame(0, 1.0, 24.0), 1)       # at least one step

    def test_rejects_nonsense_timing(self):
        with self.assertRaises(ValueError):
            bake.steps_per_frame(0, 0.0, 24.0)


class TestPlanBake(CacheTestCase):
    def test_fresh_plan_clears_old_frames_and_starts_at_first(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 5)
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 10, resume=False)
        self.assertEqual((plan.next_frame, plan.restore, plan.total, plan.already_cached), (1, None, 10, 0))
        self.assertEqual(self.disk.frames(), [])
        self.assertEqual(self.manifest()["start"], 1)

    def test_fresh_plan_replaces_a_cache_with_other_settings(self):
        self.disk.begin(1, {**SIGNATURE, "particles": 999})
        self.fill(1, 2)
        bake.plan_bake(self.disk, SIGNATURE, 1, 4, resume=False)
        self.assertEqual(self.manifest()["signature"], SIGNATURE)

    def test_resume_continues_after_the_last_cached_frame(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 4)
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 10, resume=True, particles=64)
        self.assertEqual((plan.next_frame, plan.already_cached), (5, 4))
        self.assertIsNotNone(plan.restore)
        restore = self.restore_of(plan)
        self.assertEqual(restore.frame, 4)
        self.assertEqual(restore.position.tobytes(), make_frame(4)[0].tobytes())
        self.assertEqual(self.disk.frames(), [1, 2, 3, 4])  # nothing deleted

    def test_resume_with_everything_cached_is_complete(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 6)
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 6, resume=True)
        self.assertTrue(plan.complete)
        self.assertEqual(self.disk.frames(), [1, 2, 3, 4, 5, 6])

    def test_resume_can_extend_the_end_frame(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 3)
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 8, resume=True)
        self.assertEqual((plan.next_frame, plan.total), (4, 8))

    def test_resume_refuses_changed_settings(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 3)
        with self.assertRaises(CacheMismatchError):
            bake.plan_bake(self.disk, {**SIGNATURE, "steps_per_frame": 3}, 1, 8, resume=True)
        self.assertEqual(self.disk.frames(), [1, 2, 3])  # a refusal deletes nothing

    def test_resume_refuses_a_particle_count_mismatch(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 3)
        with self.assertRaises(CacheError):
            bake.plan_bake(self.disk, SIGNATURE, 1, 8, resume=True, particles=65)

    def test_resume_drops_a_corrupt_tail_frame(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 4)
        self.disk.path(4).write_bytes(b"PK\x03\x04 truncated")
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 8, resume=True)
        self.assertEqual((plan.next_frame, plan.discarded), (4, 1))
        self.assertEqual(self.restore_of(plan).frame, 3)
        self.assertEqual(self.disk.frames(), [1, 2, 3])

    def test_resume_without_a_cache_is_a_fresh_bake(self):
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 4, resume=True)
        self.assertEqual((plan.next_frame, plan.restore), (1, None))
        self.assertEqual(self.manifest()["start"], 1)

    def test_empty_range_is_rejected(self):
        with self.assertRaises(ValueError):
            bake.plan_bake(self.disk, SIGNATURE, 5, 4, resume=False)

    def test_invalidate_then_resume_resimulates_from_that_frame(self):
        self.disk.begin(1, SIGNATURE)
        self.fill(1, 8)
        self.disk.invalidate_from(5)
        plan = bake.plan_bake(self.disk, SIGNATURE, 1, 8, resume=True)
        self.assertEqual(plan.next_frame, 5)
        self.assertEqual(self.restore_of(plan).frame, 4)


if __name__ == "__main__":
    unittest.main()
