"""Gravity signature regressions: real disk/job code, deterministic fake solver (no Blender/CUDA)."""
import json
import tempfile
import unittest
from types import SimpleNamespace

from test_cache_bake import COUNT, BakeTestCase, FakeRing, FakeSource

from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import CacheMismatchError, DiskCache


def solver(gravity=(0.0, 0.0, -9.8100004196167), unit=1.0):
    return SimpleNamespace(num_particles=COUNT, dt=0.001, h=0.04,
                           stiffness=90000.0, viscosity=0.15,
                           gravity=gravity, unit_scale=unit)


def signature(value=None):
    return bake.solver_signature(value or solver(), 3, 24.0, "Scene", "CUSTOM")


class TestGravitySignature(unittest.TestCase):
    def test_unchanged_restart_and_json_roundtrip(self):
        self.assertEqual(signature(solver()), json.loads(json.dumps(signature(solver()))))
        self.assertEqual(signature()["gravity"], [0.0, 0.0, -9.8100004196167])

    def test_every_gravity_component_matters(self):
        for vector in ((1.0, 0.0, -9.8100004196167),
                       (0.0, 1.0, -9.8100004196167), (0.0, 0.0, 0.0)):
            with self.subTest(gravity=vector):
                self.assertNotEqual(signature(), signature(solver(vector)))

    def test_nonfinite_gravity_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "gravity"):
                signature(solver((0.0, value, 0.0)))

    def test_unit_scale_contract_is_preserved(self):
        self.assertNotIn("unit_scale", signature())
        self.assertEqual(signature(solver(unit=0.01))["unit_scale"], 0.01)

    def test_legacy_cache_remains_readable_but_resume_requires_rebake(self):
        with tempfile.TemporaryDirectory() as folder:
            disk = DiskCache(folder)
            old = signature()
            del old["gravity"]
            disk.begin(1, old)
            before = disk.manifest_path.read_bytes()
            with self.assertRaisesRegex(CacheMismatchError, "gravity.*[Bb]ake"):
                bake.plan_bake(disk, signature(), 1, 4, resume=True)
            self.assertEqual(before, disk.manifest_path.read_bytes())
            manifest = disk.manifest()
            assert manifest is not None
            self.assertEqual(manifest["signature"], old)

    def test_changed_gravity_refuses_resume_without_touching_cached_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            disk = DiskCache(folder)
            disk.begin(1, signature())
            before = disk.manifest_path.read_bytes()
            with self.assertRaisesRegex(CacheMismatchError, "gravity.*[Bb]ake"):
                bake.plan_bake(disk, signature(solver((0.0, 0.0, -1.0))), 1, 4, resume=True)
            self.assertEqual(before, disk.manifest_path.read_bytes())


class TestGravityRestartResume(BakeTestCase):
    def new_job(self, first, last, *, resume=False, ring=None, source=None, max_slice=4):
        plan = bake.plan_bake(self.disk, signature(solver()), first, last,
                              resume=resume, particles=COUNT)
        from warp_sph_realtime.cache.frame_store import FrameStore
        store = FrameStore(self.disk, ring or FakeRing())
        self.addCleanup(store.close)
        return bake.BakeJob(store, source or FakeSource(first), plan, max_slice=max_slice)

    def test_cancel_restart_resume_matches_uninterrupted(self):
        whole = self.new_job(1, 16)
        self.run_to_end(whole)
        expected = {n: self.disk.read_frame(n) for n in range(1, 17)}
        part = self.new_job(1, 16)
        while part.captured < 3:
            part.run_chunk(0.0)
        part.cancel()
        self.run_to_end(part)
        cached = part.frames_done
        self.assertTrue(0 < cached < 16)
        prefix = {n: self.disk.path(n).read_bytes() for n in range(1, cached + 1)}
        restarted = FakeSource(1)
        resumed = self.new_job(1, 16, resume=True, source=restarted)
        self.assertEqual(restarted.restores, [cached])
        self.run_to_end(resumed)
        for n in range(1, 17):
            got = self.disk.read_frame(n)
            for name in ("position", "velocity", "id"):
                self.assertEqual(getattr(got, name).tobytes(), getattr(expected[n], name).tobytes())
        for n, data in prefix.items():
            self.assertEqual(self.disk.path(n).read_bytes(), data)
