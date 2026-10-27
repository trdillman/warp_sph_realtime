"""Gravity signature regressions: real disk/job code, deterministic fake solver (no Blender/CUDA)."""
import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from test_cache_bake import COUNT, BakeTestCase, FakeRing, FakeSource

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import CacheMismatchError, DiskCache
from warp_sph_realtime.scene.description import ColliderMotion, EmitterData, SceneDescription


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


def desc(collider_hi=(0.4, 0.4, 0.4), *, fluid_hi=(0.2, 0.2, 0.2), domain_hi=(1.0, 1.0, 1.0), rate=0.0,
         moving=False):
    motion = (ColliderMotion(0, "Paddle", np.eye(3), (0.3, 0.3, 0.3)),) if moving else ()
    return SceneDescription(
        (0.0, 0.0, 0.0), domain_hi, (0.0, 0.0, -9.81), (box_mesh((0.1, 0.1, 0.1), fluid_hi, "Water"),),
        (box_mesh((0.3, 0.3, 0.3), collider_hi, "Paddle"),),
        emitters=(EmitterData(box_mesh((0.6, 0.6, 0.6), (0.7, 0.7, 0.7), "Tap"), rate=rate),),
        collider_motion=motion)


class TestBoundarySignature(unittest.TestCase):
    """Colliders, emitters and domain are part of the bake signature (resume must not mix them)."""

    def test_colliders_emitters_and_domain_change_the_fingerprint(self):
        base = desc().boundary_fingerprint()
        self.assertEqual(base, desc().boundary_fingerprint())
        for name, other in (("collider", desc((0.5, 0.4, 0.4))), ("emitter", desc(rate=100.0)),
                            ("domain", desc(domain_hi=(1.0, 1.0, 2.0))), ("motion", desc(moving=True))):
            with self.subTest(name):
                self.assertNotEqual(base, other.boundary_fingerprint())

    def test_fluid_and_an_animated_colliders_start_pose_do_not(self):
        self.assertEqual(desc().boundary_fingerprint(), desc(fluid_hi=(0.25, 0.2, 0.2)).boundary_fingerprint())
        self.assertEqual(desc(moving=True).boundary_fingerprint(),
                         desc((0.5, 0.5, 0.5), moving=True).boundary_fingerprint())

    def test_signature_carries_the_fingerprint_only_for_authored_scenes(self):
        self.assertNotIn("boundaries", signature())
        authored = solver()
        authored.boundary_fingerprint = desc().boundary_fingerprint()
        self.assertEqual(signature(authored)["boundaries"], authored.boundary_fingerprint)

    def test_changed_or_unknown_boundaries_refuse_resume_without_touching_the_cache(self):
        authored = solver()
        authored.boundary_fingerprint = desc().boundary_fingerprint()
        changed = solver()
        changed.boundary_fingerprint = desc((0.5, 0.4, 0.4)).boundary_fingerprint()
        for cached, label in ((signature(authored), "changed"), (signature(), "unknown")):
            with self.subTest(label), tempfile.TemporaryDirectory() as folder:
                disk = DiskCache(folder)
                disk.begin(1, cached)
                before = disk.manifest_path.read_bytes()
                with self.assertRaisesRegex(CacheMismatchError, f"colliders, emitters or domain {label}"):
                    bake.plan_bake(disk, signature(changed), 1, 4, resume=True)
                self.assertEqual(before, disk.manifest_path.read_bytes())
