"""Actual CUDA resume probe for authored gravity; no Blender UI/transport acceptance."""
import time

import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import CacheMismatchError, DiskCache
from warp_sph_realtime.cache.frame_store import FrameStore
from warp_sph_realtime.cache.gpu import SolverSource, VramRing
from warp_sph_realtime.scene.description import SceneDescription
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def make_solver(gravity, unit):
    desc = SceneDescription((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0), gravity,
                            (box_mesh((-0.12, -0.12, 0.3), (0.12, 0.12, 0.54), "fluid"),),
                            (), unit_scale=unit)
    return RealtimeWCSPHSolver.from_scene(desc, 0.04, device="cuda:0")


def job(disk, solver, graph, resume=False):
    signature = bake.solver_signature(solver, 3, 24.0, "Scene", "CUSTOM")
    plan = bake.plan_bake(disk, signature, 1, 8, resume=resume, particles=solver.num_particles)
    store = FrameStore(disk, VramRing(wp, solver.num_particles, 3))
    return bake.BakeJob(store, SolverSource(wp, solver, 3, graph, first_frame=1), plan)


def drive(current, cancel_after=None):
    deadline = time.monotonic() + 30
    try:
        while current.active and time.monotonic() < deadline:
            current.run_chunk(0.0)
            if cancel_after is not None and current.captured >= cancel_after:
                current.cancel()
            time.sleep(0.001)
        assert not current.active, "bake timed out"
        assert current.state in (bake.DONE, bake.CANCELLED), current.error
    finally:
        current.store.close()


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("unit", [1.0, 0.01])
def test_gravity_cancel_restart_resume_cuda(tmp_path, graph, unit):
    gravity = (1.0, 0.5, -3.0)
    whole, part = DiskCache(tmp_path / "whole"), DiskCache(tmp_path / "part")
    drive(job(whole, make_solver(gravity, unit), graph))
    partial = job(part, make_solver(gravity, unit), graph)
    drive(partial, cancel_after=3)
    assert partial.state == bake.CANCELLED
    assert part.cached_count() == 3
    before = {p.name: p.read_bytes() for p in part.directory.iterdir()}
    with pytest.raises(CacheMismatchError, match="gravity.*Bake Cache"):
        job(part, make_solver((1.0, 0.5, -1.0), unit), graph, resume=True)
    assert before == {p.name: p.read_bytes() for p in part.directory.iterdir()}
    drive(job(part, make_solver(gravity, unit), graph, resume=True))
    for frame in range(1, 9):
        a, b = whole.read_frame(frame), part.read_frame(frame)
        for name in ("position", "velocity", "id"):
            left, right = getattr(a, name), getattr(b, name)
            if unit == 1.0 or name == "id":
                assert left.tobytes() == right.tobytes(), (frame, name)
            else:
                # Scene-unit float32 storage followed by metre restoration is lossy.
                error = float(np.max(np.abs(left - right)))
                bound = 16 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(left))))
                print("ROUNDTRIP", graph, unit, frame, name, error, bound)
                assert error <= bound, (frame, name, error, bound)
        assert np.isfinite(b.position).all()
        assert np.isfinite(b.velocity).all()
    assert not np.array_equal(whole.read_frame(1).position, whole.read_frame(8).position)
