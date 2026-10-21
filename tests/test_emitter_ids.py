"""Stable particle ids with emitters (P1.6 x P2.1): emitted particles get fresh unique ids, ids follow particles
through the spatial sort, and a bake records the active (growing) count with unique ids.

The CPU-device variants run in CI; the CUDA variants (graph launches, the real VRAM ring) need a GPU.
"""
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh
from tests.test_scene_seam_cpu import plane
from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import DiskCache, Frame
from warp_sph_realtime.cache.frame_store import FrameStore
from warp_sph_realtime.cache.gpu import FREE, INFLIGHT, QUEUED, SolverSource, VramRing
from warp_sph_realtime.scene.description import EmitterData, SceneDescription
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

SPACING = 0.04
TALL = ((-0.5, -0.5, 0.0), (0.5, 0.5, 1.2))
STEPS = 4  # solver steps (substeps) per cached frame

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required"))]


def emitter_scene(budget=1200) -> SceneDescription:
    """A resting block (so the sort really permutes) plus a tap pouring onto it."""
    block = box_mesh((-0.2, -0.2, 0.0), (0.2, 0.2, 0.12), "block")
    tap = EmitterData(plane((-0.08, -0.08), (0.08, 0.08), 1.0), speed=1.5, rate=0.0)
    return SceneDescription(*TALL, (0.0, 0.0, -9.81), (block,), (), emitters=(tap,), particle_budget=budget)


def active_state(solver):
    n = solver.active_count
    return solver.pid.numpy()[:n].copy(), solver.pos_sim.numpy()[:n].copy()


@pytest.mark.parametrize("device", DEVICES)
def test_emitted_particles_get_fresh_unique_ids_that_follow_them(device):
    solver = RealtimeWCSPHSolver.from_scene(emitter_scene(), SPACING, device=device)
    fill = solver.active_count
    assert 0 < fill < solver.num_particles
    out = wp.zeros(solver.num_particles, dtype=wp.vec3, device=device)
    ids, pos = active_state(solver)
    assert ids.tolist() == list(range(fill))
    counts, permuted = [fill], False
    for _ in range(12):
        previous = dict(zip(ids.tolist(), pos, strict=True))
        solver.step(out)
        wp.synchronize_device(device)
        ids, pos = active_state(solver)
        counts.append(len(ids))
        # Particles never die, so the active ids are exactly 0 .. n-1: unique, none stale or duplicated.
        assert sorted(ids.tolist()) == list(range(len(ids)))
        old = np.array([i in previous for i in ids.tolist()])
        moved = np.array([np.linalg.norm(p - previous[i]) for i, p in zip(ids[old].tolist(), pos[old], strict=True)])
        assert moved.max() < SPACING  # an id stays with its particle (a random order would jump ~0.5 m)
        new = pos[~old]
        assert (new[:, 2] > 0.8).all()  # new ids belong to particles that just left the tap
        permuted = permuted or not np.array_equal(ids, np.arange(len(ids)))
    assert counts[-1] > fill and counts == sorted(counts)  # the emitter added particles
    assert permuted  # the spatial sort really reordered the slots
    assert solver.active_count < solver.num_particles  # still inside the pool


class CpuRing:
    """VramRing's protocol on the CPU device (VramRing needs CUDA streams): synchronous host copies."""

    class Slot:
        def __init__(self, index, capacity, device):
            self.index, self.state, self.frame, self.count = index, FREE, 0, 0
            self.pos = wp.empty(capacity, dtype=wp.vec3, device=device)
            self.vel = wp.empty(capacity, dtype=wp.vec3, device=device)
            self.pid = wp.empty(capacity, dtype=wp.int32, device=device)

        def host_frame(self) -> Frame:
            n = self.count
            return Frame(self.frame, self.pos.numpy()[:n].copy(), self.vel.numpy()[:n].copy(),
                         self.pid.numpy()[:n].copy())

    def __init__(self, capacity, slots=3, device="cpu"):
        self.capacity = capacity
        self.slots = [self.Slot(i, capacity, device) for i in range(slots)]
        self._inflight = []

    def acquire(self):
        for slot in self.slots:
            if slot.state == FREE:
                slot.state = INFLIGHT
                return slot
        return None

    def cancel(self, slot):
        self.release(slot)

    def submit(self, slot, frame, count):
        assert 1 <= count <= self.capacity
        slot.frame, slot.count = frame, count
        self._inflight.append(slot)

    def ready(self):
        done, self._inflight = self._inflight, []
        for slot in done:
            slot.state = QUEUED
        return done

    def release(self, slot):
        slot.state = FREE

    def synchronize(self):
        pass


def bake_frames(disk, device, first, last, *, resume=False, stop_after=None):
    solver = RealtimeWCSPHSolver.from_scene(emitter_scene(), SPACING, device=device)
    ring = VramRing(wp, solver.num_particles, 3) if device != "cpu" else CpuRing(solver.num_particles)
    plan = bake.plan_bake(disk, {"particles": solver.num_particles}, first, last, resume=resume)
    store = FrameStore(disk, ring)
    job = bake.BakeJob(store, SolverSource(wp, solver, STEPS, first_frame=first), plan)
    deadline = time.monotonic() + 120.0
    while job.active and time.monotonic() < deadline:
        job.run_chunk(0.004)
        if stop_after is not None and job.captured >= stop_after:
            job.cancel()
            stop_after = None
        time.sleep(0.001)
    store.close()
    assert not job.active, f"bake stuck in {job.state}"
    return solver, job


@pytest.mark.parametrize("device", DEVICES)
def test_bake_of_an_emitter_scene_records_growing_counts_with_unique_ids(device):
    with tempfile.TemporaryDirectory() as tmp:
        disk = DiskCache(Path(tmp) / "c")
        solver, job = bake_frames(disk, device, 1, 8)
        assert job.state == bake.DONE and disk.cached_range() == (1, 8)
        frames = [disk.read_frame(f) for f in range(1, 9)]
        counts = [f.count for f in frames]
        assert counts == sorted(counts) and counts[-1] > counts[0]  # the active count, not the pool
        assert counts[-1] < solver.num_particles
        for a, b in zip(frames, frames[1:], strict=False):
            assert sorted(b.id.tolist()) == list(range(b.count))  # unique, fresh ids for the emitted ones
            where = {int(i): k for k, i in enumerate(b.id)}
            matched = np.array([where[int(i)] for i in a.id])
            assert np.abs(b.position[matched] - a.position).max() < 2.0 * SPACING  # ids persist with particles

        # Resume after a cancel continues the schedule from the restored frame's time and count.
        part = DiskCache(Path(tmp) / "p")
        _, cancelled = bake_frames(part, device, 1, 8, stop_after=3)
        assert cancelled.state == bake.CANCELLED
        _, resumed = bake_frames(part, device, 1, 8, resume=True)
        assert resumed.state == bake.DONE and part.cached_range() == (1, 8)
        for f in range(1, 9):
            whole, again = disk.read_frame(f), part.read_frame(f)
            assert whole.count == again.count, f"frame {f}"
            for name in ("position", "velocity", "id"):  # bit-identical to the uninterrupted bake
                assert getattr(whole, name).tobytes() == getattr(again, name).tobytes(), f"{name} of frame {f}"


def test_solver_source_feeds_collider_keys_before_each_slice_and_counts_the_active_particles():
    class Solver:
        num_particles, active_count, substeps, dt = 100, 37, 1, 0.01

        def __init__(self):
            self.calls = []

        def step(self, out, use_graph=True):
            self.calls.append(("step", self.substeps))

    solver = Solver()
    source = SolverSource(wp, solver, 6, before_step=lambda: solver.calls.append(("keys", None)))
    assert source.count == 37  # the live particles, not the pool
    slot = types.SimpleNamespace(pos=None)
    source.advance(slot, 4)
    source.advance(slot, 2)
    assert solver.calls == [("keys", None), ("step", 4), ("keys", None), ("step", 2)]
