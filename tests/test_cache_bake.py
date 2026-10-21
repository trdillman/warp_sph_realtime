"""Bake job and FrameStore (P1.6, P2.2) with a fake ring and source: chunking, cancel, resume, invalidate, failure.

The real GPU ring and solver are covered by tests/test_cache_cuda.py.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import CacheError, DiskCache, Frame
from warp_sph_realtime.cache.frame_store import FrameStore
from warp_sph_realtime.cache.gpu import FREE, INFLIGHT, QUEUED

COUNT = 32
SIGNATURE = {"particles": COUNT}


def state_at(tick: int):
    """Deterministic solver state after ``tick`` steps: what the fake solver 'simulates'."""
    base = np.arange(COUNT, dtype=np.float32)
    position = np.stack([np.full(COUNT, tick, np.float32), base, base * 0.5 + tick], axis=1)
    return position, -position, ((np.arange(COUNT) * 7 + tick) % COUNT).astype(np.int32)


class FakeSlot:
    def __init__(self, index):
        self.index, self.state, self.frame, self.count = index, FREE, 0, 0
        self.pos = np.zeros((COUNT, 3), np.float32)
        self.vel = np.zeros((COUNT, 3), np.float32)
        self.pid = np.zeros(COUNT, np.int32)
        self.host_position = np.zeros((COUNT, 3), np.float32)
        self.host_velocity = np.zeros((COUNT, 3), np.float32)
        self.host_id = np.zeros(COUNT, np.int32)

    def host_frame(self) -> Frame:
        n = self.count
        return Frame(self.frame, self.host_position[:n], self.host_velocity[:n], self.host_id[:n])


class FakeRing:
    """Same protocol as cache.gpu.VramRing; ``hold`` keeps host copies 'in flight' until ``complete()``."""

    def __init__(self, slots=3, hold=False):
        self.slots = [FakeSlot(i) for i in range(slots)]
        self.hold, self._inflight, self._lock = hold, [], threading.Lock()
        self.synchronized = 0

    def acquire(self):
        with self._lock:
            for slot in self.slots:
                if slot.state == FREE:
                    slot.state = INFLIGHT
                    return slot
        return None

    def cancel(self, slot):
        self.release(slot)

    def submit(self, slot, frame, count):
        slot.frame, slot.count = frame, count
        slot.host_position[:count], slot.host_velocity[:count], slot.host_id[:count] = slot.pos, slot.vel, slot.pid
        self._inflight.append(slot)

    def complete(self):
        self.hold = False

    def ready(self):
        if self.hold:
            return []
        done, self._inflight = self._inflight, []
        for slot in done:
            slot.state = QUEUED
        return done

    def release(self, slot):
        with self._lock:
            slot.state = FREE

    def synchronize(self):
        self.synchronized += 1
        self.hold = False


class FakeSource:
    """Fake solver: state after ``n`` frames is ``state_at(n)``; a frame takes ``steps`` solver steps."""
    count = COUNT

    def __init__(self, first=1, steps=3):
        self.first, self.steps, self.substeps_taken, self.restores, self.slices = first, steps, 0, [], []

    def steps_for(self, advance):
        return self.steps if advance else 0

    def advance(self, slot, steps):
        self.slices.append(steps)
        self.substeps_taken += steps

    def capture(self, slot, stepped):
        assert self.substeps_taken % self.steps == 0, "a frame must be captured on a frame boundary"
        slot.pos[:], slot.vel[:], slot.pid[:] = state_at(self.substeps_taken // self.steps)

    def restore(self, frame):
        self.restores.append(frame.frame)
        self.substeps_taken = (frame.frame - self.first) * self.steps


class BakeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.disk = DiskCache(Path(self._tmp.name) / "cache")

    def new_job(self, first, last, *, resume=False, ring=None, source=None, max_slice=4):
        plan = bake.plan_bake(self.disk, SIGNATURE, first, last, resume=resume, particles=COUNT)
        store = FrameStore(self.disk, ring or FakeRing())
        self.addCleanup(store.close)
        return bake.BakeJob(store, source or FakeSource(first), plan, max_slice=max_slice)

    def run_to_end(self, job, budget=0.002, timeout=60.0):
        """Drive the job the way the 1 ms timer does, waiting on the writer thread without busy-spinning."""
        deadline, calls, idle_calls, seen = time.monotonic() + timeout, 0, 0, None
        while job.active and time.monotonic() < deadline:
            job.run_chunk(budget)
            calls += 1
            progress = (job.captured, job.store.frames_written, job.state)
            idle_calls = idle_calls + 1 if progress == seen else 0
            seen = progress
            time.sleep(0.001 if idle_calls > 50 else 0)  # yield to the writer; sleep once nothing else can move
        self.assertFalse(job.active, "bake did not finish")
        job.store.close()
        return calls

    def assert_whole_prefix(self, first, expected_last):
        """Every frame first..expected_last is present, readable and exact; nothing else is left behind."""
        self.assertEqual(self.disk.frames(), list(range(first, expected_last + 1)))
        self.assertEqual(self.disk.cached_range(), (first, expected_last))
        self.assertEqual(list(self.disk.directory.glob("*.tmp")), [])
        for frame in range(first, expected_last + 1):
            read = self.disk.read_frame(frame)
            for got, want in zip((read.position, read.velocity, read.id), state_at(frame - first), strict=True):
                self.assertEqual(got.tobytes(), want.tobytes(), f"frame {frame}")


class TestBakeJob(BakeTestCase):
    def test_bakes_the_range_in_order_from_the_initial_state(self):
        job = self.new_job(1, 12)
        self.run_to_end(job)
        self.assertEqual(job.state, bake.DONE)
        self.assertEqual((job.captured, job.frames_done, job.fraction), (12, 12, 1.0))
        self.assert_whole_prefix(1, 12)  # frame 1 is the initial state (tick 0), frame 12 is tick 11

    def test_chunks_are_bounded_and_do_not_block(self):
        job = self.new_job(1, 30)
        captured_per_call = []
        while job.active:
            before = job.captured
            job.run_chunk(0.0)  # zero budget: at most one frame per call
            time.sleep(0)  # let the writer thread run
            captured_per_call.append(job.captured - before)
        self.assertLessEqual(max(captured_per_call), 1)
        self.assertGreater(len(captured_per_call), 30)  # progress needed more than one call per frame
        job.store.close()
        self.assert_whole_prefix(1, 30)

    def test_a_frame_of_many_steps_is_advanced_in_slices_across_calls(self):
        source = FakeSource(1, steps=10)
        job = self.new_job(1, 4, source=source, max_slice=4)
        calls_per_frame = []
        last_captured, calls = 0, 0
        while job.active:
            job.run_chunk(0.0)  # zero budget: one slice per call
            time.sleep(0)
            calls += 1
            if job.captured > last_captured:
                calls_per_frame.append(calls)
                last_captured, calls = job.captured, 0
        job.store.close()
        self.assertEqual(max(source.slices), 4)                      # never more than one slice of steps per call
        self.assertEqual(sum(source.slices), 30)                     # frames 2..4: 10 steps each; frame 1 none
        self.assertTrue(all(n >= 3 for n in calls_per_frame[1:]))    # ceil(10 / 4) calls per stepped frame
        self.assert_whole_prefix(1, 4)

    def test_cancel_in_the_middle_of_a_frame_returns_its_slot(self):
        ring = FakeRing(slots=2)
        job = self.new_job(1, 10, ring=ring, source=FakeSource(1, steps=9), max_slice=2)
        job.run_chunk(0.0)                       # frame 1 (initial state) captured, or its first slice started
        while job.captured < 2:
            job.run_chunk(0.0)
        job.run_chunk(0.0)                       # now inside frame 3: some slices done, not captured
        job.cancel()
        self.run_to_end(job)
        self.assertEqual(job.state, bake.CANCELLED)
        self.assertEqual([slot.state for slot in ring.slots], [FREE, FREE])
        self.assert_whole_prefix(1, job.captured)

    def test_a_full_ring_waits_for_the_next_call_instead_of_blocking(self):
        ring = FakeRing(slots=1, hold=True)
        job = self.new_job(1, 5, ring=ring)
        self.assertTrue(job.run_chunk(1.0))        # the only slot is taken
        self.assertFalse(job.run_chunk(1.0))       # no slot: returns at once, nothing captured
        self.assertEqual(job.captured, 1)
        self.assertEqual(ring.synchronized, 0)     # it never blocked on the copy
        ring.complete()
        self.run_to_end(job)
        self.assert_whole_prefix(1, 5)

    def test_progress_reports_frames_on_disk_not_frames_queued(self):
        ring = FakeRing(hold=True)
        job = self.new_job(1, 10, ring=ring)
        job.run_chunk(1.0)
        self.assertGreater(job.captured, 0)
        self.assertEqual(job.frames_done, 0)
        ring.complete()
        self.run_to_end(job)
        self.assertEqual(job.frames_done, 10)

    def test_cancel_leaves_a_consistent_whole_prefix(self):
        job = self.new_job(1, 50)
        for _ in range(7):
            job.run_chunk(0.0)
        job.cancel()
        self.assertEqual(job.state, bake.CANCELLING)
        self.run_to_end(job)
        self.assertEqual(job.state, bake.CANCELLED)
        self.assertLess(job.captured, 50)
        self.assertEqual(job.frames_done, job.captured)  # everything captured before the cancel was written
        self.assert_whole_prefix(1, job.captured)
        self.assertEqual((self.disk.manifest() or {}).get("start"), 1)

    def test_cancel_with_frames_in_flight_still_ends_consistent(self):
        ring = FakeRing(slots=3, hold=True)
        job = self.new_job(1, 20, ring=ring)
        job.run_chunk(1.0)
        job.cancel()
        ring.complete()
        self.run_to_end(job)
        self.assertEqual(job.state, bake.CANCELLED)
        self.assert_whole_prefix(1, job.captured)

    def test_cancel_after_completion_changes_nothing(self):
        job = self.new_job(1, 3)
        self.run_to_end(job)
        job.cancel()
        self.assertEqual(job.state, bake.DONE)

    def test_resume_after_cancel_equals_an_uninterrupted_bake(self):
        job = self.new_job(1, 25)
        for _ in range(9):
            job.run_chunk(0.0)
        job.cancel()
        self.run_to_end(job)
        cached = job.captured
        self.assertTrue(0 < cached < 25)

        source = FakeSource(1)
        resumed = self.new_job(1, 25, resume=True, source=source)
        self.assertEqual(resumed.plan.next_frame, cached + 1)
        self.assertEqual(source.restores, [cached])        # state came from the last cached frame
        self.run_to_end(resumed)
        self.assertEqual(resumed.state, bake.DONE)
        self.assertEqual(resumed.captured, 25 - cached)
        self.assert_whole_prefix(1, 25)                     # identical to uninterrupted (state_at is the reference)

    def test_resume_extends_a_finished_bake_to_a_later_end(self):
        self.run_to_end(self.new_job(1, 6))
        self.run_to_end(self.new_job(1, 10, resume=True))
        self.assert_whole_prefix(1, 10)

    def test_invalidate_from_frame_then_resume_resimulates_only_the_tail(self):
        self.run_to_end(self.new_job(1, 10))
        stale_marker = self.disk.path(3).stat().st_mtime_ns
        self.disk.invalidate_from(7)
        job = self.new_job(1, 10, resume=True)
        self.assertEqual((job.plan.next_frame, job.plan.already_cached), (7, 6))
        self.run_to_end(job)
        self.assert_whole_prefix(1, 10)
        self.assertEqual(self.disk.path(3).stat().st_mtime_ns, stale_marker)  # earlier frames untouched

    def test_fresh_bake_replaces_the_old_cache(self):
        self.run_to_end(self.new_job(1, 10))
        job = self.new_job(1, 4)  # fresh: the old 10-frame cache is gone
        self.assertEqual(self.disk.frames(), [])
        self.run_to_end(job)
        self.assert_whole_prefix(1, 4)

    def test_everything_already_cached_is_complete_without_work(self):
        self.run_to_end(self.new_job(1, 5))
        job = self.new_job(1, 5, resume=True)
        self.assertEqual(job.state, bake.DONE)
        self.assertFalse(job.active)

    def test_a_disk_failure_ends_the_job_with_a_consistent_prefix(self):
        real_write = self.disk.write_frame

        def failing(frame, *arrays):
            if frame == 4:
                raise CacheError("disk full")
            return real_write(frame, *arrays)

        self.disk.write_frame = failing  # type: ignore[method-assign]
        job = self.new_job(1, 10)
        self.run_to_end(job)
        self.assertEqual(job.state, bake.FAILED)
        self.assertIn("disk full", job.error)
        self.assert_whole_prefix(1, 3)

    def test_eta_appears_once_frames_are_written(self):
        job = self.new_job(1, 200, ring=FakeRing(hold=True))
        job.run_chunk(0.0)
        self.assertIsNone(job.eta_seconds())
        job.store.ring.complete()
        while job.store.frames_written < 3 and job.active:
            job.run_chunk(0.0)
        self.assertIsNotNone(job.eta_seconds())
        job.cancel()
        self.run_to_end(job)


class TestFrameStore(BakeTestCase):
    def test_close_drains_submitted_frames(self):
        ring = FakeRing(slots=3, hold=True)
        store = FrameStore(self.disk, ring)
        for frame in (1, 2, 3):
            slot = store.acquire()
            slot.pos[:], slot.vel[:], slot.pid[:] = state_at(frame)
            store.submit(slot, frame, COUNT)
        self.assertEqual(store.pending(), 3)
        store.close()  # synchronizes the ring, then waits for the writer
        self.assertEqual(store.pending(), 0)
        self.assertEqual(self.disk.frames(), [1, 2, 3])
        self.assertEqual(store.frames_written, 3)
        self.assertGreater(store.bytes_written, 3 * COUNT * 28)

    def test_acquire_reports_busy_without_blocking(self):
        ring = FakeRing(slots=2, hold=True)
        store = FrameStore(self.disk, ring)
        self.addCleanup(store.close)
        for frame in (1, 2):
            slot = store.acquire()
            self.assertIsNotNone(slot)
            store.submit(slot, frame, COUNT)
        self.assertIsNone(store.acquire())

    def test_slots_are_recycled_after_the_write(self):
        store = FrameStore(self.disk, FakeRing(slots=1))
        self.addCleanup(store.close)
        for frame in range(1, 6):
            slot = None
            for _ in range(2000):
                slot = store.acquire()
                if slot is not None:
                    break
                threading.Event().wait(0.001)
            self.assertIsNotNone(slot, f"slot never recycled for frame {frame}")
            slot.pos[:], slot.vel[:], slot.pid[:] = state_at(frame)
            store.submit(slot, frame, COUNT)
        store.close()
        self.assertEqual(self.disk.frames(), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
