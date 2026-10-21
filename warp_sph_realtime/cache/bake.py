"""Bake planning and the chunked bake job (P1.6 record, P2.2 bake). No ``bpy``, no Warp.

A bake simulates frames ``first`` ... ``last`` and records each to a ``FrameStore``. Frame ``first`` is the
initial state; frame ``k`` is the state ``(k - first) * steps_per_frame`` solver steps later. ``plan_bake`` decides
where to start (fresh, or resume after the last intact cached frame) and ``BakeJob.run_chunk`` does a bounded
amount of work per call, so a timer can drive it without ever blocking the UI.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from .disk import CacheError, DiskCache, Frame

RUNNING, CANCELLING, FINISHING = "RUNNING", "CANCELLING", "FINISHING"
DONE, CANCELLED, FAILED = "DONE", "CANCELLED", "FAILED"
ACTIVE_STATES = (RUNNING, CANCELLING, FINISHING)


def steps_per_frame(setting: int, dt: float, fps: float) -> int:
    """Solver steps per cached frame: ``setting`` when positive, else real time (``1 / fps`` s of simulated time)."""
    if setting > 0:
        return int(setting)
    if dt <= 0.0 or fps <= 0.0:
        raise ValueError("time step and frame rate must be positive")
    return max(1, math.ceil(1.0 / (fps * dt) - 1e-9))


@dataclass(frozen=True)
class BakePlan:
    first: int                 # first frame of the timeline (the initial state)
    last: int                  # last frame wanted
    next_frame: int            # first frame still to capture (last + 1 when everything is cached)
    restore: Frame | None      # state to load before capturing next_frame (resume), else None
    discarded: int = 0         # corrupt tail frames removed while planning a resume

    @property
    def total(self) -> int:
        return self.last - self.first + 1

    @property
    def already_cached(self) -> int:
        return self.next_frame - self.first

    @property
    def complete(self) -> bool:
        return self.next_frame > self.last


def plan_bake(disk: DiskCache, signature: dict, first: int, last: int, *, resume: bool,
              particles: int | None = None) -> BakePlan:
    """Prepare ``disk`` for a bake of frames ``first`` ... ``last``.

    Fresh (``resume`` False, or nothing to resume): the cache is cleared and a new manifest written.
    Resume: the manifest must match ``signature`` (CacheMismatchError otherwise); capturing continues after the
    last intact frame, whose state is returned in ``restore``. A last frame that cannot be read (power loss) is
    dropped first. ``particles`` (optional) must equal the cached frame's particle count.
    """
    if last < first:
        raise ValueError("the bake range is empty (end is before start)")
    if resume and disk.manifest() is not None:
        disk.begin(first, signature)
        discarded = 0
        while True:
            span = disk.cached_range()
            if span is None:
                break
            try:
                tail = disk.read_frame(span[1])
            except CacheError:
                disk.invalidate_from(span[1])
                discarded += 1
                continue
            if particles is not None and tail.count != particles:
                raise CacheError(f"cached frames have {tail.count} particles, the scene has {particles}")
            if span[1] >= last:
                return BakePlan(first, last, last + 1, None, discarded)  # already complete
            return BakePlan(first, last, span[1] + 1, tail, discarded)
        disk.invalidate_from(first)  # nothing intact at the start: bake from scratch
        return BakePlan(first, last, first, None, discarded)
    disk.clear()
    disk.begin(first, signature)
    return BakePlan(first, last, first, None)


class BakeJob:
    """State machine for one bake. ``run_chunk`` never blocks; ``cancel`` takes effect at the next chunk.

    ``store`` is a FrameStore; ``source`` provides ``count``, ``steps_for``, ``advance`` and ``capture``
    (``SolverSource``). A frame is advanced ``max_slice`` solver steps at a time and the clock is checked after each
    slice, so one call costs about ``budget_s`` however many steps a frame needs. Cancel and failure leave the cache
    as a prefix of whole frames: frames already captured are still written.
    """

    def __init__(self, store, source, plan: BakePlan, clock: Callable[[], float] = time.perf_counter,
                 max_slice: int = 4):
        self.store = store
        self.source = source
        self.plan = plan
        self._clock = clock
        self.max_slice = max(1, max_slice)
        self.next_frame = plan.next_frame
        self._slot = None        # ring slot of the frame being advanced
        self._steps_left = 0
        self._stepped = False
        self.captured = 0        # frames this job has handed to the store
        self.error = ""
        self.state = RUNNING
        self._started = clock()
        if plan.restore is not None:
            source.restore(plan.restore)
        if plan.complete:
            self.state = DONE

    # -- progress -------------------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def frames_done(self) -> int:
        """Frames of the range safely on disk."""
        return self.plan.already_cached + self.store.frames_written

    @property
    def fraction(self) -> float:
        return min(1.0, self.frames_done / self.plan.total)

    def eta_seconds(self) -> float | None:
        written = self.store.frames_written
        if written < 1 or not self.active:
            return None
        elapsed = self._clock() - self._started
        return elapsed / written * (self.plan.total - self.frames_done)

    # -- control --------------------------------------------------------------------------------------

    def cancel(self) -> None:
        if self.state == RUNNING:
            self.state = CANCELLING

    def run_chunk(self, budget_s: float) -> bool:
        """Capture frames for up to ``budget_s`` seconds, then update the state. Returns True if a frame was
        captured. Raises only what ``source.advance`` or ``capture`` raise (solver errors); cache errors end the job as FAILED."""
        captured = False
        try:
            if self.state == RUNNING:
                deadline = self._clock() + budget_s
                while self.next_frame <= self.plan.last:
                    if self._slot is None:
                        self._slot = self.store.acquire()
                        if self._slot is None:
                            break
                        self._steps_left = self.source.steps_for(self.next_frame > self.plan.first)
                        self._stepped = False
                    try:
                        if self._steps_left > 0:
                            n = min(self._steps_left, self.max_slice)
                            self.source.advance(self._slot, n)
                            self._steps_left -= n
                            self._stepped = True
                        if self._steps_left == 0:
                            self.source.capture(self._slot, self._stepped)
                            slot, self._slot = self._slot, None
                            self.store.submit(slot, self.next_frame, self.source.count)
                            self.next_frame += 1
                            self.captured += 1
                            captured = True
                    except BaseException:
                        self._drop_frame()
                        raise
                    if self._clock() >= deadline:
                        break
                if self.next_frame > self.plan.last:
                    self.state = FINISHING
            else:
                self._drop_frame()
            if self.state in (CANCELLING, FINISHING) and self.store.idle():
                self.state = DONE if self.state == FINISHING else CANCELLED
            else:
                self.store.pump()
        except CacheError as exc:
            self.fail(str(exc))
        return captured

    def _drop_frame(self) -> None:
        """Give back the slot of a frame that was started but will not be finished (cancel, failure)."""
        slot, self._slot = self._slot, None
        if slot is not None:
            self.store.abandon(slot)

    def fail(self, message: str) -> None:
        self.error = message
        self.state = FAILED
        self._drop_frame()
