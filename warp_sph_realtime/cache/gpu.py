"""VRAM ring buffer and solver adapter for recording (P1.6).

Warp is never imported here (it is imported only by ``addon._activate_runtime()``): the caller passes the
loaded ``warp`` module as ``wp``.

Recording one frame costs the solver stream three device-to-device copies into a ring slot (position, velocity,
id: ~28 B per particle). A second stream then copies the slot to pinned host memory and records an event, so
the solver never waits for PCIe or disk. The host polls the event without blocking (``Event.is_complete``).
The readback is an explicit user action (record or bake); it is never part of a live-view transport.
Nothing here runs inside CUDA graph capture.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from .disk import Frame

FREE, INFLIGHT, QUEUED = "free", "inflight", "queued"


class Slot:
    """One ring entry: device snapshot arrays, pinned host arrays and the event that marks the host copy done."""

    def __init__(self, wp, index: int, capacity: int, device: str):
        self.index = index
        self.state = FREE
        self.frame = 0
        self.count = 0
        self.pos = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.vel = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.pid = wp.empty(capacity, dtype=wp.int32, device=device)
        self._host = (wp.empty(capacity, dtype=wp.vec3, device="cpu", pinned=True),
                      wp.empty(capacity, dtype=wp.vec3, device="cpu", pinned=True),
                      wp.empty(capacity, dtype=wp.int32, device="cpu", pinned=True))
        self.event = wp.Event(device)
        # NumPy views of the pinned memory: valid once ``event`` is complete.
        self.host_position, self.host_velocity, self.host_id = (arr.numpy() for arr in self._host)

    def host_frame(self) -> Frame:
        """Views (not copies) of the first ``count`` host rows; valid until the slot is released."""
        n = self.count
        return Frame(self.frame, self.host_position[:n], self.host_velocity[:n], self.host_id[:n])


class VramRing:
    """``slots`` snapshot buffers for up to ``capacity`` particles.

    Main-thread protocol: ``acquire`` -> fill ``slot.pos`` / ``vel`` / ``pid`` on the solver stream ->
    ``submit`` -> later ``ready`` hands the slot over once its host copy is complete -> whoever consumes the
    host data calls ``release`` (any thread).
    """

    def __init__(self, wp, capacity: int, slots: int = 3, device: str = "cuda:0"):
        if capacity < 1 or slots < 1:
            raise ValueError("ring needs at least one slot and one particle")
        self.wp = wp
        self.device = device
        self.capacity = capacity
        self.copy_stream = wp.Stream(device)
        self.slots = [Slot(wp, i, capacity, device) for i in range(slots)]
        self._inflight: deque[Slot] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> Slot | None:
        """A free slot, or None when every slot is still being copied or written (never blocks)."""
        with self._lock:
            for slot in self.slots:
                if slot.state == FREE:
                    slot.state = INFLIGHT  # reserved; submit() queues it
                    return slot
        return None

    def cancel(self, slot: Slot) -> None:
        """Give back an acquired slot that was never submitted."""
        self.release(slot)

    def submit(self, slot: Slot, frame: int, count: int) -> None:
        """Queue the slot's device snapshot for the asynchronous host copy. ``count`` <= capacity rows."""
        if not 1 <= count <= self.capacity:
            raise ValueError(f"frame has {count} particles, the ring holds 1 to {self.capacity}")
        wp = self.wp
        slot.frame, slot.count = frame, count
        compute = wp.get_stream(self.device)
        self.copy_stream.wait_stream(compute)  # the snapshot copies are ordered before the host copies
        host_pos, host_vel, host_pid = slot._host
        wp.copy(host_pos, slot.pos, count=count, stream=self.copy_stream)
        wp.copy(host_vel, slot.vel, count=count, stream=self.copy_stream)
        wp.copy(host_pid, slot.pid, count=count, stream=self.copy_stream)
        self.copy_stream.record_event(slot.event)
        self._inflight.append(slot)

    def ready(self) -> list[Slot]:
        """Slots (in submit order) whose host copy has finished; each is handed over exactly once."""
        done = []
        while self._inflight and self._inflight[0].event.is_complete:
            slot = self._inflight.popleft()
            slot.state = QUEUED
            done.append(slot)
        return done

    def release(self, slot: Slot) -> None:
        with self._lock:
            slot.state = FREE

    def in_use(self) -> int:
        with self._lock:
            return sum(1 for slot in self.slots if slot.state != FREE)

    def synchronize(self) -> None:
        """Block until every queued host copy is done (teardown only)."""
        self.copy_stream.synchronize()

    def vram_bytes(self) -> int:
        return len(self.slots) * self.capacity * 28


class SolverSource:
    """Takes frames from a ``RealtimeWCSPHSolver``: the initial state, or the state ``steps_per_frame`` later.

    A frame is advanced in slices (``advance``) so a timer tick never spends a whole frame of host time launching
    solver steps, then ``capture`` copies the state into the ring slot."""

    def __init__(self, wp, solver: Any, steps_per_frame: int, use_graph: bool = True):
        if steps_per_frame < 1:
            raise ValueError("steps_per_frame must be at least 1")
        self.wp = wp
        self.solver = solver
        self.steps_per_frame = steps_per_frame
        self.use_graph = use_graph

    @property
    def count(self) -> int:
        return int(self.solver.num_particles)

    def steps_for(self, advance: bool) -> int:
        """Solver steps before the next frame: ``steps_per_frame``, or 0 for the initial state."""
        return self.steps_per_frame if advance else 0

    def advance(self, slot: Slot, steps: int) -> None:
        """Run ``steps`` solver steps (stream-ordered, no host wait)."""
        solver = self.solver
        solver.substeps = steps
        solver.step(slot.pos, use_graph=self.use_graph)  # step() ends by copying pos_sim into slot.pos

    def capture(self, slot: Slot, stepped: bool) -> None:
        """Copy the solver state into the slot (device to device). ``stepped`` False: positions not copied yet."""
        wp, solver = self.wp, self.solver
        if not stepped:
            wp.copy(slot.pos, solver.pos_sim)
        wp.copy(slot.vel, solver.vel)
        wp.copy(slot.pid, solver.pid)

    def restore(self, frame: Frame) -> None:
        """Load a cached frame as the solver state (resume). Raises ValueError on a particle-count mismatch."""
        solver = self.solver
        if frame.count != solver.num_particles:
            raise ValueError(f"cached frame has {frame.count} particles, the scene has {solver.num_particles}")
        solver.pos_sim.assign(frame.position)
        solver.vel.assign(frame.velocity)
        solver.pid.assign(frame.id)
        self.wp.synchronize_device(solver.pos_sim.device)
