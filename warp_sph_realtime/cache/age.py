"""Particle age from the cache (P2.3). No ``bpy``, no Warp.

The cache files hold no emission time, and emitters only ever add particles (ids are unique and never reused), so a
particle's emission frame is the first cached frame that contains its id. ``EmissionIndex`` finds it incrementally:
one pass over the id arrays from the start of the cache up to the frame being shown (an id lookup table, ``-1`` =
not seen yet), extended as the timeline moves forward and kept when it moves backward or jumps. Age is then
``(frame - emission frame) * seconds per frame``; particles present in the first cached frame have age 0 there.

The cache format is unchanged, so caches baked before this existed work as they are. A re-baked or invalidated frame
is detected by its file timestamp (one directory listing per update) and the scan rolls back to the earliest
changed frame.
"""
from __future__ import annotations

import numpy as np

from .disk import CacheError, DiskCache
from .output import positive

MAX_ID = 1 << 26  # ids index a lookup table; larger ones (never produced by the solver pool) are rejected


def seconds_per_frame(manifest: dict | None, fallback_fps: float = 24.0) -> float:
    """Simulated seconds between two cached frames: ``steps_per_frame * dt`` from the bake signature, else
    ``1 / fps`` (signature or ``fallback_fps``)."""
    signature = (manifest or {}).get("signature") or {}
    steps, dt = signature.get("steps_per_frame"), signature.get("dt")
    if positive(steps) and positive(dt):
        return float(steps) * float(dt)
    fps = signature.get("fps")
    return 1.0 / float(fps) if positive(fps) else 1.0 / float(fallback_fps if positive(fallback_fps) else 24.0)



class EmissionIndex:
    """First-seen frame of every particle id of one cache directory."""

    def __init__(self, disk: DiskCache):
        self.disk = disk
        self._first = np.full(0, -1, dtype=np.int32)
        self._start: int | None = None
        self._last: int | None = None           # newest frame scanned
        self._stamps: dict[int, int] = {}       # scanned frame -> file mtime_ns at scan time

    @property
    def scanned_to(self) -> int | None:
        return self._last

    def reset(self) -> None:
        self._first = np.full(0, -1, dtype=np.int32)
        self._start = self._last = None
        self._stamps.clear()

    def _rollback(self, frame: int) -> None:
        """Forget everything learned from frame ``frame`` onward."""
        if self._start is None or frame <= self._start:
            self.reset()
            return
        self._first[self._first >= frame] = -1
        for number in [n for n in self._stamps if n >= frame]:
            del self._stamps[number]
        self._last = frame - 1

    def sync(self, start: int, frame: int) -> None:
        """Make the index valid through ``frame`` (a cached frame, ``frame >= start``). Raises CacheError."""
        if self._start != start:
            self.reset()
        stamps = self.disk.frame_stamps()  # one directory listing, taken before any read
        stale = min((n for n, stamp in self._stamps.items() if stamps.get(n) != stamp), default=None)
        if stale is not None:  # a scanned frame was rewritten or deleted (invalidate + re-bake): forget from there
            self._rollback(stale)
        if self._start is None:
            self._start = start
        scanned = self._last
        for number in range(start if scanned is None else scanned + 1, frame + 1):
            stamp = stamps.get(number)
            if stamp is None:
                raise CacheError(f"frame {number} is not cached")
            self._learn(number, self.disk.read_ids(number))
            self._stamps[number] = stamp
            self._last = number

    def _learn(self, frame: int, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or high >= MAX_ID:
            raise CacheError(f"frame {frame} has particle ids outside 0..{MAX_ID - 1}")
        if high >= self._first.size:
            grown = np.full(max(high + 1, 2 * self._first.size), -1, dtype=np.int32)
            grown[:self._first.size] = self._first
            self._first = grown
        fresh = ids[self._first[ids] < 0]
        self._first[fresh] = frame

    def emission_frames(self, ids: np.ndarray) -> np.ndarray:
        """First cached frame of each id; ``-1`` for ids not seen by the scan."""
        out = np.full(ids.shape, -1, dtype=np.int32)
        known = (ids >= 0) & (ids < self._first.size)
        out[known] = self._first[ids[known]]
        return out

    def ages(self, ids: np.ndarray, frame: int, per_frame: float) -> np.ndarray:
        """Age in seconds of particles ``ids`` at cached frame ``frame`` (float32; 0 for unknown ids)."""
        first = self.emission_frames(ids)
        return np.where(first >= 0, (frame - first) * per_frame, 0.0).astype(np.float32)
