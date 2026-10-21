"""FrameStore (P1.6): a disk adapter (``DiskCache``) fed by a VRAM ring (``VramRing``) through a writer thread.

The main thread never blocks on a copy or a disk write: ``acquire`` returns None while every ring slot is busy,
``pump`` moves finished host copies to the writer thread, which writes them in order (so the files on disk are
always a prefix of the timeline) and then frees the slot. ``close`` drains what was already submitted.

Thread safety: only the writer thread touches the pinned host arrays after ``pump`` hands a slot over, and it
uses no Warp or Blender API. Disk failures stop the writer; ``pump`` then raises ``CacheError``.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from .disk import CacheError, DiskCache

_STOP = object()


class FrameStore:
    def __init__(self, disk: DiskCache, ring: Any = None):
        self.disk = disk
        self.ring = ring
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._outstanding = 0        # submitted, not yet written (or dropped after a failure)
        self._error: CacheError | None = None
        self.frames_written = 0      # this store's writes (not frames already on disk)
        self.bytes_written = 0
        self.last_written = None     # frame number of the newest file written by this store

    # -- main thread ----------------------------------------------------------------------------------

    def acquire(self):
        """A free ring slot to fill, or None (every slot busy). Never blocks."""
        self.pump()
        return self.ring.acquire()

    def abandon(self, slot) -> None:
        """Return an acquired slot that will not be submitted."""
        self.ring.cancel(slot)

    def submit(self, slot, frame: int, count: int) -> None:
        """Queue a filled slot: asynchronous host copy now, disk write when the copy is done."""
        self.ring.submit(slot, frame, count)
        with self._lock:
            self._outstanding += 1

    def pump(self) -> None:
        """Hand finished host copies to the writer. Never blocks. Raises CacheError after a write failure."""
        if self.ring is not None:
            for slot in self.ring.ready():
                self._ensure_thread()
                self._queue.put(slot)
        self._raise_error()

    def pending(self) -> int:
        with self._lock:
            return self._outstanding

    def idle(self) -> bool:
        """True when everything submitted has been written."""
        self.pump()
        return self.pending() == 0

    def close(self, timeout: float = 60.0) -> None:
        """Wait for submitted frames to reach disk (bounded by ``timeout``), then stop the writer.

        Teardown only: this blocks. A failed writer or CUDA error ends the wait early; the cache stays
        consistent either way (a prefix of whole frames)."""
        deadline = time.monotonic() + timeout
        try:
            if self.ring is not None and self.pending():
                self.ring.synchronize()
            while self.pending() and time.monotonic() < deadline:
                self.pump()
                time.sleep(0.001)
        except Exception:
            pass  # CUDA or disk error during teardown: stop waiting, the written prefix is consistent
        finally:
            if self._thread is not None:
                self._queue.put(_STOP)
                self._thread.join(timeout=max(deadline - time.monotonic(), 1.0))
                self._thread = None

    # -- writer thread --------------------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="warp-sph-cache-writer", daemon=True)
            self._thread.start()

    def _raise_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def _run(self) -> None:
        while True:
            slot = self._queue.get()
            if slot is _STOP:
                return
            try:
                with self._lock:
                    failed = self._error is not None
                if not failed:  # after a failure later frames are dropped: files stay a prefix
                    data = slot.host_frame()
                    size = self.disk.write_frame(data.frame, data.position, data.velocity, data.id)
                    with self._lock:
                        self.frames_written += 1
                        self.bytes_written += size
                        self.last_written = data.frame
            except Exception as exc:  # CacheError, or anything unexpected: never lose it silently
                with self._lock:
                    self._error = exc if isinstance(exc, CacheError) else CacheError(f"cache writer failed: {exc}")
            finally:
                self.ring.release(slot)
                with self._lock:
                    self._outstanding -= 1
