"""Run a Warp install on a worker thread so Blender's UI stays responsive.

Pure Python: no ``bpy`` here. The worker runs ``warp_installer.run_plan`` (which
touches no Blender state) and only writes plain values into this object under a
lock. The main thread polls ``snapshot()`` from a ``bpy.app.timers`` timer and
draws the progress bar; it never blocks on the worker.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from . import warp_installer
from .warp_installer import InstallPlan, InstallResult

Runner = Callable[..., InstallResult]


@dataclass(frozen=True)
class JobSnapshot:
    """A consistent view of the job for drawing; ``result`` is set once the job is over."""

    message: str
    fraction: float
    cancelling: bool
    result: InstallResult | None

    @property
    def finished(self) -> bool:
        return self.result is not None


class InstallJob:
    """One install on one daemon thread. Start once; poll with ``snapshot()``; stop with ``cancel()``."""

    def __init__(self, plan: InstallPlan, runner: Runner | None = None):
        self._plan = plan
        self._runner = runner or warp_installer.run_plan
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._message = "Starting..."
        self._fraction = 0.0
        self._result: InstallResult | None = None
        self._thread = threading.Thread(target=self._work, name="warpsph-install", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """Ask the worker to stop at its next checkpoint; nothing is promoted after this."""
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(self._message, self._fraction, self._cancel.is_set(), self._result)

    def _on_progress(self, message: str, fraction: float) -> None:
        with self._lock:
            self._message = message
            self._fraction = fraction

    def _work(self) -> None:
        try:
            result = self._runner(self._plan, progress=self._on_progress, cancel=self._cancel.is_set)
        except Exception as exc:  # noqa: BLE001 - a worker must always end with a result for the UI
            result = InstallResult(success=False, message=f"Installation failed: {exc}")
        with self._lock:
            self._result = result
