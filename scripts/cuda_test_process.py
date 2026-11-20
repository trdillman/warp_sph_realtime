"""Run a test command with an owned process tree that can be drained before lease release."""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

CLEANUP_SECONDS = 5


class ProcessCleanupError(RuntimeError):
    """The test runner could not verify that its process tree stopped."""


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
        ("process_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
        ("priority", wintypes.DWORD), ("schedule", wintypes.DWORD),
    ]


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _JobBasicLimits), ("io_counters", ctypes.c_ulonglong * 6),
        ("memory_limits", ctypes.c_size_t * 4),
    ]


class _JobAccounting(ctypes.Structure):
    _fields_ = [
        ("times", ctypes.c_longlong * 4), ("faults", wintypes.DWORD),
        ("total", wintypes.DWORD), ("active", wintypes.DWORD), ("terminated", wintypes.DWORD),
    ]


class _WindowsJob:
    def __init__(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            "GetModuleFileNameW": ([wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD], wintypes.DWORD),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                         wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                           wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = argtypes, restype
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _JobExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def interpreter_image(self) -> str:
        # sys.executable can refer to a venv redirector that starts a process before GO.
        filename = ctypes.create_unicode_buffer(32768)
        length = self.kernel.GetModuleFileNameW(None, filename, len(filename))
        if not 0 < length < len(filename):
            raise ctypes.WinError(ctypes.get_last_error())
        return filename.value

    def assign(self, pid: int) -> None:
        # The launcher owns a Popen handle and the bootstrap cannot run the test before admission.
        process = self.kernel.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE.
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def drain(self) -> None:
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + CLEANUP_SECONDS
        while True:
            accounting = _JobAccounting()
            if not self.kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.active == 0:
                return
            if time.monotonic() >= deadline:
                raise ProcessCleanupError("Windows test job still has active processes.")
            time.sleep(0.02)

    def close(self) -> None:
        self.kernel.CloseHandle(self.handle)


def _run_windows(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> int:
    job = _WindowsJob()
    try:
        bootstrap = [job.interpreter_image(), "-I", "-S", str(Path(__file__).resolve()), json.dumps(command)]
        child = subprocess.Popen(bootstrap, cwd=cwd, env=env, stdin=subprocess.PIPE)
        assigned = False
        try:
            job.assign(child.pid)
            assigned = True
            assert child.stdin is not None
            child.stdin.write(b"GO\n")
            child.stdin.close()
            return child.wait(timeout=timeout)
        finally:
            try:
                if assigned:
                    job.drain()
                elif child.poll() is None:
                    child.kill()  # The unassigned bootstrap never received GO.
                child.wait(timeout=CLEANUP_SECONDS)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ProcessCleanupError("Windows test process cleanup failed.") from error
            finally:
                if child.stdin is not None and not child.stdin.closed:
                    child.stdin.close()
    finally:
        job.close()


def _drain_process_group(group: int) -> None:
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + CLEANUP_SECONDS
    while time.monotonic() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise ProcessCleanupError("The test process group did not exit.")


def run_owned(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> int:
    if os.name == "nt":
        return _run_windows(command, cwd=cwd, env=env, timeout=timeout)
    child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    finally:
        try:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=CLEANUP_SECONDS)
            _drain_process_group(child.pid)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProcessCleanupError("Test process group cleanup failed.") from error


if __name__ == "__main__":
    # This child cannot launch pytest until its parent puts it in the non-breakaway job.
    if sys.stdin.buffer.readline() != b"GO\n":
        raise SystemExit(2)
    raise SystemExit(subprocess.call(json.loads(sys.argv[1]), stdin=subprocess.DEVNULL))
