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
SUPERVISOR_FAILED = 125


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
            "IsProcessInJob": ([wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
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

    def _process_ids(self, deadline: float) -> list[int]:
        capacity = 16
        while True:
            # JOBOBJECT_BASIC_PROCESS_ID_LIST ends with a variable-length ULONG_PTR array.
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("listed", wintypes.DWORD),
                            ("ids", ctypes.c_size_t * capacity)]

            processes = ProcessIds()
            if self.kernel.QueryInformationJobObject(
                self.handle, 3, ctypes.byref(processes), ctypes.sizeof(processes), None,
            ):
                return list(processes.ids[:processes.listed])
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            if time.monotonic() >= deadline:
                raise ProcessCleanupError("Could not enumerate the Windows test job before termination.")
            capacity = max(capacity * 2, processes.assigned)

    def drain(self) -> None:
        deadline = time.monotonic() + CLEANUP_SECONDS
        processes = []
        try:
            for pid in self._process_ids(deadline):
                # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
                process = self.kernel.OpenProcess(0x101000, False, pid)
                if not process:
                    error = ctypes.get_last_error()
                    if error == 87:  # The process exited before OpenProcess.
                        continue
                    raise ctypes.WinError(error)
                processes.append(process)
                member = wintypes.BOOL()
                if not self.kernel.IsProcessInJob(process, self.handle, ctypes.byref(member)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not member.value:
                    self.kernel.CloseHandle(processes.pop())  # A recycled PID outside this job.

            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            self._wait_until_empty(deadline)
            # Job accounting can reach zero before the process objects are signaled.
            for process in processes:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if self.kernel.WaitForSingleObject(process, remaining_ms) != 0:
                    raise ProcessCleanupError("A Windows test process did not signal termination.")
        finally:
            for process in processes:
                self.kernel.CloseHandle(process)

    def _wait_until_empty(self, deadline: float) -> None:
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


def _drain_linux_children(children: Path) -> None:
    deadline = time.monotonic() + CLEANUP_SECONDS
    while True:
        try:
            reaped, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return  # No direct children means no descendants left to adopt.
        if reaped:
            continue
        if time.monotonic() >= deadline:
            raise ProcessCleanupError("Linux test supervisor still has active children.")
        # Only signal our own unreaped children. Their PIDs cannot have been reused,
        # even if they exited between reading the file and sending the signal.
        for token in children.read_text(encoding="ascii").split():
            try:
                os.kill(int(token), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.02)


def _linux_supervisor(command: list[str]) -> int:
    cancelled = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, cancel)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        children.read_text(encoding="ascii")  # Verify kernel support before starting pytest.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER.
            raise OSError(ctypes.get_errno(), "Could not enable Linux test descendant ownership")
        if cancelled:
            return 0
        child = subprocess.Popen(command)
        try:
            while not cancelled:
                try:
                    status = child.wait(timeout=0.1)
                    return status if status >= 0 else 128 - status
                except subprocess.TimeoutExpired:
                    pass
            return 0
        finally:
            _drain_linux_children(children)
            child.poll()
    except (OSError, ProcessCleanupError) as error:
        print(f"Linux test supervisor: {error}", file=sys.stderr)
        return SUPERVISOR_FAILED


def _cancel_linux_supervisor(supervisor: subprocess.Popen) -> None:
    try:
        supervisor.terminate()
        status = supervisor.wait(timeout=CLEANUP_SECONDS + 1)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProcessCleanupError("Linux test supervisor could not confirm cleanup.") from error
    if status == SUPERVISOR_FAILED or status < 0:
        raise ProcessCleanupError("Linux test supervisor did not verify removal after cancellation.")


def _run_linux(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> int:
    cancelled = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    # The supervisor is in a separate session. Keep control here through cleanup
    # so the caller can release its lease once the supervisor has reaped its tree.
    previous_handler = signal.signal(signal.SIGTERM, cancel)
    try:
        supervisor = subprocess.Popen(
            [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--linux-supervisor", json.dumps(command)],
            cwd=cwd, env=env, start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while not cancelled:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    status = supervisor.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if status == SUPERVISOR_FAILED or status < 0:
                    raise ProcessCleanupError("Linux test supervisor did not verify completion.")
                return 128 + signal.SIGTERM if cancelled else status
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            _cancel_linux_supervisor(supervisor)
            raise
        _cancel_linux_supervisor(supervisor)
        return 128 + signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def run_owned(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> int:
    if os.name == "nt":
        return _run_windows(command, cwd=cwd, env=env, timeout=timeout)
    if sys.platform == "linux":
        return _run_linux(command, cwd=cwd, env=env, timeout=timeout)
    raise OSError(f"Managed CUDA test containment is unavailable on {sys.platform}.")


if __name__ == "__main__":
    if sys.argv[1] == "--linux-supervisor":
        raise SystemExit(_linux_supervisor(json.loads(sys.argv[2])))
    # This child cannot launch pytest until its parent puts it in the non-breakaway job.
    if sys.stdin.buffer.readline() != b"GO\n":
        raise SystemExit(2)
    raise SystemExit(subprocess.call(json.loads(sys.argv[1]), stdin=subprocess.DEVNULL))
