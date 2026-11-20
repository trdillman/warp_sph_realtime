"""Regression coverage for broker entry and exit around headless pytest."""
import ctypes
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts.cuda_test_process import ProcessCleanupError, run_owned
from scripts.run_cuda_tests import REPO, main


@pytest.fixture(autouse=True)
def clear_inherited_cuda_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.mark.parametrize("mask", ["", "-1", " -1,0"])
@patch("scripts.run_cuda_tests.subprocess.run")
def test_cpu_mask_refuses_before_lease(run, mask, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    assert main([]) == 2
    run.assert_not_called()
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


@patch("scripts.run_cuda_tests.subprocess.run")
def test_broker_refusal_prevents_test_launch(run):
    run.return_value = subprocess.CompletedProcess([], 75, stdout="GPU reserved\n")

    assert main(["--no-wait", "-q"]) == 75

    run.assert_called_once()
    assert run.call_args.args[0][-1] == "--no-wait"


@patch("scripts.run_cuda_tests.run_owned")
@patch("scripts.run_cuda_tests.subprocess.run")
def test_leased_pytest_inherits_scope_and_releases_after_failure(run, run_tests, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-explicit-device")
    run.side_effect = [
        subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "id": "lease-one", "state": "active", "cls": "shared", "scope_pid": os.getpid(),
        })),
        subprocess.CompletedProcess([], 0),
    ]
    run_tests.return_value = 1

    assert main(["-q", "-k", "graph"]) == 1

    acquire, release = run.call_args_list
    tests = run_tests.call_args
    command = acquire.args[0]
    for flag in ("--holder-pid", "--scope-pid"):
        assert command[command.index(flag) + 1] == str(os.getpid())
    assert tests.args[0][1:] == ["-m", "pytest", "-q", "-k", "graph"]
    assert tests.kwargs["cwd"] == REPO
    assert tests.kwargs["env"]["WARPSPH_TEST_CUDA"] == "1"
    assert tests.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-explicit-device"
    assert release.args[0] == ["bgate", "release", "lease-one"]


@patch("scripts.run_cuda_tests.subprocess.run")
def test_foreign_scope_prevents_test_launch_and_is_preserved(run):
    run.return_value = subprocess.CompletedProcess([], 0, stdout=json.dumps({
        "id": "another-job", "state": "active", "cls": "shared", "scope_pid": os.getpid() + 1,
    }))

    assert main([]) == 2

    run.assert_called_once()


@pytest.mark.parametrize(
    ("failure", "release_failure", "expected", "released"),
    [
        (subprocess.TimeoutExpired("pytest", 1), False, 124, True),
        (ProcessCleanupError("still active"), False, 1, False),
        (None, True, 1, True),
        (subprocess.TimeoutExpired("pytest", 1), True, 124, True),
    ],
)
@patch("scripts.run_cuda_tests.run_owned")
@patch("scripts.run_cuda_tests.subprocess.run")
def test_cleanup_and_release_failures(run, run_tests, failure, release_failure, expected, released, capsys):
    grant = subprocess.CompletedProcess([], 0, stdout=json.dumps({
        "id": "lease-one", "state": "active", "cls": "shared", "scope_pid": os.getpid(),
    }))
    run.side_effect = [grant, FileNotFoundError("bgate") if release_failure else subprocess.CompletedProcess([], 0)]
    run_tests.side_effect = failure
    run_tests.return_value = 0

    assert main([]) == expected
    assert run.call_count == (2 if released else 1)
    assert capsys.readouterr().err


@pytest.mark.skipif(os.name != "nt", reason="requires Windows Job Objects")
@pytest.mark.parametrize("timeout", [False, True], ids=["parent exits", "parent times out"])
def test_windows_owned_tree_stops_with_live_descendant(tmp_path, timeout):
    marker = tmp_path / "descendant.txt"
    sleeper = "import time; time.sleep(90)"
    source = (
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[2]])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "if sys.argv[3] == 'wait': child.wait()\n"
    )
    command = [sys.executable, "-c", source, str(marker), sleeper, "wait" if timeout else "exit"]
    outsider = subprocess.Popen([sys.executable, "-c", sleeper])
    try:
        if timeout:
            with pytest.raises(subprocess.TimeoutExpired):
                run_owned(command, cwd=tmp_path, env=dict(os.environ), timeout=2.0)
        else:
            assert run_owned(command, cwd=tmp_path, env=dict(os.environ), timeout=5.0) == 0
        assert outsider.poll() is None
        pid = int(marker.read_text())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            assert ctypes.get_last_error() == 87
        else:
            try:
                assert kernel.WaitForSingleObject(handle, 0) == 0
            finally:
                kernel.CloseHandle(handle)
    finally:
        outsider.terminate()
        outsider.wait(timeout=5)
