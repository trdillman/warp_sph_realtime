"""Regression coverage for broker entry and exit around headless pytest."""
import json
import os
import subprocess
from unittest.mock import patch

from scripts.run_cuda_tests import REPO, main


@patch("scripts.run_cuda_tests.subprocess.run")
def test_broker_refusal_prevents_test_launch(run):
    run.return_value = subprocess.CompletedProcess([], 75, stdout="GPU reserved\n")

    assert main(["--no-wait", "-q"]) == 75

    run.assert_called_once()
    assert run.call_args.args[0][-1] == "--no-wait"


@patch("scripts.run_cuda_tests.subprocess.run")
def test_leased_pytest_inherits_scope_and_releases_after_failure(run):
    run.side_effect = [
        subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "id": "lease-one", "state": "active", "cls": "shared", "scope_pid": os.getpid(),
        })),
        subprocess.CompletedProcess([], 1),
        subprocess.CompletedProcess([], 0),
    ]

    assert main(["-q", "-k", "graph"]) == 1

    acquire, pytest, release = run.call_args_list
    command = acquire.args[0]
    for flag in ("--holder-pid", "--scope-pid"):
        assert command[command.index(flag) + 1] == str(os.getpid())
    assert pytest.args[0][1:] == ["-m", "pytest", "-q", "-k", "graph"]
    assert pytest.kwargs["cwd"] == REPO
    assert pytest.kwargs["env"]["WARPSPH_TEST_CUDA"] == "1"
    assert release.args[0] == ["bgate", "release", "lease-one"]


@patch("scripts.run_cuda_tests.subprocess.run")
def test_foreign_scope_prevents_test_launch_and_is_preserved(run):
    run.return_value = subprocess.CompletedProcess([], 0, stdout=json.dumps({
        "id": "another-job", "state": "active", "cls": "shared", "scope_pid": os.getpid() + 1,
    }))

    assert main([]) == 2

    run.assert_called_once()
