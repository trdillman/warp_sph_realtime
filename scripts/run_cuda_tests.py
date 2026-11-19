"""Run headless CUDA tests within a shared blender-gate lease."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 3600
TEST_TIMEOUT_SECONDS = LEASE_SECONDS - 60


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wait", action="store_true", help="fail if a shared lease cannot start immediately")
    options, pytest_args = parser.parse_known_args(argv)

    pid = str(os.getpid())
    acquire = [
        "bgate", "acquire", "--class", "shared", "--label", "warpsph-headless-cuda-tests",
        "--holder-pid", pid, "--scope-pid", pid,
        "--ttl", str(LEASE_SECONDS), "--max-seconds", str(LEASE_SECONDS),
    ]
    if options.no_wait:
        acquire.append("--no-wait")

    try:
        grant = subprocess.run(acquire, check=False, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("CUDA tests require blender-gate (bgate on PATH).", file=sys.stderr)
        return 2
    if grant.returncode:
        print(grant.stdout, end="", file=sys.stderr)
        return grant.returncode
    try:
        lease = json.loads(grant.stdout)
    except ValueError:
        lease = None
    if not (
        isinstance(lease, dict)
        and isinstance(lease.get("id"), str)
        and lease["id"]
        and lease.get("state") == "active"
        and lease.get("cls") == "shared"
        and lease.get("scope_pid") == os.getpid()
    ):
        print("blender-gate did not return an active shared lease for the test runner.", file=sys.stderr)
        return 2
    lease_id = lease["id"]

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *pytest_args],
            cwd=REPO,
            env=dict(os.environ, WARPSPH_TEST_CUDA="1"),
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"CUDA tests exceeded their {TEST_TIMEOUT_SECONDS}s runtime limit.", file=sys.stderr)
        return 124
    finally:
        release = subprocess.run(["bgate", "release", lease_id], check=False)

    return result.returncode or release.returncode


if __name__ == "__main__":
    raise SystemExit(main())
