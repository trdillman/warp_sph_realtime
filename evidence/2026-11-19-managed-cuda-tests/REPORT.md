# Headless CUDA test admission

Plain pytest now sets a CPU-only device mask before test collection. The
headless CUDA launcher acquires a shared blender-gate lease scoped to its own
process, starts pytest after confirming that scope, and releases its lease when
pytest exits. CUDA test cases remain available on dedicated GPU hosts by opt-in.

## Verification

- On Windows, the native environment retained the corrected `-1` mask after
  pytest configuration, and Warp 1.17.0 reported no visible CUDA device.
- The full Windows host suite passed with Python 3.11.9 and Warp 1.17.0.
  Pytest reported 723 passed, 64 skipped, and 72 passing subtests.
- An actual broker refusal while an exclusive job was running returned exit
  75 before pytest started. Tests also cover rejecting a foreign scope and
  releasing the owned lease after a failing pytest run.
- The documented launcher passed the full leased headless suite at
  `de5e1ef247e540b8b6c8045da11b6f52aecfb6e9` with Windows Python 3.11.9,
  Warp 1.17.0, and the pinned SDK. Pytest reported 785 passed, 2 skipped, and
  76 passing subtests. The launcher exited 0 and released shared lease
  `bb7b1a06253f`. The two skipped tests require an isolated Blender 5.2 install
  or a certified OpenGL wheel. The shared lease did not collect telemetry.
- Ruff, targeted Pyright, and the staged whitespace check passed.

## Installed Blender checks

Both checks skipped by the headless suite passed separately at
`32041e8d15e6376c7b4b8e675cbefd7a6e84e20b` under Blender 5.2.0 LTS
build `fbe6228777e7`, in fresh disposable profiles. The registration check
exercised building, installing, enabling, offline refusal and disabling. The
CUDA/OpenGL check used the SHA-locked Warp 1.17.0 wheel, ran the real WCSPH
solver with and without CUDA graph replay, and mapped, copied, unmapped and
destroyed a CUDA/OpenGL buffer. Its build, install and runtime leases
`da49ff9bc4c1`, `d0383b39f2ba` and `4fc6fa9a8998` were released.
The fixture substitutes a local wheel transfer for the network download.

The default host installation contained a foreign Warp 1.18.0 copy, which
WarpSPH flagged. These checks used a clean portable copy with the same
Blender executable SHA-256. The host copy was preserved.

A first validation attempt used an empty CUDA mask, which Windows removed from
the native process environment. The corrected mask and a fresh test run
supersede that attempt.
