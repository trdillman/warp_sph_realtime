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

A first validation attempt used an empty CUDA mask, which Windows removed from
the native process environment. The corrected mask and a fresh test run
supersede that attempt.
