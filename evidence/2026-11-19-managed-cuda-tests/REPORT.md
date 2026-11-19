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
- Ruff, targeted Pyright, and the staged whitespace check passed.

A first validation attempt used an empty CUDA mask, which Windows removed from
the native process environment. The corrected mask and a fresh test run
supersede that attempt. A granted live CUDA suite remains pending a free GPU
window.
