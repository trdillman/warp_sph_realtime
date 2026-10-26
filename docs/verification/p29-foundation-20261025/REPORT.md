# P2.9 foundation draft verification

Base: `82cf500231a2065b8cf3b4a3e26062aaa20acdd4` (main).

This draft publishes the existing Phase 1 foundation snapshot. The 13 files in
the original foundation-import manifest were verified byte-identical in the
source and imported checkouts before copying. Both existing checkouts and their
indexes were preserved. This is an isolated publication lane, not a transfer of
ownership of either original checkout. The separate cache-provenance work is
excluded, including its independent changes to sim_session.py.

Reviewed scope: shared edit classification and panel hints; Reset and compatible
output reuse; live gravity with one graph recapture; undo/redo scene and object
resolution; animated-collider rename/deletion safety. Collider movement, shape,
set changes, emitters/domain live edits, recording and rewind remain later phases.

Publication review removed the unmeasured claim that Reset is faster than Stop
and Start from the operator tooltip and manual. No performance result is claimed.

## Local verification

Python 3.11. All checks use existing dependencies, one CPU affinity, one worker,
a 1 GiB Windows Job memory limit and a 120-second per-command timeout.

- `python -m pytest tests/test_live_edits.py tests/test_authoring_ui.py tests/test_blender_scene.py tests/test_transport_session.py -q`:
  56 passed, 13 subtests passed, 6.70 seconds. Includes mocked lifecycle controls.
- `ruff check .`: passed.
- `git diff --check`: passed.
- `pyright`: 0 errors, 0 warnings, 0 informations.
- Exact-head hosted CI is recorded in the PR.

The full local solver suite was not run under the no-GPU assignment. Hosted CI
provides its own CPU-only full suite and Python 3.13 checks.

## Acceptance gates still open

This is a source-review draft, not completed Phase 1 product acceptance.

- R1: real Blender session_uid behavior across undo/redo, transforms, roles and deletion.
- R2/R3: modal editing tick continuity and frame_set interaction on both backends.
- R4: 1M Reset <=2 seconds and faster than Stop + Start on the same scene.
- Real CUDA gravity acceleration and eager/graph equivalence.
- OpenGL/Vulkan at 100k and 1M: editing >=90% of baseline steps/s, change handling
  <=1 ms/tick for 32 objects, and captured interaction evidence.
- Bake, save/reopen, and GPU lifecycle qualification against the final candidate.

No GPU work, DCC launch, package installation, merge or release was performed.
