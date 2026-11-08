# PR31 review repairs

Codex review found two valid boundaries on original head2ad949f:

- Scene FPS is a reset-only edit. Timeline now keeps the accepted solver FPS
  captured at rebuild, aligning its frame target with animated collider sampling.
  Reset accepts the new rate for both clocks.
- A Single Step clamped to an already-completed end frame now clears step_once,
  preventing a stale request from blocking native animation bake admission.

The actual SimSession subprocess regression reproduces premature FPS retiming
before the fix, then verifies the old FPS until Reset, the new FPS afterward,
and end-frame request cleanup. Final affected suite21 tests/2 subtests passed;
Ruff passed. Exact-head CI is required before merge.

Installed GPU receipts remain bound to sourcef56999f/package30ae8c23. These review
repairs have focused source evidence; no new GPU/package acceptance is inferred.
