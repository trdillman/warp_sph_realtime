# PR31 review repairs

Codex review found two valid boundaries on original head2ad949f:

- Scene FPS is a reset-only edit. Timeline now keeps the accepted scene FPS
  captured at rebuild, aligning its frame target with animated collider sampling.
  Reset accepts the new rate for both clocks.
- A Single Step clamped to an already-completed end frame now clears step_once,
  preventing a stale request from blocking native animation bake admission.

CodeRabbit additionally found repeated failed rewind/post-bake rebuilds on the
modal timer. A rejected replay now remains required but is held without repeated
rebuilds/reports; playback pauses and the panel requests scene correction plus
explicit Reset. A successful Reset clears the held failure and replay state.
Regression verifies one attempt over30ticks, retained replay requirement, paused
playback and recovery after explicit Reset.

Independent follow-up caught the real demo solver's legacy hardcoded fps=24.
The accepted clock now comes from scene render FPS at rebuild for both scene
types. Custom solver construction accepts that same rate. The regression sets
the fake solver fps=24 while the scene uses25 then50, verifying the real boundary.

The actual SimSession subprocess regression reproduces premature FPS retiming
before the fix, then verifies the old FPS until Reset, the new FPS afterward,
and end-frame request cleanup. Final affected suite21 tests/2 subtests passed;
Ruff passed. Exact-head CI is required before merge.

Installed GPU receipts remain bound to sourcef56999f/package30ae8c23. These review
repairs have focused source evidence; no new GPU/package acceptance is inferred.
