# PR26 qualification follow-up — 25 October 2026

## Candidate and scope

Main and PR base: `82cf500231a2065b8cf3b4a3e26062aaa20acdd4`.
Reviewed original PR head: `890ddc897c7f60d7ed9c0e2d4bf477cc726d973e`.
The follow-up source is bound by `source-sha256.json` and the containing commit.
Work used a fresh Linux clone and isolated qualification branch. No Windows
checkout, installed extension, user profile, or separate cache-provenance lane
was modified. No solver kernels or transport adapters changed.

## Reproduced source defects and repairs

1. **Paused Reset under backpressure:** output reuse ignored a skipped publication.
   The replacement solver committed, but paused ticks never published its initial
   state. A pending publication now retries before the pause/step path; it does not
   advance physics. Existing session, timer, renderer and pause state are retained.
2. **Undo/redo geometry false negatives:** clearing pending geometry flags and
   suppressing undo updates hid interior edits with unchanged mesh identity,
   counts and bounds. Geometry uncertainty now survives until Reset, and undo
   geometry flags are consumed. This intentionally permits conservative Reset
   hints for unchanged geometry; the manual explains the tradeoff.
3. **Failed Reset timeline drift:** the scene rewound before validation, then a
   rejected rebuild kept the old solver at the wrong visible frame. Rejected
   Reset now restores the prior scene frame inside the update-muting scope.

Each regression was executed and observed failing before its production repair.
`reset-red.log` and `undo-frame-red.log` retain those failures. The existing
subprocess lifecycle fixture exercises real SimSession and TransportSession
control flow with fake Blender, solver and graphics dependencies. These are
source/lifecycle tests, not GUI or GPU evidence.

## Executed verification

Linux x86_64, Python 3.12.14, Warp 1.17.0. This local Python differs from Blender's
3.13; exact-head hosted Linux/Windows CI must separately confirm that lane.

| Check | Result | Evidence |
| --- | --- | --- |
| Focused existing suite | VERIFIED: 58 passed, 13 subtests | focused-green.log |
| Full `python -m pytest -q --junitxml=.../suite.xml` | VERIFIED: 556 passed, 65 skipped, 50 subtests; 33.54 s | suite-final.log, suite.xml |
| `python -m ruff check .` | VERIFIED: passed | ruff.log |
| `python -m pyright` (1.1.408) | VERIFIED: 0 errors/warnings | pyright.log |
| `git diff --check` | VERIFIED: passed | command executed before publication |
| Independent source review | Three confirmed findings addressed; focused suite independently passed | REVIEW.md |

The first full-suite attempt began before the final repairs; its partial log is
retained as `suite-baseline.log` and is not the final-candidate receipt. Skipped
tests do not establish GPU, Blender or platform qualification.

## Runtime acceptance remains blocked

The tunnel diagnostic returned `UNAVAILABLE / Connection timed out`; a subsequent
read-only command returned `Session terminated`. The alternate local gateway's
read-only process command returned an internal error. These are observed tool
failures, not a claim that the desktop is physically disconnected. No Blender
executable or NVIDIA runtime was available in the Linux execution environment.
No DCC launch, package installation on the desktop, or GPU workload occurred.

| Required acceptance | Status |
| --- | --- |
| Blender undo identity, role/delete/transform undo/redo (R1) | BLOCKED |
| Modal edit tick continuity and frame_set interaction (R2/R3) | BLOCKED |
| CUDA next-step gravity acceleration; one recapture; eager/graph equivalence | BLOCKED |
| 1M Reset <=2 s and faster than paired Stop+Start (R4) | BLOCKED; no latency measured |
| OpenGL and Vulkan, 100k/1M editing >=90% of untouched steps/s | BLOCKED; no throughput measured |
| Change handling <=1 ms/tick for 32 objects | BLOCKED; no runtime measurement |
| Real viewport rendering, session preservation and failure cleanup | BLOCKED |
| Save/reopen and bake integration | BLOCKED |
| Interaction screen capture | BLOCKED |

PR26 must remain a draft until these mandatory runtime gates pass. Source review
does not satisfy Phase 1 product acceptance. Live geometry/collider transforms,
emitters/domain editing, recording and rewind remain later phases. The next step
is the existing isolated Blender harness, one Blender process at a time, through
the actual installed candidate on both graphics backends. Preserve the separate
cache-provenance candidate and qualify its eventual integration independently.
