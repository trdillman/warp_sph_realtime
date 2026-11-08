# Independent source review

Reviewer: existing /root/timeline_review, read-only, 8 November 2026.

Plan review required per-step collider sampling, actual-step accounting, capped
host submission and step count, target/frame duration limit, pause/bake ownership
and truthful telemetry. These constraints were implemented and tested.

First diff review found post-admission publication may return false after the
producer executes. Fixed by accounting actual taken steps regardless of output
status and retrying copy_positions under _output_pending without repeated physics.
Regression covers both pre-admission rejection and post-production deferral.

Followup verdict PASS: no further blocking findings in bounded batching diff.
Reviewer independently ran focused checks (22 tests,2subtests reported).
Parent focused command tests/test_timeline.py tests/test_transport_session.py:
21 passed,2subtests. Source review preceded candidate package/live GPU comparison.

Final saved-proof audit: PASS. Reviewer recomputed both medians and verified all
17rows/arm,13steps,100Kparticles,matching dt/substeps and exact positions/IDs.
Candidate archive matches source; native/add-on/UI-click pause receipts support
the control claims. Scope excludes real-time/platform/release acceptance. Minor
planned-paused vs actual-closed baseline chronology was clarified as an explicit
execution deviation in CONTRACT.md. No further runtime or source changes.
