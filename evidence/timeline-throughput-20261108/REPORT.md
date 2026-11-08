# Timeline throughput repair — 8 November 2026

The avoidable per-physics-step publication bottleneck is fixed. Installed native
single-frame completion fell from median3.41295s to0.48866s across16 measured reset
trials per arm (6.984x ratio of medians). Final solver positions and stable IDs
match byte-for-byte for every warmup and measured trial. This is a bounded
development comparison, not a counterbalanced hardware or24fps certificate.

## Cause and change

Merged main01d034e calls the unchanged solver once per admitted output frame.
At the demo's dt0.001648026890777408 and2substeps, each solver call advances only
0.003296s. One24fps timeline frame needs13calls. Shared SDK snapshots block the
next admission until capture/commit completes, so those13small steps each incurred
output scheduling/readback/native mesh work. The six-second intrusive baseline
profile recorded26solver calls and26native commits; native-output service took
most measured add-on CPU time. Cumulative profile metrics also contain earlier
session history; they are not clean six-second deltas or acceptance timings.

Source commit fda42d4d7e47c147ab6e5cb091528fa5387201e7 batches unchanged solver
calls after transport admission, publishes their final state once, and yields
after at most16steps, one timeline frame plus <one-step rounding, or8ms of host
submission time checked between steps. The8ms cap is not a GPU-duration guarantee.
Rejected admission does no physics. Animated collider sampling occurs before
every step and restores the requested playhead. No kernel, timestep, substeps,
particle count, quality, dependencies, native validation or memory guards changed.

Actual physics counters/elapsed and step-interval telemetry use the number of
steps taken. An adapter deferring publication after admitted compute causes a
copy-only output retry, without repeating physics. Native/legacy bake ownership,
pause cancellation, reset/replay failure holds and unbound legacy behavior remain.
No intermediate physics steps are skipped. Pause processes before the next batch;
already submitted work can retire at the boundary.

## Exact installed comparison

- Baseline product main01d034e, package
  3db5a2a1013064e26e5df32610cdad15bda01aa036ab75027c832ce6e13af44b,
  owned PID48636. Candidate package
  785c5248c7ec5f2cc194633c1c97a9ebf330577b6f3c3fadcdedb192b75c4717,
  fresh owned PID54788, port9877, session784bcc9b57e34c7e8fdaef71e83bf108.
- All60 installed files matched the exact candidate archive at installation and
  final handoff. Module bl_ext.demo.warp_sph_realtime is enabled/in preferences.
- Blender5.2.0 LTS fbe6228777e7, Vulkan, RTX4080SUPER cuda:0, Warp1.17.0,
  pinned private SDK0.1.0a2. Double Dam + Pillars,100Kparticles, WATER,24fps,
  CUDA-graph setting and live GN12Hz unchanged. No private SDK is redistributed.
- Public Reset/frame_offset(delta=1), initial output settled, then measure until
  solver reaches1/24s and the latest native snapshot has committed. One warmup
  and16recorded trials per arm. Diagnostic profile excluded. Correctness readback
  occurs after timing, not per step. Exact final result:13steps,0.04284869916s.
- Baseline range2.46166–3.71736s; candidate0.42882–0.76958s. All trials retained;
  no exclusions/retries. Baseline adjacent A/A ratios0.944–1.231 expose noise.
  Sequential processes/reset trials are not independent fresh-process pairs;
  there is no general speedup confidence or all-workloads claim.
- `CONTRACT.md`, `baseline-trials.json`, `candidate-trials.json`, and
  `comparison.json` preserve the contract/raw results. `analyze_comparison.py`
  rejects wrong positions, changed substeps, missing trials, stale package IDs
  and failed attempts. These are receipt-validator negative controls, not injected
  runtime GPU faults. Benchmark runner is scripts/benchmark_timeline_frame.py.

## Controls, tests and surviving demo

`runtime.json` records all public workflow checks PASS in the candidate process:
native Play, Pause while behind, Jump to Start, frame offset and stationary held
single step; add-on Step, Pause/Resume, Reset, Viscous/Water edits, Stop/Start.
Native Pause held52steps/time0.17139479664 and identical positions across1.2sec.
Add-on Pause held71steps/time0.23401981849 and identical positions across1.2sec.
Both normal single steps still produced exactly the expected13-step frame result.

An actual UI click of add-on Pause also froze frame1552,10421steps,
time34.34817645758 and all positions across two samples1.3sec apart
(`pause-button-click.json`). The first zero-duration synthetic mouse event did
not activate the button; that failed attempt is retained in pause-click-attempt.json.
A normal held click succeeded. An initial setup attempt to assign Blender's
read-only active_panel_category failed; the visible tab was selected with a mouse.
No product error or CUDA recovery was needed for these tooling mistakes.

Final source tests:741passed,2skipped,76subtests. Ruff and Pyright pass. The
focused lifecycle/batching suite passed21tests+2subtests; independent review PASS
after fixing the post-production publication-deferral accounting case. Regressions
cover caps, target rounding, collider sampling/restoration, backpressure, deferred
output, partial-batch fatal handling and the prior timeline controls.

Owned baseline session was stopped/closed before candidate measurements. Final
status/file_state/scene_info/execute_code gate and visible-window identity are
preserved. `handoff.json`, `demo.png`, candidate-launch.json, installed.json,
the exact archive and saved WarpSPH-Live-Merged-Demo.blend identify the delivered
demo. It remains open/running with the editable WarpSPH panel visible.

## Remaining scope and delivery

The100Klive-GN scene still runs below wall-clock timeline speed (the visible
step-based speed estimate was about0.14–0.15x during this run). Full native mesh
conversion, Geometry Nodes evaluation/rendering and output ownership/scheduling
remain substantial costs. This fix removes the unnecessary output frequency;
it does not establish real-time performance or conceal the remaining lag by
moving the playhead, changing physics/quality, dropping steps or disabling live GN.
Long-run/platform/device qualification remains open. No public release/tag or
regular-profile installation. Source fix is committed locally; not pushed/merged.
