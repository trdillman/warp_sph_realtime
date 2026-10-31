# WarpSPH shared-SDK adoption checkpoint

**31 October 2026. Real consumer implementation and native qualification, not another synthetic SDK demo.**

## What changed for the artist

WarpSPH now has an optional **Shared SDK Output (Experimental)** setting and a **Snapshot to Geometry Nodes** button. The existing renderer remains the default. With the exact private SDK installed in the qualified environment, the workflow is: enable the shared output while stopped, Start, Pause after simulation has advanced, request the native snapshot, then Stop to work with the saved result.

The new route uses the actual RealtimeWCSPHSolver and the existing modal operators. It publishes positions, solver velocity, last-evaluated solver density, artist radius/colour and stable particle IDs through the shared SDK. The snapshot is an ordinary Blender point mesh with named attributes and an editable GN instancer. Its ID filtering follows real solver identities through the solver's spatial sort, not array positions.

**The saved result is usable without installing WarpSPH, Warp or the SDK.** A fresh isolated Blender process with all three modules absent reopened all 100,000 particles with identical native array hashes and selected three real solver IDs through Geometry Nodes.

## Implementation boundaries

The consumer bridge is new; the physics is not. All **21 source files in the solver, scene-input and cache modules remain byte-for-byte unchanged** from consumer main 52a383a. This implementation adapts existing sorted nonnegative int32 solver IDs to uint64 losslessly; it does not expand the solver's ID range or substitute a different allocator/physics model.

Normal simulation publication remains GPU-only. CPU readback is explicit, bounded and requested by the artist for a selected completed frame. Native mesh/GN mutation occurs on the main timer boundary, not inside the draw callback. Position and radius respect the existing scene-unit convention; velocity retains metre-per-second meaning. Density is explicitly the last SPH density evaluation, not a fabricated new post-step measurement.

The exact SDK is admitted by all 33 payload hashes before activation. Missing or incompatible dependencies produce an actionable error; they do not silently switch to another renderer. SDK code is not copied into WarpSPH. The public SDK v2 gate remains unchanged: this is a deliberately pinned private consumer opt-in.

## Actual Blender/CUDA/Vulkan checks

Environment: Blender **5.2.0 LTS build fbe6228777e7**, Warp **1.17.0**, **Vulkan**, RTX **4080 SUPER**, cuda:0. The extension was installed through Blender's extension-install operator into a canonical isolated profile. All 55 installed extension files matched the exact ZIP and checkout bytes.

| Scenario | Evidence-backed result |
|---|---|
| Real100,000-particle solver | At least 24 steps through actual Start/Pause operators; the final run reached 25 before pausing. Native positions, velocity, density and stable ID words matched the actual solver arrays. Radius and colour matched artist controls. |
| GN from actual solver data | Selected three actual particle IDs, each yielding exactly12 sphere vertices at the matching particle centre and radius. This is representative evaluated-GN selection on the100,000-particle native snapshot, not a full numerical inspection of all1.2million generated vertices. |
| Snapshot state preservation | Correctness capture did not change the paused solver's arrays. |
| Reset and Step | Reset reused the compatible bridge only after producer completion, rebound the new real solver, advanced its epoch and remained paused. One Step advanced exactly once. |
| Authored collider A/B | A 288-particle column ran 300 steps with a plate and 300 without it at the same 0.7649463 simulated seconds. With the plate, the minimum signed distance stayed positive at 0.032414m and reached the contact zone; without it, particles crossed the former plate region to -0.099217m. |
| Authored emitter A/B | At 80 particles/s the 200-particle scene reached 208; at 160 particles/s it reached 216 over the same 40 steps / 0.10199284seconds. IDs remained unique and followed particles through sorting; the native snapshot exported 216 live particles, not the 1,200-capacity buffer. |
| Running extension disable | The100,000-particle SSFR session completed 2,686 solver steps and 2,685 real draws before disable. Transport, timer, draw handler and solver references closed cleanly. These counts are lifecycle evidence, not a frame-rate benchmark. |
| Running file load | A live POINTS session closed and released resources when a different saved file was opened. |
| Deliberately pending file load | The test queued 64 real solver advances into a bridge publication, then opened a file before the producer finished. One slot was genuinely pending before drain, both were complete afterward, and teardown succeeded. Observed drain was approximately 30 ms. No completion event or status was fabricated. |
| Fresh-process persistence | All100,000 native points and seven arrays were hash-identical after reopening with no WarpSPH/Warp/SDK available; exact GN selection still worked. |

## Defects found and repaired during adoption

**Cross-scene snapshot targeting.** A generic collection name could resolve a different scene's output collection. The SDK correctly rejected that unsafe target. The consumer now uses a persistent per-scene ownership UUID in the collection name and rejects foreign targets.

**Teardown with pending GPU work.** The pinned SDK's close method safely quarantines outstanding work rather than waiting. A real running-file-load test therefore exposed a missing consumer drain. The consumer now polls every recorded publisher slot before owner revocation, using the existing five-second shutdown budget. Pending leases/fences are not discarded; timeout, driver-query failure or renderer-cleanup failure remains sticky and retains ownership. The failed process was exited without clearing its fatal registry before testing a fresh corrected candidate.

**Test-oracle corrections.** The actual Stop operator schedules modal teardown on the next event. The initial acceptance script asserted close too early; it now waits for the actual timer removal. The original failed receipt and corrected successful run are retained.

## Host verification

The final source suite passed **636 tests and 72 subtests**, with 64 GPU-dependent checks skipped in the CPU-only run. Actual native checks above are separate evidence, not relabelled skips. Ruff passed and ordinary project Pyright reported zero errors and warnings without requiring the optional SDK in the developer environment.

New regressions cover dependency-byte mismatch, lazy loading, duplicate Start, GPU backpressure before solver mutation, current sorted solver-array binding, safe reset epochs, snapshot request/cancel/failure behavior, cross-scene target separation, all-slot drains, timeout preservation and cleanup exceptions.

An independent read-only reviewer was requested, but its separate connector returned 401/manual reauthentication required. This checkpoint has author self-review and executable evidence, not external review sign-off.

## What is deliberately not claimed

This is a useful **partial ADOPT-01 vertical slice**, not acceptance of the entire 34-slice expansion. Real-time performance and the full high-count animated-collider matrix remain unqualified. Arbitrary SDK SceneDelta input is reported REBUILD_REQUIRED; actual effects use WarpSPH's existing authoring/live-input/Reset workflow. No astrophysics consumer was changed.

**Cache baking through Shared SDK Output is explicitly blocked for now.** Stop and choose the existing output route to use the original bake workflow. The cache implementation and default behavior remain unchanged. A native snapshot is not an animation bake, simulation continuation cache, live zero-copy GN pipeline or Cycles/motion-blur qualification.

The supplied extension is an experimental developer candidate. Its SDK dependency is a specific private wheel. Do not replace a production installation or the original published SDK a2 package merely because the metadata retains the old version label.

## Files and identity

- Consumer branch: `interop/warpsph-adoption-20261031`.
- Consumer payload commit: `411ce892fc1713c287aedcd4feecc19fd6961ba2`.
- Consumer ZIP SHA256: `cd0b648ffb70d55cb50c73122338d850525c388875a9c86f75ade5991a72096a`.
- Pinned SDK wheel SHA256: `24a4e65a290a4f95e9cba7ee54e31f3cecd066c17e2897c8e6cf4ddf8e7e7bda`.
- Worktree: `C:\Users\Tyler\projects\warpsph-sdk-adoption-20261031`.
- Evidence: `evidence/sdk-adoption-20261031/`, including failed attempts and final candidate-specific receipts.
- Openable native scene: `C:\Users\Tyler\Downloads\WarpSPH-Actual-Solver-GN-Snapshot-2026-10-31.blend`.

All four owned Blender processes exited, port 19977 is free and the shared GPU lease was released. Main, production preferences, other worktrees and unrelated jobs were preserved. No merge, push or production installation occurred. The Pages tracker could not be reached through an available provider; its pending update is recorded locally rather than reported as synchronized.
