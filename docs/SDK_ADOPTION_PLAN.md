# ADOPT-01: WarpSPH shared rich-frame output and native snapshots

Baseline consumer52a383a (fresh GitHub main verified), SDK0eee721/reviewed wheel24a4e65a. Original34-slice plan and ADOPT-01 remain authoritative. Existing solver, int32 stable IDs, scene interaction, cache provenance and physics are preserved. Astro adoption is not undertaken while the relevant project is held.

Architecture: one optional consumer-local InteropBridge and GPU pack kernel adapt the real RealtimeWCSPHSolver output to the exact installed SDK. No copied SDK code. A render-session-compatible wrapper uses existing POINTS/SPHERES/SSFR renderer inside SDK draw leases, so real GPU completion is tracked. Position scratch retains existing scene-unit convention; velocity is m/s; density is solver density; IDs widen losslessly from existing nonnegative int32 to uint64. No identity-by-array-index. One explicit experimental setting selects this route at Start; unavailable SDK never silently substitutes a different transport.

Artist workflow: Pause running solver; request Snapshot to Geometry Nodes. A draw-context readback pins the exact most recently published frame and returns an immutable CPU payload. Native conversion runs at the timer boundary outside draw callbacks. Updates replace only this scene's owned SDK snapshot, retain GN modifiers, and reject transformed/foreign targets. File load/Stop/unregister cancel pending work and close the SDK owner; sticky failure propagates to both consumer and SDK state. Ordinary solver tick has no host readback.

Reset: compatible capacity reuse requires prior producer completion, then rebind solver with a new epoch. Otherwise construct a new transport transactionally. Existing gravity/physics edits and animated collider updates remain at original solver boundaries and record applied-input sequence. Arbitrary SDK scene deltas that require rebuild report REBUILD_REQUIRED, not fictitious physics application. Baking continues through existing solver/cache path; partial adoption acceptance is explicit until all full ADOPT/performance requirements pass.

Tasks:
- [ ] Host RED tests: lazy SDK import, exact payload mismatch, aligned stable IDs after reordering, backpressure prevents step, duplicate Start, reset epoch, source release, snapshot staging/rollback/close and explicit unsupported scene edits.
- [ ] Implement pinned dependency loader, Warp GPU adapter, InteropBridge, narrow SimSession integration and optional UI. Keep sim/ and cache/ source hashes unchanged.
- [ ] Build exact extension; install wheel/extension only in canonical isolated Blender5.2 fbe6228777e7+Warp1.17 CUDA0/Vulkan; one shared GPU lease.
- [ ] Actual100000-particle solver run, snapshot attrs/GN compared to actual solver arrays by pid, reset/pausing/step, two process reopen, stop/disable/load cleanup. Include actual collider/emitter A/B where current authoring supports it; never use gravity-only motion as collision evidence.
- [ ] Record wall timing separately from correctness readbacks; no24FPS claim without measured UI cadence. Save source, installed identity, scene and raw receipts; commit local checkpoint. No shared-main merge, push, production installation or public-v2 activation.

Tracker: authenticated Pages provider was not found via plugin discovery. Fallback snapshot sequence65 fetched2026-10-31T19:16:59.779Z read; pending update will be queued, not reported as live synchronization. ctx gateway tools were not exposed; short derived output and explicit edit reads used.
