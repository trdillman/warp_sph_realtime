# WarpSPH native snapshot adoption implementation plan

Goal: an explicit Snapshot Paused Simulation to Geometry Nodes action over the real current WarpSPH solver, using one exact shared SDK artifact. This is a useful first ADOPT-01 sub-slice, not completion of the whole live-frame bridge, collider A/B, performance or bake/export contract.

Architecture: preserve the solver, existing live transport and cache pipeline. A lazy consumer module validates that the actual session is paused and not baking, captures only the active prefix of authoritative position/velocity/pid/density arrays with an explicit bounded readback, and passes one immutable typed payload to the existing SDK SnapshotService/Geometry Nodes consumer. A thin UI operator creates or explicitly updates only an SDK-owned snapshot. No physics kernels or sim/sim_core.py changes.

Pinned consumer base:52a383aa45ba14acfba0e5c2c7c86a91f1498181. SDK base:0eee7214bab679d0033c15b21a9e3490a6ffd1d4; wheel24a4e65a290a4f95e9cba7ee54e31f3cecd066c17e2897c8e6cf4ddf8e7e7bda. Qualified environment:Windows Blender5.2.0LTS fbe6228777e7/Warp1.17.0/RTX4080SUPER cuda:0. Canonical isolated harness and one shared GPU lease; no input/focus takeover.

Ruling: start with paused native-output adoption rather than replacing the existing interactive renderer. It delivers actual solver data in GN while keeping zero-readback live behavior and approved physics unchanged. Full ADOPT-01 remains open and separately reported. Existing cache output is preserved, not reimplemented.

1. Consumer contract and identity (TDD)
Files:warp_sph_realtime/sdk_snapshot.py, warp_sph_realtime/sdk_snapshot.lock.json, tests/test_sdk_snapshot.py.
- [ ] Fail tests for lazy imports, unpaused/baking/foreign scene/stale solver rejection, active prefix capture, exact int32->uint64 numerical promotion, zero/negative/overflow/duplicate IDs, coherent stamp/units/metadata, byte limits before readback, wrong SDK payload identity, unchanged solver state on success/failure.
- [ ] Implement explicit paused capture with full package hash validation before SDK import; no fallback. Reuse solver.pid, never assign indices after spatial sorting. Preserve unit semantics:position in metres, velocity m/s, density from final substep force evaluation, radius/color explicit display properties. New generation for replaced solver.
- [ ] Capture only on user action. No readback in SimSession.tick/draw, no new background handlers, no imported Warp at add-on registration.

2. Artist-facing native output (TDD)
Files:warp_sph_realtime/sdk_snapshot_ui.py; narrow cache_ui.register/unregister/Output panel integration; tests/test_sdk_snapshot_ui.py.
- [ ] Require explicit create/update; never replace an unrelated object. Link only into active scene. Keep GN modifier, exact ID words and failure rollback.
- [ ] On first creation stage native output and GN transactionally so GN failure removes only newly owned output. Normal update must retain all old data on rejection.
- [ ] UI poll does not allocate session/runtime or import SDK. Show pause/not-installed/qualified-candidate limits clearly.

3. Real Blender qualification and checkpoint
SDK example examples/consumers/warpsph.py and validation/adopt_warpsph_snapshot.py; paired source+installed identities and receipts.
- [ ] Source/full focused tests; package isolated consumer, audit all copied bytes; use one shared SDK wheel unchanged.
- [ ] Actual100k WarpSPH simulation advances and sorts IDs; pause/capture; compare real arrays by IDs with native and GN output. Resume, advance, pause/update and prove unchanged live state across each capture. Include growth/emitter case using existing approved APIs where time allows.
- [ ] Wrong-context/budget/replacement failure preserves current output. Stop/disable/load removes no unrelated data. Save snapshot scene and reopen without solver/SDK to verify persisted native output.
- [ ] Test library remains no mock-bpy substitute for native acceptance. Record correctness readback separately from performance; do not invent real-time or collider acceptance.
- [ ] Commit SDK example/evidence and consumer code independently; save cross-repo manifest and checkpoint to Downloads; no main merge/push/production install. Queue tracker update if authenticated Page tools unavailable.
