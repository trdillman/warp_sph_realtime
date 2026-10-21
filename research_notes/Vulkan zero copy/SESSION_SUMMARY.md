# Session summary — Vulkan zero-copy M0

## Outcome

The bounded Windows exact-build M0 prerequisite checklist passes. M1 and product
implementation have not started. Final reconciliation, gate evidence and limitations:
[m0_qualification_results.md](m0_qualification_results.md). Final audit:
`evidence/vulkan-zero-copy/20261017-contained-m0/verification-final.json` (19 checks,
8 gate groups); `manifest-final.json` binds the final artifacts. All prior failures and
source snapshots remain preserved.23 package Python files/inventory are unchanged from
preflight; unrelated pre-existing changes were retained. No commit or publication.

## What this session covered

1. **Plan review and revision.** Reviewed the original Vulkan zero-copy proposal and
   addressed six findings: incomplete native ABI proof, cleanup lifetime, fatal CUDA
   fallback, device identity before import, unsafe resolver caching, and missing sizing/
   stream/scheduling/performance evidence. Reconciled AGENTS, release and product guidance.
   The original review is `docs/VULKAN_ZERO_COPY_PLAN_REVIEW.md`; the current authority is
   `docs/VULKAN_ZERO_COPY_PLAN.md`. Zero-copy means no per-frame CPU readback, with a GPU
   buffer-to-texture copy still present.

2. **Public prerequisite probes.** Isolated Vulkan and OpenGL sessions proved public
   instanced quads with gpu_InstanceIndex/texelFetch and every RGBA component/padding at
   N=1,4095,4096,4097,4194304; zero count performed no allocation/draw. A bootstrap staging
   failure was preserved and corrected after staging completed. Report: `m0_spikes.md`;
   evidence: `evidence/vulkan-zero-copy/20261017-m0/`.

3. **Offline ABI and bounded synchronization.** Matched executable/PDB/source evidence
   established exact native layouts, hidden return ABI, texture wrapper extraction and
   UUID access. The original GPU_fence_wait path uses UINT64_MAX and was rejected.
   Revised completion uses raw CUDA event queries and zero-timeout Vulkan timeline queries
   with explicit status handling. Report: `m0_offline_followup.md`; evidence:
   `evidence/vulkan-zero-copy/20261017-offline-followup/`.

4. **Process containment and native core.** An external Windows Job supervisor validates
   PID/executable/creation time/nonce, owns a kill-on-close Job, enforces deadlines and
   confirms exit. Uncertain callbacks park without returning into Blender destruction.
   The first CUDA import exposed omitted reserved ABI fields; corrected exact header
   layouts then passed native transfer,10000 stress frames and50 recreate cycles. Real
   final-window synthetic-timeout containment also passed. Report: `m0_contained_results.md`.

5. **Cleanup ownership diagnosis.** Two guarded lifecycle attempts failed before unsafe
   release. An identity-only trace established that viewport drawing uses shared DRW
   context while timer/unregister use the window context. The paired locked DRW enable/
   disable scope then passed unregister, producer stop, load_pre and last-VIEW_3D loss,
   restoring the exact incoming context in10–14ms. Six rollback points and a4097-error
   negative control passed. Report: `m0_context_results.md`.

6. **Final M0 reconciliation and missing qualification.** Reused all proven rows rather
   than repeating them. A120-frame supplement proved current texture-cap/free-memory
   checks, injected pre-allocation budget/limit rejection, actual constructor UUID
   rejection, two allocation generations,1/3/7-tick cadence,299 no-free skips,118 stale
   retirements,533 repeated draws, per-tick query caps and common-clock completion
   observations. One final GPU error readback was zero. New process exited naturally0.
   Report: `m0_qualification_results.md`.

## What remains unproved

- The M5 four-million-particle<=2ms performance target is NOT MEASURED. Supplemental
  N4097 host completion-observed bounds include launch/cadence/polling/consumer overhead:
  p50 20.5849ms,p95 72.2876ms,max685.4183ms. They are not exact GPU durations or a500ms
  end-to-end bound.
- Production integration, universal bounded native-call duration, multi-window support,
  physical Vulkan deallocation, real fatal-GPU fault handling/isolation, Linux and release
  acceptance remain open. Disposable-process containment is an experimental boundary.
- M1 requires a separate request. No next GPU run is queued or left running.
- Tracker access repeatedly returned an empty Page; all updates are queued in
  `docs/TRACKER_UPDATE_PENDING.md`. The tracker is not synchronized.

The requested handoff is stored separately in the Windows temporary directory and points
to these durable artifacts rather than duplicating them.
