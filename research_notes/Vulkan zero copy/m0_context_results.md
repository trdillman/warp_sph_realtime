# M0 context ownership continuation

Scope: identify lifecycle callback GPU context, prove paired ownership/restoration,
and complete blocked cleanup probes. Scratch harness only; M1 remains unauthorized.

Exact candidate: Blender fbe6228777e7, executable SHA256
0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327.
Evidence root: `evidence/vulkan-zero-copy/20261017-contained-m0/`.

Identity-only probe PASS (`context-identity/identity.json`): same main thread;
POST_VIEW is DRW shared viewport context, timer/unregister are the live window
context. No GPU allocations or context switches. Supervisor confirms natural exit0.
Source and matching binary explain why previous make/reset attempts could never
recover the captured draw context: window and shared draw contexts are distinct.

Completed: matched-PDB disassembly and pinned source support paired
`DRW_gpu_context_enable/disable` (0xb458b0/0xb459b0). Entry acquires both locks and
frame boundaries; exit restores the window drawable and releases both locks.
Strict main-thread, one-live-window, non-nested scope guards precede acquisition;
the active context must equal the captured shared draw context before release.
Exact incoming window context and DRW-disabled state must be restored before return.
After file replacement, only the next authenticated draw recaptures context identity.
Do not use release/activate as an ownership acquisition substitute.

| Gate | Result | Cleanup including restoration |
|---|---|---|
| Actual addon unregister | PASS | 13.945 ms |
| Producer stop before upload | PASS | 10.106 ms |
| Actual file-load load_pre | PASS | 12.241 ms |
| Last VIEW_3D changed to text editor | PASS | 10.041 ms |
| Context restoration | PASS | 4/4 exact pointer transitions |
| Synthetic partial setup rollback | PASS | six acquisition points |
| Deliberate GPU mismatch | PASS | 4097 expected, 4097 detected |
| Process exit | PASS | PID65140 natural exit0 |

Evidence: `lifecycle-owned/lifecycle.json`, `lifecycle-owned/watch/result.json`,
`source-lifecycle-owned/`, `source-manifest-lifecycle-owned.json`, launch receipts
and `retained-launch-logs/`. Identity probe PID63704 also exited naturally with0.
`drw-context-abi.json`, `drw-disable-body.json`, and `draw_gpu_context.cc` preserve
offline proof. Independent read-only review found no receipt safety inconsistency.

`m0_owned_verify.py` audits 15 checks, including the earlier unchanged native core,
50 recreates, 10000 stress frames and containment controls. It requires exactly four
restoration rows with prior==live-window context!=captured DRW context. All pass.
All eight owned probe PIDs exited. All 23 package Python files and inventory match
the original preflight. No product source or M1 implementation changed.

Prior passing core and failed lifecycle receipts remain preserved. New audit and
hash manifest use `verification-owned.json` and `manifest-owned.json`; historical
failed verification is not overwritten. The blocked cleanup slice is complete.
The full original M0 checklist is not certified by this 15-check audit (notably its
both-backend draw/count coverage must be reconciled separately); timing acceptance
remains unmeasured. Milestone checkboxes remain open and M1 requires a new request.

Limits: 100ms is an observed healthy-cleanup budget, not a hard bound on locking or
driver calls. External Job containment still supplies the experimental failure
boundary. Optional shader printf handling can read a GPU storage buffer, so no
universal no-blocking-wait claim is made for the entire DRW wrapper. Blender frees
to deferred pools; physical deallocation is not established by wrapper release.
Final-window timeout handling is the retained synthetic containment test, not
production safe shutdown. No shared-GPU fault isolation or release acceptance.

Tracker read again returned an empty shell; update remains queued in
`docs/TRACKER_UPDATE_PENDING.md`. No further GPU retry is needed for these passed
cleanup cases. Remaining work is broader qualification under separate scope.
