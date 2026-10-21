# Contained M0 results — 2026-10-17

**M0 exit remains BLOCKED; M1 was not started.** Process containment is now demonstrated,
and the native transport core passed. Normal cleanup from Blender lifecycle callbacks
still fails its context-ownership guard. No guard was removed to obtain a passing result.

Evidence root: `evidence/vulkan-zero-copy/20261017-contained-m0/`.
Candidate: Windows x64 Blender 5.2.0 `fbe6228777e7`, Python3.13.13, Warp1.17.0,
RTX4080SUPER, driver610.88. Executable SHA256
`0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327`.
The inventory and hashes of all 23 recorded package Python files remain unchanged.
All six launcher-owned Blender processes exited; no probe or supervisor remains running.

## Gate results

| Gate | Result | Evidence / scope |
|---|---|---|
| External process containment | PASS | Child natural/deadline controls, identity rejection, injected assignment rejection, supervisor loss and receipt-write failure controls. Windows Job membership is required before native calls; creation identity and nonce are checked. |
| Actual final-window timeout path | PASS, synthetic pending work | PID7876 closed its only window, reached exit_pre then modal cancel, parked without returning; supervisor confirmed exit197. No native GPU fault was injected. `final-window/`. |
| Blender/CUDA UUID equality | PASS | Both raw16-byte values `5d09f77ba936a6d9054d0324aef67adb`; comparison precedes all imports. Negative fixture rejected by the same equality guard. This is a scratch guard, not future M1 resolver qualification. |
| Exact ABI / import / texture extraction | PASS, bounded candidate | Corrected native run made112 imports. Explicit aggregate-result ABI, exact Python type and wrapper offset guards succeeded. CUDA structs independently compiled against installed12.9 header:104/88 bytes, reserved offsets36/20. |
| Full transport-position correctness | PASS | N=1,4095,4096,4097,4194304; every RGBA component including padding compared, zero mismatches. Four-million row compares16,777,216 components. Zero and invalid/capped count guard exercised. |
| Bounded timer completion queries | PASS, observed path | Raw CUDA event queries10056, Vulkan timeout-zero queries10055; one not-ready result. No infinite fence wait, GPU_finish or per-frame position readback. |
| Two-slot scheduling | PASS, bounded4097-particle lane | 10000 frames, GPU-accumulated mismatch sum0, one final correctness readback,10112 draw callbacks. Does not qualify four-million-particle latency or all possible cadence/resize interleavings. |
| 50 recreate cycles | PASS, bounded no-growth samples | Process handle count1351 throughout; reported free CUDA device memory15,165,505,536 bytes throughout. Deferred Blender ownership remains distinct from physical retirement of each allocation. |
| Six partial-acquisition rollback points | PASS, healthy synthetic interruptions | After pixel-buffer, export handle, import, mapping, event and fence. Both lifecycle attempts completed all six before the lifecycle failure. These are not injected driver/device failures. |
| Actual add-on unregister cleanup | FAIL | `addon_utils.disable` reaches scratch module unregister, but callback graphics-context identity is not the captured viewport context. Both attempts stop before normal resource release. |
| Producer-stop, load_pre, last-view-loss normal cleanup | BLOCKED / NOT RUN | Ordered behind failed unregister; do not infer passes from containment or healthy draw-context cleanup. |
| Negative GPU mismatch detector control | NOT CERTIFIED | The assertion was upstream of the lifecycle failure, but its result was not durably written before failure. No standalone passing receipt. |
| Completed-transport timing acceptance | NOT MEASURED | Query mechanism demonstrated; no latency distribution or4M performance acceptance. |
| Overall M0 exit | BLOCKED | Lifecycle and associated remaining acceptance evidence are not complete. |

The corrected core PID68016 exited naturally with code0. Killed candidates are failed
subcases with successful containment, never normal-cleanup passes.

## Lifetime / containment solution

The independent supervisor owns an unnamed, non-inheritable Windows Job with
KILL_ON_JOB_CLOSE and no breakaway, retains the verified process handle, and enforces
absolute and phase deadlines. Callback failure attempts durable diagnostics and then
parks in a nonreturning finally path. It cannot unwind into Blender context destruction.
The supervisor terminates only its owned process and waits at most10 seconds for exit.
An unconfirmed termination would stop all further GPU work.

The final-window source route was checked: WM_exit_ex calls exit_pre then removes modal
handlers before add-on/database/GPU destruction. The real final-window control observed
that ordering. `exit-pre.json` names its boolean argument `is_background`; this is merely
the test's field label and does not establish background execution. Launcher evidence
records the isolated GUI session.

This solution permits disposable native experiments. It does not provide shared-GPU/TDR
isolation or permission to terminate a user's production Blender during cleanup. Normal
product lifecycle acceptance still requires a verified context lease and bounded release.

Primary references: [Windows Jobs](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects),
[asynchronous process termination](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess),
[exact Blender exit ordering](https://github.com/blender/blender/blob/fbe6228777e7/source/blender/windowmanager/intern/wm_init_exit.cc#L477).

## Preserved failures and bounded corrections

1. PID56608, `native/`: first import returned `CUresult=1`. The harness had omitted
   reserved[16] fields from CUDA descriptors, following abbreviated web declarations.
   Installed cuda.h and a native compile established the defect. One separately reviewed
   corrected core candidate passed; both source snapshots and receipts remain intact.
   Runtime loading uses System32 nvcuda.dll, not the developer Toolkit DLL path; the
   Toolkit header is offline ABI evidence only.
2. PID27160, `lifecycle/`: actual unregister called make_drawable but GPU context equality
   failed. Blender's windrawable cache can bypass reactivation. The original traceback
   is retained in launcher logs; addon_utils caught it, then the harness parked on its
   missing-unregister-return assertion.
3. PID50468, `lifecycle-corrected/`: matched-PDB/binary evidence approved the forced
   make_drawable→reset_drawable route, with an unchanged ownership guard. The callback
   instead entered with a non-null context different from the viewport's captured context,
   failing `unexpected foreign active context` before forced activation or release.
   Supervisor exit197 is confirmed. No further lifecycle retry was made.

Offline `context-abi.json` identifies reset_drawable RVA0x43d590 and nullable
GPU_context_active_set RVA0x412cc0. It also contains one incidental string symbol;
that data entry was never treated as callable. Source shows no explicit infinite wait in
this activation route, but mutex/driver calls remain subject to supervision. Source
eligibility did not establish callback-context ownership at runtime.

Current script includes an **unrun diagnostic-only correction** to preserve the incoming
context address and original lifecycle error before parking. Measured sources remain
immutable under `source*` with separate manifests. Do not substitute current script bytes
for a measured snapshot.

## Verification and resume condition

`m0_contained_verify.py` independently checks receipts, source snapshots, owned-process
exit and unchanged package inventory. It exits1 because lifecycle-dependent gates fail;
this is an honest failed M0 audit, not a weakened test. Raw failure receipts, launch
identities/logs, CUDA header excerpt/hash, compiler failures and successful ABI compilation
are retained. `manifest.json` binds the retained evidence.

Before another native lifecycle run, identify the callback's incoming context and prove
its ownership/lifetime and required graphics-state restoration. It must be safe to borrow
the owner window and restore that incoming context without relying on another redraw or
destroying a secondary context via unbounded device-idle. Keep the exact identity guard
until that evidence supports a revised scoped rule. Then rerun the failed lifecycle lane
and its blocked successor cases; no need to repeat the unchanged passing core first.

Tracker Page remains unreadable and has no available update connector. The pending queue
records this outcome; no tracker synchronization or M1 authorization is claimed.
