# M0 final reconciliation and bounded supplement

Scope: reconcile existing proof and fill missing prerequisite evidence; no M1.
Exact candidate remains Blender fbe6228777e7 / Warp1.17.0 / RTX4080SUPER on Windows.
Earlier failed and passing receipts remain immutable.

## Reconciled evidence before this run

| Original M0 requirement | Retained proof | Remaining gap |
|---|---|---|
| Exact executable/PDB/runtime/isolation | 20261017-m0 host-preflight, offline-identity; offline-followup verification; launch receipts | none for pinned probe |
| Texture ABI, UUID equality and rejection | offline ABI plus native-corrected receipt | exercise actual constructor with mismatched/short UUID |
| Fences, timer polling, bounded reuse | native-corrected core, 10000 stress frames | explicit cadence/drop/generation accounting |
| Unregister/load/last-view cleanup | lifecycle-owned receipt and verification-owned | none for bounded healthy cases |
| Both-backend quads, texelFetch, full counts/padding | public-vulkan.json and public-opengl.json, source m0_public_probe.py | actual reported texture limit and budget rejection |
| 10000-frame no-per-frame-readback lane | native-corrected receipt, one final GPU-error readback | short supplemental cadence trace, not another10000 run |
| 50 recreates and partial acquisition | native-corrected cycles plus lifecycle-owned six partial cases | none for bounded probe |
| Compatible-clock completion instrumentation | zero-timeout completion query already proven | host-clock publication/completion-observation brackets |

Earlier uncertainty about both-backend rendering coverage was a reconciliation gap,
not missing execution. The retained public harness uses gpu_InstanceIndex/texelFetch
and six-vertex instanced quads, including full padding comparisons through4194304.
Native invalid counts [-1,4194305,1.5] and zero-shape handling are already recorded.

## Predeclared supplemental acceptance

One isolated supervised Vulkan run, at most180 seconds, existing500ms pending-work
limit.120 consumed4097-particle frames over two allocation generations, cadence1/3/7
ticks. No deliberate GPUfault/OOM. Current reported texture cap and free-memory query;
pure pre-allocation guard budgets2PB+texture+scratch+equal retained resize overlap plus
1GiBreserve. Restricted cap/free inputs must reject with zero native/import deltas.
Same actual Slot constructor must reject short/mismatched UUID before native calls.
Record raw query results, per-tick query caps, drops, stale READY retirement, repeated
texture draws, and generation-stamped comparisons. One final accumulated-error readback.
Clock is CPU perf_counter_ns throughout. Completion-observed bounds include polling
delay and consumer work; they are not precise GPU durations or the4M<=2msM5 benchmark.

## Result: PASS_BOUNDED_M0

Supplement PID68068 exited naturally0 under the180-second supervisor.120 consumedframes,
60 per allocation generation;238 accepted publications,299 no-free skips,118 stale READY
retirements,533 repeated texture draws,378 deliberately deferred draw callbacks.539 ticks
and539 draw callbacks. Cadence1/3/7 each consumed40 frames. Two stale-generation inputs
rejected. One final GPU error readback:0. All120 timestamp chains are ordered and complete;
raw CUDA/Vulkan query outcomes and per-tick caps are retained. No per-frame readback.

Reported device extent and current free memory are in `qualification/qualification.json`.
Invalid types/counts, zero allocation, injected device cap/free-budget restrictions and
actual Slot constructor short/mismatched UUID tests pass with zero added native/import
calls. Budget is conservative scratch feasibility, not production admission control:
actual export rounding and other-process VRAM races still require production handling.

Completion-observed upper bounds atN4097: p50=20.5849ms,p95=72.2876ms,max=685.4183ms.
Clock: QueryPerformanceCounter,100ns reported resolution, monotonic/nonadjustable. These
include host launch costs, selected cadence, polling delay and the texture consumer.
The685ms sample includes673.305ms from publish entry to observed CUDA readiness; these
stamps cannot assign that delay to JIT, driver execution or polling separately. The500ms
pending-query budget starts after launch, so it is not a500ms end-to-end publication
bound. These are not hard real-time bounds. The<=2ms4M M5
performance target remains NOT MEASURED; these data do not constitute a pass or fail of it.

Final audit `verification-final.json`:19 checks and8 gate groups PASS. The seven original
M0 checkbox requirements are reconciled and checked for this exact disposable experiment;
the eighth gate separately records completion instrumentation. All23 package Python files
and inventory remain unchanged. All9 contained-run owned PIDs exited. Historical public
run exit evidence and failures remain in their original roots. No product changes or M1.

Artifacts beneath `evidence/vulkan-zero-copy/20261017-contained-m0/`:
`qualification/qualification.json`, `qualification/watch/result.json`,
`source-qualification/`, `source-manifest-qualification.json`, `launch-qualification.json`,
`retained-launch-logs/launch-qualification/`, `verification-final.json`, `manifest-final.json`.
Previous receipts/manifests remain untouched. The report records local Oct17; UTC run
timestamps cross into Oct18. Tracker read returned empty shell again; update remains queued.

M0 completion is not shipping authorization. Future production ownership, fault handling,
bounded-call caveats, physical Vulkan retirement, multi-window support, Linux and M5
benchmarks remain in their later gates. Next step is a separately authorized M1 task.

Independent read-only final review accepted this bounded M0 conclusion without a blocking
receipt/code finding, with the timing/containment/production limits above retained.
