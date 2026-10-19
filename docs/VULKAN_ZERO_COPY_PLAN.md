# Vulkan zero-copy transport — implementation plan

Status: bounded exact-build M0 prerequisite checklist PASS (2026-10-17 local date).
Final reconciliation: `research_notes/Vulkan zero copy/m0_qualification_results.md`.
M1 profiles/resolver implemented and verified (2026-10-18). M2–M4 implemented and
headless-verified 2026-10-18 (pytest/ruff/pyright, two independent reviews). M5 v3 run
2026-10-18 (dedicated upload submission): parity, images, lifecycle, fault matrix and mocked
fatal PASS; latency gate FAILS (4194304 completed p95 11.66 ms vs ≤2 ms; ≤2.27 ms up to 2M).
At 4M the copy queues behind the viewport's own GPU point rendering. Awaiting the owner's
target decision. M6 blocked (no Linux hardware); M7 proposal drafted, not posted. Evidence:
`evidence/vulkan-zero-copy/20261018-m5-fence/m5-verdict.json` (v2:
`evidence/vulkan-zero-copy/20261018-m2plus/`).
Owner: engineering. Parent: `docs/RELEASE_PLAN.md` P0.11/P0.11b.
Historical review: `docs/VULKAN_ZERO_COPY_PLAN_REVIEW.md`.
Retained evidence: `research_notes/Vulkan zero copy/`.

The user completed bounded M0 continuation, then separately authorized M1 implementation.
Latest results: `research_notes/Vulkan zero copy/m0_qualification_results.md`.
Exact UUID equality, native full-pattern transfer,10000-frame scheduling,50 recreates,
partial setup rollback and final-window process containment passed their bounded checks.
Normal unregister, producer-stop, file-load and last-view cleanup now pass with exact
context restoration. Historical failures remain preserved. This is a disposable-process
prerequisite result, not product integration. Current M1 work is tracked in
`research_notes/Vulkan zero copy/m1_task.md`; M0 receipts remain historical and preserved.

## Outcome and authority

On a qualified NVIDIA/Blender Vulkan build, Warp publishes into Blender-owned GPU memory,
followed by a GPU pixel-buffer-to-texture copy and viewport draw. Zero-copy here means no
per-frame CPU readback; it does not mean no GPU copy or host wait. Unknown builds and
recoverable interop failures use labelled Readback only while the CUDA solver context is
usable. Fatal CUDA errors stop the session and require restart. A self-test cannot establish
a universal crash-free or nonblocking guarantee.

AGENTS.md governs backend isolation and the narrow Readback exception. Keep Warp pinned to
1.17.0 and the solver on cuda:0. Vulkan interop requires the physical-device UUID to match
CUDA device 0. Do not silently switch compute devices. Device selection, non-NVIDIA compute
and changing the runtime pin are outside this plan.

## Retained evidence and limits

The retained probe reports Blender 5.2.0 build fbe6228777e7, Vulkan, Warp 1.16.0 and
4,194,304 vec4 positions (64 MiB). Accompanying notes identify RTX 4080 SUPER/driver 610.88.
It imported a pixel buffer into CUDA, wrote a pattern and found zero mismatches in sampled
CPU checks and a shader sample. This establishes feasibility only for that candidate.

The 1.55 ms median (1.47–3.06 ms) measures texture update plus GPU_finish after Warp writing
and CUDA synchronization. It excludes pack, slot waits and publish costs. It is not full
transport performance. Sparse samples do not prove every particle. No pinned-1.17,
double-buffer, lifecycle, Linux runtime or release acceptance is implied. Preserve these
original receipts. Source/ELF observations in open_risks.md are discovery evidence, not ABI
guarantees for future builds.

## Shared interface and adapter boundaries

P0.11 defines publish(positions), positions_texture(), capabilities() and close().
Positions remain flat wp.vec3 SoA in the solver. The renderer receives a GPUTexture plus
logical count and dimensions. A not-ready result is explicit and draws nothing.
Capabilities report transport, zero_copy, maximum count, readiness and unavailability
reason. close() is idempotent and distinguishes completed cleanup from failed/quarantined
ownership; queued cleanup is never reported as freed.

Selection uses the running gpu.platform.backend_type_get(), not preferences:
1. VulkanInterop only after ABI, functions, device identity, cleanup route and self-test pass.
2. GLInterop only on OpenGL and after its own interop/self-test gates pass.
3. Readback only with a usable CUDA context and measured particle-count budget.
4. Otherwise stop startup with a specific diagnostic. Never call raw GL on Vulkan.

Readback alone may transfer positions to preallocated host staging, then upload a texture
in a GPU context. Keep transfers inside the adapter, outside solver kernels/graph capture;
prove stream ordering and staging lifetime before claiming overlap. GLInterop must gain a
verified texture-producing bridge in P0.11; today's VBO does not satisfy the new interface.
The legacy OpenGL SSFR path stays explicitly separate until P0.13. Vulkan shows spheres
with a note; do not pass the texture interface into an incompatible VBO renderer.

## C1 — Native ABI and safe cache (findings 1 and 5)

Ship reviewed profiles per exact binary/platform/architecture/Python ABI. Each records
executable identity, Windows PDB GUID and age or Linux ELF build-id, calling conventions,
function signatures, structure sizes/alignments/offsets, enum values, external allocation
assumptions and the GPUTexture wrapper extraction rule. Include provenance/schema version.
id(texture)+16 is not a portable rule. Width/height functions dereference the pointer and
cannot safely validate an unknown layout. Device UUID/context access needs the same profile.

Reject unknown profiles before pointer extraction or native calls. Matching decorated
names alone is insufficient. Establish profiles offline from matching source/debug data,
then validate on an isolated exact-build candidate; never guess a layout at runtime.

Windows resolver:
1. Identify loaded executable and approved profile, locate matching PDB, and verify RSDS
   GUID/age. Require matched debug data and expected decorated names.
2. Resolve symbols in an isolated helper process using DbgHelp. A private session handle
   alone does not isolate DbgHelp from other threads/components in Blender. Bound the helper,
   validate its results, and make no Blender GPU calls from it.
3. Cache module-relative RVAs only, never process addresses. Key by executable digest,
   build hash, PE timestamp/SizeOfImage, GUID/age, architecture and ABI-profile version.
   Rebase against the current loaded module each activation.
4. Validate JSON schema/types/identity and executable-section bounds. Independently compare
   every callable RVA against a shipped reviewed exact-build manifest or freshly resolve
   it from the matched PDB. A checksum beside mutable JSON is not independent validation.
   A well-formed, in-range wrong RVA must be rejected before any native call.
5. Discard invalid cache and re-resolve once from trusted inputs. Missing/unverifiable
   inputs mean Unavailable. Self-test is downstream, never the first safety guard.

2026-10-18 revision (owner-delegated decision): for an executable whose SHA-256 and PE/RSDS
identity match a shipped reviewed profile, the profile's RVAs are the runtime authority;
steps 1–3 (PDB, DbgHelp helper, cache) become an optional deep verification used to build
and audit profiles. The digest pins every code/data byte, so a PDB adds no runtime safety
while requiring users to download a 1+ GB file. Resolution runs at session Start, never in
register(). Unknown digests remain Unavailable.

Required functions: pixel-buffer create/free/size/native-handle, texture update from pixel
buffer, fence create/signal/free, verified zero-timeout timeline query (never GPU_fence_wait),
width/height after validated extraction, and verified
device UUID/context access. Linux uses reviewed build-specific tables, never runtime
heuristic scanning. Account for ELF load bias/type per profile; one non-PIE observation
does not authorize absolute addresses for future builds.

## C2 — Device, sizes, streams and slots (findings 4 and 6)

Before export/import, obtain Blender's actual Vulkan physical-device UUID through the
profile's verified route and compare it with CUDA device 0. Absent/ambiguous/different
identity means Unavailable with zero imports. An import error/self-test is not a substitute.
Bind Warp arrays, pack launches, external imports, events and synchronization to that same
verified device/context. Record identity and stream policy in receipts.

For N>0, width=4096 and height=ceil(N/4096), checked against actual texture limits.
Allocate/map 16*width*height bytes per pixel buffer with checked arithmetic and required
allocation rounding; upload the same padded extent. Pack only N positions, initialize
padding, draw exactly N instances. N=0 allocates nothing and draws nothing. Budget both
pixel buffers, texture, solver, staging and temporary resize overlap against currently
available VRAM with reserve; 75% of total VRAM alone is not an adequate check.
Recoverable resize failure preserves the old candidate.

Pack on the solver's explicit Warp stream, or establish an event dependency across streams.
Record an event on that actual producer stream after packing. Query its raw CUresult once
per tick: success (0) permits READY, NOT_READY (600) retains WRITING, every other result
stops the session. Warp Event.is_complete() collapses errors to false and is unsuitable
for this classification. Do not call synchronize_stream or assume default-stream ordering.
No allocation or host wait enters CUDA capture.

Slots follow FREE → WRITING → READY → IN_FLIGHT → FREE, with monotonic frame and allocation
generation IDs. Draw consumes the newest READY slot, uploads and signals its fence.
Reclaim an IN_FLIGHT slot only after a verified vkWaitSemaphores query with timeout=0
returns VK_SUCCESS (0); VK_TIMEOUT (2) retains ownership and every other result stops.
The exact-build GPU_fence_wait calls UINT64_MAX and is forbidden. Read the signalled
fence's timeline plus the matching live backend device/semaphore only through the reviewed
profile; never alter Blender's fields. Retire stale READY slots
without overwriting WRITING/IN_FLIGHT data. If no safe slot is available, skip publication
and retain the last texture rather than block indefinitely. M0 must prove a bounded reuse
policy; otherwise VulkanInterop stays unavailable until that policy is revised and proven.

Bound each owner-thread tick to at most two producer queries and two Vulkan queries, once
per slot, with no polling loop, sleeps or retry. A predeclared 500 ms pending-work deadline
stops publication and quarantines uncertain ownership; it never authorizes reuse. Shutdown
has a 100 ms overall polling budget, zero-timeout queries, and no new submissions except
the final retirement fence. Timeout/error is failed cleanup, never fallback or success.
These are experiment limits, not latency acceptance or a hard wall-clock guarantee for
driver calls. Do not use GPU_finish, device-idle or infinite fence waits to satisfy them.

Initialize in a GPU context before publication. Timer-before-first-draw touches no missing
resources. Multiple ticks may replace an unconsumed READY frame; multiple draws reuse the
last texture without re-signalling consumed slots. Serialize state on Blender's owner
thread. Prove multi-window context ownership or restrict use to the owning viewport and
stop safely on its loss.

## C3 — Teardown and error classification (findings 2 and 3)

A session resource owner retains every acquisition, import, fence and handler from first
allocation. Its synchronous cleanup context route must remain valid through unregister,
load_pre and last-viewport closure, independent of future redraw. M0 must demonstrate
that route before interop is enabled. No proven route means Unavailable; deferring to the
next draw is not sufficient for release.

Offline follow-up: `research_notes/Vulkan zero copy/m0_offline_followup.md`. Restrict the
next eligible probe to one owner window/context; global draw handlers must reject every
other window. Outside cancellation, require current live-window-list membership before
using an owner pointer. In the authenticated modal cancel callback, the window is already
unlinked: require matching callback context.window, session/operator token and main thread
instead. Window context activation does not recover the shared viewport draw context.
The qualified normal callback route below supersedes that earlier hypothesis. Never
dereference a cached stale pointer or generalize the normal route to unlinked windows.

Disable further draws/publication, then signal a final retirement fence after the final
texture read, not just the pixel-buffer upload. Both CUDA production and Vulkan consumption
must be positively complete before handback. GPUOffScreen.bind is not a context lease.
Reject secondary-context ownership as the bounded cleanup solution: its GHOST Vulkan
destructor calls vkDeviceWaitIdle. Normal Blender resource deletion may enter a deferred
pool: record CUDA_RELEASED / BLENDER_DEFERRED separately from physical allocation release;
M0/M5 numeric retirement/no-growth evidence remains required.

Blocking edge: cancel cannot veto window destruction and a Python reference does not keep
the GPU backend alive. On unresolved CUDA work, quarantine alone cannot make final-context
destruction safe. Before that native lifecycle experiment, establish a surviving backend
lifetime or an independently supervised disposable-process exit contract. Neither is proved
by this offline work. Do not free potentially CUDA-active allocations into Blender's pool,
force its shared discard pool, or claim this gate passed.

2026-10-17 contained M0 continuation: an external Windows Job supervisor now provides an
experiment-only failure boundary. Before native calls, compare launcher PID, executable,
creation time and fresh nonce; require assigned Job membership and KILL_ON_JOB_CLOSE.
The supervisor owns the sole non-inheritable Job handle and retains the process handle.
On uncertain completion, cancel/load_pre callbacks persist diagnostics and park without
returning, even on receipt-write failure. The supervisor terminates the isolated process
and waits at most 10 seconds for confirmed exit; unconfirmed exit blocks further GPU work.
Absolute 600-second and fixed per-phase deadlines cannot be extended by heartbeats.
No real illegal-access/device-loss/TDR injection is authorized by this containment design.
This closes the experimental callback-return hazard, not normal lifecycle acceptance or
shared-GPU fault isolation. A killed candidate records failed subcase/containment passed.
Evidence: `evidence/vulkan-zero-copy/20261017-contained-m0/`.

Observed limitation: make_drawable can skip activation when windrawable is cached. A
matched-PDB reset_drawable route (RVA0x43d590) forces activation; nullable GPU context setter
(RVA0x412cc0) can restore an originally inactive GPU pointer. However the corrected real
unregister callback entered with a non-null context different from the captured viewport
context. The guard rejected it before switching/freeing. Do not treat a thread-local raw
current-context pointer as a lifetime lease or restore arbitrary foreign contexts. The
follow-up identity trace identifies that incoming pointer as the live window context;
POST_VIEW uses the shared DRW viewport context. For the qualified one-window/main-thread
normal callbacks, require incoming==live-window context and DRW disabled; enter the paired
DRW_gpu_context_enable/disable scope (RVAs 0xb458b0/0xb459b0). Enable acquires Blender's
draw/context locks and frame boundary; require active==captured DRW context before cleanup.
Disable restores the window drawable and releases the locks; require exact incoming
pointer restoration and DRW disabled before return. Recapture after file replacement only
inside the next authenticated draw. Do not use release/activate as lock acquisition.

All four normal cases passed in 10–14 ms, including restoration, within the 100 ms healthy
probe budget. This is not a hard deadline for mutexes/driver calls: external containment
remains required. Optional shader printf handling can read a GPU storage buffer; the full
wrapper is not certified universally free of blocking waits. Final-window timeout proof
remains the separate process-containment control, not healthy product teardown acceptance.

Normal teardown: stop new solver/publish/draw work and scheduling; complete producer and
submitted GPU reads through the verified context route; free mapped CUDA pointers, destroy
external-memory imports, free pixel buffers/fences and release textures; then remove the
cleanup context/handlers and clear ownership. Retain the cleanup owner while stopping the
timer. Drain before file replacement/context destruction. Reverse partial setup using an
acquisition ledger; repeated close must not double-free.

Export once per allocation; never GPU_pixel_buffer_map an exported buffer. Close Win32
handles after import attempts in a finally path. Successful Linux import transfers fd
ownership to CUDA; on failed import the owner closes the fd. Dedicated flags and sizes
must match the verified exported allocation.

- Recoverable availability/interop errors with usable CUDA: safely close, switch once to
  Readback, record/display reason. Do not repeatedly retry interop.
- Fatal CUDA/context loss/illegal address/launch failure, or uncertain context health:
  stop, preserve original diagnostics, require restart, and issue no further solver,
  Readback or speculative CUDA cleanup calls. Quarantine resources that cannot safely be
  freed until process exit. Do not reset Blender's shared device or claim normal leak-free
  teardown. Fatal acceptance proves stopped work and truthful status; normal cleanup gates
  separately require complete release.
- Healthy-context self-test mismatch: safe cleanup then fallback. Native access violations
  cannot be recovered by Python exceptions; prevent invalid calls through C1.

## Milestones

### M0 — Isolated prerequisite probes

Artifacts: versioned scripts and m0_spikes.md under research_notes/Vulkan zero copy/;
raw logs/receipts under evidence/vulkan-zero-copy/<candidate>/<run>/.

- [x] Freeze executable/package hashes, approved ABI evidence, Blender/Python, Warp 1.17.0,
  driver/GPU UUIDs, backend and isolation paths. Isolate preferences/extension storage.
  Runtime loading must not depend on the old probe's developer CUDA Toolkit path.
- [x] Prove approved texture extraction, UUID matching and zero-import rejection.
- [x] Prove context-valid fences, timer wait legality and bounded slot reuse. Create,
  upload, signal and free remain context-bound.
- [x] Demonstrate synchronous teardown for unregister, file load and last-view closure.
  Failure blocks M1; do not implement around an unproven cleanup route.
- [x] Prove instanced quads with gpu_InstanceIndex/texelFetch on both backends. Fully compare
  deterministic positions at N=1,4095,4096,4097,4194304; test zero/invalid/limit counts,
  initialized padding and drawing exactly N.
- [x] Separate correctness readback from a 10,000-frame scheduling lane with no per-frame
  CPU readback or GPU_finish. Accumulate generation/frame mismatch errors on GPU, inspect
  once after drain. Record waits, drops and multiple-tick/draw cadence.
- [x] Run 50 recreate cycles and partial-setup failure cases; collect numeric handle/VRAM
  samples. Task Manager screenshots supplement measurements, not replace them.

Each answer needs a receipt or explicit failure. Preserve failed probes; do not automatically
retry GPU runs after candidate/harness changes. Amend the contract before advancing.

Final reconciliation uses original public receipts for both-backend quads/counts, retained
native core for 10000 frames/50 recreates, lifecycle-owned for cleanup, and qualification
for missing limit/budget/UUID guards plus120-frame cadence/generation instrumentation.
`verification-final.json` under the contained evidence root audits19 checks and8 gate
groups. Limit/budget negatives inject restricted inputs before native allocation, not real
OOM. Completion uses one QueryPerformanceCounter clock and zero-timeout fence observations;
the resulting upper bounds include cadence, polling, consumer work and cold-start costs.
They do not qualify precise GPU duration or the M5 four-million-particle<=2ms target.
These checked boxes qualify the disposable exact-build prerequisite experiment only.
Production integration, general timeout safety, Linux and M2–M7 remain unimplemented/open.

### M1 — Profiles and resolver

Delivered files: package render/blender_symbols.py, packaged profiles/helper and
tests/test_blender_symbols.py. Linux stub returns Unavailable until M6.

- [x] Add failing tests for unknown profile/layout/enum/signature, missing/mismatched PDB
  GUID/age, helper timeout, cache schema/types/bounds, in-range wrong RVA and stale identity.
  Assert zero native calls and pointer extraction on preflight rejection.
- [x] Implement C1, pass tests and fresh-process ASLR/rebasing checks.
- [x] Resolve every required function/identity route on the approved Blender candidate.

Evidence: `evidence/vulkan-zero-copy/20261018-m1/` and
`research_notes/Vulkan zero copy/m1_task.md`; usage: `docs/VULKAN_RESOLVER.md`.
Two fresh processes resolved 20 callable and four data addresses (helper then cache).
Both received the same nonpreferred ASLR base; distinct-base arithmetic is unit-tested.
These checks made no GPU native calls or texture pointer extractions. Package built,
not installed. M1 does not qualify transport, product cleanup or performance.

### M2 — Vulkan transport

Planned files: package render/transport_vulkan.py and tests/test_transport_vulkan.py;
integrate through P0.11, not directly into solver code.

- [x] Test slot/stream transitions, padded sizes, generations, partial ownership cleanup
  and error classes before implementation.
- [x] Implement C2/C3 after M0/M1. Use two pixel buffers, two fences and one texture.
  A 4×4, 256-byte known-answer self-test gates activation.
- [x] Verify first-draw ordering, multiple ticks/draws, resize and healthy fallback
  (headless fakes in tests/test_transport_vulkan.py and test_transport_session.py; runtime in M5).

### M3 — Texture renderer (P0.12)

Planned files: package render/gpu_points.py, render/viewport_draw.py and renderer tests.

- [x] Use one six-vertex quad batch; texelFetch by instance, explicit count/dimensions,
  MVP, size and colour. Verify depth/occlusion and position parity for all adapters
  (position/image parity is an M5 runtime gate).
- [x] Keep legacy OpenGL SSFR routing explicit until P0.13; Vulkan uses spheres with a note.

### M4 — Lifecycle integration

Planned files: package addon.py, render/viewport_draw.py and env-gated Blender tests.

- [x] Replace direct buffer ownership with the session/transport owner; initialize before
  publication. Keep the solver independent of transport.
- [x] Integrate close into stop, unregister, load_pre, viewport loss and transactional
  resize. Diagnostics distinguish recoverable fallback from restart-required failure.
- [x] Test immediate disable, in-flight disable, repeated close, no-view startup,
  last-view closure, file load and failure after every setup acquisition. Successful
  normal close leaves no owned handles, timers, callbacks or allocations (headless
  subprocess add-on tests with fakes; real-Blender cycles are the M5 gate).

### M5 — Runtime qualification and measurement

Use exact packaged candidates, Warp 1.17.0 and RTX 4080 SUPER for Windows qualification.
Record source/package identities and environment; mocks do not certify runtime behavior.

- [x] Full deterministic transport-position parity with Readback, plus rendered-image
  comparisons with a predeclared tolerance on both backends, including padded counts.
- [x] 50 cycles each of start/stop, resize, disable and file load. Account for allocator
  warm-up; record owned resources, handles and numeric VRAM. Require no ownership growth
  and final VRAM within a predeclared measured idle-baseline envelope.
- [x] Fault matrix: missing/stale/corrupt cache, in-range wrong RVA, missing PDB, unknown
  ABI, UUID mismatch, allocation/import failure and self-test mismatch. Assert pre-call/
  pre-import rejection where applicable and exactly one fallback report.
- [x] Fatal error tests assert stop, no fallback and no further CUDA calls. Real fatal
  injection requires separate explicit runtime authorization and a disposable process;
  mocked errors prove classification only.
- [x] Benchmark 0.5M/1M/2M/4M, recording exact counts (4M gate=4194304), fixed scene/camera/
  cadence and all applicable adapters/backends. Warm up at least 300 frames and retain
  at least 3000 samples per case under reports/benchmarks/ with candidate references.

Report CPU submission, slot wait, pack GPU time, upload GPU time, drops, total frame time,
and publish-to-draw latency separately. Completed transport latency runs from publish entry
(solver data ready) through completion of that frame's texture upload, including pack,
producer synchronization, waits and upload. Establish verified nonblocking completion
instrumentation on a compatible clock in M0. Do not add per-frame full drains to measure it;
if instrumentation is unavailable, timing acceptance remains unmeasured.

Acceptance: completed transport latency p95 ≤2 ms at 4194304 particles, with p50/p95/max
reported; no GPU_finish or per-frame readback in accelerated steady state. Benchmark at
declared cadence with zero publication drops; skipped frames cannot improve the score.
Report user-visible publish-to-draw latency separately. The old copy-only median does not
predict this stricter gate. Failure requires explicit reviewed target/design changes,
never redefining metrics after measurement.

**M5 result (2026-10-18, frozen manifest v2 `7c7d67bb…17e2`, Blender 5.2.0 fbe6228777e7,
Warp 1.17.0, RTX 4080 SUPER, driver 610.88).** Criteria pre-registered in
`evidence/vulkan-zero-copy/20261018-m2plus/M5_PROTOCOL.md`; verdict `m5-verdict.json`;
benchmarks `reports/benchmarks/vulkan-zero-copy-m5-20261018.{json,md}`.
- Parity: interop and Readback textures bit-exact for 1/4095/4096/4097/4194304, padding zero,
  both backends. Images: 20 pairs, 100% of pixels within tolerance (same- and cross-backend).
- Lifecycle: 4×50 cycles, no owned objects left, quarantine empty, every close `completed`;
  VRAM flat (1728.5 MiB warm = final), handle slope ≤0.25/cycle.
- Fault matrix: 15/15 cases (resolver rejections never construct the Vulkan adapter; cache
  faults recover via one helper; activation faults fall back once with a balanced ledger).
- Mocked fatal (CUresult 700): stop, no fallback, zero further native calls, quarantined.
  Mock only — real fault injection not authorized.
- **Latency: FAIL.** Completed transport latency (publish → upload-fence timeline value,
  blocking vkWaitSemaphores observer) p50/p95/max ms: 0.5M 3.10/3.86/7.70, 1M 3.12/4.17/42.5,
  2M 3.49/4.45/38.1, 4194304 9.72/11.03/12.51; 3000 samples each, zero drops. The fence is
  Blender's frame submission, so the figure includes that frame's viewport rendering
  (frame p50 ≈4 ms); publish→draw p95 at 4M is 3.64 ms. Pack GPU p95 ≤0.22 ms, CPU submit
  p95 ≤0.33 ms. Upload GPU time remains unmeasured.
- Readback budget (CPU copy): publish→draw p95 Vulkan 10.9/16.4/27.7/49.9 ms and OpenGL
  16.0/20.8/28.4/46.6 ms at 0.5M/1M/2M/4M — a 60 Hz budget holds to ≈1M particles.
- Next (owner): revise the 4M target or its measurement definition (e.g. a dedicated upload
  fence) through a reviewed change; do not reinterpret this run. → Owner chose the dedicated
  upload fence (design change, metric unchanged): see the v3 result below.
- Harness defects fixed before the counted run (product untouched): see
  `.codex/context/vulkan-m2-loop.md` failure register; attempt 1 kept in `runs-attempt1/`.

**M5 v3 design change (2026-10-18, owner "Option 1").** `publish` packs, then waits up to
`PRODUCER_WAIT` = 2 ms for its own CUDA event. It then uploads and signals the slot fence from
the timer inside the qualified DRW context scope, so the fence submission carries only the copy,
not the frame. The draw uploads only a READY slot the timer could not submit (slow producer, or
no qualified scope such as a second window). C2 amendment: `publish` may query its own slot
repeatedly within `PRODUCER_WAIT`. Trade-off: up to 2 ms of main-thread spin per tick when the
producer is slow, and each tick submits its own copy (no coalescing across ticks between draws).
Texture write-after-read across submissions is covered by Blender's device-level resource
tracker and its single queue (source evidence at fbe6228777e7; no validation-layer run).

**M5 v3 result (2026-10-18, frozen manifest v3.1 `3b363bc1…ee2a`, same candidate).** Protocol
`evidence/vulkan-zero-copy/20261018-m5-fence/M5_PROTOCOL.md` (criteria unchanged from v2);
verdict `m5-verdict.json`; benchmarks
`reports/benchmarks/vulkan-zero-copy-m5-20261018-m5-fence.{json,md}`.
- Parity, images, lifecycle, fault matrix and mocked fatal: PASS.
- **Latency: FAIL.** Completed transport latency p50/p95 ms (3000 samples, zero drops, no
  missing completions): 0.5M 0.91/1.16, 1M 1.15/1.45, 2M 1.99/2.27, 4194304 10.94/11.66.
  Timer-submitted uploads 3300/3300 at every count except 3299/3300 at 4M. Compared with v2,
  latency is 3× lower up to 2M, but unchanged at 4M.
- Diagnosis (pre-freeze smoke, not a gate): at 4M with point drawing disabled, p95 was 2.37 ms.
  The remaining ~9 ms is the copy waiting on the GPU queue behind the viewport's own rendering
  of 4M points. No transport fence change can remove that. Publish→draw p95 at 4M is 3.82 ms;
  CPU submit p95 is 1.49 ms (now includes the producer wait and the upload submission).
- Readback (Vulkan, CPU copy) publish→draw p95: 7.2/12.5/23.2/44.5 ms at 0.5M/1M/2M/4M.
- Attempts: 1 (displays off: no viewport redraws) and 2 (harness `step_fatal` captured only
  the second tick) are kept in `runs-attempt1/`, `runs-attempt2/`; the product was unchanged
  across them.
- Next (owner): the 2 ms at 4M target conflicts with drawing 4M points in the same GPU queue.
  Options are a frame-budget target (e.g. completed ≤ one 60 Hz frame), gating 2 ms at ≤1M, or
  measuring with rendering excluded as a separate copy-only metric. Each is a reviewed target
  change, not a reinterpretation of this run.

### M6 — Linux after Windows qualification

Status 2026-10-18: BLOCKED — no Linux hardware/candidate available in this environment.

- [ ] Generate reviewed profiles/tables offline with matching source, RTTI/string anchors
  and disassembly. Verify wrapper/enum/signature/device identity and ELF relocation model;
  no runtime scans or unverified dummy-this calls.
- [ ] Repeat applicable M0/M5 gates on each official Linux candidate and real hardware,
  including fd ownership on successful and failed imports. Windows results do not qualify
  Linux. Unknown builds use Readback only with a healthy CUDA context.

### M7 — Public upstream API

- [x] Draft GPUPixelBuffer/native_handle, texture update and fence API proposal. Owner
  approval is required before external posting. Draft: `docs/VULKAN_UPSTREAM_API_PROPOSAL.md`
  (not posted).
- [ ] Qualify any future public API adapter before replacing internal resolution.

## Dependencies, scope and risks

Critical path: this guidance revision → separately authorized M0 → lifecycle P0.1/P0.2
and P0.11 texture seam → M1/M2 → M3/M4 → M5. M6 follows Windows qualification. Re-estimate
after ABI, UUID access, bounded waits and cleanup feasibility; the old three-week estimate
is provisional. Missing prerequisites keep VulkanInterop unavailable, not partially enabled.

Out of scope: AMD/Intel/macOS compute, removing the GPU copy, Vulkan SSFR (P0.13), device
selection, pin changes, publication and implementation under this revision request.

## Review resolution and task record

| Finding | Revised contract | Future evidence |
|---|---|---|
| 1: incomplete ABI | C1 profiles before extraction/calls | M0 profile; M1 rejection tests |
| 2: deferred cleanup | C3 synchronous independent drain | M0 feasibility; M4/M5 lifecycle |
| 3: fatal CUDA fallback | C3 stop/quarantine classification | M2/M5 error matrix |
| 4: late device match | C2 UUID equality before import | M0/M1 rejection; M5 mismatch |
| 5: unsafe cache | C1 independently verified RVAs/rebasing | M1/M5 corruption/restart |
| 6: evidence gaps | C2 sizing/streams/slots; M0/M5 timing | Pinned runtime parity/benchmarks |

Documentation resolution is complete; implementation/runtime gates remain open.
AGENTS.md, RELEASE_PLAN.md and PRODUCT_DELIVERY.md are reconciled with these contracts.
Revision validation: five edited documents checked for whitespace/conflict markers;
all six findings mapped, all eight milestones present and uncompleted, and the obsolete
vertex_buffer() interface removed from the release plan. These documentation checks passed;
they do not validate implementation or runtime behavior.
The review and original probe evidence remain historical and unchanged. No implementation,
probe, build or runtime test was performed. Tracker access returned no readable Page;
synchronization is queued in docs/TRACKER_UPDATE_PENDING.md.
The preceding paragraph records the documentation-only revision. Later M0 execution,
its initial failures and their resolution are recorded in `research_notes/Vulkan zero copy/`
(m0_spikes, m0_offline_followup, m0_contained_results, m0_context_results and
m0_qualification_results). M0 passed its bounded checklist on 2026-10-17; M1 is above.
