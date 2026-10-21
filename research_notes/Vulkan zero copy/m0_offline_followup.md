# M0 offline follow-up — 2026-10-17

Outcome: bounded-wait plan revised; exact-build ABI and UUID access route established
offline. M0 is **NOT PASSED**. No native Blender/GPU calls, new Blender launches, product
implementation or M1 work occurred in this follow-up. Prior runtime evidence is unchanged.

Candidate: Windows x64 Blender 5.2.0 fbe6228777e7, executable SHA-256
`0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327`.
Matched PE/PDB GUID `d357c018-4455-89ed-4c4c-44205044422e`, age 1, retained in
`evidence/vulkan-zero-copy/20261017-m0/offline-identity.json`.
Follow-up evidence: `evidence/vulkan-zero-copy/20261017-offline-followup/`.
This is a research profile, not an approved shipped resolver/profile.

## Evidence and ABI

`decorated-symbols.json` records matched-PDB symbols; `binary-disassembly.json` and
`.txt` record 21 exact PE function bodies and hashes. `source-capture.json` inventories
21 pinned source files. Scripts `m0_abi_symbols.py` and `m0_abi_binary.py` reproduce the
offline extraction. REA analysis timed out after 300 seconds; `rea-attempt.json` preserves
the failure. One distinct fallback, task-local Capstone 5.0.6, completed; REA was closed.

All RVAs below require this exact executable and current module-base rebasing. They do
not authorize runtime pointer access. Win64 aggregate returns use an explicit output
pointer; do not guess a ctypes automatic C++ structure-return ABI.

| Item | Exact-build proof |
|---|---|
| GPU_platform_uuid, RVA 0x41b040 | RCX is hidden result pointer, RAX returns it; copies 16 bytes. Span has borrowed byte pointer at 0, signed 64-bit count at 8, size 16, alignment 8. Require count 16; copy bytes immediately; do not free. |
| pixel-buffer native handle, 0xd96d20 | RCX output, RDX pixel-buffer; wrapper invokes virtual method with this RCX/output RDX. Result has int64 handle at 0, size_t size at 8, size 16. |
| BPyGPUTexture | CreatePyObject and dealloc both use native pointer at +16. Require exact Python type (type object RVA 0x66ac390; PyObject type pointer +8), strong reference and exact profile before extraction. Width/height are not pointer validators. |
| VKFence | Allocation size 24; vtable 0, signalled byte 8, timeline uint64 at 16. signal submits active context and stores timeline. |
| Bounded Vulkan route | GPUBackend::get RVA 0x4123a0; VKBackend device member +16, VKDevice native VkDevice +16 and timeline semaphore +144. VKDevice::wait_for_timeline RVA 0x1c152b0 proves these operands but uses infinite timeout: never call it. vkWaitSemaphores thunk RVA 0x4c0da00, IAT RVA 0x5d4d570. Verify live backend vtable/profile, fence ownership and nonzero handles before query. |
| VkSemaphoreWaitInfo | sType 1000207004 at 0, pNext 8, flags 16, count 20, semaphore pointer 24, values pointer 32; size 40, alignment 8. One semaphore and fence's captured timeline value. |
| Secondary context | Return object size16 with GHOST pointer0/GPU pointer8; ABI is inspectable, but destruction calls device-idle and is rejected as the bounded cleanup route. |

UUID contract: compare all 16 Vulkan physical-device UUID bytes with CUDA device 0's
16 bytes. NVIDIA text UUID normalization strips optional `GPU-` and hyphens only; do not
use GUID little-endian conversion or compare names/PCI strings. Missing, malformed or
different values reject before any export/import. The earlier NVIDIA inventory UUID is
not evidence of the Blender native UUID value. Actual equality and zero-import runtime
rejection remain unrun.

## Synchronization

Plan C2 now specifies zero-timeout Vulkan queries, raw CUDA event result classification,
at most two queries per API per tick, 500 ms progress deadline and 100 ms shutdown budget.
Warp 1.17 staged `_src/context.py` Event.is_complete converts query results into a bool;
`native/warp.cu` wp_cuda_event_query returns cuEventQuery's CUresult. Preserve 0/success,
600/not-ready and every other error separately. A producer event follows the pack on its
explicit stream. No producer synchronize_stream or GPU_fence_wait enters the protocol.

The [Vulkan contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitSemaphores.html)
specifies immediate checking for timeout zero. This eliminates explicit infinite waits;
it does not guarantee a malfunctioning driver cannot stall. Aggregate ABI interpretation
also follows [Microsoft's x64 ABI](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170).

## Cleanup proof and remaining blocker

Independent read-only cleanup review used exact fbe6228777e7 sources:
`wm_window.cc` 558–573 unlinks the owner before cancel, then destroys it; `wm_event_system.cc`
2331–2348 calls cancel synchronously. Callback-scoped owner identity is therefore required,
not live-list membership inside cancel. Outside cancel, list membership is required.
`wm_window.cc` 1624–1661 supports activation of a still-live owner, followed by active-context
identity verification. Restrict all work to that one context.

Disable draws/publication before a final fence covering texture reads as well as uploads.
After positively completed CUDA and Vulkan work, release CUDA mapping/imports and handles,
then hand Blender resources to its deferred pool. `vk_resource_pool.hh` and
`vk_device_submission.cc` 78–85 establish deferred ownership; wrapper deletion is not
physical VRAM retirement. Do not force the pool or weaken numeric no-growth acceptance.

`GHOST_ContextVK.cc` 849–881 calls device-idle during secondary-context destruction.
`gpu_context.cc` 229–235 destroys the backend with the final context. Cancel returns void
and cannot delay destruction. Thus timeout/error while closing the final context has no
proved safe retention route. A Python object reference is insufficient. The next native
experiment requires a proven surviving backend lifetime or independently supervised
disposable-process containment before testing that edge. No runtime retry is justified
by this offline evidence alone.

| Gate | Result |
|---|---|
| Exact-build symbol/aggregate/wrapper/UUID access proof | PASS offline, scoped above; not a complete shipped profile |
| Bounded wait design and native query operands | PASS offline; live ordering/timing unrun |
| Healthy single-owner cleanup ordering | CONDITIONAL source-supported candidate; live teardown unrun |
| Timeout-safe final-context destruction | BLOCKED: no lifetime/containment proof |
| Actual Vulkan/CUDA UUID equality and zero-import rejection | NOT RUN |
| Scheduling 10,000 frames, 50 recreate cycles, physical retirement | NOT RUN |
| M0 exit / M1 entry | FAIL / BLOCKED |

Next action is an offline lifetime/containment design for the final-context timeout edge,
then a separately bounded isolated native probe after its safety preconditions are met.
Tracker remains inaccessible; update queued in `docs/TRACKER_UPDATE_PENDING.md`.
