# Vulkan zero-copy plan review — 2026-10-17

Verdict: revise before M1 implementation. The retained probe supports feasibility on one build; it does not yet establish safe production transport. M0 remains useful after its safety and acceptance contracts are tightened.

Scope: read-only assessment of `docs/VULKAN_ZERO_COPY_PLAN.md`, its retained probe and risk notes, release/product records, selected live source, and primary upstream documentation. Reviewed working tree at HEAD `617b4eabfb24c89618122523eff3f057d3e3f58a`; the plan and research directory are untracked and the checkout contains unrelated changes. No implementation, GPU runs, tests, or acceptance claims were made. Only this review and the required tracker queue were written.

## Findings

### 1. P1 — Symbol matching does not establish the complete native ABI

Plan lines 76 and 99–119 promise safe native calls after binary/symbol identification, but omit how a Python GPUTexture becomes a native Texture pointer. The actual probe (`research_notes/Vulkan zero copy/vk_zero_copy_probe.py:86–87`) reads `id(pytex) + 16`. `GPU_texture_width/height` then dereference that pointer; they cannot safely validate an incorrect layout. A matching PDB and decorated function names do not prove Python wrapper layout, enum values, or returned structure layouts. A new build can preserve those function names while changing an internal structure.

Require an explicitly verified ABI profile for each supported build/platform, covering texture extraction, ctypes calling conventions, structure sizes/offsets, and enum values. Reject an unknown profile before any native call. Test rejection without invoking native functions. See upstream [BPyGPUTexture layout](https://raw.githubusercontent.com/blender/blender/main/source/blender/python/gpu/gpu_py_texture.hh).

### 2. P1 — Deferred close lacks a guaranteed cleanup owner

Plan lines 137–140 defer destruction to the next draw callback while promising safe unregister/load_pre. Unregister is also required to remove that callback (release-plan P0.1); closing the last viewport can likewise prevent another draw. Merely copying `viewport_draw.py:599–610` does not guarantee cleanup will run. The plan needs to specify who retains resources and drains pending cleanup before callback/context loss, including partial initialization and repeated close.

Define teardown ordering and a guaranteed context-valid cleanup path. Add acceptance for disable immediately after startup, disable while a slot is in flight, last-view closure, file load, and failed setup after each allocation. Do not count scheduled cleanup as completed cleanup.

### 3. P1 — “Any CUDA error → Readback” is not a valid recovery contract

Plan lines 144–145 require fallback for every CUDA failure. Illegal-address and launch failures can leave CUDA unusable for subsequent work; the solver and Readback still depend on that device/context. Re-entering CUDA cleanup or publishing can repeat the error rather than recover.

Classify recoverable interop failures separately from fatal CUDA failures. Permit Readback only while the CUDA context remains usable; fatal errors must stop the session, retain diagnostics, and explain the restart requirement. Add both classes to fault-injection acceptance. NVIDIA documents the inconsistent-state/restart behavior in its [CUDA Runtime API](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_Runtime_API.pdf).

### 4. P1 — Device compatibility is checked too late

The Risks table treats mismatched Vulkan/CUDA devices as something to discover through an import error or self-test mismatch. M2 has no explicit device identity check before import. NVIDIA requires external memory and synchronization objects to be imported/mapped on the same device that created them; an error or mismatched pattern is not the specified identity check.

Add a Vulkan/CUDA UUID comparison before import and bind allocation, Warp execution, and synchronization to that verified device/context. If Blender's device UUID cannot be obtained safely for a supported build, return Unavailable. Add a mismatched-device rejection test. Source: [NVIDIA graphics interoperability requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html).

### 5. P2 — Cached addresses need a pre-call validation contract

V7 and M1 lines 108–109 say addresses are cached, but do not specify module-relative RVAs versus process addresses, range validation, or validation of corrupted entries before invocation. Reusing absolute addresses across ASLR changes is unsafe. Deleting a cache after a self-test fails is too late if the first function call crashes. M5 explicitly expects corrupt-cache injection to recover without a crash.

Specify cached RVAs, rebasing against the current module, exact identity including PDB age, executable-section bounds, schema/type checks, and trustworthy symbol revalidation on invalid data. A syntactically valid in-range but incorrect RVA must also be rejected or independently re-resolved; bounds alone are insufficient. Test corruption and fresh-process restarts before native invocation.

### 6. P2 — Existing evidence and planned measurements leave acceptance gaps

The retained `vk_probe.json` reports Warp 1.16.0, whereas the project pins 1.17.0. The script checks sampled positions; its 1.55 ms interval begins after the Warp launch and CUDA synchronization. It measures copy plus finish, not the entire proposed pack/wait/publish/upload path. M0's per-frame shader readback may also serialize the queue and hide the steady-state scheduling problem being tested.

Require the pinned runtime in M0/M5, define transport timing boundaries and percentile criteria, and separately measure completed GPU work and CPU submission time. Keep a no-readback steady-state stress lane distinct from correctness readback. Include non-row-aligned counts such as 4097: the plan must allocate/copy the padded texture extent safely and draw exactly N particles. Add explicit stream identity/order and slot-state cases for timer-before-first-draw, multiple timer ticks per draw, and multiple draws per tick.

## Additional prerequisite

The supplied AGENTS.md mandates OpenGL and forbids per-frame CPU readback, while the release plan introduces Vulkan and Readback. Reconcile the authoritative project guidance before implementing the fallback; the plan alone should not silently override those invariants.

## Evidence limits and next action

The existing probe is useful evidence of sampled write/read and shader sampling on Blender build `fbe6228777e7`. No new runtime or performance evidence was generated in this review. Its results must not be expanded into crash-free operation, 1.17 compatibility, lifecycle acceptance, or a full transport performance result.

Next action: revise the contracts above, then run bounded M0 probes against an exact isolated Blender/Warp candidate. Implementation and execution are outside this review request. The canonical tracker was attempted but returned no readable content, and no Page read/update connector was available; synchronization is pending in `docs/TRACKER_UPDATE_PENDING.md`.
