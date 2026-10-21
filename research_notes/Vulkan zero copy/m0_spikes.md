# M0 bounded probes — 2026-10-17

**Result: M0 exit gate NOT PASSED; M1 not started.** Public shader/runtime prerequisites
passed. The proposed direct fence-wait path fails the bounded-wait requirement by exact-build
source inspection. Native interop tests remain blocked by incomplete ABI/device/cleanup proof.
These are distinct from a GPU crash or an observed transport corruption: neither occurred.

## Candidate and ownership

- Repository HEAD: `617b4eabfb24c89618122523eff3f057d3e3f58a`, with pre-existing dirty/untracked
  work preserved. Package source hashes are frozen in the host receipt and compared at exit.
- Blender 5.2.0 LTS build `fbe6228777e7`, Windows x64; executable and PDB SHA-256 in
  `evidence/vulkan-zero-copy/20261017-m0/host-preflight.json`.
- Matching executable RSDS/PDB GUID `d357c018-4455-89ed-4c4c-44205044422e`, age 1, verified
  by reading the PE debug directory and MSF PDB information stream independently.
- Warp 1.17.0, isolated staged import root; 570 non-cache files hash-compared with the
  installed source. `warp.dll` SHA-256: `7a5ce6e5540a7d7e07004221c742f4e4cd7db1f09a076407bde4a1bb6240ea20`.
  This is staged runtime evidence, not extension installation or locked-wheel qualification.
- RTX 4080 SUPER, driver 610.88, CUDA UUID `GPU-5d09f77b-a936-a6d9-054d-0324aef67adb`.
- Isolated MCP Vulkan PID 76576 and OpenGL PID 28384, sequentially, port 9899. Both passed
  bundled status/file/scene/execute gates and dependency identity/stdio checks. Both were
  closed after probing; their persistent logs and launch receipts are retained.
- Nine pre-existing Blender sessions belonged to another project; no commands were sent to
  them and none were closed. Initial VRAM was 8739/16376 MiB. This was not a clean benchmark
  or leak-measurement environment. No profile save, startup save or product add-on enable
  occurred. Sessions used isolated launcher app templates, not a regular-profile reset.

## Gate results

| M0 gate | Status | Actual evidence and limit |
|---|---|---|
| Exact candidate/runtime/isolation | PARTIAL PASS | Binary/PDB identity and runtime pin/import/stream checks pass; complete native ABI profile is not established. See host-preflight, offline-identity, launch and public receipts. |
| Texture-pointer ABI and safe native calls | BLOCKED, NOT RUN | Historical `id(texture)+16` and raw RVAs are insufficient. No pointer extraction or Blender native call was made. Source headers were captured at the exact commit; full layout/return-ABI verification remains open. |
| Vulkan/CUDA UUID equality before import | BLOCKED, NOT RUN | CUDA UUID read successfully. Public Blender API returns `10de/2702/0`, not a UUID. Offline PDB discovery found `GPU_platform_uuid`, but it returns C++ `Span<uint8_t>`; that return ABI is unverified. Zero external-memory imports. No claim that no native route exists. |
| Resolve fence functions | PASS, DISCOVERY ONLY | All four names found in matched PDB; undecorated listing is not signature/layout qualification. `symbol-discovery.txt`. |
| Bounded wait/reuse policy | FAIL, SOURCE CONTRACT | Exact-build `VKFence::wait` calls `VKDevice::wait_for_timeline`, which uses `vkWaitSemaphores(..., UINT64_MAX)`. Direct wait has no finite timeout and logs errors without returning a status to the caller. A verified completion query/finite-time alternative is required before calling this in the modal path. This is not an observed hang. |
| Synchronous cleanup on unregister/load/last-view closure | BLOCKED, NOT RUN | No verified independent context route/ownership profile established; closing a public offscreen is not proof for imported buffers/fences. Context symbols were found but not called. |
| Instanced quads + texelFetch, both backends | PASS, BOUNDED | Six-vertex batch, gpu_InstanceIndex, public RGBA32F textures; counts 0, 1, 4095, 4096, 4097, 4194304. All nonzero rows compare every RGBA component including padding: zero mismatches on both backends. This uses CPU-created textures, not CUDA external memory. |
| Invalid sizes/device-limit rejection | NOT RUN | No native allocator/transport exists under this M0 scope. Positive padded-tail rendering passed; it does not prove transport allocation rejection. |
| 10,000-frame no-readback double-buffer lane | BLOCKED, NOT RUN | Depends on approved ABI/UUID and bounded synchronization. The correctness probe deliberately reads back and is not a substitute. |
| 50 imported-buffer recreate/leak cycles and partial-allocation failures | BLOCKED, NOT RUN | Depends on approved native resource/cleanup route; other live sessions also prevent clean global VRAM attribution. |
| Nonblocking completed-transport timing | BLOCKED, NOT RUN | No qualified transport/completion instrumentation. Diagnostic draw-plus-readback samples are explicitly not performance acceptance. |

All GPU operations used public Blender GPU APIs or Warp's own API. `native_calls=0` in
receipts specifically means no directly invoked internal Blender functions, not that public
GPU operations avoid native code internally. No production fallback or rejection tests were
run: refusing to call unverified addresses is probe safety, not proof of a future resolver.

## Preserved failure and correction

Attempt 1 (PID 69012) failed bootstrap with `No module named 'warp._src'`: the dependency
copy was still running when launch began. The copy subsequently reported a __pycache__
collision caused by that overlapping import. No GPU probe ran. The process exited.
`launch-attempt1-error.json` and `launch-attempt1.log` preserve the error. After completion,
all 570 non-cache files matched the source; exactly one corrected Vulkan startup was made.
There were no retries of GPU probes or changes to the measured public shader harness.
Workflow correction: dependency staging must finish and hash comparison pass before launch.

## Reproducibility and evidence

Evidence root: `evidence/vulkan-zero-copy/20261017-m0/`.
Scripts: `m0_preflight.py`, `m0_offline.py`, `m0_public_probe.py`, `m0_verify.py` and
`m0_launch.yaml` beside this document. `manifest.json` records hashes of receipts, scripts
and captured source. `verification.json` audits results, runtime copy, package-source
preservation, isolation paths and owned-process exit. These scripts are scratch probes only.

Public receipts: `public-vulkan.json` and `public-opengl.json`; each contains full candidate,
runtime paths, explicit-stream known-answer status, every tested count and mismatch total.
The four million row compares 16,777,216 float components on each backend. Reported
draw-plus-readback times include synchronization/readback and do not measure the ≤2 ms gate.

Exact-build primary sources are retained under `upstream-fbe6228777e7/` with URLs/hashes in
`offline-identity.json`. Relevant upstream references:

- [Fence wait](https://raw.githubusercontent.com/blender/blender/fbe6228777e7/source/blender/gpu/vulkan/vk_fence.cc)
- [Submission wait implementation](https://raw.githubusercontent.com/blender/blender/fbe6228777e7/source/blender/gpu/vulkan/vk_device_submission.cc)
- [UUID return contract](https://raw.githubusercontent.com/blender/blender/fbe6228777e7/source/blender/gpu/GPU_platform.hh)
- [Public device API](https://raw.githubusercontent.com/blender/blender/fbe6228777e7/source/blender/python/gpu/gpu_py_platform.cc)

## Stop decision and exact resume work

Stop before M1 as requested. Do not mark the combined M0 renderer/transport milestone passed
from public rendering alone. The bounded-wait source finding requires a plan decision before
any native runtime continuation: choose and verify a completion-query/finite-wait mechanism,
then finish the exact-build Span/texture/handle ABI profile and synchronous teardown route.
Resume with those offline M0 prerequisites; only after they pass may a separately bounded
native probe import external memory. No further GUI/GPU run is queued or running.
Tracker update remains pending because the linked Page returned no readable content.
