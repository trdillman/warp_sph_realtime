# Open risks: Blender CUDA->Vulkan pixel-buffer zero copy (research, read-only)

Sources: Blender main shallow clone (HEAD 3fe0f2ab, 2026-10-17), raw files from
projects.blender.org branches blender-v5.2-release / blender-v5.3-release / main,
official blender-5.2.0-linux-x64.tar.xz (sha256 96f6c181...1c48, verified against
mirror .sha256; mirror.clarkson.edu because download.blender.org is behind a Cloudflare challenge).
Scratch tools (not in repo): scratchpad/elfscan.py, elfscan2.py, anchor.py.

## 1. LINUX symbols  (confidence: high for the facts, medium for the resolver design)

Facts (ELF inspection of `blender-5.2.0-linux-x64/blender`, 166 MB):
- ET_EXEC (type 2), x86-64, NON-PIE: code lives at fixed absolute VAs (.text starts 0x4b74c0). No ASLR slide for the main binary.
- Stripped: NO `.symtab` (count 0), no `.debug_*`, no `.gnu_debuglink`; only `.note.gnu.build-id`.
- `.dynsym` has 4902 entries, 1483 defined functions: 1027 `Py*` (static CPython exports), 446 plain C, 10 other C++.
  ZERO `blender::` / `GPU_*` symbols. So none of GPU_pixel_buffer_create / GPU_pixel_buffer_get_native_handle /
  GPU_texture_update_sub_from_pixel_buffer / GPU_finish can be dlsym()'d.
- Tarball contains no debug-symbol/.dbg/.sym file (tar -t grep). No separate symbols download found for Linux.
  (Windows is different: blender.pdb ships.)
- RTTI is present (typeinfo name strings), and the CLOG strings survive in .rodata.

Prototype proving a signature resolver is feasible (5.2.0 Linux):
- String "Failed to get file descriptor for Vulkan pixel buffer" @0x80120f0; one RIP-relative `lea` xref @0x153b156;
  enclosing FDE (from .eh_frame_hdr table, 253,304 FDEs) start = 0x153b0a0 = VKPixelBuffer::get_native_handle.
- RTTI route cross-checks it: typeinfo name `N7blender3gpu13VKPixelBufferE` -> typeinfo 0xa816c68 -> vtable 0xa816c90;
  vtable slot 4 = 0x153b0a0 (same function). Slot 2/3 = map/unmap-ish, slot 5 = get_size.
- Same trick found vtables for VKBackend (0xa816828) and VKTexture (0xa8171f0).

Recommended resolver (Linux), most to least robust:
 1. Resolve virtual-method entries, not the GPU_* wrappers: RTTI name -> typeinfo -> vtable -> slot. get_native_handle = VKPixelBuffer vtable[4]
    (cross-check by string xref). Call it as a C++ method (this=PixelBuffer*, SysV returns the 16-byte {int64,size_t} in RAX:RDX; works with a ctypes Structure restype).
    Slot indices are ABI-derived from the class declaration order (PixelBuffer base: map, unmap, get_native_handle, get_size) so they only change if upstream reorders virtuals.
 2. Allocation: VKBackend::pixelbuf_alloc (VKBackend vtable slot, found as the function that references the VKPixelBuffer vtable). Its `this` is unused (`new VKPixelBuffer(size)`) - verify by disassembly (capstone) before relying on a dummy `this`; GPU_pixel_buffer_create additionally rounds size up to 256 (gpu_texture.cc:1035-1044), so do that rounding yourself.
 3. update_sub_from_pixel_buffer: VKTexture::update_sub(offset,extent,format,PixelBuffer*) virtual (gpu_texture.cc:523-537 shows it is a direct virtual call). Pick the VKTexture vtable slot whose body takes the PixelBuffer overload (disassembly check: it calls the shared 8-arg helper with a non-null pixel_buffer); this is the weakest link.
 4. Sync without GPU_finish: cuStreamSynchronize + GPU fence (see Q2) or find VKContext::finish (flags SUBMIT|WAIT_FOR_COMPLETION).
 5. Always: key everything by build-id (.note.gnu.build-id) + sha256 of the binary. Ship a precomputed offset table for each official release (5.2.0, 5.2.x ...) and use the scanner only as a fallback with self-test (create a 256-byte pixel buffer, get handle, import, write, update a 1x1 texture, read back) before enabling. Non-PIE makes table offsets stable per build.
 Not recommended: Python-C-API reachability (no Python type exposes these pointers; Py* exports are only CPython).
 Feasibility: medium-high. Needs ~200 lines (ELF + eh_frame_hdr + capstone). Unverified: actual calls on Linux, NVIDIA Linux VK external-memory OPAQUE_FD behaviour with CUDA, and 5.2.1/5.2.2 byte layouts.
Linux-specific semantics (gpu/GPU_texture.hh:1380-1393, vk_pixel_buffer.cc:96-112): handle is an OPAQUE_FD; "caller is responsible for closing"; cuImportExternalMemory takes ownership of the fd on Linux (cycles graphics_interop.cpp:71-73) but NOT of the Win32 handle (graphics_interop.cpp:65-69, CloseHandle at :173-177). Close the NT handle on Windows after cuDestroyExternalMemory.

## 2. SYNC  (confidence: high, from source; Vulkan-side ordering medium)

Cycles does NOT use any GPU-GPU semaphore between CUDA and Vulkan:
- graphics_interop.cpp (cuda): Vulkan buffer imported once with cuImportExternalMemory+cuExternalMemoryGetMappedBuffer (:77-97); `map()` just returns the cached pointer (:131-134); `unmap()` is a NO-OP for Vulkan (:145-153, only the GL path calls cuGraphicsUnmapResources). No cuImportExternalSemaphore anywhere.
- integrator/pass_accessor_gpu.cpp:77 `queue_->synchronize()` (host cuStreamSynchronize) right after the film-convert kernel, before update_end().
- display_driver.cpp (main): `update_end()` -> update_tile_texture_pixels() records GPU_texture_update_sub_from_pixel_buffer (:480-497, called :523), then `GPU_fence_signal(gpu_upload_sync_)` (:527) then `GPU_flush()` (:529).
- Reuse of the same pixel buffer: update_begin() `GPU_fence_wait(gpu_upload_sync_)` + `GPU_fence_wait(gpu_render_sync_)` (:431-432) BEFORE CUDA writes again; comment :429-430 says exactly this ("wait for the previous PBO-to-texture copy before reusing the PBO"). draw() also signals gpu_render_sync_ (:823).
- Vulkan side: update_sub with a pixel buffer only adds a VKCopyBufferToImageNode to the render graph (vk_texture.cc:510-532); nothing is submitted until a flush. VKFence::signal() = context.flush_render_graph(SUBMIT|RENEW) and stores the timeline value (vk_fence.cc); VKFence::wait() = vkWaitSemaphores on the device timeline (CPU wait, vk_device_submission.cc:99-111). GPU_flush() = VKContext::flush() = SUBMIT|RENEW without wait (vk_context.cc:162-167). GPU_finish() = SUBMIT|WAIT_FOR_COMPLETION (vk_context.cc:212-216).
- The copy reads the buffer at GPU EXECUTION time, not at record time. So until the fence signals, CUDA must not overwrite that buffer.

Minimal correct per-frame protocol (inferred from the above; not yet tested here):
  1. before CUDA touches buffer[i]: GPU_fence_wait(fence[i]) (no-op if never signalled)
  2. Warp/CUDA kernel writes buffer[i]; cuStreamSynchronize (or cudaEventSynchronize on the write event). This host sync is REQUIRED - nothing else orders CUDA before Vulkan.
  3. on the Blender GPU context thread: GPU_texture_update_sub_from_pixel_buffer(tex, ..., buffer[i], ...)
  4. GPU_fence_signal(fence[i])  (this also submits; GPU_flush() afterwards is redundant but harmless)
Double buffering (2 pixel buffers, 2 fences) is sufficient and avoids GPU_finish: step 1 waits on the copy from 2 frames ago, which is normally already complete (no stall). With a single buffer the fence wait in step 1 still replaces GPU_finish and is cheaper than a full finish. GPU_finish is a correct but heavier superset (your 1.55 ms figure includes it). A single buffer WITHOUT any fence/finish is a data race (copy may execute after the next CUDA write). Without a flush/fence the copy sits in the render graph until the next natural submit (frame swap), which is not a correctness bug for ordering but delays visibility and widens the race window for buffer reuse.
Caveats: fence signal must run on a thread with the Blender GPU context active (VKFence::signal uses VKContext::get()); wait is context-free. A Vulkan memory barrier for the external write is not emitted by Blender; host-side completion of the CUDA work (step 2) is what makes it visible on the same device. Unverified: cross-API cache-coherence on non-NVIDIA/multi-GPU (same-device only is assumed; Cycles checks device UUID via graphics_interop_get_device, display_driver.cpp:580-605).

Gotchas found in vk_pixel_buffer.cc:
- get_native_handle() calls create(true) which FREES and reallocates the buffer if it was previously created as non-export (e.g. after GPU_pixel_buffer_map, which does create(false)). And map() after export frees the exported buffer again (vk_pixel_buffer.cc:21-44). Never call map/unmap on a buffer you export; get the handle once and cache it. Each get_native_handle call returns a NEW OS handle to the same memory.
- Allocation failure crash: upstream issue #164095 (open, 5.2.1, Windows, RTX 5090): intermittent null-deref `vmaGetAllocationInfo <- VKBuffer::export_memory_get <- VKPixelBuffer::get_native_handle` in Cycles interop. Treat a failed/huge allocation as fatal for your path: validate size against device limits (maxBufferSize) beforehand and retry-guard. https://projects.blender.org/blender/blender/issues/164095

## 3. STABILITY 5.2 -> 5.3 -> main  (confidence: high on public API, medium on internals)

Diffed blender-v5.2-release vs blender-v5.3-release vs main (raw files):
- Signatures UNCHANGED: GPU_pixel_buffer_create(size_t), GPU_pixel_buffer_get_native_handle, struct GPUPixelBufferNativeHandle{int64 handle; size_t size}, GPU_texture_update_sub_from_pixel_buffer(gpu::Texture*, eGPUDataFormat, GPUPixelBuffer*, int x6) - identical in all three (GPU_texture.hh 5.2:1374-1382, 5.3/main:1401-1409). Still in namespace blender, so Itanium/MSVC mangling is unchanged. GPU_fence_* unchanged in GPU_state.hh (main:246-249).
- 5.3 branch and main are identical for all pixel-buffer files I diffed (5.3 header diff adds only unrelated mipmap/compressed-format functions).
- Internals that did change (5.2 -> 5.3): VKPixelBuffer calls vkGetMemoryWin32HandleKHR/vkGetMemoryFdKHR (renamed pointers, same behaviour); VKBuffer now holds VKResourceWithHandle<VkBuffer> and the copy node uses buffer.resource() (render-graph resource tracking, vk_texture.cc 5.3:~511), VKBuffer::create gains `alignment` and rounds alloc size up to 16; display_driver.cpp gains a reset()/update_pending_ handshake. Handle ownership unchanged (caller closes). vk_fence.cc, vk_memory_pool.cc, cuda graphics_interop.cpp: no diff.
- Implication: same mangled names/signatures should work on 5.3, but RVAs/offsets change per build (Windows: re-resolve from the pdb; Linux: rescan). Keep a startup self-test.
- Planned/related changes: none found that alter this API. Related open work: PR #165103 (vertex-buffer staging), issue #141555 "Vulkan: Better way to upload data to the GPU", PR #153686 (chunked staging). None touch VKPixelBuffer. Search was by keyword only; absence of evidence, not proof.

## 4. AVOIDING THE COPY  (confidence: high)

No internal path exists today:
- Only two things are ever created exportable: VKPixelBuffer (vk_pixel_buffer.cc: create(memory_export) -> VKBuffer::create(..., export_memory=true), which sets VkExternalMemoryBufferCreateInfo, DEDICATED_MEMORY and the `external_memory_pixel_buffer` VMA pool, vk_buffer.cc:87-94) and VKTexture images created with GPU_TEXTURE_USAGE_MEMORY_EXPORT (vk_texture.cc:759-763, VKTexture::export_memory :555, used only by the OpenXR path vk_context.cc:604-621).
- VKVertexBuffer::wrap_handle is `NOT_YET_IMPLEMENTED` (vk_vertex_buffer.cc:74-77) and VKVertBuf::allocate passes export_memory=false (:243-250). VKStorageBuffer::allocate passes export_memory=false (vk_storage_buffer.cc:86-93). No GPU_storagebuf_*/GPU_vertbuf_* function returns a native handle; GPU_vertbuf_wrap_handle (GPU_vertex_buffer.hh:359) only imports a GL/Metal id.
- GPU_TEXTURE_USAGE_GENERAL explicitly EXCLUDES MEMORY_EXPORT (GPU_texture.hh:~776-781) and Python's gpu.types.GPUTexture always uses GENERAL (gpu_py_texture.cc:362), so a Python-created texture cannot be exported even with its pointer.
- Hence the pixel buffer + vkCmdCopyBufferToImage (your measured 1.55 ms/64 MiB) is the only supported route. Texture-image export (CUDA mipmapped array over an optimal-tiling image) would need a texture created by C++ with MEMORY_EXPORT; not reachable from Python without more hacking.
- Possible but hacky (unverified, not recommended): call VKBuffer::create(export=true) directly on a VKStorageBuffer's `buffer_` member via raw offsets - fragile ABI poking and would break ownership/discard handling.
- If you want the vertex data path to go through the buffer: sample the texture in the vertex/fragment shader (texelFetch), as the working probe already does ("verify_shader_sample"), so the pixel-buffer->texture copy is the only extra step.

## 5. UPSTREAM / claim check  (confidence: medium - keyword search of projects.blender.org + devtalk only)

No Blender PR, design task or devtalk thread proposes a Python API for pixel buffers, external memory or CUDA/Warp interop. Related prior art only:
- Cycles' own interop (display_driver.cpp / cuda graphics_interop.cpp) - the only consumer.
- #164095 interop crash (above) https://projects.blender.org/blender/blender/issues/164095
- #145179 "pyGPU: Add support for SSBO" (open tracking) https://projects.blender.org/blender/blender/issues/145179 ; PR #161447 (SSBO, closed; #164034 says it is merged in 5.3 alpha) https://projects.blender.org/blender/blender/pulls/161447 ; #164034 storage_buf API inconsistencies https://projects.blender.org/blender/blender/issues/164034
- PR #161912 Python: GPUTexture.update for RGBA8 byte buffers (open) https://projects.blender.org/blender/blender/pulls/161912 (and #160528 WIP)
- PR #146505 WIP deform render-cache VBOs with Python compute shaders (open) https://projects.blender.org/blender/blender/pulls/146505
- PR #163332 WIP GPU Pipeline API https://projects.blender.org/blender/blender/pulls/163332
- #141555 Vulkan: Better way to upload data to the GPU https://projects.blender.org/blender/blender/issues/141555
- devtalk: GPU-module feedback thread https://devtalk.blender.org/t/suggestions-feedback-on-the-extensions-for-the-gpu-module/17706 (no interop proposal found); search hits for "cuda interop"/"external memory" returned only unrelated Blender Today / meeting posts.
Recommendation: file nothing yet; if you want upstream support, a devtalk/design post proposing `gpu.types.GPUPixelBuffer` is net-new.

Claim checks:
- "PR 165103 in Blender 5.3 adds asynchronous vertex buffer uploads via dedicated transfer queues": FALSE as stated. #165103 (HooglyBoogly, 2026-10-13) is OPEN, base=main, no milestone, titled "Vulkan: Skip CPU staging buffer copy when uploading vertex buffers". Its description says one fewer copy step and that "the final upload happens asynchronously"; the words "transfer"/"queue" do not occur in the description and its 14 changed files (vk_vertex_buffer, vk_staging_buffer, draw extractors) contain no queue-family work as far as the file list shows (diff not read line by line). Not merged, not in 5.3. https://projects.blender.org/blender/blender/pulls/165103
- "blender-web uses GHOST_ContextVK virtual swapchains with NVENC": UNVERIFIED / no evidence. No "blender-web" repo in blender/*; no source hit for NVENC or "virtual swapchain" in the main clone; issue/PR searches only show unrelated items (NVENC: ImBuf hardware video encoding PRs #165032, #163453 which are about the VSE, not GHOST). Treat as false until a source is given.
- "gpu.capabilities.ray_query_support_get() exists": TRUE on main/5.3 (gpu_py_capabilities.cc:388-403, registered :504), FALSE on blender-v5.2-release (0 occurrences in that file), so it is absent in 5.2.0.

## Recommended actions
1. Linux: implement the RTTI/vtable + string-anchor resolver with per-build-id offset table and a runtime self-test; do not rely on dynsym.
2. Sync: adopt wait-fence(i) -> write -> cuStreamSynchronize -> update_sub -> signal-fence(i) with 2 buffers; drop GPU_finish.
3. Never call GPU_pixel_buffer_map on an exported buffer; cache the handle; close the Win32 handle yourself; guard against allocation failure (#164095).
4. Re-run the Windows pdb resolution on 5.3; signatures are unchanged.
