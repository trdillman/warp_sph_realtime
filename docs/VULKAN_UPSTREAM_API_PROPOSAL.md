# DRAFT — Blender `gpu` API proposal: exportable pixel buffers and GPU fences

Status: draft for owner review (plan M7). **Not posted.** External posting to
projects.blender.org / devtalk requires owner approval.

## Problem
Python add-ons that simulate on CUDA (NVIDIA Warp, PyTorch, CuPy) can only show results in the
viewport through a CPU round trip: device → host copy, `gpu.types.Buffer`, `GPUTexture(data=)`.
At 4 M particles that is 64 MiB per frame through system memory. Blender's Vulkan backend
already has the pieces to avoid it (`GPUPixelBuffer` with an exportable memory handle,
`GPU_texture_update_sub_from_pixel_buffer`, timeline-semaphore `GPUFence`), but none is exposed
to Python. This add-on currently reaches them through an exact-build symbol profile (one
reviewed Blender build, SHA-pinned, Readback fallback everywhere else). A public API removes
that maintenance burden and makes the capability available to every build and platform.

## Proposed Python surface (minimal)
```python
pb = gpu.types.GPUPixelBuffer(size_bytes)          # device-local, exportable
handle, size = pb.export_handle()                   # Win32 HANDLE (NT) / Linux opaque fd; caller owns it
uuid = gpu.platform.device_uuid_get()               # 16 bytes, to match the CUDA device
tex.update_from_pixel_buffer(pb, width, height, format='RGBA32F')
fence = gpu.types.GPUFence()                        # timeline value on the backend queue
fence.signal()                                      # after commands that read pb/tex
fence.query() -> bool                               # non-blocking; never an infinite wait
pb.free(); fence.free()                             # explicit, also on GC
```
Rules: all calls on the main thread with a valid GPU context (draw callbacks, timers, operators);
`export_handle` duplicates the handle so the caller closes it; `query` is the only completion
primitive (no blocking wait exposed); unsupported backends raise `NotImplementedError` so add-ons
fall back cleanly. OpenGL can implement the same surface with `GL_EXT_memory_object` where
available, or report unsupported.

## Why this shape
- Matches what the qualified internal route uses (two pixel buffers, two fences, one texture),
  so the add-on can switch adapters behind its existing transport interface without behaviour
  change.
- Keeps synchronization explicit and bounded: producers use their own API (cuEvent) on the
  imported memory; the fence covers Blender's read so the buffer can be reused.
- No new ownership model inside Blender: the pixel buffer stays a Blender object; the external
  API imports a duplicate handle and must release its import before `pb.free()`.

## Evidence to attach (after owner approval)
M0/M5 results from `evidence/vulkan-zero-copy/` — parity, lifecycle cycles, latency at
4 194 304 particles on Blender 5.2.0 / RTX 4080 SUPER — with the caveat that they come from one
build and one GPU.

## Open questions for Blender developers
1. Is `GPUPixelBuffer` export acceptable as public API, or should export live on `GPUTexture`?
2. Timeline semaphore export (for CUDA-side waits) vs. Blender-side fence query only.
3. macOS/Metal equivalent (IOSurface / MTLSharedEvent) — out of scope for this draft.
