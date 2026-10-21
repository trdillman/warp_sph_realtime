"""VulkanInterop position transport for the exact profiled Windows build (plan C2/C3, M2).

Warp packs positions into Blender-owned pixel buffers imported into CUDA; the newest READY
pixel buffer is copied into one RGBA32F texture. No per-frame CPU readback, but a GPU copy
remains. Two slots move FREE -> WRITING -> READY -> IN_FLIGHT -> FREE; each public call
makes at most one CUDA event query or one zero-timeout Vulkan query per slot, except that
publish polls its own pack event for up to PRODUCER_WAIT, and nothing else sleeps outside
the bounded setup/shutdown budgets.

The timer submits each upload on its own (DRW scope, then GPU_fence_signal, which flushes
only that copy), so a slot fence covers its upload rather than the next viewport frame.
When the producer is slower than PRODUCER_WAIT or the scope route is unavailable (e.g. a
second window), the draw uploads instead and the fence lands in that frame's submission.

Activation and close run from the window context (operator, timer, unregister, load_pre)
inside Blender's DRW_gpu_context_enable/disable scope qualified in M0; draws are accepted
only in the owner window's captured DRW context. Setup errors on the recoverable allowlist
leave CUDA usable and raise RecoverableTransportError after a complete rollback. Every other
CUDA/Vulkan error, timeout or context-restoration failure raises FatalTransportError: the
transport stops, makes no further calls and keeps all objects referenced until exit.
"""
from __future__ import annotations

import ctypes as c
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import warp as wp

from .transport import (
    COMPLETED, FAILED, MAX_COUNT, QUARANTINED, TEXEL_BYTES, Capabilities, FatalTransportError, Frame,
    RecoverableTransportError, dimensions, process_state,
)

FREE, WRITING, READY, IN_FLIGHT = 'FREE', 'WRITING', 'READY', 'IN_FLIGHT'
CUDA_NOT_READY = 600
VK_TIMEOUT = 2
# Setup-phase CUresults that leave the context usable: INVALID_VALUE, OUT_OF_MEMORY,
# OPERATING_SYSTEM, INVALID_HANDLE, NOT_SUPPORTED. Everything else is fatal.
RECOVERABLE_SETUP = frozenset({1, 2, 304, 400, 801})
_POP_SAFE = RECOVERABLE_SETUP | {0, CUDA_NOT_READY}  # CUresults after which the CUDA context is still usable
PENDING_DEADLINE = 0.5   # seconds a slot may stay WRITING/IN_FLIGHT before quarantine
SETUP_BUDGET = 0.1       # self-test producer/consumer completion
PRODUCER_WAIT = 0.002    # publish waits this long for its pack before leaving the upload to the draw
CLOSE_BUDGET = 0.1       # whole shutdown polling budget
SELF_TEST = (4, 4)       # 16 texels = 256-byte known-answer transfer gates activation
VRAM_RESERVE = 512 << 20

_QUARANTINE: list = process_state().quarantine  # never freed; survives add-on reload until exit
_generation = 0


@wp.kernel
def _pack(src: wp.array[wp.vec3], dst: wp.array[wp.vec4], n: int):
    i = wp.tid()
    if i < n:
        p = src[i]
        dst[i] = wp.vec4(p[0], p[1], p[2], 1.0)
    else:
        dst[i] = wp.vec4(0.0)


@wp.kernel
def _known(dst: wp.array[wp.vec4], n: int):
    i = wp.tid()
    if i < n:
        dst[i] = wp.vec4(float(i) * 0.5 + 1.0, 3.0, float(i % 7), 1.0)
    else:
        dst[i] = wp.vec4(0.0)


def known_answer(texels: int) -> np.ndarray:
    ids = np.arange(texels, dtype=np.float32)
    return np.stack((ids * 0.5 + 1.0, np.full_like(ids, 3.0), ids % 7, np.ones_like(ids)), axis=1)


class _RouteUnavailable(RecoverableTransportError):
    """The qualified window-context -> DRW cleanup route is not available here."""


class _Slot:
    __slots__ = ('size', 'texels', 'pb', 'ext', 'ptr', 'array', 'event', 'fence', 'state', 'frame', 'started')
    pb: int | None
    ext: int | None
    ptr: int | None
    array: Any
    event: int | None
    fence: int | None

    def __init__(self, texels: int):
        self.texels, self.size = texels, texels * TEXEL_BYTES
        self.pb = self.ext = self.ptr = self.array = self.event = self.fence = None
        self.state, self.frame, self.started = FREE, -1, 0.0


class VulkanInteropTransport:
    """Two pixel buffers, two fences, one texture. Construct from the window context."""

    def __init__(self, native, ops, count: int, *, window: int, clock=time.monotonic, sleep=time.sleep):
        global _generation
        self.count = count
        self.width, self.height = dimensions(count)  # ValueError before any native call
        self._n, self._ops, self._window = native, ops, window
        self._clock, self._sleep = clock, sleep
        self._slots: list[_Slot] = []
        self._texture = None
        self._test_texture = None  # held until the self-test's GPU work is known complete
        self._tp = 0
        self._drw_ctx = 0
        self._fatal = ''
        self._closed = ''
        self._frame = -1
        self._shown: int | None = None
        self._polled = False
        self.drops = 0
        self.uploads = 0
        self.timer_uploads = 0  # uploads submitted on their own; the rest rode a viewport frame
        _generation += 1
        self.generation = _generation
        with self._fatal_guard():
            self._activate()

    # -- public interface (P0.11) -------------------------------------------------------
    def can_publish(self) -> bool:
        """Timer backpressure: False while a step is still being produced (one step in flight,
        so a shown step is at most one solver step old) or both slots are being read."""
        if self._fatal or self._closed:
            return False
        with self._fatal_guard():
            for slot in self._slots:
                self._poll(slot)
            self._promote()
        available = not self._slots or (all(s.state != WRITING for s in self._slots)
                                         and any(s.state in (FREE, READY) for s in self._slots))
        self._polled = available  # the publish of this tick reuses this observation (C2 query budget)
        return available

    def publish(self, positions) -> bool:
        """Timer: pack into a FREE (else oldest READY) slot, then submit its upload once the
        pack completes within PRODUCER_WAIT; False skips without blocking."""
        if self._fatal or self._closed:
            return False
        if len(positions) != self.count:
            raise ValueError(f'expected {self.count} positions, got {len(positions)}')
        if not self.count:
            return True
        with self._fatal_guard():
            if not self._polled:
                for slot in self._slots:
                    self._poll(slot)
            self._polled = False
            free = [s for s in self._slots if s.state == FREE]
            ready = [s for s in self._slots if s.state == READY]
            slot = free[0] if free else min(ready, key=lambda s: s.frame) if ready else None
            if slot is None:
                self.drops += 1
                return False
            self._ops.pack(positions, slot.array, self.count)
            self._check('cuEventRecord', self._n.event_record(slot.event, self._ops.stream_handle()))
            self._frame += 1
            slot.frame = self._frame
            self._enter(slot, WRITING)
            start = self._clock()
            while slot.state == WRITING and self._clock() - start < PRODUCER_WAIT:
                self._sleep(0)
                self._poll(slot)
            self._promote()
            return True

    def positions_texture(self) -> Frame | None:
        """Draw: upload a READY slot the timer could not submit, return the shown frame."""
        self._polled = False  # a draw between can_publish and publish invalidates the observation
        if self._fatal or self._closed or not self.count:
            return None
        n = self._n
        if n.current_window() != self._window or n.active_context() != self._drw_ctx:
            return None  # foreign window/context: touch nothing
        with self._fatal_guard():
            for slot in self._slots:
                if slot.state == WRITING:
                    self._poll(slot)
            self._upload_newest()
        if self._shown is None:
            return None
        return Frame(self._texture, self.count, self.width, self.height, self.generation, self._shown)

    def newest_frame(self) -> int:
        """Newest frame a draw can show (uploaded or ready to upload); -1 before the first."""
        ready = [s.frame for s in self._slots if s.state in (READY, IN_FLIGHT)]
        return max(ready + [-1 if self._shown is None else self._shown])

    def capabilities(self) -> Capabilities:
        reason = self._fatal or (f'closed: {self._closed}' if self._closed else '')
        return Capabilities('VULKAN_INTEROP', True, MAX_COUNT, not reason, reason)

    def close(self) -> str:
        """Idempotent. completed = all released; quarantined = CUDA released, Blender objects
        retained until exit; failed = uncertain ownership retained, restart required."""
        if self._closed:
            return self._closed
        if self._fatal:
            self._closed = FAILED
            return self._closed
        try:
            with self._fatal_guard():
                self._closed = self._close()
        except FatalTransportError:
            self._closed = FAILED
        return self._closed

    @property
    def fatal_reason(self) -> str:
        return self._fatal

    # -- activation -----------------------------------------------------------------------
    def _activate(self):
        n = self._n
        if not n.backend_is_vulkan():
            raise RecoverableTransportError('running GPU backend is not the profiled Vulkan backend')
        limit = n.max_texture_size()
        if max(self.width, self.height) > limit:
            raise RecoverableTransportError(f'{self.width}x{self.height} exceeds texture limit {limit}')
        blender_uuid = n.blender_uuid()
        code, cuda_uuid = n.cuda_uuid()
        self._check('cuDeviceGetUuid', code, setup=True)
        if not blender_uuid or len(blender_uuid) != 16 or blender_uuid != cuda_uuid:
            raise RecoverableTransportError('Blender Vulkan device UUID differs from CUDA device 0')
        code, free, total = n.mem_info()
        self._check('cuMemGetInfo', code, setup=True)
        need = 3 * TEXEL_BYTES * self.width * self.height + max(VRAM_RESERVE, total // 10)
        if free < need:
            raise RecoverableTransportError(f'needs {need >> 20} MiB free VRAM, {free >> 20} MiB available')
        with self._scope():
            try:
                self._self_test()
                if self.count:
                    self._texture = n.create_texture(self.width, self.height)
                    self._tp = n.texture_pointer(self._texture)
                    for _ in range(2):
                        self._new_slot(self.width * self.height)
            except RecoverableTransportError:
                self._release_all()
                self._texture = self._test_texture = None
                raise

    def _self_test(self):
        n = self._n
        w, h = SELF_TEST
        texture = self._test_texture = n.create_texture(w, h)  # a fatal timeout quarantines it with self
        tp = n.texture_pointer(texture)
        slot = self._new_slot(w * h)
        self._ops.fill_known(slot.array, w * h)
        self._check('cuEventRecord', n.event_record(slot.event, self._ops.stream_handle()), setup=True)
        self._enter(slot, WRITING)
        self._settle(slot, SETUP_BUDGET)
        n.texture_update(tp, slot.pb, w, h)
        n.fence_signal(slot.fence)
        self._enter(slot, IN_FLIGHT)
        self._settle(slot, SETUP_BUDGET)
        observed = n.read_texture(texture, w, h)
        self._slots.remove(slot)
        self._release(slot)
        self._test_texture = None
        if not np.array_equal(np.asarray(observed, dtype=np.float32).reshape(-1, 4), known_answer(w * h)):
            raise RecoverableTransportError('256-byte known-answer self-test mismatch')

    def _new_slot(self, texels: int) -> _Slot:
        """Acquire one slot; each field is the ledger entry released by _release."""
        n = self._n
        slot = _Slot(texels)
        self._slots.append(slot)
        slot.pb = n.pb_create(slot.size) or None
        if slot.pb is None or n.pb_size(slot.pb) < slot.size:
            raise RecoverableTransportError('pixel buffer allocation failed')
        handle, export_size = n.pb_export(slot.pb)
        if not handle or export_size < slot.size:
            if handle:
                n.close_handle(handle)
            raise RecoverableTransportError('pixel buffer export failed')
        try:
            code, ext = n.import_memory(handle, export_size)
        finally:
            closed = n.close_handle(handle)
        self._check('cuImportExternalMemory', code, setup=True)
        slot.ext = ext
        if not closed:
            raise RecoverableTransportError('CloseHandle failed after import')
        code, ptr = n.map_buffer(ext, slot.size)
        self._check('cuExternalMemoryGetMappedBuffer', code, setup=True)
        slot.ptr = ptr
        slot.array = self._ops.wrap(ptr, texels)
        code, event = n.event_create()
        self._check('cuEventCreate', code, setup=True)
        slot.event = event
        slot.fence = n.fence_create() or None
        if slot.fence is None:
            raise RecoverableTransportError('fence creation failed')
        return slot

    # -- state machine ------------------------------------------------------------------
    def _upload_newest(self) -> bool:
        """In the DRW context: copy the newest READY slot and signal its fence (which submits)."""
        ready = [s for s in self._slots if s.state == READY]
        if not ready:
            return False
        newest = max(ready, key=lambda s: s.frame)
        for stale in ready:
            if stale is not newest:
                stale.state = FREE  # produced, never consumed
        self._n.texture_update(self._tp, newest.pb, self.width, self.height)
        self._n.fence_signal(newest.fence)  # covers the pixel-buffer read by this upload
        self._enter(newest, IN_FLIGHT)
        self._shown = newest.frame
        self.uploads += 1
        return True

    def _promote(self):
        """Timer: submit a READY upload on its own; without the route the draw uploads it."""
        if not any(s.state == READY for s in self._slots):
            return
        try:
            with self._scope():
                if self._upload_newest():
                    self.timer_uploads += 1
        except _RouteUnavailable:  # raised before any upload work
            pass

    def _poll(self, slot: _Slot):
        if slot.state == WRITING:
            code = self._n.event_query(slot.event)
            if code == 0:
                self._leave(slot, READY)
            elif code != CUDA_NOT_READY:
                raise FatalTransportError(f'cuEventQuery returned CUresult {code}')
        elif slot.state == IN_FLIGHT:
            code = self._n.fence_query(slot.fence)
            if code == 0:
                self._leave(slot, FREE)
            elif code != VK_TIMEOUT:
                raise FatalTransportError(f'vkWaitSemaphores returned VkResult {code}')
        if slot.state in (WRITING, IN_FLIGHT) and self._clock() - slot.started > PENDING_DEADLINE:
            raise FatalTransportError(f'{slot.state} work pending beyond 500 ms; ownership quarantined')

    def _enter(self, slot: _Slot, state: str):
        slot.state, slot.started = state, self._clock()

    def _leave(self, slot: _Slot, state: str):
        """Work completes in queue order, so the deadline of work queued behind restarts now."""
        prior, slot.state = slot.state, state
        now = self._clock()
        for other in self._slots:
            if other is not slot and other.state == prior:
                other.started = max(other.started, now)

    def _settle(self, slot: _Slot, budget: float):
        start = self._clock()
        while True:
            self._poll(slot)
            if slot.state not in (WRITING, IN_FLIGHT):
                return
            if self._clock() - start >= budget:
                raise FatalTransportError(f'{slot.state} work incomplete after {budget * 1000:.0f} ms')
            self._sleep(0.001)

    def _check(self, name: str, code: int, *, setup: bool = False):
        if code == 0:
            return
        if setup and code in RECOVERABLE_SETUP:
            raise RecoverableTransportError(f'{name} failed with CUresult {code}')
        raise FatalTransportError(f'{name} failed with CUresult {code}')

    # -- teardown -------------------------------------------------------------------------
    def _close(self) -> str:
        try:
            with self._scope():
                self._retire()
        except _RouteUnavailable:  # raised only before any retirement work
            return self._close_cuda_only()
        return COMPLETED

    def _retire(self):
        """Inside the scope: one final retirement fence, bounded polling, M0-order release.

        A single timeline fence signalled after the last texture read covers every earlier
        read of the texture and of both pixel buffers. If both slots are still being written,
        the producers finish first so the fence lands on a slot that is not WRITING.
        """
        start = self._clock()
        final = self._shown is not None and bool(self._slots)

        def settled(done) -> bool:
            for slot in self._slots:
                self._poll(slot)
                if slot.state == READY:
                    slot.state = FREE
            if done():
                return True
            if self._clock() - start >= CLOSE_BUDGET:
                raise FatalTransportError('shutdown exceeded 100 ms; ownership quarantined')
            self._sleep(0.001)
            return False

        for slot in self._slots:
            if slot.state == READY:
                slot.state = FREE
        if final:
            while not settled(lambda: any(s.state != WRITING for s in self._slots)):
                pass
            slot = next(s for s in self._slots if s.state != WRITING)
            self._n.fence_signal(slot.fence)  # final retirement after the last texture read
            self._enter(slot, IN_FLIGHT)
        while not settled(lambda: all(s.state == FREE for s in self._slots)):
            pass
        self._release_all()
        self._texture = None

    def _close_cuda_only(self) -> str:
        """No GPU-context route: finish producers, release CUDA objects, retain Blender ones."""
        start = self._clock()
        while any(s.state == WRITING for s in self._slots):
            for slot in self._slots:
                if slot.state == WRITING:
                    self._poll(slot)
            if any(s.state == WRITING for s in self._slots):
                if self._clock() - start >= CLOSE_BUDGET:
                    raise FatalTransportError('producer work incomplete at shutdown; ownership quarantined')
                self._sleep(0.001)
        for slot in self._slots:
            self._release_cuda(slot)
        _QUARANTINE.append((self._slots, self._texture))
        return QUARANTINED

    def _release_cuda(self, slot: _Slot):
        n = self._n
        slot.array = None
        if slot.ptr is not None:
            self._check('cuMemFree', n.mem_free(slot.ptr))
            slot.ptr = None
        if slot.ext is not None:
            self._check('cuDestroyExternalMemory', n.destroy_external(slot.ext))
            slot.ext = None
        if slot.event is not None:
            self._check('cuEventDestroy', n.event_destroy(slot.event))
            slot.event = None

    def _release(self, slot: _Slot):
        """Release in M0 order; requires the GPU scope and completed producer/consumer work."""
        self._release_cuda(slot)
        if slot.fence is not None:
            self._n.fence_free(slot.fence)
            slot.fence = None
        if slot.pb is not None:
            self._n.pb_free(slot.pb)
            slot.pb = None

    def _release_all(self):
        while self._slots:
            self._release(self._slots[-1])
            self._slots.pop()

    @contextmanager
    def _scope(self):
        """Qualified M0 route: window context -> locked DRW context -> exact restoration."""
        n = self._n
        if not n.is_main_thread():
            raise _RouteUnavailable('GPU cleanup route requires the main thread')
        if n.live_windows() != {self._window}:
            raise _RouteUnavailable('owner window is not the single live window')
        prior = n.active_context()
        if not prior or prior != n.window_context(self._window) or prior == self._drw_ctx or n.drw_enabled():
            raise _RouteUnavailable('unclassified or nested GPU context')
        n.drw_enable()
        pending: BaseException | None = None  # only an error raised inside this scope
        try:
            active = n.active_context()
            if not self._drw_ctx and active and active != prior:
                self._drw_ctx = active
            if not active or active != self._drw_ctx:
                raise _RouteUnavailable('DRW scope did not activate the captured viewport context')
            yield
        except BaseException as exc:
            pending = exc
            raise
        finally:
            n.drw_disable()
            if n.active_context() != prior or n.drw_enabled():
                raise FatalTransportError('window GPU context restoration failed' + (
                    f' (while handling {type(pending).__name__}: {pending})' if pending else ''))

    @contextmanager
    def _fatal_guard(self):
        try:
            yield
        except RecoverableTransportError:
            raise
        except Exception as exc:
            # Anything not on the recoverable allowlist leaves GPU/CUDA state uncertain.
            if not self._fatal:
                self._fatal = str(exc) if isinstance(exc, FatalTransportError) else f'{type(exc).__name__}: {exc}'
                _QUARANTINE.append(self)  # keep every acquisition referenced; no further calls
            if isinstance(exc, FatalTransportError):
                raise
            raise FatalTransportError(self._fatal) from exc


# -- real native seam (Windows, exact profiled build) ----------------------------------------
P = c.c_void_p
U = c.c_uint64


class _Span(c.Structure):
    _fields_ = [('ptr', P), ('count', c.c_int64)]


class _NativeHandle(c.Structure):
    _fields_ = [('handle', c.c_int64), ('size', c.c_size_t)]


class _WinHandle(c.Structure):
    _fields_ = [('handle', P), ('name', P)]


class _HandleUnion(c.Union):
    _fields_ = [('fd', c.c_int), ('win32', _WinHandle), ('sci', P)]


class _MemDesc(c.Structure):
    _fields_ = [('type', c.c_int), ('handle', _HandleUnion), ('size', U), ('flags', c.c_uint),
                ('reserved', c.c_uint * 16)]


class _BufferDesc(c.Structure):
    _fields_ = [('offset', U), ('size', U), ('flags', c.c_uint), ('reserved', c.c_uint * 16)]


class _WaitInfo(c.Structure):
    _fields_ = [('type', c.c_uint), ('next', P), ('flags', c.c_uint), ('count', c.c_uint),
                ('semaphores', c.POINTER(U)), ('values', c.POINTER(U))]


_CUDA_SIGNATURES = {
    'cuDeviceGetUuid_v2': (P, c.c_int), 'cuCtxPushCurrent_v2': (P,), 'cuCtxPopCurrent_v2': (c.POINTER(P),),
    'cuImportExternalMemory': (c.POINTER(P), c.POINTER(_MemDesc)),
    'cuExternalMemoryGetMappedBuffer': (c.POINTER(U), P, c.POINTER(_BufferDesc)),
    'cuMemFree_v2': (U,), 'cuDestroyExternalMemory': (P,), 'cuEventCreate': (c.POINTER(P), c.c_uint),
    'cuEventRecord': (P, P), 'cuEventQuery': (P,), 'cuEventDestroy_v2': (P,),
    'cuMemGetInfo_v2': (c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)),
}


class BlenderVulkanNative:
    """ctypes binding of a resolved exact-build profile, the CUDA driver and kernel32."""

    def __init__(self, resolved, device: str = 'cuda:0'):
        try:
            self._bind(resolved, device)
        except (ImportError, KeyError, AttributeError, OSError, TypeError, ValueError) as exc:
            # Nothing has been called through the profile yet: CUDA and Blender are untouched.
            raise RecoverableTransportError(f'native binding unavailable: {type(exc).__name__}: {exc}') from exc

    def _bind(self, resolved, device: str):
        import bpy
        import gpu
        self._bpy, self._gpu = bpy, gpu
        fn, self._data, self._enums = resolved.callables, resolved.data, resolved.enums
        lay = resolved.layouts
        for struct, name in ((_Span, 'Span'), (_NativeHandle, 'GPUPixelBufferNativeHandle'),
                             (_MemDesc, 'CUDA_EXTERNAL_MEMORY_HANDLE_DESC'),
                             (_BufferDesc, 'CUDA_EXTERNAL_MEMORY_BUFFER_DESC'), (_WaitInfo, 'VkSemaphoreWaitInfo')):
            if c.sizeof(struct) != lay[name]['size']:
                raise RecoverableTransportError(f'{name} layout differs from the profile')
        off = {name: entry['offsets'] for name, entry in lay.items()}
        self._tex_type, self._tex_ptr = off['BPyGPUTexture']['ob_type'], off['BPyGPUTexture']['texture']
        self._backend_device = off['VKBackend']['device']
        self._vk_device, self._vk_semaphore = off['VKDevice']['vk_device'], off['VKDevice']['submission_finished_semaphore']
        self._fence_signalled, self._fence_value = off['VKFence']['signalled'], off['VKFence']['timeline_value']
        self._win_runtime, self._runtime_ctx = off['wmWindow']['runtime'], off['wmWindowRuntime']['gpu_context']

        def bind(name, ret, *args):
            return c.CFUNCTYPE(ret, *args)(fn[name])
        self._uuid = bind('GPU_platform_uuid', P, c.POINTER(_Span))
        self._pb_create = bind('GPU_pixel_buffer_create', P, c.c_size_t)
        self._pb_free = bind('GPU_pixel_buffer_free', None, P)
        self._pb_size = bind('GPU_pixel_buffer_size', c.c_size_t, P)
        self._pb_handle = bind('GPU_pixel_buffer_get_native_handle', P, c.POINTER(_NativeHandle), P)
        self._update = bind('GPU_texture_update_sub_from_pixel_buffer', None, P, c.c_int, P, *([c.c_int] * 6))
        self._fence_create = bind('GPU_fence_create', P)
        self._fence_signal = bind('GPU_fence_signal', None, P)
        self._fence_free = bind('GPU_fence_free', None, P)
        self._backend = bind('GPUBackend_get', P)
        self._context = bind('GPU_context_active_get', P)
        self._drw_enable = bind('DRW_gpu_context_enable', None)
        self._drw_disable = bind('DRW_gpu_context_disable', None)
        self._drw_enabled = bind('DRW_gpu_context_is_enabled', c.c_bool)
        self._wait = bind('vkWaitSemaphores', c.c_int, P, c.POINTER(_WaitInfo), U)
        self._k32 = c.WinDLL('kernel32', use_last_error=True)
        self._k32.CloseHandle.argtypes, self._k32.CloseHandle.restype = [P], c.c_int
        self._cu = c.WinDLL(str(Path(os.environ['SystemRoot']) / 'System32' / 'nvcuda.dll'))
        for name, args in _CUDA_SIGNATURES.items():
            func = getattr(self._cu, name)
            func.argtypes, func.restype = args, c.c_int
        self._cuda_context = wp.get_device(device).context

    def _cuda(self, name, *args) -> int:
        if self._cu.cuCtxPushCurrent_v2(self._cuda_context):
            raise FatalTransportError('cuCtxPushCurrent failed')
        code = 0  # a ctypes argument error is raised before the driver call: the context is untouched
        try:
            code = getattr(self._cu, name)(*args)
            return code
        finally:
            # Any other CUresult (e.g. sticky 700) leaves the context uncertain: make no further CUDA
            # call, not even the pop, and let the caller's check report this code and quarantine.
            if code in _POP_SAFE:
                prior = P()
                if self._cu.cuCtxPopCurrent_v2(c.byref(prior)):
                    raise FatalTransportError(f'cuCtxPopCurrent failed after {name} returned CUresult {code}')

    # host
    def is_main_thread(self) -> bool:
        return threading.current_thread() is threading.main_thread()

    def live_windows(self) -> set[int]:
        if len(self._bpy.data.window_managers) != 1:
            return set()
        return {w.as_pointer() for w in self._bpy.context.window_manager.windows}

    def current_window(self) -> int | None:
        window = self._bpy.context.window
        return window.as_pointer() if window else None

    def max_texture_size(self) -> int:
        return self._gpu.capabilities.max_texture_size_get()

    # Blender GPU
    def backend_is_vulkan(self) -> bool:
        if self._gpu.platform.backend_type_get() != 'VULKAN':
            return False
        backend = self._backend()
        return bool(backend) and P.from_address(backend).value == self._data['VKBackend_vtable']

    def blender_uuid(self) -> bytes | None:
        span = _Span()
        if self._uuid(c.byref(span)) != c.addressof(span) or not span.ptr or span.count != 16:
            return None
        return c.string_at(span.ptr, 16)

    def active_context(self) -> int:
        return self._context() or 0

    def window_context(self, window: int) -> int:
        runtime = P.from_address(window + self._win_runtime).value  # caller verified liveness
        return (P.from_address(runtime + self._runtime_ctx).value or 0) if runtime else 0

    def drw_enable(self):
        self._drw_enable()

    def drw_disable(self):
        self._drw_disable()

    def drw_enabled(self) -> bool:
        return bool(self._drw_enabled())

    def create_texture(self, width: int, height: int):
        return self._gpu.types.GPUTexture((width, height), format='RGBA32F')

    def read_texture(self, texture, width: int, height: int):
        data = texture.read()
        data.dimensions = width * height * 4
        return np.asarray(data, dtype=np.float32)

    def texture_pointer(self, texture) -> int:
        if type(texture) is not self._gpu.types.GPUTexture or \
                P.from_address(id(texture) + self._tex_type).value != self._data['BPyGPUTexture_Type']:
            raise RecoverableTransportError('GPUTexture wrapper type differs from the profile')
        pointer = P.from_address(id(texture) + self._tex_ptr).value
        if not pointer:
            raise RecoverableTransportError('GPUTexture has no native texture')
        return pointer

    def pb_create(self, size: int) -> int:
        return self._pb_create(size) or 0

    def pb_size(self, pb: int) -> int:
        return self._pb_size(pb)

    def pb_export(self, pb: int) -> tuple[int, int]:
        handle = _NativeHandle()
        self._pb_handle(c.byref(handle), pb)
        return handle.handle, handle.size

    def pb_free(self, pb: int):
        self._pb_free(pb)

    def texture_update(self, texture_pointer: int, pb: int, width: int, height: int):
        self._update(texture_pointer, self._enums['GPU_DATA_FLOAT'], pb, 0, 0, 0, width, height, 1)

    def fence_create(self) -> int:
        return self._fence_create() or 0

    def fence_signal(self, fence: int):
        self._fence_signal(fence)

    def fence_free(self, fence: int):
        self._fence_free(fence)

    def fence_query(self, fence: int) -> int:
        """Zero-timeout timeline query; never GPU_fence_wait (infinite)."""
        backend = self._backend()
        if not backend or P.from_address(backend).value != self._data['VKBackend_vtable'] or \
                P.from_address(fence).value != self._data['VKFence_vtable']:
            raise FatalTransportError('Vulkan backend/fence identity changed')
        device = backend + self._backend_device
        vk_device = P.from_address(device + self._vk_device).value
        semaphore = U(U.from_address(device + self._vk_semaphore).value)
        value = U(U.from_address(fence + self._fence_value).value)
        if not (vk_device and semaphore.value and value.value and c.c_ubyte.from_address(fence + self._fence_signalled).value):
            raise FatalTransportError('signalled fence has no timeline value')
        info = _WaitInfo(self._enums['VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO'], None, 0, 1,
                         c.pointer(semaphore), c.pointer(value))
        return self._wait(vk_device, c.byref(info), 0)

    def close_handle(self, handle: int) -> bool:
        return bool(self._k32.CloseHandle(handle))

    # CUDA driver (raw CUresult)
    def cuda_uuid(self) -> tuple[int, bytes]:
        uuid = (c.c_ubyte * 16)()
        return self._cuda('cuDeviceGetUuid_v2', uuid, 0), bytes(uuid)

    def mem_info(self) -> tuple[int, int, int]:
        free, total = c.c_size_t(), c.c_size_t()
        return self._cuda('cuMemGetInfo_v2', c.byref(free), c.byref(total)), free.value, total.value

    def import_memory(self, handle: int, size: int) -> tuple[int, int]:
        desc = _MemDesc()
        desc.type = self._enums['CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32']
        desc.handle.win32.handle, desc.size = handle, size
        desc.flags = self._enums['CUDA_EXTERNAL_MEMORY_DEDICATED']
        ext = P()
        return self._cuda('cuImportExternalMemory', c.byref(ext), c.byref(desc)), ext.value or 0

    def map_buffer(self, ext: int, size: int) -> tuple[int, int]:
        ptr, desc = U(), _BufferDesc(0, size, 0)
        return self._cuda('cuExternalMemoryGetMappedBuffer', c.byref(ptr), ext, c.byref(desc)), ptr.value

    def mem_free(self, ptr: int) -> int:
        return self._cuda('cuMemFree_v2', ptr)

    def destroy_external(self, ext: int) -> int:
        return self._cuda('cuDestroyExternalMemory', ext)

    def event_create(self) -> tuple[int, int]:
        event = P()
        return self._cuda('cuEventCreate', c.byref(event), 2), event.value or 0  # CU_EVENT_DISABLE_TIMING

    def event_record(self, event: int, stream: int) -> int:
        return self._cuda('cuEventRecord', event, stream)

    def event_query(self, event: int) -> int:
        return self._cuda('cuEventQuery', event)

    def event_destroy(self, event: int) -> int:
        return self._cuda('cuEventDestroy_v2', event)


class WarpOps:
    """Warp launches on the solver's current cuda:0 stream; the event is recorded on it."""

    def __init__(self, device: str = 'cuda:0'):
        self.device = wp.get_device(device)

    def wrap(self, ptr: int, texels: int):
        return wp.array(ptr=ptr, dtype=wp.vec4, shape=texels, device=self.device, copy=False)

    def stream_handle(self) -> int:
        return wp.get_stream(self.device).cuda_stream

    def pack(self, positions, dst, count: int):
        wp.launch(_pack, dim=dst.shape[0], inputs=[positions, dst, count], device=self.device)

    def fill_known(self, dst, count: int):
        wp.launch(_known, dim=dst.shape[0], inputs=[dst, count], device=self.device)


def create(resolved, count: int, *, window: int, device: str = 'cuda:0') -> VulkanInteropTransport:
    """Bind the resolved profile and activate; raises Recoverable/FatalTransportError."""
    return VulkanInteropTransport(BlenderVulkanNative(resolved, device), WarpOps(device), count, window=window)


__all__ = ['VulkanInteropTransport', 'BlenderVulkanNative', 'WarpOps', 'create', 'known_answer']
