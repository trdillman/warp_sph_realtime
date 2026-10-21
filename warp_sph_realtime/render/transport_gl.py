"""GLInterop position transport on Blender's OpenGL backend (P0.11 GL half).

The seam is the same as VulkanInterop's, with no CPU readback. Warp packs the solver's flat
vec3 positions into one preallocated vec4 staging buffer; a device-to-device copy moves the
staging rows into a Blender ``GPUTexture`` (RGBA32F, 4096 wide) that is registered with CUDA
through ``cudaGraphicsGLRegisterImage``. The copy is a GPU copy, so the label says
"GPU copy, no CPU readback", never "zero copy".

Per publish, all on the GL thread inside the timer and all on Warp's CUDA stream::

    pack kernel -> cudaGraphicsMapResources -> SubResourceGetMappedArray
                -> cudaMemcpy2DToArrayAsync -> cudaGraphicsUnmapResources -> event record

Ordering argument (CUDA runtime, graphics interop): cudaGraphicsMapResources(stream) makes
every GL call issued before it complete before any later work on ``stream`` begins, so a draw
from the previous frame has finished reading the texture before the copy overwrites it.
cudaGraphicsUnmapResources(stream) makes every CUDA operation issued on ``stream`` before it
(the solver step, the pack and the copy) complete before any GL call issued after it begins.
The draw runs after the timer returns, hence after the unmap, and so sees a finished copy
without any host wait. A single event recorded after the unmap tells the next tick whether
that publish has completed: while it has not, ``can_publish`` is False and the tick is
skipped, so at most one solver step is queued ahead of the viewport.

Blender's Python ``GPUTexture`` has no GL name (``bindcode``), so the native seam identifies
the name by content (see ``interop_bridge.GLTextureNative.identify``). A known-answer copy
through a 4x4 texture gates activation, and its readback is the only readback.

Setup errors on the recoverable allowlist leave CUDA usable: everything acquired is released
and RecoverableTransportError is raised (the caller falls back once to Readback). Every other
CUDA error is fatal: the transport stops, makes no further CUDA calls and stays referenced.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp

from .transport import (
    COMPLETED, FAILED, MAX_COUNT, TEXEL_BYTES, Capabilities, FatalTransportError, Frame,
    RecoverableTransportError, dimensions, process_state,
)

CUDA_NOT_READY = 600
# Setup-phase cudart errors that leave the context usable: InvalidValue, MemoryAllocation,
# InvalidGraphicsContext, OperatingSystem, InvalidResourceHandle, NotSupported.
RECOVERABLE_SETUP = frozenset({1, 2, 219, 304, 400, 801})
SELF_TEST = (4, 4)  # 16 texels = 256-byte known-answer copy gates activation
VRAM_RESERVE = 512 << 20
NAME = 'GL_INTEROP'

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


class _Target:
    """One registered texture: Blender texture, its GL name and the CUDA resource."""
    __slots__ = ('width', 'height', 'texture', 'name', 'resource')

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.texture: Any = None
        self.name = 0
        self.resource = 0


class GLInteropTransport:
    """One staging buffer, one registered texture. Construct from a context with the GL context current."""

    def __init__(self, native, ops, count: int):
        global _generation
        self.count = count
        self.width, self.height = dimensions(count)  # ValueError before any native call
        self._n, self._ops = native, ops
        self._target: _Target | None = None
        self._staging: Any = None
        self._event = 0
        self._pending = False
        self._fatal = ''
        self._closed = ''
        self._frame = -1
        self.publishes = 0
        _generation += 1
        self.generation = _generation
        with self._fatal_guard():
            try:
                self._activate()
            except RecoverableTransportError:
                if self._staging is not None:
                    self._ops.synchronize()  # setup work may be queued on the stream
                self._release_all()
                raise

    # -- public interface (P0.11) -------------------------------------------------------
    def can_publish(self) -> bool:
        """Timer backpressure: False while the previous publish is still running on the GPU."""
        if self._fatal or self._closed:
            return False
        with self._fatal_guard():
            return not self._busy()

    def publish(self, positions) -> bool:
        """Timer, GL thread: pack and copy ``positions`` into the texture; False skips without blocking."""
        if self._fatal or self._closed:
            return False
        if len(positions) != self.count:
            raise ValueError(f'expected {self.count} positions, got {len(positions)}')
        if not self.count:
            return True
        target = self._target
        assert target is not None
        with self._fatal_guard():
            if self._busy():
                return False
            stream = self._ops.stream_handle()
            self._ops.pack(positions, self._staging, self.count)
            self._copy(target, stream)
            self._check('cudaEventRecord', self._n.event_record(self._event, stream))
            self._pending = True
            self._frame += 1
            self.publishes += 1
            return True

    def positions_texture(self) -> Frame | None:
        if self._fatal or self._closed or self._frame < 0 or self._target is None:
            return None
        return Frame(self._target.texture, self.count, self.width, self.height, self.generation, self._frame)

    def newest_frame(self) -> int:
        """Every published frame is drawable: the unmap orders the copy before later GL work."""
        return self._frame

    def capabilities(self) -> Capabilities:
        reason = self._fatal or (f'closed: {self._closed}' if self._closed else '')
        return Capabilities(NAME, True, MAX_COUNT, not reason, reason)

    def close(self) -> str:
        """Idempotent. completed = all released; failed = uncertain ownership retained, restart required."""
        if self._closed:
            return self._closed
        if self._fatal:
            self._closed = FAILED
            return self._closed
        try:
            with self._fatal_guard():
                self._ops.synchronize()  # one host wait at teardown: the last copy must finish before unregister
                self._release_all()
            self._closed = COMPLETED
        except FatalTransportError:
            self._closed = FAILED
        return self._closed

    @property
    def fatal_reason(self) -> str:
        return self._fatal

    # -- activation -----------------------------------------------------------------------
    def _activate(self):
        n = self._n
        if not n.backend_is_opengl():
            raise RecoverableTransportError('running GPU backend is not OpenGL')
        reason = n.check_gl_device()
        if reason:
            raise RecoverableTransportError(reason)
        if not self.count:
            return
        limit = n.max_texture_size()
        if max(self.width, self.height) > limit:
            raise RecoverableTransportError(f'{self.width}x{self.height} exceeds texture limit {limit}')
        code, free, total = n.mem_info()
        self._check('cudaMemGetInfo', code, setup=True)
        need = 2 * TEXEL_BYTES * self.width * self.height + max(VRAM_RESERVE, total // 10)
        if free < need:
            raise RecoverableTransportError(f'needs {need >> 20} MiB free VRAM, {free >> 20} MiB available')
        code, self._event = n.event_create()
        self._check('cudaEventCreateWithFlags', code, setup=True)
        self._staging = self._ops.alloc(self.width * self.height)
        self._self_test()
        self._target = self._new_target(self.width, self.height)

    def _self_test(self):
        w, h = SELF_TEST
        self._ops.fill_known(self._staging, w * h)
        target = self._new_target(w, h)
        try:
            self._copy(target, self._ops.stream_handle())
            self._ops.synchronize()
            observed = self._n.read_texture(target.texture)
            ok = np.array_equal(np.asarray(observed, dtype=np.float32).reshape(-1, 4), known_answer(w * h))
        finally:
            self._release_target(target)
        if not ok:
            raise RecoverableTransportError('256-byte known-answer self-test mismatch')

    def _new_target(self, width: int, height: int) -> _Target:
        n = self._n
        target = _Target(width, height)
        try:
            target.texture = n.create_texture(width, height)
            target.name = n.identify(target.texture, width, height)
            if not target.name:
                raise RecoverableTransportError('could not identify the OpenGL name of the position texture')
            code, target.resource = n.register_image(target.name)
            self._check('cudaGraphicsGLRegisterImage', code, setup=True)
        except BaseException:
            self._release_target(target)
            raise
        return target

    # -- per-frame copy -------------------------------------------------------------------
    def _copy(self, target: _Target, stream: int):
        """Map, copy staging rows into the texture's mip 0 array, unmap; see the module ordering argument."""
        n = self._n
        self._check('cudaGraphicsMapResources', n.map(target.resource, stream))
        code, array = n.mapped_array(target.resource)
        self._check('cudaGraphicsSubResourceGetMappedArray', code)
        row = target.width * TEXEL_BYTES
        self._check('cudaMemcpy2DToArrayAsync',
                    n.copy_to_array(array, self._ops.pointer(self._staging), row, row, target.height, stream))
        self._check('cudaGraphicsUnmapResources', n.unmap(target.resource, stream))

    def _busy(self) -> bool:
        """True while the last publish has not finished; one event query, never blocks."""
        if not self._pending:
            return False
        code = self._n.event_query(self._event)
        if code == CUDA_NOT_READY:
            return True
        self._check('cudaEventQuery', code)
        self._pending = False
        return False

    # -- teardown -------------------------------------------------------------------------
    def _release_target(self, target: _Target):
        if target.resource:
            self._check('cudaGraphicsUnregisterResource', self._n.unregister(target.resource))
            target.resource = 0
        target.texture = None  # drops Blender's texture after CUDA released its registration

    def _release_all(self):
        if self._target is not None:
            self._release_target(self._target)
            self._target = None
        if self._event:
            self._check('cudaEventDestroy', self._n.event_destroy(self._event))
            self._event = 0
        self._staging = None
        self._pending = False

    def _check(self, name: str, code: int, *, setup: bool = False):
        if code == 0:
            return
        if setup and code in RECOVERABLE_SETUP:
            raise RecoverableTransportError(f'{name} failed with CUDA error {code}')
        raise FatalTransportError(f'{name} failed with CUDA error {code}')

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


class WarpOps:
    """Warp launches on the solver's current stream of the device; the copy is issued on that stream."""

    def __init__(self, device: str = 'cuda:0'):
        self.device = wp.get_device(device)

    def alloc(self, texels: int):
        return wp.zeros(texels, dtype=wp.vec4, device=self.device)  # padding stays zero

    def pointer(self, array) -> int:
        return array.ptr

    def stream_handle(self) -> int:
        return wp.get_stream(self.device).cuda_stream

    def pack(self, positions, dst, count: int):
        wp.launch(_pack, dim=dst.shape[0], inputs=[positions, dst, count], device=self.device)

    def fill_known(self, dst, count: int):
        wp.launch(_known, dim=dst.shape[0], inputs=[dst, count], device=self.device)

    def synchronize(self):
        wp.synchronize_device(self.device)


def create(count: int, *, device: str = 'cuda:0') -> GLInteropTransport:
    """Bind the real cudart/GL seam and activate; raises Recoverable/FatalTransportError."""
    from .interop_bridge import GLTextureNative
    try:
        native = GLTextureNative(device)
    except RuntimeError as exc:  # no CUDA 12.9 runtime or GL entry points: interop unavailable
        raise RecoverableTransportError(str(exc)) from exc
    return GLInteropTransport(native, WarpOps(device), count)


__all__ = ['GLInteropTransport', 'WarpOps', 'create', 'known_answer']
