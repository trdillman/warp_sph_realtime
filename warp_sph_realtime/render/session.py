"""Transport session owner (plan M4): one adapter, its positions scratch and its renderer.

The solver stays independent of transport: each tick it writes flat wp.vec3 positions into
the session scratch, which the adapter publishes. Selection follows the running backend:
Vulkan tries the exact-build interop adapter and OpenGL tries the GLInterop texture adapter;
each falls back once to labelled Readback on a recoverable failure, and a fatal error propagates
(the caller must stop and require restart). The caller chooses whether OpenGL comes through
here at all: the legacy VBO path (with its raw-GL SSFR) stays outside it. Session frames draw
SSFR through the gpu-module renderer (gpu_ssfr, P0.13).
This module is bpy-free; Blender-specific factories are imported lazily.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .transport import COMPLETED, FatalTransportError, RecoverableTransportError

VULKAN_LABEL = 'Vulkan interop (GPU copy, no CPU readback)'
GL_LABEL = 'OpenGL interop (GPU copy, no CPU readback)'
READBACK_LABEL = 'Readback (CPU copy)'


class TransportSession:
    def __init__(self, transport, positions, renderer, backend: str, reason: str = '', count: int | None = None):
        self.transport = transport
        self.count = count  # particles per frame; a solver of the same size can reuse the session
        self.positions = positions
        self.renderer = renderer
        self.backend = backend
        self.reason = reason
        self.waits = 0
        self.close_result = ''
        self._redrawn = -1

    @property
    def label(self) -> str:
        if not self.transport.capabilities().zero_copy:
            return READBACK_LABEL
        return GL_LABEL if self.backend == 'OPENGL' else VULKAN_LABEL

    @property
    def fatal_reason(self) -> str:
        return self.transport.fatal_reason

    def tick(self, produce: Callable[[Any], None]) -> bool:
        """Timer: skip (never block) while the adapter has no slot; else produce and publish."""
        if not self.transport.can_publish():
            self.waits += 1
            return False
        try:
            produce(self.positions)
        except Exception as exc:  # Warp surfaces sticky CUDA errors as plain exceptions
            raise FatalTransportError(f'solver step failed; CUDA state unknown: {exc}') from exc
        return self.transport.publish(self.positions)

    def take_new_frame(self) -> bool:
        """Timer: True once per newly drawable frame. Redrawing only then keeps the viewport
        from contending with the solver on ticks that skipped or are still producing."""
        frame = self.transport.newest_frame()
        if frame <= self._redrawn:
            return False
        self._redrawn = frame
        return True

    def draw(self, settings) -> bool:
        """Viewport draw: the adapter rejects foreign windows/contexts and draws nothing."""
        frame = self.transport.positions_texture()
        if frame is not None:  # shown by Blender's own redraw: no extra redraw for it
            self._redrawn = max(self._redrawn, frame.frame)
        return self.renderer.draw(frame, settings)

    def close(self) -> str:
        """Idempotent; the transport result is authoritative (completed/quarantined/failed)."""
        if not self.close_result:
            self.close_result = self.transport.close()
            self.renderer.free()
            if self.close_result == COMPLETED:
                self.positions = None
        return self.close_result


def _device_positions(count: int):
    import warp as wp
    return wp.empty(count, dtype=wp.vec3, device='cuda:0')


def open_session(backend: str, count: int, window: int, *, interop_unavailable: str = '', resolve=None,
                 create_vulkan=None, create_gl=None, create_readback=None, alloc=None,
                 renderer_factory=None) -> TransportSession:
    """Select and activate an adapter. Raises FatalTransportError (no fallback) or setup errors.

    A non-empty ``interop_unavailable`` (an earlier recoverable failure) skips interop:
    the fallback happens once per process, never as a per-session retry loop.
    """
    if resolve is None and backend == 'VULKAN':
        from .blender_symbols import resolve_current_blender as resolve
    if create_vulkan is None and backend == 'VULKAN':
        from .transport_vulkan import create as create_vulkan
    if create_gl is None and backend == 'OPENGL':
        from .transport_gl import create as create_gl
    if create_readback is None:
        from .transport import ReadbackTransport as create_readback
    if alloc is None:
        alloc = _device_positions
    if renderer_factory is None:
        from .gpu_ssfr import ViewportRenderer as renderer_factory

    transport, reason = None, ''
    if backend == 'VULKAN' and interop_unavailable:
        reason = interop_unavailable
    elif backend == 'VULKAN':
        resolution = resolve()
        if resolution.available:
            try:
                transport = create_vulkan(resolution.resolved, count, window=window)
            except RecoverableTransportError as exc:
                reason = f'Vulkan interop unavailable: {exc}'
        else:
            reason = f'Vulkan interop unavailable: {resolution.reason}' + \
                (f' ({resolution.detail})' if resolution.detail else '')
    elif backend == 'OPENGL' and interop_unavailable:
        reason = interop_unavailable
    elif backend == 'OPENGL':
        try:
            transport = create_gl(count)
        except RecoverableTransportError as exc:
            reason = f'OpenGL interop unavailable: {exc}'
    else:
        reason = f'no interop adapter for the {backend} backend'
    if transport is None:
        transport = create_readback(count, reason=reason)
    try:
        return TransportSession(transport, alloc(count), renderer_factory(), backend, reason, count)
    except BaseException:
        transport.close()
        raise
