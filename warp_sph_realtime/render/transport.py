"""P0.11 position transport seam: flat wp.vec3 solver positions in, a GPUTexture out.

Every adapter exposes can_publish() and publish(positions) on the timer,
positions_texture() in the viewport draw, capabilities() and an idempotent close(). Positions are packed as
RGBA32F texels in a 4096-wide texture; the renderer draws exactly ``count`` instances.
This module is bpy-free; the gpu module is passed in or imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from typing import Any

WIDTH = 4096
MAX_COUNT = WIDTH * 1024  # 4,194,304 positions = 64 MiB of RGBA32F per buffer
TEXEL_BYTES = 16

# close() results. Queued or quarantined cleanup is never reported as freed.
COMPLETED = 'completed'
FAILED = 'failed'
QUARANTINED = 'quarantined'


class RecoverableTransportError(RuntimeError):
    """Interop is unavailable but the CUDA context is healthy: fall back once to Readback."""


class FatalTransportError(RuntimeError):
    """CUDA/GPU state is uncertain: stop, make no further CUDA calls, restart Blender."""


_PROCESS_STATE = '_warp_sph_realtime_process_state'


def process_state():
    """Process-lifetime store that survives add-on disable/enable (module reload).

    Quarantined GPU objects and the sticky restart-required / interop-unavailable flags
    must outlive every module instance, or a re-enable would free or retry them.
    """
    state = sys.modules.get(_PROCESS_STATE)
    if state is None:
        state = types.ModuleType(_PROCESS_STATE)
        state.quarantine, state.restart_required, state.interop_unavailable = [], '', ''
        sys.modules[_PROCESS_STATE] = state
    return state


@dataclass(frozen=True)
class Frame:
    texture: Any
    count: int
    width: int
    height: int
    generation: int
    frame: int


@dataclass(frozen=True)
class Capabilities:
    transport: str
    zero_copy: bool  # no host round trip; VulkanInterop still does one GPU buffer-to-texture copy per frame
    max_count: int
    ready: bool
    reason: str = ''


def dimensions(count: int, max_count: int = MAX_COUNT) -> tuple[int, int]:
    """Padded (width, height); N=0 allocates and draws nothing."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > max_count:
        raise ValueError(f'particle count {count!r} outside 0..{max_count}')
    return (WIDTH, -(-count // WIDTH)) if count else (0, 0)


class ReadbackTransport:
    """Labelled CPU fallback, never zero-copy: device-to-host copy in publish, texture upload in draw.

    The host wait in publish is the documented Readback exception. Staging is preallocated
    and padding stays zero, so the uploaded extent matches the interop adapters.
    """

    def __init__(self, count: int, *, reason: str = '', gpu_module: Any = None):
        import numpy as np
        import warp as wp
        self._wp = wp
        self._gpu = gpu_module
        self.count = count
        self.width, self.height = dimensions(count)
        self._reason = reason
        self._staging = np.zeros((self.width * self.height, 4), dtype=np.float32)
        self._host = wp.empty(count, dtype=wp.vec3, device='cpu', pinned=True) if count else None
        self._texture = None
        self._dirty = False
        self._frame = -1
        self._closed = False

    def can_publish(self) -> bool:
        return not self._closed  # publish is synchronous: never in flight

    def publish(self, positions) -> bool:
        if self._closed:
            return False
        if not self.count:
            return True
        if len(positions) != self.count:
            raise ValueError(f'expected {self.count} positions, got {len(positions)}')
        try:
            self._wp.copy(self._host, positions)
            if positions.device.is_cuda:
                self._wp.synchronize_device(positions.device)
        except Exception as exc:  # Warp reports sticky CUDA errors as plain exceptions
            raise FatalTransportError(f'Readback device copy failed: {exc}') from exc
        self._staging[:self.count, :3] = self._host.numpy()
        self._staging[:self.count, 3] = 1.0
        self._frame += 1
        self._dirty = True
        return True

    def positions_texture(self) -> Frame | None:
        if self._closed or self._frame < 0:
            return None
        if self._dirty:
            gpu = self._gpu or __import__('gpu')
            data = gpu.types.Buffer('FLOAT', self._staging.size, self._staging.ravel())
            self._texture = gpu.types.GPUTexture((self.width, self.height), format='RGBA32F', data=data)
            self._dirty = False
        return Frame(self._texture, self.count, self.width, self.height, 0, self._frame)

    def newest_frame(self) -> int:
        return self._frame  # publish is synchronous: every published frame is drawable

    def capabilities(self) -> Capabilities:
        return Capabilities('READBACK', False, MAX_COUNT, not self._closed, self._reason)

    @property
    def fatal_reason(self) -> str:
        return ''

    def close(self) -> str:
        self._closed = True
        self._texture = None
        self._host = None
        return COMPLETED
