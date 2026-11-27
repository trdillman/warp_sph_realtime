from collections import deque
from types import SimpleNamespace

from warp_sph_realtime.cache.gpu import INFLIGHT, QUEUED, VramRing


class _Stream:
    def wait_stream(self, stream):
        pass

    def record_event(self, event):
        event.is_complete = True


class _Warp:
    def __init__(self):
        self.copy_calls = []

    def get_stream(self, device):
        return object()

    def copy(self, *args, **kwargs):
        self.copy_calls.append((args, kwargs))


def test_vram_ring_submits_empty_initial_frame_without_copying():
    ring = VramRing.__new__(VramRing)
    ring.wp = _Warp()
    ring.device = "cuda:0"
    ring.capacity = 4
    ring.copy_stream = _Stream()
    ring._inflight = deque()
    slot = SimpleNamespace(
        frame=0,
        count=0,
        state=INFLIGHT,
        pos=object(),
        vel=object(),
        pid=object(),
        _host=(object(), object(), object()),
        event=SimpleNamespace(is_complete=False),
    )

    ring.submit(slot, 1, 0)

    assert ring.wp.copy_calls == []
    assert ring.ready() == [slot]
    assert slot.state == QUEUED
    assert slot.count == 0
