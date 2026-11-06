"""File-load teardown drains all actual publisher slots, never just one free slot."""

from types import SimpleNamespace

import pytest
from test_sdk_bridge import ready

from warp_sph_realtime import interop_bridge


def helper():
    assert hasattr(interop_bridge, "_drain_pinned_native"), "shared native drain missing"
    return interop_bridge._drain_pinned_native


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = 0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds
        self.sleeps += 1


def test_drain_waits_for_every_slot_not_any_free_slot():
    clock = Clock()
    calls = []
    slots = [object(), object()]

    def available(slot):
        calls.append(slot)
        return slot is slots[0] or clock.value >= 0.003

    native = SimpleNamespace(_publisher=SimpleNamespace(_slots=slots, _available=available))
    assert helper()(native, budget=0.01, clock=clock.now, sleep=clock.sleep)
    assert clock.sleeps >= 3 and all(slot in calls for slot in slots)


def test_drain_timeout_does_not_invent_completion_or_clear_live_leases():
    clock = Clock()
    slot = SimpleNamespace(leases=1)
    native = SimpleNamespace(_publisher=SimpleNamespace(_slots=[slot], _available=lambda value: False))
    assert not helper()(native, budget=0.004, clock=clock.now, sleep=clock.sleep)
    assert slot.leases == 1 and 0.004 <= clock.value <= 0.006


def test_drain_failure_propagates_without_calling_retire():
    def failure(slot):
        raise RuntimeError("driver completion query failed")

    native = SimpleNamespace(_publisher=SimpleNamespace(_slots=[object()], _available=failure))
    with pytest.raises(RuntimeError, match="completion"):
        helper()(native, budget=0.01)


def test_bridge_drains_before_owner_close():
    bridge, runtime, _ = ready()
    events = []
    runtime.drain = lambda native: events.append("drained") or True
    runtime.owner.close = lambda: events.append("closed") or "completed"
    assert bridge.close() == "completed" and events == ["drained", "closed"]


def test_failed_drain_is_sticky_and_retains_source_scratch():
    bridge, runtime, _ = ready()
    runtime.drain = lambda native: False
    scratch = bridge.positions
    assert bridge.close() == "failed" and runtime.fatal and bridge.positions is scratch
    assert runtime.freed == 0


def test_renderer_cleanup_exception_cannot_report_completed_close():
    bridge, runtime, _ = ready()
    runtime.drain = lambda native: True

    def fail():
        raise RuntimeError("renderer free failed")

    runtime.free_renderer = fail
    assert bridge.close() == "failed" and runtime.fatal and bridge.positions is not None
