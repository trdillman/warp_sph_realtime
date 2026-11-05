"""Native-output state machine: no new physics step under capture backpressure."""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from warp_sph_realtime import native_output as m
from warp_sph_realtime.cache.native_sequence import NativeFrame, SequenceReader, SequenceWriter


class Bridge:
    is_shared_sdk = True

    def __init__(self):
        self._stamp = None
        self.snapshot_status = "IDLE"
        self.snapshot_message = ""
        self.request = None
        self.time = 0.0
        self.produces = 0
        self.ready = True

    def tick(self, produce):
        if not self.can_publish():
            return False
        produce(None)
        self.produces += 1
        self._stamp = NS(source_id="s", epoch=0, frame_id=self.produces, simulation_time_s=self.time, scene_revision=0)
        return True

    def can_publish(self):
        return self.ready and self.snapshot_status not in ("QUEUED", "CAPTURED")

    def request_snapshot(self, request):
        assert self.can_publish()
        self.request = request
        self.snapshot_status = "QUEUED"

    def service_snapshot(self):
        return False

    def close(self):
        return "completed"

    def cancel_snapshot(self, why):
        self.snapshot_status = "CANCELLED"
        self.snapshot_message = why


def fixture(tmp_path):
    bridge = Bridge()
    settings = NS(substeps=2, use_cuda_graph=True, sdk_native_status="", sdk_live_geometry=False)
    session = NS(
        session=bridge,
        solver=NS(substeps=2, copy_positions=lambda out: None),
        scene=NS(warp_sph=settings),
        _output_pending=True,
        step_count=0,
        active_demo="DEMO",
        running=True,
        paused=True,
        step_once=False,
        bake=None,
    )
    scene_frames = []
    session._set_frame = scene_frames.append
    session.solver.step = lambda out, **kw: setattr(bridge, "time", bridge.time + session.solver.substeps * 0.01)
    writer = SequenceWriter(tmp_path / "animation", first=1, last=3, fps=25, signature={})
    live_bridge = Bridge()
    session.session = live_bridge
    original_solver = session.solver
    clone = NS(substeps=2, copy_positions=lambda out: None)
    clone.step = lambda out, **kw: setattr(bridge, "time", bridge.time + clone.substeps * 0.01)

    def advance(steps, max_steps):
        clone.substeps = min(steps, max_steps)
        clone.step(None)
        return clone.substeps

    replay = NS(solver=clone, advance_steps=advance, synchronize=lambda: None, close=lambda: "completed")
    job = m.NativeBake(session, writer, 2, 1, 3, replay=replay, bridge=bridge)
    job.original_solver = original_solver
    session.native_bake = job
    context = NS(screen=NS(areas=[]))
    return session, job, bridge, context, scene_frames


def consume(job, bridge):
    n = 1
    stamp = vars(bridge._stamp)
    values = {
        "position": np.full((n, 3), stamp["simulation_time_s"], dtype=np.float32),
        "velocity": np.zeros((n, 3), dtype=np.float32),
        "color": np.ones((n, 4), dtype=np.float32),
        "radius": np.ones(n, dtype=np.float32),
        "density": np.ones(n, dtype=np.float32),
        "id": np.ones(n, dtype=np.uint64),
    }
    job.writer.append(bridge.request["scene_frame"], NativeFrame(values, stamp))
    bridge.snapshot_status = "COMPLETED"


def test_pending_capture_never_advances_solver_or_skips_animation_frame(tmp_path):
    session, job, bridge, ctx, frames = fixture(tmp_path)
    job.tick(ctx, lambda *a: None)
    assert bridge.produces == 1 and job.phase == "CAPTURE"
    for _ in range(10):
        job.tick(ctx, lambda *a: None)
    assert bridge.produces == 1
    consume(job, bridge)
    job.tick(ctx, lambda *a: None)
    assert frames == []
    job.tick(ctx, lambda *a: None)
    consume(job, bridge)
    job.tick(ctx, lambda *a: None)
    assert frames == [] and bridge.time == pytest.approx(0.02)
    job.tick(ctx, lambda *a: None)
    consume(job, bridge)
    # finish writes COMPLETE even when no bpy playback is available in host tests.
    job.tick(ctx, lambda *a: None)
    assert job.status == "COMPLETE" and session.native_bake is None and session.paused
    reader = SequenceReader(job.writer.directory)
    assert [reader.read(i).stamp["simulation_time_s"] for i in range(1, 4)] == pytest.approx([0, 0.02, 0.04])


def test_cancel_preserves_prefix_and_cancels_pending_frame(tmp_path):
    session, job, bridge, ctx, _ = fixture(tmp_path)
    job.tick(ctx, lambda *a: None)
    consume(job, bridge)
    job.tick(ctx, lambda *a: None)
    job.tick(ctx, lambda *a: None)
    assert bridge.snapshot_status == "QUEUED"
    job.cancelled = True
    job.tick(ctx, lambda *a: None)
    reader = SequenceReader(job.writer.directory)
    assert reader.span == (1, 1) and reader.status == "CANCELLED" and bridge.snapshot_status == "CANCELLED"


def test_stop_disk_error_does_not_block_existing_gpu_teardown(tmp_path):
    session, job, bridge, ctx, _ = fixture(tmp_path)

    def fail(_status):
        raise OSError("disk removed")

    job.writer.close = fail
    m.stop(session)
    assert session.native_bake is None and job.closed
    assert "disk" in job.error


def test_disabled_native_output_never_requests_readback(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", NS(app=NS(is_job_running=lambda _: False)))
    bridge = Bridge()
    bridge.tick(lambda _: None)
    scene = NS(warp_sph=NS(sdk_live_geometry=False))
    session = NS(session=bridge, scene=scene, running=True, bake=None)
    m.service(session)
    assert bridge.request is None


def test_render_does_not_service_native_mutation(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", NS(app=NS(is_job_running=lambda _: True)))
    bridge = Bridge()
    bridge.snapshot_status = "QUEUED"
    session = NS(session=bridge, scene=NS(warp_sph=NS(sdk_live_geometry=True)))
    m.service(session)
    assert bridge.snapshot_status == "QUEUED"


def test_live_disable_cancels_pending_update_before_mutation(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", NS(app=NS(is_job_running=lambda _: False)))
    bridge = Bridge()
    bridge.snapshot_status = "QUEUED"
    session = NS(session=bridge, scene=NS(warp_sph=NS(sdk_live_geometry=False)), _native_live_pending=True)
    m.service(session)
    assert bridge.snapshot_status == "CANCELLED" and not session._native_live_pending


def test_isolated_bake_never_changes_live_solver_steps_or_output(tmp_path):
    session, job, bridge, ctx, frames = fixture(tmp_path)
    original = session.solver
    output = session.session
    for _ in range(20):
        job.tick(ctx, lambda *a: None)
        if bridge.snapshot_status == "QUEUED":
            consume(job, bridge)
        if job.status == "COMPLETE":
            break
    assert job.status == "COMPLETE"
    assert session.solver is original and session.session is output
    assert output.produces == 0 and session.step_count == 0 and frames == []
