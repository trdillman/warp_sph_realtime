"""ADOPT-01 host contracts. Native solver/GPU evidence is a separate gate."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def bridge_api():
    assert importlib.util.find_spec("warp_sph_realtime.interop_bridge"), "shared SDK consumer bridge missing"
    from warp_sph_realtime import interop_bridge

    return interop_bridge


def dependency_api():
    assert importlib.util.find_spec("warp_sph_realtime.sdk_dependency"), "exact shared SDK identity loader missing"
    from warp_sph_realtime import sdk_dependency

    return sdk_dependency


def solver(count=8, time=0.0):
    return SimpleNamespace(
        num_particles=count,
        active_count=count,
        sim_time=time,
        unit_scale=0.1,
        _next_id=count,
        vel="velocity",
        density_sorted="density",
        pid="stable_pid",
    )


class Receipt:
    status = "ACCEPTED"

    def __init__(self):
        self.released = True

    def source_released(self):
        return self.released


class Native:
    def __init__(self):
        self.ready = True
        self.frames = []
        self.close_status = "completed"
        self.closes = 0
        self.receipt = Receipt()

    def can_publish(self):
        return self.ready

    def publish(self, frame):
        self.frames.append(frame)
        return self.receipt

    def close(self):
        self.closes += 1
        return self.close_status


class Bindings:
    def __init__(self):
        self.native = Native()
        self.owner = SimpleNamespace(close=self.native.close)
        self.fatal = ""
        self.packs = []
        self.creates = 0
        self.freed = 0
        self.snapshots = []

    def create(self, capacity):
        self.creates += 1
        return self.owner, self.native, {"position": object(), "id": object(), "radius": object(), "color": object()}

    def pack(self, solver, arrays, scene):
        self.packs.append((solver, arrays, scene))
        return {
            "position": arrays["position"],
            "id": arrays["id"],
            "radius": arrays["radius"],
            "color": arrays["color"],
            "velocity": solver.vel,
            "density": solver.density_sorted,
        }

    def frame(self, source, epoch, number, revision, solver, channels):
        return SimpleNamespace(
            stamp=SimpleNamespace(
                source_id=source,
                epoch=epoch,
                frame_id=number,
                simulation_time_s=solver.sim_time,
                scene_revision=revision,
            ),
            active_count=solver.active_count,
            channels=channels,
        )

    def poison(self, reason, owner):
        self.fatal = reason

    def fatal_reason(self):
        return self.fatal

    def free_renderer(self):
        self.freed += 1

    def commit_snapshot(self, payload, request, scene):
        self.snapshots.append((payload, request))
        return SimpleNamespace(outputs=())

    def apply_receipt(self, revision, status, reason):
        return SimpleNamespace(revision=revision, status=status, reason=reason)


def ready(count=8):
    m = bridge_api()
    runtime = Bindings()
    scene = SimpleNamespace(session_uid=77)
    s = solver(count)
    bridge = m.InteropBridge(s, scene, window=10, _bindings=runtime)
    bridge.start()
    return bridge, runtime, s


def test_lazy_import_does_not_load_warp_blender_or_sdk():
    bridge_api()
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys;sys.path.insert(0,"
        + repr(str(root))
        + ');import warp_sph_realtime.interop_bridge;import warp_sph_realtime.sdk_dependency;assert not set(("bpy","gpu","warp","blender_vulkan_interop")).intersection(sys.modules)'
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_wrong_package_bytes_rejected_before_import(tmp_path):
    m = dependency_api()
    package = tmp_path / "blender_vulkan_interop"
    package.mkdir()
    (package / "__init__.py").write_text("wrong candidate")
    with pytest.raises(RuntimeError, match="identity|candidate|hash"):
        m.verify_payload(package, {"__init__.py": "0" * 64})


def test_payload_identity_accepts_exact_bytes_and_rejects_missing(tmp_path):
    import hashlib

    m = dependency_api()
    package = tmp_path / "blender_vulkan_interop"
    package.mkdir()
    data = b"correct exact SDK"
    (package / "__init__.py").write_bytes(data)
    m.verify_payload(package, {"__init__.py": hashlib.sha256(data).hexdigest()})
    with pytest.raises(RuntimeError, match="identity|candidate|missing"):
        m.verify_payload(package, {"missing.py": "0" * 64})


def test_duplicate_start_never_allocates_again():
    bridge, runtime, _ = ready()
    with pytest.raises(RuntimeError, match="started"):
        bridge.start()
    assert runtime.creates == 1


def test_backpressure_never_steps_solver_or_packs():
    bridge, runtime, _ = ready()
    runtime.native.ready = False
    calls = []
    assert not bridge.tick(lambda out: calls.append(out))
    assert not calls and not runtime.packs and bridge.waits == 1


def test_frame_uses_current_solver_arrays_and_simulation_time():
    bridge, runtime, s = ready()
    s.active_count = 3

    def step(out):
        assert out is bridge.positions
        s.sim_time = 0.2
        s.vel = "reordered_vel"
        s.pid = "reordered_ids"
        s.density_sorted = "current_density"

    assert bridge.tick(step)
    frame = runtime.native.frames[-1]
    assert frame.active_count == 3 and frame.stamp.simulation_time_s == 0.2
    assert frame.channels["velocity"] == "reordered_vel" and frame.channels["density"] == "current_density"
    assert runtime.packs[-1][0].pid == "reordered_ids"


def test_rebind_waits_for_source_release_and_advances_epoch():
    bridge, runtime, s = ready()
    bridge.tick(lambda out: None)
    old_source = runtime.native.frames[-1].stamp.source_id
    new = solver(time=0.0)
    runtime.native.receipt.released = False
    assert not bridge.try_bind_solver(new) and bridge.solver is s
    runtime.native.receipt.released = True
    assert bridge.try_bind_solver(new)
    assert bridge.tick(lambda out: None)
    frame = runtime.native.frames[-1]
    assert frame.stamp.epoch == 1 and frame.stamp.source_id == old_source and frame.stamp.frame_id == 1
    assert not bridge.try_bind_solver(solver(9))


def test_scene_delta_never_claims_unimplemented_physics():
    bridge, _, _ = ready()
    receipt = bridge.apply_scene(SimpleNamespace(snapshot=SimpleNamespace(revision=5)))
    assert receipt.status == "REBUILD_REQUIRED" and receipt.revision == 5 and receipt.reason


def test_snapshot_request_pins_last_frame_and_blocks_steps_until_serviced():
    bridge, runtime, _ = ready()
    bridge.tick(lambda out: None)
    bridge.request_snapshot("output")
    calls = []
    assert not bridge.tick(lambda out: calls.append(out))
    assert not calls
    assert bridge.snapshot_status == "QUEUED"
    bridge._snapshot_payload = "captured"
    bridge.snapshot_status = "CAPTURED"
    assert bridge.service_snapshot()
    assert runtime.snapshots == [("captured", "output")] and bridge.snapshot_status == "COMPLETED"
    assert bridge.tick(lambda out: None)


def test_pending_snapshot_cancelled_on_close_idempotently():
    bridge, runtime, _ = ready()
    bridge.tick(lambda out: None)
    bridge.request_snapshot("output")
    assert bridge.close() == bridge.close() == "completed"
    assert runtime.native.closes == 1 and runtime.freed == 1 and bridge.snapshot_status == "CANCELLED"
    assert bridge.positions is None


def test_uncertain_close_poisoned_and_scratch_retained():
    bridge, runtime, _ = ready()
    runtime.native.close_status = "quarantined"
    scratch = bridge.positions
    assert bridge.close() == "quarantined" and runtime.fatal and bridge.positions is scratch
    assert runtime.freed == 0


def test_solver_exception_is_fatal_and_quarantines_before_cleanup():
    m = bridge_api()
    bridge, runtime, _ = ready()

    def broken(out):
        raise RuntimeError("CUDA state unknown")

    with pytest.raises(m.FatalTransportError):
        bridge.tick(broken)
    assert runtime.fatal and not runtime.native.frames


def test_snapshot_exception_preserves_old_output_and_resumes_publication():
    bridge, runtime, _ = ready()
    bridge.tick(lambda out: None)
    bridge.request_snapshot("output")
    bridge._snapshot_payload = "payload"
    bridge.snapshot_status = "CAPTURED"

    def fail(*args):
        raise ValueError("target changed")

    runtime.commit_snapshot = fail
    assert not bridge.service_snapshot() and bridge.snapshot_status == "FAILED"
    assert bridge.tick(lambda out: None)
