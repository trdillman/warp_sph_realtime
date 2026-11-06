"""Actual installed extension disable/load lifecycle qualification helpers.

Run each phase from the canonical owned Blender process. File opening is performed
by the harness between arm_file_load and verify_file_load. No process-state reset
or synthetic completion is used.
"""

import importlib
import json
import os
import time
from pathlib import Path

import addon_utils
import bpy

MODULE = "bl_ext.user_default.warp_sph_realtime"
_state = {}


def _write(name, value):
    (_state["output"] / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def start_disable(output):
    from blender_vulkan_interop._impl.transport import process_state

    smod = importlib.import_module(MODULE + ".sim_session")
    assert smod.active() is None and not process_state().restart_required
    _state["output"] = Path(output)
    _state["output"].mkdir(parents=True, exist_ok=True)
    scene = bpy.data.scenes.new("WarpSPH SDK Lifecycle Verified")
    bpy.context.window.scene = scene
    settings = scene.warp_sph
    settings.use_shared_sdk = True
    settings.particle_count = 100000
    settings.render_mode = "SSFR_LIQUID"
    settings.scene_source = "DEMO"
    settings.use_cuda_graph = True
    assert bpy.ops.wm.warp_sph_start() == {"RUNNING_MODAL"}
    _state["session"] = smod.get(scene)
    return {"pid": os.getpid(), "status": "running_SSFR_before_disable"}


def disable_and_start_load():
    from blender_vulkan_interop._impl.transport import process_state

    session = _state["session"]
    bridge = session.session
    assert session.running and session.step_count > 0 and bridge.metrics["draws"] > 0
    before = {
        "pid": os.getpid(),
        "solver_count": session.solver.num_particles,
        "solver_steps": session.step_count,
        "real_draws": bridge.metrics["draws"],
        "render_mode": session.scene.warp_sph.render_mode,
    }
    addon_utils.disable(MODULE, default_set=False)
    report = {
        **before,
        "extension_disabled": not addon_utils.check(MODULE)[1],
        "bridge_close": bridge.close_result,
        "session_close": session.last_close,
        "timer_removed": session.timer is None,
        "draw_removed": session.draw_handle is None,
        "solver_released": session.solver is None,
        "sdk_fatal": process_state().restart_required,
        "close_drain_s": bridge.metrics.get("close_drain_s"),
    }
    assert report["extension_disabled"] and report["bridge_close"] == report["session_close"] == "completed"
    assert report["timer_removed"] and report["draw_removed"] and report["solver_released"] and not report["sdk_fatal"]
    report["status"] = "passed"
    _write("running-extension-disable.json", report)
    assert addon_utils.enable(MODULE, default_set=False, persistent=False) is not None
    scene = bpy.data.scenes.new("WarpSPH SDK Running File Load")
    bpy.context.window.scene = scene
    scene.warp_sph.use_shared_sdk = True
    scene.warp_sph.render_mode = "POINTS"
    scene.warp_sph.particle_count = 100000
    assert bpy.ops.wm.warp_sph_start() == {"RUNNING_MODAL"}
    smod = importlib.import_module(MODULE + ".sim_session")
    _state["load_session"] = smod.get(scene)
    _state["load_bridge"] = _state["load_session"].session
    return report


def arm_file_load():
    session, bridge = _state["load_session"], _state["load_bridge"]
    assert session.running and session.step_count > 0 and bridge.metrics["draws"] > 0
    original_drain = bridge._bindings.drain

    def observed_drain(native):
        publisher = native._publisher
        started = time.perf_counter()
        before = [publisher._available(slot) for slot in publisher._slots]
        result = original_drain(native)
        after = [publisher._available(slot) for slot in publisher._slots]
        _write(
            "fileload-drain-observation.json",
            {
                "pid": os.getpid(),
                "ready_before": before,
                "ready_after": after,
                "completed": result,
                "elapsed_s": time.perf_counter() - started,
                "method": "actual SDK readiness predicates, no forged completion",
            },
        )
        return result

    bridge._bindings.drain = observed_drain
    report = {
        "pid": os.getpid(),
        "steps": session.step_count,
        "draws": bridge.metrics["draws"],
        "running": session.running,
        "render_mode": session.scene.warp_sph.render_mode,
    }
    _write("fileload-before.json", report)
    return report


def verify_file_load():
    from blender_vulkan_interop._impl.transport import process_state

    session, bridge = _state["load_session"], _state["load_bridge"]
    smod = importlib.import_module(MODULE + ".sim_session")
    report = {
        "pid": os.getpid(),
        "bridge_close": bridge.close_result,
        "session_close": session.last_close,
        "timer_removed": session.timer is None,
        "draw_removed": session.draw_handle is None,
        "solver_released": session.solver is None,
        "sdk_fatal": process_state().restart_required,
        "active_consumer": smod.active() is not None,
        "close_drain_s": bridge.metrics.get("close_drain_s"),
        "loaded_file": bpy.data.filepath,
    }
    report["status"] = (
        "passed"
        if (
            report["bridge_close"] == report["session_close"] == "completed"
            and report["timer_removed"]
            and report["draw_removed"]
            and report["solver_released"]
            and not report["sdk_fatal"]
            and not report["active_consumer"]
        )
        else "failed"
    )
    _write("running-file-load.json", report)
    assert report["status"] == "passed", report
    return report


def force_inflight_file_load(filepath, queued_solver_steps=64):
    """Queue real solver GPU work and open a file before it completes.

    This is a diagnostic stress producer, not ordinary UI cadence: one bridge
    publication contains queued_solver_steps real solver advances. No driver
    fence, event, status or completion result is fabricated or overwritten.
    """
    session, bridge = _state["load_session"], _state["load_bridge"]
    assert session.running and session.paused and not session._output_pending
    assert bridge._receipt.source_released() and bridge.can_publish()
    original = bridge._bindings.drain

    def observed(native):
        publisher = native._publisher
        before = [publisher._available(slot) for slot in publisher._slots]
        started = time.perf_counter()
        result = original(native)
        after = [publisher._available(slot) for slot in publisher._slots]
        _write(
            "forced-fileload-drain.json",
            {
                "pid": os.getpid(),
                "ready_before": before,
                "ready_after": after,
                "completed": result,
                "elapsed_s": time.perf_counter() - started,
                "real_solver_steps_queued": queued_solver_steps,
                "scope": "diagnostic queued real WCSPH work, not performance cadence",
            },
        )
        return result

    bridge._bindings.drain = observed

    def queued_producer(output):
        for _ in range(queued_solver_steps):
            session.solver.step(output, use_graph=True)

    assert bridge.tick(queued_producer)
    _write(
        "forced-fileload-before.json",
        {
            "pid": os.getpid(),
            "queued_solver_steps": queued_solver_steps,
            "published_frame": bridge._stamp.frame_id,
            "simulation_time": session.solver.sim_time,
            "source_released_immediately_after_publish": bridge._receipt.source_released(),
        },
    )
    bpy.ops.wm.open_mainfile(filepath=str(filepath))
    return None
