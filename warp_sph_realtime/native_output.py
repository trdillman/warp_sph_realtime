"""Explicit native-output controller. GPU-only default does no native readback."""

import math
import time


class LiveGate:
    def __init__(self):
        self.last_stamp = None
        self.last_time = -math.inf

    def due(self, stamp, fps, now, pending):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("positive finite live output rate required")
        return stamp is not None and stamp != self.last_stamp and not pending and now - self.last_time >= 1.0 / fps

    def queued(self, stamp, now):
        self.last_stamp, self.last_time = stamp, now


def _stamp_key(stamp):
    return None if stamp is None else (stamp.source_id, stamp.epoch, stamp.frame_id)


def _owner(scene):
    import uuid

    if not scene.warp_sph.sdk_snapshot_owner:
        scene.warp_sph.sdk_snapshot_owner = uuid.uuid4().hex
    return scene.warp_sph.sdk_snapshot_owner


def live_request(scene):
    owner = _owner(scene)
    settings = scene.warp_sph
    if not settings.sdk_live_name.strip():
        raise ValueError("name the live Geometry Nodes object")
    return {
        "name": settings.sdk_live_name,
        "collection": "WarpSPH SDK Output " + owner,
        "kind": "LIVE",
        "max_bytes": int(settings.sdk_native_budget_mb) * 1024 * 1024,
        "max_native_bytes": int(settings.sdk_native_budget_mb) * 1024 * 1024,
    }


def service(session):
    """One main-thread boundary: complete queued work then request latest live frame."""
    bridge = session.session
    if not getattr(bridge, "is_shared_sdk", False):
        return
    settings = session.scene.warp_sph
    if getattr(session, "_native_live_pending", False) and not getattr(settings, "sdk_live_geometry", False):
        bridge.cancel_snapshot("Live Geometry Nodes disabled; last good output retained")
        session._native_live_pending = False
    import bpy

    if getattr(bpy.app, "is_job_running", lambda kind: False)("RENDER"):
        return
    bridge.service_snapshot()
    if getattr(session, "_native_live_pending", False):
        if bridge.snapshot_status in ("COMPLETED", "FAILED", "CANCELLED"):
            session._native_live_pending = False
            settings.sdk_native_status = bridge.snapshot_message
            if bridge.snapshot_status == "FAILED":
                settings.sdk_live_geometry = False
                settings.sdk_native_status = "Live output stopped: " + bridge.snapshot_message
    if getattr(session, "native_bake", None) is not None:
        return
    if not getattr(settings, "sdk_live_geometry", False) or not session.running or session.bake is not None:
        return
    gate = getattr(session, "_native_live_gate", None)
    if gate is None:
        gate = session._native_live_gate = LiveGate()
    stamp = _stamp_key(bridge._stamp)
    now = time.monotonic()
    pending = bridge.snapshot_status in ("QUEUED", "CAPTURED")
    if gate.due(stamp, float(settings.sdk_live_rate), now, pending):
        try:
            bridge.request_snapshot(live_request(session.scene))
            gate.queued(stamp, now)
            session._native_live_pending = True
            settings.sdk_native_status = "Live native update queued (bounded GPU readback)"
        except (ValueError, RuntimeError) as exc:
            settings.sdk_live_geometry = False
            settings.sdk_native_status = f"Live output stopped: {exc}"


def commit_request(payload, request, scene, bindings):
    if request.get("kind") != "ANIMATION":
        return None
    from .cache.native_sequence import from_payload

    frame = from_payload(payload)
    return request["_writer"].append(request["scene_frame"], frame)


class NativeBake:
    """Every sample comes from independent solver storage; live state is frozen."""

    def __init__(self, session, writer, steps, first, last, *, replay, bridge):
        self.session, self.writer, self.replay, self.bridge = session, writer, replay, bridge
        self.steps = steps
        self.first, self.last, self.frame = first, last, first
        self.phase = "INITIAL"
        self.remaining = 0
        self.cancelled = self.paused = False
        self.queued_at = None
        self.captured = 0
        self.status = "RUNNING"
        self.error = ""
        self.closed = False

    def queue(self):
        self.bridge.request_snapshot(
            {
                "kind": "ANIMATION",
                "_writer": self.writer,
                "scene_frame": self.frame,
                "max_bytes": self.writer.max_frame_bytes,
            }
        )
        self.queued_at = time.monotonic()
        self.phase = "CAPTURE"

    def tick(self, context, report):
        session, bridge = self.session, self.bridge
        try:
            if self.cancelled:
                bridge.cancel_snapshot("Animation cancelled; complete frames retained")
                self.finish("CANCELLED", context, report)
                return True
            bridge.service_snapshot()
            if self.paused and self.phase in ("INITIAL", "ADVANCE"):
                return True
            if self.phase == "INITIAL":
                if not bridge.tick(self.replay.solver.copy_positions):
                    return True
                self.queue()
            elif self.phase == "CAPTURE":
                if bridge.snapshot_status == "FAILED":
                    raise ValueError(bridge.snapshot_message)
                if bridge.snapshot_status == "CANCELLED":
                    self.finish("CANCELLED", context, report)
                    return True
                if bridge.snapshot_status == "COMPLETED":
                    self.captured += 1
                    if self.frame == self.last:
                        self.finish("COMPLETE", context, report)
                        return True
                    self.frame += 1
                    self.remaining = self.steps
                    self.phase = "ADVANCE"
                elif time.monotonic() - self.queued_at > 60:
                    raise ValueError("No completed viewport frame for 60 seconds; capture stopped")
            elif self.phase == "ADVANCE":
                if not bridge.can_publish():
                    return True
                taken = self.replay.advance_steps(self.remaining, max_steps=4)
                # Explicit bake-only wait orders private clone work before the
                # normal stream performs pack/publication. Never changes live dt.
                self.replay.synchronize()
                if not bridge.tick(self.replay.solver.copy_positions):
                    raise RuntimeError("replay publication admission changed after private advance")
                self.remaining -= taken
                if not self.remaining:
                    self.queue()
            session.scene.warp_sph.sdk_native_status = (
                f"Baking independent animation: {self.captured}/{self.last - self.first + 1} frames"
            )
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
            return True
        except Exception as exc:
            from .render.transport import FatalTransportError

            self.error = str(exc)
            bridge.cancel_snapshot("Animation failed")
            self.finish("FAILED", context, report)
            if isinstance(exc, FatalTransportError) or session.restart_required:
                session.mark_fatal(str(exc))
                session.stop(context)
                return False
            return True

    def close_resources(self, status):
        if self.closed:
            return
        try:
            self.writer.close(status)
        except Exception as exc:
            self.status, self.error = "FAILED", str(exc)
        # Both owners retain their allocations after an uncertain completion.
        try:
            result = self.bridge.close()
            if result != "completed":
                self.session.mark_fatal(f"Animation GPU output cleanup: {result}")
        except Exception as exc:
            self.session.mark_fatal(f"Animation GPU output cleanup: {exc}")
        result = self.replay.close()
        if result != "completed":
            self.session.mark_fatal(f"Animation replay cleanup: {result}")
        self.closed = True

    def finish(self, status, context, report):
        self.status = status
        self.close_resources(status)
        session = self.session
        session.native_bake = None
        settings = session.scene.warp_sph
        settings.sdk_native_status = f"Animation {self.status}: {self.captured} complete frames. {self.error}"
        session.bake_result = settings.sdk_native_status
        if self.status == "COMPLETE" and not getattr(session, "restart_required", ""):
            try:
                from .cache import native_playback

                native_playback.attach(
                    session.scene,
                    self.writer.directory,
                    settings.sdk_animation_name,
                    session.session._bindings._sdk["output.geometry_nodes"].create_particle_instancer,
                    max_bytes=int(settings.sdk_native_budget_mb) * 1024 * 1024,
                    configure_range=False,
                )
                settings.sdk_native_status = (
                    f"Animation baked: {self.first}-{self.last}; live simulation unchanged. Stop then scrub/embed."
                )
            except Exception as exc:
                settings.sdk_native_status = f"Frames saved; playback setup failed: {exc}"
        report({"INFO"} if self.status == "COMPLETE" else {"WARNING"}, settings.sdk_native_status)


def begin_bake(session, context, report, *, first, last, resume=False):
    import bpy

    from .cache.bake import solver_signature, steps_per_frame
    from .cache.disk import free_bytes
    from .cache.native_sequence import SequenceWriter
    from .cache.solver_replay import SolverReplay
    from .interop_bridge import open_session
    from .sim_session import runtime

    if resume:
        report({"ERROR"}, "Rich animation resume is not supported; choose a new empty output directory")
        return False
    if not session.running or session.solver is None or not getattr(session.session, "is_shared_sdk", False):
        report({"ERROR"}, "Start a Shared SDK simulation before baking native animation")
        return False
    if not session.paused or session.step_once or session._output_pending:
        report({"ERROR"}, "Pause the simulation and finish its pending frame before baking a private copy")
        return False
    if session.bake is not None or getattr(session, "native_bake", None) is not None:
        report({"ERROR"}, "A bake is already running")
        return False
    bridge = session.session
    if bridge.snapshot_status in ("QUEUED", "CAPTURED"):
        report({"ERROR"}, "A native frame is pending; retry after it completes")
        return False
    settings = session.scene.warp_sph
    writer = replay = output = None
    try:
        if bpy.context.scene is not session.scene or bpy.app.is_job_running("RENDER"):
            raise ValueError("animation bake requires the active scene outside a render")
        path = settings.sdk_animation_dir.strip()
        if not path or (path.startswith("//") and not bpy.data.filepath):
            raise ValueError("choose a durable animation directory or save the .blend before a relative path")
        fps = session.scene.render.fps / session.scene.render.fps_base
        steps = steps_per_frame(settings.bake_steps, session.solver.dt, fps)
        budget = int(settings.sdk_native_budget_mb) * 1024 * 1024
        estimate = SolverReplay.estimate_clone_bytes(
            runtime.wp, session.solver, expected_type=runtime.RealtimeWCSPHSolver
        )
        # Charge the separate two-slot rich output in addition to the clone.
        if estimate + session.solver.num_particles * 512 > budget:
            raise ValueError("private solver plus output estimate exceeds native memory budget")
        signature = solver_signature(session.solver, steps, fps, session.scene.name, session.active_demo)
        signature.update(
            initial_simulation_time_s=float(session.solver.sim_time),
            sample_step_seconds=float(steps * session.solver.dt),
            requested_frame_seconds=1.0 / fps,
            origin="current paused solver cloned; live state and timeline untouched",
            density_sampling="last solver force evaluation, not recalculated at the final positions",
        )
        available = free_bytes(bpy.path.abspath(path))
        need = (int(last) - int(first) + 1) * (session.solver.num_particles * 56 + 8192)
        if available is not None and need > available:
            raise ValueError(f"not enough disk space; native frames need up to {need} bytes")
        writer = SequenceWriter(
            bpy.path.abspath(path), first=first, last=last, fps=fps, signature=signature, max_frame_bytes=budget
        )
        if not bridge._bindings.drain(bridge._native):
            raise RuntimeError("live GPU output did not settle before clone capture")
        replay = SolverReplay.clone_current(
            runtime.wp, session.solver, expected_type=runtime.RealtimeWCSPHSolver, max_clone_bytes=budget
        )
        output = open_session(replay.solver, session.scene, context.window.as_pointer())
        _owner(session.scene)
        session.native_bake = NativeBake(session, writer, steps, first, last, replay=replay, bridge=output)
        settings.sdk_native_status = "Baking a private copy from the paused state; live simulation untouched"
        return True
    except Exception as exc:
        if output is not None:
            output.close()
        if replay is not None:
            replay.close()
        if writer is not None:
            writer.close("FAILED")
        report({"ERROR"}, f"Cannot bake native animation: {exc}")
        return False


def stop(session):
    job = getattr(session, "native_bake", None)
    if job is not None:
        try:
            job.bridge.cancel_snapshot("Animation stopped")
            job.status = "CANCELLED"
            job.close_resources("CANCELLED")
            session.bake_result = f"Animation stopped: {job.captured} complete frames retained. {job.error}"
        finally:
            session.native_bake = None
    session._native_live_pending = False
    session._native_live_gate = None
