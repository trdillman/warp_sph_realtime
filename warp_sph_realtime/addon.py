"""Blender operators, panel, scene properties, and the modal simulation runtime.

NVIDIA Warp and every CUDA-dependent module are imported only in
``_activate_runtime()``, after the owned/version/device checks, so the
extension registers (and can install Warp) on machines without it.
"""

from __future__ import annotations

import importlib
import logging
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty

from . import diagnostics, warp_installer, warp_runtime
from .render import session as transport_session
from .render.gpu_points import settings_from_scene
from .render.transport import COMPLETED, FAILED, QUARANTINED, FatalTransportError, process_state

_LOG = logging.getLogger(__name__)

wp = None
ZeroCopyGLBuffer52 = None
RealtimeWCSPHSolver = None
build_blender_demo_helpers = None
launch_demo_scene_init = None
cleanup_viewport_renderer = None
draw_particles_viewport = None
_startup_error = ""


def _activate_runtime():
    """Import CUDA-dependent modules only after owned/version/device checks."""
    global wp, ZeroCopyGLBuffer52, RealtimeWCSPHSolver
    global build_blender_demo_helpers, launch_demo_scene_init
    global cleanup_viewport_renderer, draw_particles_viewport
    if wp is not None:
        return  # already active; keep existing bindings
    loaded = warp_runtime.require_cuda_runtime()
    bridge = importlib.import_module(".render.interop_bridge", __package__)
    viewport = importlib.import_module(".render.viewport_draw", __package__)
    solver = importlib.import_module(".sim.sim_core", __package__)
    demos = importlib.import_module(".scene.demo_scenes", __package__)
    wp = loaded
    ZeroCopyGLBuffer52 = bridge.ZeroCopyGLBuffer52
    RealtimeWCSPHSolver = solver.RealtimeWCSPHSolver
    build_blender_demo_helpers = demos.build_blender_demo_helpers
    launch_demo_scene_init = demos.launch_demo_scene_init
    cleanup_viewport_renderer = viewport.cleanup_viewport_renderer
    draw_particles_viewport = viewport.draw_particles_viewport

class _SimRuntime:
    gl_buf: ZeroCopyGLBuffer52 | None = None   # legacy OpenGL VBO path only
    session: transport_session.TransportSession | None = None  # every other backend
    solver: RealtimeWCSPHSolver | None = None
    draw_handle = None
    timer = None
    token: int = 0            # bumped on every start/stop; orphaned modals exit
    running: bool = False
    paused: bool = False      # physics frozen; GPU state and drawing kept alive
    step_once: bool = False   # advance exactly one frame while paused
    active_demo: str = "DOUBLE_DAM_PILLARS"
    fps_avg: float = 0.0
    frame_time_ms: float = 0.0
    vram_mb: float = 0.0
    step_count: int = 0
    last_tick: float = 0.0
    last_close: str = ""
    last_error: str = ""
    draw_errors: set = set()  # distinct draw failures already logged this run

    # Process-lifetime flags: they outlive add-on disable/enable (module reload).
    @property
    def restart_required(self) -> str:  # fatal GPU/CUDA state: no further CUDA work this process
        return process_state().restart_required

    @restart_required.setter
    def restart_required(self, value: str):
        process_state().restart_required = value

    @property
    def interop_unavailable(self) -> str:  # recoverable interop failure: Readback until restart
        return process_state().interop_unavailable

    @interop_unavailable.setter
    def interop_unavailable(self, value: str):
        process_state().interop_unavailable = value


RT = _SimRuntime()
_QUARANTINE: list = process_state().quarantine  # objects never freed until exit, across reloads


def _gpu_backend() -> str:
    import gpu
    return gpu.platform.backend_type_get()


def _log_dir():
    return bpy.utils.extension_path_user(__package__, path="logs", create=True)


def _addon_version() -> str:
    import tomllib
    from pathlib import Path
    with open(Path(__file__).with_name("blender_manifest.toml"), "rb") as handle:
        return tomllib.load(handle)["version"]


def _warp_owner() -> str:
    blocked = warp_installer.foreign_warp_block_reason()
    if blocked:
        return "foreign: " + blocked
    return "owned (WarpSPH-managed install)" if warp_installer.owned_install_status() else "not installed"


def _diagnostics_report() -> str:
    """Never raises; every probe is isolated (see diagnostics.build_report)."""
    data = {**diagnostics.system_probes(), **diagnostics.warp_probes(wp)}
    data.update(
        addon_version=_addon_version,
        blender_version=lambda: bpy.app.version_string,
        blender_build_hash=lambda: bpy.app.build_hash.decode(errors="replace"),
        gpu_backend=_gpu_backend,
        warp_source=_warp_owner,  # P1.11 will extend this with the chosen Warp source
        transport=lambda: RT.session.label if RT.session is not None else "none",
        last_error=RT.last_error,
        last_close=RT.last_close,
        interop_unavailable=RT.interop_unavailable,
    )
    if wp is None:
        data["warp_version"] = lambda: warp_runtime.status().version or "not loaded (GPU details need one Start)"
    return diagnostics.build_report(data, lambda: diagnostics.read_log_tail(diagnostics.log_path()))


def _report_draw_failure(exc: Exception):
    """Stop the run and log each distinct draw failure once, never once per redraw."""
    RT.last_error = f"Viewport draw failed: {exc}"
    RT.running = False  # the modal timer stops the session; never raise every redraw
    key = f"{type(exc).__name__}: {exc}"
    if key not in RT.draw_errors:
        RT.draw_errors.add(key)
        _LOG.error("WarpSPH viewport draw failed (logged once per distinct error): %s", key, exc_info=exc)


def draw_particles_blender52():
    session = RT.session
    try:
        if session is None:
            # Legacy OpenGL path: the SSFR/sphere passes restore GL state in a finally block
            # (_preserve_gl_state) and draw_points restores its own, so a failure leaves none behind.
            draw_particles_viewport(RT.gl_buf, RT.running)
            return
        scene = bpy.context.scene
        if not RT.running or not scene.warp_sph_show_particles:
            return
        session.draw(settings_from_scene(scene, session.backend))
    except FatalTransportError as exc:
        _mark_fatal(str(exc))
    except Exception as exc:
        _report_draw_failure(exc)


def _mark_fatal(reason: str):
    RT.restart_required = RT.restart_required or reason
    RT.running = False


def _retire_session(session):
    result = session.close()
    RT.last_close = result
    if result != COMPLETED:
        _QUARANTINE.append(session)
    if result == QUARANTINED:  # Blender objects retained: do not create more interop objects
        RT.interop_unavailable = RT.interop_unavailable or \
            "Vulkan interop cleanup was quarantined (no GPU-context route)"
    if result == FAILED:
        _mark_fatal(session.fatal_reason or "transport cleanup did not complete; ownership retained")


def _new_solver(scene, num_p: int, demo_type: str):
    solver = RealtimeWCSPHSolver(
        num_particles=num_p,
        h=0.04,
        substeps=scene.warp_sph_substeps,
        enable_obstacles=1 if demo_type == "DOUBLE_DAM_PILLARS" else 0,
    )
    solver.stiffness = scene.warp_sph_stiffness
    solver.cfl_factor = scene.warp_sph_time_step_scale
    # Initialize GPU SoA particle distribution for the chosen demo
    launch_demo_scene_init(solver, demo_type)
    solver.rebuild_cuda_graph()
    return solver


def _reallocate_and_init_scene(scene, num_p: int, demo_type: str, window=None):
    """Transactional: the new output and solver are built while the old ones stay live;
    on failure the new ones are released and the running session is untouched."""
    backend = _gpu_backend()
    if backend == 'OPENGL':
        gl_buf = ZeroCopyGLBuffer52(num_p)
        try:
            solver = _new_solver(scene, num_p, demo_type)
            with gl_buf.mapped() as pos_vbo_wp:
                wp.copy(pos_vbo_wp, solver.pos_sim)
        except BaseException:
            gl_buf.destroy()
            raise
        # Commit point: output and solver switch together, before the old output is released.
        old_buf, RT.gl_buf = RT.gl_buf, gl_buf
        RT.solver, RT.active_demo = solver, demo_type
        if old_buf:
            old_buf.destroy()
    else:
        solver = _new_solver(scene, num_p, demo_type)
        window = window or bpy.context.window
        try:
            session = transport_session.open_session(
                backend, num_p, window.as_pointer(), interop_unavailable=RT.interop_unavailable)
        except FatalTransportError:
            _QUARANTINE.append(solver)
            raise
        try:
            session.tick(lambda out: wp.copy(out, solver.pos_sim))  # publish before first draw
        except FatalTransportError:
            _QUARANTINE.extend((solver, session))  # no CUDA cleanup after a fatal error
            raise
        except BaseException:
            _retire_session(session)
            if RT.restart_required:  # its cleanup failed: CUDA state unknown, keep the solver too
                _QUARANTINE.append(solver)
            raise
        zero_copy = session.transport.capabilities().zero_copy
        old_session, old_solver = RT.session, RT.solver
        if backend == 'VULKAN' and not zero_copy:
            if old_session is not None and old_session.transport.capabilities().zero_copy:
                # A resize must not silently downgrade a working interop session.
                _retire_session(session)
                if RT.restart_required:
                    _QUARANTINE.append(solver)
                    raise FatalTransportError(RT.restart_required)
                raise RuntimeError(f"new size cannot use Vulkan interop ({session.reason}); kept the current size")
            RT.interop_unavailable = session.reason  # fall back once; no interop retries
        # Commit point: output and solver switch together, before the old session is retired.
        RT.session = session
        RT.solver, RT.active_demo = solver, demo_type
        if old_session is not None:
            _retire_session(old_session)
            if RT.restart_required:  # old cleanup failed: stop; _stop_session quarantines the new pair
                _QUARANTINE.append(old_solver)
                raise FatalTransportError(RT.restart_required)

    RT.vram_mb = solver.get_vram_usage_mb()
    RT.step_count = 0
    RT.last_tick = 0.0
    # Spawn matching Blender 3D viewport wireframe domain & obstacle collider meshes
    build_blender_demo_helpers(demo_type)


def _stop_session(context=None):
    """Idempotent teardown for stop, cancel, unregister, file load and viewport loss."""
    context = context or getattr(bpy, "context", None)
    RT.running = False
    RT.paused = False
    RT.step_once = False
    RT.token += 1
    if RT.timer is not None:
        wm = getattr(context, "window_manager", None)
        if wm is not None:
            try:
                wm.event_timer_remove(RT.timer)
            except (ReferenceError, RuntimeError, ValueError):
                pass  # already freed with its window
        RT.timer = None
    if RT.draw_handle:
        bpy.types.SpaceView3D.draw_handler_remove(RT.draw_handle, 'WINDOW')
        RT.draw_handle = None
    session, solver = RT.session, RT.solver
    RT.session = None
    RT.solver = None
    if session is not None:
        if RT.restart_required:
            _QUARANTINE.append(session)  # no speculative CUDA cleanup after a fatal error
        else:
            _retire_session(session)
    if RT.restart_required and solver is not None:
        _QUARANTINE.append(solver)  # dropping it would free CUDA allocations
    if RT.gl_buf:
        if cleanup_viewport_renderer is not None:
            cleanup_viewport_renderer()
        RT.gl_buf.destroy()
        RT.gl_buf = None
    screen = getattr(context, "screen", None)
    for area in getattr(screen, "areas", ()):
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _apply_scene(op, context, count: int, demo: str) -> bool:
    try:
        _reallocate_and_init_scene(context.scene, count, demo, context.window)
        return True
    except FatalTransportError as exc:
        _mark_fatal(str(exc))
        op.report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
    except Exception as exc:
        op.report({'ERROR'}, f"WarpSPH could not rebuild the scene: {exc}")
        if RT.solver is not None:  # keep the running session; do not retry the same change
            context.scene.warp_sph_particle_count = RT.solver.num_particles
            context.scene.warp_sph_demo_scene = RT.active_demo
    return False


class WM_OT_warp_sph_start(bpy.types.Operator):
    bl_idname = "wm.warp_sph_start"
    bl_label = "Start WarpSPH Realtime"
    _token = -1

    def modal(self, context, event):
        if self._token != RT.token:
            return {'CANCELLED'}  # already stopped by unregister, file load or cancel
        if not RT.running or event.type == 'ESC':
            self.cleanup(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            screen = context.screen
            if screen is None or not any(area.type == 'VIEW_3D' for area in screen.areas):
                self.report({'WARNING'}, "WarpSPH stopped: its window has no 3D viewport")
                self.cleanup(context)
                return {'CANCELLED'}
            scene = context.scene
            target_count = scene.warp_sph_particle_count
            target_demo = scene.warp_sph_demo_scene

            # Dynamic Buffer & Demo Manager: reallocate/reset on-the-fly if slider or demo changed
            if RT.solver.num_particles != target_count or RT.active_demo != target_demo:
                if not _apply_scene(self, context, target_count, target_demo):
                    if not RT.running:
                        self.cleanup(context)
                        return {'CANCELLED'}
                    for area in context.screen.areas:  # kept the running session; no step this tick
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                    return {'PASS_THROUGH'}
            elif (RT.solver.stiffness != scene.warp_sph_stiffness
                  or RT.solver.cfl_factor != scene.warp_sph_time_step_scale):
                try:
                    RT.solver.stiffness = scene.warp_sph_stiffness
                    RT.solver.cfl_factor = scene.warp_sph_time_step_scale
                    RT.solver.update_time_step()
                    RT.solver.rebuild_cuda_graph()
                except Exception as exc:
                    if RT.session is not None:  # as a failed step (session.tick): CUDA state unknown
                        _mark_fatal(f"CUDA graph rebuild failed; CUDA state unknown: {exc}")
                        self.report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
                    else:
                        self.report({'ERROR'}, f"WarpSPH stopped: {exc}")
                    self.cleanup(context)
                    return {'CANCELLED'}

            if RT.paused and not RT.step_once:
                # Keep redrawing so display settings stay live on the frozen frame
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                return {'PASS_THROUGH'}

            t0 = time.perf_counter()

            RT.solver.substeps = scene.warp_sph_substeps
            solver, use_graph = RT.solver, scene.warp_sph_use_cuda_graph
            try:
                if RT.session is not None:
                    # No host sync: the adapter orders publication on the solver stream and
                    # skips the step (backpressure) while a step is in flight or both slots are busy.
                    stepped = RT.session.tick(lambda out: solver.step(out, use_graph=use_graph))
                else:
                    with RT.gl_buf.mapped() as pos_vbo_wp:
                        solver.step(pos_vbo_wp, use_graph=use_graph)
                    wp.synchronize_device("cuda:0")
                    stepped = True
            except FatalTransportError as exc:
                _mark_fatal(str(exc))
                self.report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
                self.cleanup(context)
                return {'CANCELLED'}
            except Exception as exc:
                self.report({'ERROR'}, f"WarpSPH stopped: {exc}")
                self.cleanup(context)
                return {'CANCELLED'}
            now = time.perf_counter()
            if RT.session is None:
                dt_ms = (now - t0) * 1000.0
            elif stepped and RT.last_tick:
                dt_ms = (now - RT.last_tick) * 1000.0  # asynchronous: wall-clock step interval
            else:
                dt_ms = None
            if stepped and RT.session is not None:
                RT.last_tick = now
            if dt_ms is not None:
                RT.frame_time_ms = 0.9 * RT.frame_time_ms + 0.1 * dt_ms if RT.step_count > 0 else dt_ms
                RT.fps_avg = 1000.0 / max(RT.frame_time_ms, 0.001)
            if stepped:
                RT.step_count += 1
                RT.step_once = False  # a single-step request survives a backpressure skip

            # GL draws synchronously after every step; a session redraws only for a new frame
            if RT.session is None or RT.session.take_new_frame():
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

        return {'PASS_THROUGH'}

    def execute(self, context):
        if RT.running:
            return {'FINISHED'}
        if RT.restart_required:
            self.report({'ERROR'}, f"WarpSPH needs a Blender restart: {RT.restart_required}")
            return {'CANCELLED'}

        RT.last_error = ""
        RT.draw_errors.clear()
        try:
            _activate_runtime()
            scene = context.scene
            _reallocate_and_init_scene(scene, scene.warp_sph_particle_count, scene.warp_sph_demo_scene,
                                       context.window)
        except FatalTransportError as exc:
            _mark_fatal(str(exc))
            _stop_session(context)
            self.report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
            return {'CANCELLED'}
        except Exception as exc:
            _stop_session(context)
            self.report({'ERROR'}, f"WarpSPH cannot start: {exc}")
            return {'CANCELLED'}

        RT.running = True
        RT.paused = False
        RT.step_once = False
        RT.token += 1
        self._token = RT.token
        try:
            RT.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_particles_blender52, (), 'WINDOW', 'POST_VIEW'
            )
            wm = context.window_manager
            RT.timer = wm.event_timer_add(0.001, window=context.window)
            wm.modal_handler_add(self)
        except Exception as exc:
            _stop_session(context)
            self.report({'ERROR'}, f"WarpSPH cannot start: {exc}")
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        # Blender cancels the modal (window close, file load); the owner window may be gone,
        # in which case the transport releases CUDA objects and quarantines Blender ones.
        if self._token == RT.token:
            _stop_session(context)

    def cleanup(self, context):
        _stop_session(context)


class WM_OT_warp_sph_load_demo(bpy.types.Operator):
    bl_idname = "wm.warp_sph_load_demo"
    bl_label = "Quick-Launch Demo Scene"

    demo_id: bpy.props.StringProperty(default="DOUBLE_DAM_PILLARS")

    def execute(self, context):
        scene = context.scene
        scene.warp_sph_demo_scene = self.demo_id
        if RT.running:
            if not _apply_scene(self, context, scene.warp_sph_particle_count, self.demo_id):
                return {'CANCELLED'}
        else:
            bpy.ops.wm.warp_sph_start()
        return {'FINISHED'}


class WM_OT_warp_sph_stop(bpy.types.Operator):
    bl_idname = "wm.warp_sph_stop"
    bl_label = "Stop Simulation"

    def execute(self, context):
        RT.running = False
        return {'FINISHED'}


class WM_OT_warp_sph_pause(bpy.types.Operator):
    bl_idname = "wm.warp_sph_pause"
    bl_label = "Pause / Resume Simulation"
    bl_description = "Freeze or continue the physics without releasing GPU buffers"

    @classmethod
    def poll(cls, context):
        return RT.running

    def execute(self, context):
        RT.paused = not RT.paused
        return {'FINISHED'}


class WM_OT_warp_sph_step_frame(bpy.types.Operator):
    bl_idname = "wm.warp_sph_step_frame"
    bl_label = "Step One Frame"
    bl_description = "Pause (if running) and advance the simulation by exactly one frame"

    @classmethod
    def poll(cls, context):
        return RT.running

    def execute(self, context):
        RT.paused = True
        RT.step_once = True
        return {'FINISHED'}


def _readback_active() -> bool:
    """True while the running transport is the Readback compatibility path (from what runs, not prefs)."""
    session = RT.session
    return session is not None and not session.transport.capabilities().zero_copy


class WM_OT_warp_sph_use_opengl(bpy.types.Operator):
    bl_idname = "wm.warp_sph_use_opengl"
    bl_label = "Use OpenGL fast mode"
    bl_description = ("Set Blender's graphics backend preference to OpenGL for zero-copy particle transport; "
                      "takes effect after a restart")

    @classmethod
    def poll(cls, context):
        return _readback_active()

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, message="Switch Blender's graphics backend to OpenGL and save preferences? "
                                 "Blender must be restarted.")

    def execute(self, context):
        system = context.preferences.system
        previous = system.gpu_backend
        system.gpu_backend = 'OPENGL'
        try:
            saved = bpy.ops.wm.save_userpref()
        except Exception as exc:
            saved, error = {'CANCELLED'}, str(exc)
        else:
            error = "preferences were not saved"
        if 'FINISHED' not in saved:
            system.gpu_backend = previous
            self.report({'ERROR'}, f"WarpSPH could not switch to OpenGL: {error}")
            return {'CANCELLED'}
        self.report({'WARNING'}, "Graphics backend set to OpenGL and saved: restart Blender to apply it")
        return {'FINISHED'}


class WM_OT_warp_sph_install(bpy.types.Operator):
    bl_idname = "wm.warp_sph_install"
    bl_label = "Install NVIDIA Warp"
    bl_description = "Download and checksum-verify Warp 1.17.0 CUDA 12 into this extension's isolated user directory"

    repair: BoolProperty(name="Repair Warp", default=False)

    def execute(self, context):
        if RT.running:
            self.report({'ERROR'}, "Stop the simulation before changing the Warp runtime")
            return {'CANCELLED'}
        result = warp_installer.install_warp(force_reinstall=self.repair)
        self.report({'INFO'} if result.success else {'ERROR'}, result.message)
        return {'FINISHED'} if result.success else {'CANCELLED'}


class WM_OT_warp_sph_copy_diagnostics(bpy.types.Operator):
    bl_idname = "wm.warp_sph_copy_diagnostics"
    bl_label = "Copy Diagnostics"
    bl_description = "Copy GPU, driver, Warp, Blender and recent error details to the clipboard for bug reports"

    def execute(self, context):
        context.window_manager.clipboard = _diagnostics_report()
        self.report({'INFO'}, "WarpSPH diagnostics copied to the clipboard")
        return {'FINISHED'}


class WARPSPH_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or "warp_sph_realtime"

    def draw(self, context):
        box = self.layout.box()
        box.label(text="NVIDIA Warp Runtime (CUDA 12.9)", icon='PREFERENCES')
        state = warp_runtime.status()
        box.label(text=state.message, icon='CHECKMARK' if state.available else 'ERROR')
        if _startup_error:
            box.label(text=_startup_error, icon='ERROR')
        if not state.available:
            box.operator("wm.warp_sph_install", text="Install NVIDIA Warp 1.17.0", icon='IMPORT')
        row = box.row()
        op = row.operator("wm.warp_sph_install", text="Repair Warp (restart required)", icon='FILE_REFRESH')
        op.repair = True
        box.label(text="WarpSPH never modifies Blender's Python or other extensions.")
        box.operator("wm.warp_sph_copy_diagnostics", icon='COPYDOWN')


class VIEW3D_PT_warp_sph(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WarpSPH"
    bl_label = "WarpSPH Realtime Fluid"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        state = warp_runtime.status()
        status_box = layout.box()
        status_box.label(text="Runtime: " + ("Installed (GPU check on Start)" if state.available else "Not ready"), icon='CHECKMARK' if state.available else 'ERROR')
        status_box.label(text=state.message)
        if _startup_error:
            status_box.label(text=_startup_error, icon='ERROR')
        if not state.available:
            status_box.operator("wm.warp_sph_install", text="Install NVIDIA Warp", icon='IMPORT')
        op = status_box.operator("wm.warp_sph_install", text="Repair Warp", icon='FILE_REFRESH')
        op.repair = True
        if RT.restart_required:
            status_box.label(text="GPU transport failed: restart Blender to run WarpSPH again", icon='ERROR')
            status_box.label(text=RT.restart_required)
        if RT.session is not None:
            status_box.label(text="Transport: " + RT.session.label,
                             icon='CHECKMARK' if RT.session.transport.capabilities().zero_copy else 'INFO')
            if RT.session.reason:
                status_box.label(text=RT.session.reason)
            if _readback_active():
                status_box.operator("wm.warp_sph_use_opengl", icon='PREFERENCES')
        elif RT.interop_unavailable:
            status_box.label(text=RT.interop_unavailable + " (Readback until restart)", icon='INFO')
        if RT.last_close and RT.last_close != COMPLETED:
            status_box.label(text=f"Last GPU cleanup: {RT.last_close} (retained until exit)", icon='ERROR')
        if RT.last_error:
            status_box.label(text=RT.last_error, icon='ERROR')
        status_box.operator("wm.warp_sph_copy_diagnostics", icon='COPYDOWN')

        # Quick-Run Demo Scene Buttons
        box_demo = layout.box()
        box_demo.label(text="Quick-Run Demo Scenes", icon='SCENE_DATA')
        col_demo = box_demo.column(align=True)
        op1 = col_demo.operator(
            "wm.warp_sph_load_demo",
            icon='MOD_FLUIDSIM',
            text="Demo 1: Double Dam + Pillars"
        )
        op1.demo_id = "DOUBLE_DAM_PILLARS"

        op2 = col_demo.operator(
            "wm.warp_sph_load_demo",
            icon='FORCE_TURBULENCE',
            text="Demo 2: Block Flop Crown Splash"
        )
        op2.demo_id = "BLOCK_FLOP_SPLASH"

        layout.separator()
        col = layout.column(align=True)
        col.prop(scene, "warp_sph_demo_scene", text="Scene")
        col.prop(scene, "warp_sph_particle_count", slider=True)
        col.prop(scene, "warp_sph_use_cuda_graph")

        box_speed = layout.box()
        box_speed.label(text="Simulation Speed vs. Stability", icon='TIME')
        col_speed = box_speed.column(align=True)
        col_speed.prop(scene, "warp_sph_substeps")
        col_speed.prop(scene, "warp_sph_time_step_scale", slider=True)
        col_speed.prop(scene, "warp_sph_stiffness")
        if RT.solver:
            col_speed.label(
                text=f"dt = {RT.solver.dt * 1000:.3f} ms  |  sim speed ≈ "
                     f"{RT.solver.dt * RT.solver.substeps * RT.fps_avg:.2f}x real time"
            )

        box_vis = layout.box()
        box_vis.label(text="Viewport Display & SSFR", icon='SHADING_RENDERED')
        col_vis = box_vis.column(align=True)
        col_vis.prop(scene, "warp_sph_show_particles")
        col_vis.prop(scene, "warp_sph_render_mode", text="Mode")
        col_vis.prop(scene, "warp_sph_particle_color", text="")
        col_vis.prop(scene, "warp_sph_depth_test")
        if RT.session is not None:
            note = settings_from_scene(scene, RT.session.backend).note
            if note:
                col_vis.label(text=note, icon='INFO')

        if scene.warp_sph_render_mode == "POINTS":
            col_vis.prop(scene, "warp_sph_point_size")
        elif scene.warp_sph_render_mode == "SPHERES":
            col_vis.prop(scene, "warp_sph_particle_radius")
        else:
            col_vis.prop(scene, "warp_sph_particle_radius")
            col_vis.prop(scene, "warp_sph_filter_iterations")
            col_vis.prop(scene, "warp_sph_filter_sigma")
            col_vis.prop(scene, "warp_sph_absorption")

        layout.separator()
        row = layout.row(align=True)
        if not RT.running:
            row.enabled = state.available and not bool(_startup_error) and not RT.restart_required
            row.operator("wm.warp_sph_start", icon='PLAY', text="Start Realtime Sim")
        else:
            if RT.paused:
                row.operator("wm.warp_sph_pause", icon='PLAY', text="Resume")
            else:
                row.operator("wm.warp_sph_pause", icon='PAUSE', text="Pause")
            row.operator("wm.warp_sph_step_frame", icon='FRAME_NEXT', text="Step")
            row.operator("wm.warp_sph_stop", icon='SNAP_FACE', text="Stop (ESC)")

        # Live GPU Telemetry Readout
        box = layout.box()
        box.label(text="GPU Performance Telemetry", icon='PREFERENCES')
        col_stat = box.column(align=True)
        status = 'IDLE' if not RT.running else ('PAUSED' if RT.paused else 'RUNNING')
        col_stat.label(text=f"Status: {status}")
        col_stat.label(text=f"Active Demo: {RT.active_demo}")
        col_stat.label(text=f"FPS Avg: {RT.fps_avg:.1f} FPS")
        col_stat.label(text=f"Frame Time: {RT.frame_time_ms:.2f} ms")
        col_stat.label(text=f"SoA VRAM Footprint: {RT.vram_mb:.1f} MB")
        if RT.running and RT.solver:
            throughput = (RT.solver.num_particles / max(RT.frame_time_ms, 0.001))
            col_stat.label(text=f"Throughput: {throughput:,.0f} particles/ms")
        if RT.running and RT.session is not None:
            drops = getattr(RT.session.transport, "drops", 0)
            col_stat.label(text=f"Transport waits: {RT.session.waits}  |  drops: {drops}")


_STOP_HANDLERS = ("load_pre", "exit_pre", "undo_pre", "redo_pre")


def _on_load_pre(*_args):
    """Stop the run before the session's Blender state is replaced (plan M4).

    A file load frees the old window/screen. Undo/redo also stop it: they swap in an earlier
    scene state (properties, helper objects) under a running solver and its window-bound
    session, and no ordering guarantee makes keeping the run alive provably safe.
    """
    _stop_session(getattr(bpy, "context", None))


def _app_handlers():
    return getattr(getattr(bpy, "app", None), "handlers", None)


def _install_handlers():
    handlers = _app_handlers()
    if handlers is None:
        return
    persistent = getattr(handlers, "persistent", None)
    if persistent is not None and not getattr(_on_load_pre, "_bpy_persistent", False):
        persistent(_on_load_pre)
    for name in _STOP_HANDLERS:
        hooks = getattr(handlers, name, None)
        if hooks is not None and _on_load_pre not in hooks:
            hooks.append(_on_load_pre)


def _remove_handlers():
    handlers = _app_handlers()
    if handlers is None:
        return
    for name in _STOP_HANDLERS:
        hooks = getattr(handlers, name, None)
        while hooks is not None and _on_load_pre in hooks:
            hooks.remove(_on_load_pre)


def register():
    global _startup_error
    _startup_error = ""
    try:
        warp_installer.apply_pending_repair_on_startup()
    except Exception as exc:
        _startup_error = str(exc)
    bpy.types.Scene.warp_sph_demo_scene = EnumProperty(
        name="Demo Scene",
        items=[
            ("DOUBLE_DAM_PILLARS", "Demo 1: Double Dam + Pillars", "Two opposing fluid columns colliding with a central sphere and 4 pillar obstacles"),
            ("BLOCK_FLOP_SPLASH", "Demo 2: Block Flop Crown Splash", "Heavy elevated fluid cube dropping into a shallow resting puddle basin"),
        ],
        default="DOUBLE_DAM_PILLARS",
    )
    bpy.types.Scene.warp_sph_particle_count = IntProperty(
        name="Particles",
        description="Total fluid particles (dynamically reallocates GPU SoA buffers on change)",
        default=1_000_000,
        min=100_000,
        max=4_000_000,
        step=100_000,
    )
    bpy.types.Scene.warp_sph_substeps = IntProperty(
        name="Sub-steps / Frame",
        description="Physics sub-steps executed per viewport redraw",
        default=2,
        min=1,
        max=10,
    )
    bpy.types.Scene.warp_sph_time_step_scale = FloatProperty(
        name="Time Step (CFL)",
        description="Physics time step as a fraction of the stability limit. Higher = faster simulation, "
                    "but above ~0.5 high-speed impacts can blow up",
        default=0.4,
        min=0.1,
        max=1.0,
        step=1,          # Blender steps are 1/100 units: 0.01 per click
        precision=3,
    )
    bpy.types.Scene.warp_sph_stiffness = FloatProperty(
        name="Fluid Stiffness",
        description="Pressure stiffness (sets sound speed). Lower = larger time step and faster simulation, "
                    "but more compressible and less stable on impacts",
        default=90000.0,
        min=10000.0,
        max=400000.0,
        step=500000,
    )
    bpy.types.Scene.warp_sph_show_particles = BoolProperty(
        name="Show Particles",
        default=True,
    )
    bpy.types.Scene.warp_sph_render_mode = EnumProperty(
        name="Render Mode",
        items=[
            ("SSFR_LIQUID", "SSFR Liquid (Narrow-Range + Beer-Lambert)", "5-pass Screen-Space Fluid Renderer with Truong & Yuksel Narrow-Range Filter and Fresnel shading"),
            ("SPHERES", "Shaded Sphere Impostors", "Single-pass eye-space shaded 3D spheres with per-fragment depth correction"),
            ("POINTS", "Raw Points (Max FPS)", "Fastest raw GL_POINTS mode for telemetry benchmarking"),
        ],
        default="SSFR_LIQUID",
    )
    bpy.types.Scene.warp_sph_point_size = FloatProperty(
        name="Point Size",
        description="Particle point size in screen pixels (Raw Points mode)",
        default=2.5,
        min=1.0,
        max=20.0,
        step=10,         # 0.1 px per click
        precision=1,
    )
    bpy.types.Scene.warp_sph_particle_radius = FloatProperty(
        name="Particle Radius (m)",
        description="World-space sphere radius for SSFR and Sphere Impostor modes",
        default=0.022,
        min=0.002,
        max=0.10,
        soft_min=0.004,  # narrow drag range so mouse travel maps to useful radii
        soft_max=0.04,
        step=0.05,       # 0.0005 m (0.5 mm) per click
        precision=4,
    )
    bpy.types.Scene.warp_sph_filter_iterations = IntProperty(
        name="Narrow-Range Filter Passes",
        description="Number of 1D separable Horizontal + Vertical Narrow-Range smoothing iterations",
        default=2,
        min=1,
        max=5,
    )
    bpy.types.Scene.warp_sph_filter_sigma = FloatProperty(
        name="Filter Sigma (px)",
        description="Screen-space Gaussian standard deviation inside the Narrow-Range depth window",
        default=4.0,
        min=1.0,
        max=12.0,
        step=10,         # 0.1 px per click
        precision=1,
    )
    bpy.types.Scene.warp_sph_absorption = FloatVectorProperty(
        name="Beer-Lambert Absorption (RGB)",
        description="Wavelength-dependent light extinction coefficients exp(-a * thickness)",
        size=3,
        default=(0.85, 0.22, 0.05),
        min=0.0,
        max=5.0,
        soft_max=2.0,
        step=1,          # 0.01 per click
        precision=3,
    )
    bpy.types.Scene.warp_sph_particle_color = FloatVectorProperty(
        name="Particle Color",
        subtype='COLOR',
        size=4,
        default=(0.12, 0.68, 1.0, 1.0),
        min=0.0,
        max=1.0,
    )
    bpy.types.Scene.warp_sph_depth_test = BoolProperty(
        name="Hide Behind Objects",
        description="Particles are hidden behind scene geometry (off = always draw on top)",
        default=True,
    )
    bpy.types.Scene.warp_sph_use_cuda_graph = BoolProperty(
        name="Warp 1.17 CUDA Graph Capture",
        description="Execute Coalesced Density + WCSPH Force + Copy-back via captured CUDA Graph",
        default=True,
    )
    bpy.utils.register_class(WM_OT_warp_sph_start)
    bpy.utils.register_class(WM_OT_warp_sph_load_demo)
    bpy.utils.register_class(WM_OT_warp_sph_stop)
    bpy.utils.register_class(WM_OT_warp_sph_pause)
    bpy.utils.register_class(WM_OT_warp_sph_step_frame)
    bpy.utils.register_class(WM_OT_warp_sph_use_opengl)
    bpy.utils.register_class(WM_OT_warp_sph_install)
    bpy.utils.register_class(WM_OT_warp_sph_copy_diagnostics)
    bpy.utils.register_class(WARPSPH_Preferences)
    bpy.utils.register_class(VIEW3D_PT_warp_sph)
    _install_handlers()
    try:
        diagnostics.setup_logging(_log_dir())
    except Exception:  # logging is best effort and must not block registration
        pass


def unregister():
    diagnostics.teardown_logging()
    _remove_handlers()
    _stop_session(getattr(bpy, "context", None))
    bpy.utils.unregister_class(VIEW3D_PT_warp_sph)
    bpy.utils.unregister_class(WM_OT_warp_sph_step_frame)
    bpy.utils.unregister_class(WM_OT_warp_sph_pause)
    bpy.utils.unregister_class(WARPSPH_Preferences)
    bpy.utils.unregister_class(WM_OT_warp_sph_copy_diagnostics)
    bpy.utils.unregister_class(WM_OT_warp_sph_install)
    bpy.utils.unregister_class(WM_OT_warp_sph_use_opengl)
    bpy.utils.unregister_class(WM_OT_warp_sph_stop)
    bpy.utils.unregister_class(WM_OT_warp_sph_load_demo)
    bpy.utils.unregister_class(WM_OT_warp_sph_start)
    del bpy.types.Scene.warp_sph_use_cuda_graph
    del bpy.types.Scene.warp_sph_depth_test
    del bpy.types.Scene.warp_sph_particle_color
    del bpy.types.Scene.warp_sph_absorption
    del bpy.types.Scene.warp_sph_filter_sigma
    del bpy.types.Scene.warp_sph_filter_iterations
    del bpy.types.Scene.warp_sph_particle_radius
    del bpy.types.Scene.warp_sph_point_size
    del bpy.types.Scene.warp_sph_render_mode
    del bpy.types.Scene.warp_sph_show_particles
    del bpy.types.Scene.warp_sph_stiffness
    del bpy.types.Scene.warp_sph_time_step_scale
    del bpy.types.Scene.warp_sph_substeps
    del bpy.types.Scene.warp_sph_particle_count
    del bpy.types.Scene.warp_sph_demo_scene
