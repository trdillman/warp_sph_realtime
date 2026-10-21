"""Blender operators, panels and the modal timer: thin UI over ``sim_session`` and ``props``.

All runtime state lives in one ``sim_session.SimSession`` per scene; operators, panels and the timer
call it. Scene settings are the ``props.WARPSPH_SceneSettings`` group at ``scene.warp_sph``. NVIDIA
Warp and every CUDA-dependent module are imported only in ``_activate_runtime()``, after the
owned/version/device checks, so the extension registers (and can install Warp) on machines without it.
"""

from __future__ import annotations

import importlib

import bpy
from bpy.props import EnumProperty, PointerProperty

from . import diagnostics, install_ui, props, sim_session, warp_installer, warp_runtime
from .render.gpu_points import settings_from_scene
from .render.transport import COMPLETED
from .sim_session import CUSTOM_SCENE

_startup_error = ""


def _activate_runtime():
    """Import CUDA-dependent modules only after owned/version/device checks."""
    runtime = sim_session.runtime
    if runtime.wp is not None:
        return  # already active; keep existing bindings
    loaded = warp_runtime.require_cuda_runtime()
    bridge = importlib.import_module(".render.interop_bridge", __package__)
    viewport = importlib.import_module(".render.viewport_draw", __package__)
    solver = importlib.import_module(".sim.sim_core", __package__)
    demos = importlib.import_module(".scene.demo_scenes", __package__)
    runtime.wp = loaded
    runtime.ZeroCopyGLBuffer52 = bridge.ZeroCopyGLBuffer52
    runtime.RealtimeWCSPHSolver = solver.RealtimeWCSPHSolver
    runtime.build_blender_demo_helpers = demos.build_blender_demo_helpers
    runtime.remove_blender_demo_helpers = demos.remove_blender_demo_helpers
    runtime.launch_demo_scene_init = demos.launch_demo_scene_init
    runtime.cleanup_viewport_renderer = viewport.cleanup_viewport_renderer
    runtime.draw_particles_viewport = viewport.draw_particles_viewport


def _scene_api():
    """NumPy-backed scene helpers (scene.blender_scene), imported on first use so the add-on still
    registers when no third-party package can be imported."""
    return importlib.import_module(".scene.blender_scene", __package__)


def _session(context) -> sim_session.SimSession:
    """The session of the context's scene."""
    return sim_session.get(context.scene)


def _log_dir():
    return bpy.utils.extension_path_user(__package__, path="logs", create=True)


def _addon_version() -> str:
    import tomllib
    from pathlib import Path
    with open(Path(__file__).with_name("blender_manifest.toml"), "rb") as handle:
        return tomllib.load(handle)["version"]


def _warp_owner() -> str:
    return warp_installer.describe_source()


def _diagnostics_report(session: sim_session.SimSession) -> str:
    """Never raises; every probe is isolated (see diagnostics.build_report)."""
    wp = sim_session.runtime.wp
    data = {**diagnostics.system_probes(), **diagnostics.warp_probes(wp)}
    data.update(
        addon_version=_addon_version,
        blender_version=lambda: bpy.app.version_string,
        blender_build_hash=lambda: bpy.app.build_hash.decode(errors="replace"),
        gpu_backend=sim_session.gpu_backend,
        warp_source=_warp_owner,
        transport=lambda: session.session.label if session.session is not None else "none",
        last_error=session.last_error,
        last_close=session.last_close,
        interop_unavailable=session.interop_unavailable,
    )
    if wp is None:
        data["warp_version"] = lambda: warp_runtime.status().version or "not loaded (GPU details need one Start)"
    return diagnostics.build_report(data, lambda: diagnostics.read_log_tail(diagnostics.log_path()))


class WM_OT_warp_sph_start(bpy.types.Operator):
    bl_idname = "wm.warp_sph_start"
    bl_label = "Start WarpSPH Realtime"
    _token = -1
    _session: sim_session.SimSession | None = None

    def modal(self, context, event):
        session = self._session
        if session is None or self._token != session.token:
            return {'CANCELLED'}  # already stopped by unregister, file load or cancel
        if not session.running or event.type == 'ESC':
            session.stop(context)
            return {'CANCELLED'}
        if event.type == 'TIMER' and not session.tick(context, self.report):
            return {'CANCELLED'}  # the session stopped itself and reported why
        return {'PASS_THROUGH'}

    def execute(self, context):
        session = _session(context)
        if session.running:
            return {'FINISHED'}
        other = sim_session.active()
        if other is not None:  # one run per process: the solver, timer and draw handler are shared
            self.report({'ERROR'}, f"WarpSPH is already running in scene '{sim_session.scene_name(other.scene)}'; "
                                   "stop it first")
            return {'CANCELLED'}
        token = session.start(context, self.report, _activate_runtime)
        if token is None:
            return {'CANCELLED'}
        try:
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            session.stop(context)
            self.report({'ERROR'}, f"WarpSPH cannot start: {exc}")
            return {'CANCELLED'}
        self._session, self._token = session, token
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        # Blender cancels the modal (window close, file load); the owner window may be gone,
        # in which case the transport releases CUDA objects and quarantines Blender ones.
        if self._session is not None and self._token == self._session.token:
            self._session.stop(context)


class WM_OT_warp_sph_load_demo(bpy.types.Operator):
    bl_idname = "wm.warp_sph_load_demo"
    bl_label = "Quick-Launch Demo Scene"

    demo_id: bpy.props.StringProperty(default="DOUBLE_DAM_PILLARS")

    def execute(self, context):
        settings = context.scene.warp_sph
        settings.scene_source = 'DEMO'
        settings.demo_scene = self.demo_id
        session = _session(context)
        if session.running:
            if not session.apply_scene(context, settings.particle_count, self.demo_id, self.report):
                return {'CANCELLED'}
        else:
            bpy.ops.wm.warp_sph_start()
        return {'FINISHED'}


class WM_OT_warp_sph_stop(bpy.types.Operator):
    bl_idname = "wm.warp_sph_stop"
    bl_label = "Stop Simulation"

    def execute(self, context):
        _session(context).running = False  # the modal timer tears the session down
        return {'FINISHED'}


class WM_OT_warp_sph_pause(bpy.types.Operator):
    bl_idname = "wm.warp_sph_pause"
    bl_label = "Pause / Resume Simulation"
    bl_description = "Freeze or continue the physics without releasing GPU buffers"

    @classmethod
    def poll(cls, context):
        return _session(context).running

    def execute(self, context):
        session = _session(context)
        session.paused = not session.paused
        return {'FINISHED'}


class WM_OT_warp_sph_step_frame(bpy.types.Operator):
    bl_idname = "wm.warp_sph_step_frame"
    bl_label = "Step One Frame"
    bl_description = "Pause (if running) and advance the simulation by exactly one frame"

    @classmethod
    def poll(cls, context):
        return _session(context).running

    def execute(self, context):
        session = _session(context)
        session.paused = True
        session.step_once = True
        return {'FINISHED'}


class WM_OT_warp_sph_use_opengl(bpy.types.Operator):
    bl_idname = "wm.warp_sph_use_opengl"
    bl_label = "Use OpenGL fast mode"
    bl_description = ("Set Blender's graphics backend preference to OpenGL for zero-copy particle transport; "
                      "takes effect after a restart")

    @classmethod
    def poll(cls, context):
        return _session(context).readback_active()

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


ROLE_ITEMS = [
    ("NONE", "None", "Not part of the WarpSPH simulation"),
    ("DOMAIN", "Domain", "Simulation box: the object's world-space axis-aligned bounding box (use a cube). "
                         "Exactly one per scene"),
    ("FLUID", "Fluid", "Closed mesh volume, filled with fluid particles when the simulation starts"),
    ("COLLIDER", "Collider", "Closed, static mesh obstacle"),
]


class WARPSPH_ObjectSettings(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="WarpSPH Role",
        description="How this object takes part in a Custom WarpSPH scene (read when the simulation starts)",
        items=ROLE_ITEMS,
        default="NONE",
    )


def _assign_role(obj, role: str):
    obj.warp_sph.role = role
    if obj.get(_scene_api().ROLE_PROP) is not None:
        del obj[_scene_api().ROLE_PROP]  # the legacy property must not override a cleared role
    if role in ("DOMAIN", "FLUID"):
        obj.display_type = 'WIRE'  # a solid box would hide the particles inside it


class OBJECT_OT_warp_sph_set_role(bpy.types.Operator):
    bl_idname = "object.warp_sph_set_role"
    bl_label = "Set WarpSPH Role"
    bl_description = ("Set the WarpSPH role of the selected mesh objects (Domain and Fluid are shown as "
                      "wireframe so the particles stay visible)")
    bl_options = {'REGISTER', 'UNDO'}

    role: EnumProperty(name="Role", items=ROLE_ITEMS, default="NONE")

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in (getattr(context, "selected_objects", None) or ()))

    def execute(self, context):
        selected = list(getattr(context, "selected_objects", None) or ())
        meshes = [o for o in selected if o.type == 'MESH']
        if not meshes:
            self.report({'ERROR'}, "Select one or more mesh objects")
            return {'CANCELLED'}
        if self.role == "DOMAIN":
            if len(meshes) != 1:
                self.report({'ERROR'}, "Select exactly one mesh to use as the domain")
                return {'CANCELLED'}
            for other in context.scene.objects:  # exactly one domain: the new one replaces the old
                if other is not meshes[0] and _scene_api().object_role(other) == "DOMAIN":
                    _assign_role(other, "NONE")
        for obj in meshes:
            _assign_role(obj, self.role)
        label = dict((key, name) for key, name, _ in ROLE_ITEMS)[self.role]
        message = f"WarpSPH role {label}: {', '.join(o.name for o in meshes)}"
        if len(meshes) < len(selected):
            message += f" ({len(selected) - len(meshes)} non-mesh object(s) skipped)"
        session = _session(context)
        if session.running and session.active_demo == CUSTOM_SCENE:
            message += "; Stop and Start to apply"
        self.report({'INFO'}, message)
        return {'FINISHED'}


# Corner i of a box has x/y/z from bit 0/1/2; quads are counter-clockwise seen from outside.
_BOX_QUADS = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))

# name, size (m), location, rotation about Z (radians), role. Domain [-1, 1] x [-1, 1] x [0, 2]; the fluid
# column (~90k particles at the default 0.02 m spacing) collapses against a tilted block.
_STARTER_SCENE = (
    ("WarpSPH Domain", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0), 0.0, "DOMAIN"),
    ("WarpSPH Fluid", (0.6, 1.2, 1.0), (-0.6, 0.0, 0.55), 0.0, "FLUID"),
    ("WarpSPH Collider", (0.3, 0.6, 0.5), (0.4, 0.0, 0.3), 0.5236, "COLLIDER"),
)


def _add_box(collection, name, size, location, rotation_z):
    hx, hy, hz = (0.5 * v for v in size)
    corners = [((hx if i & 1 else -hx), (hy if i & 2 else -hy), (hz if i & 4 else -hz)) for i in range(8)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], _BOX_QUADS)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = (0.0, 0.0, rotation_z)
    collection.objects.link(obj)
    return obj


class OBJECT_OT_warp_sph_add_fluid_scene(bpy.types.Operator):
    bl_idname = "object.warp_sph_add_fluid_scene"
    bl_label = "Add Fluid Scene"
    bl_description = ("Add a starter Custom scene: a domain cube, a fluid block and one collider, then switch "
                      "the scene source to Custom")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        collection = getattr(context, "collection", None) or context.scene.collection
        if any(_scene_api().object_role(o) == "DOMAIN" for o in context.scene.objects):
            self.report({'ERROR'}, "This scene already has a WarpSPH domain; clear its role first")
            return {'CANCELLED'}
        for name, size, location, rotation_z, role in _STARTER_SCENE:
            _assign_role(_add_box(collection, name, size, location, rotation_z), role)
        context.scene.warp_sph.scene_source = 'CUSTOM'
        self.report({'INFO'}, "Added a WarpSPH domain, fluid and collider; press Start Realtime Sim")
        return {'FINISHED'}


class WM_OT_warp_sph_copy_diagnostics(bpy.types.Operator):
    bl_idname = "wm.warp_sph_copy_diagnostics"
    bl_label = "Copy Diagnostics"
    bl_description = "Copy GPU, driver, Warp, Blender and recent error details to the clipboard for bug reports"

    def execute(self, context):
        context.window_manager.clipboard = _diagnostics_report(sim_session.active() or _session(context))
        self.report({'INFO'}, "WarpSPH diagnostics copied to the clipboard")
        return {'FINISHED'}


class OBJECT_PT_warp_sph(bpy.types.Panel):
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"
    bl_label = "WarpSPH Realtime Fluid"

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and obj.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        layout.prop(obj.warp_sph, "role", expand=True)
        role = _scene_api().object_role(obj)
        help_text = {
            "DOMAIN": "Simulation box: this object's world bounding box",
            "FLUID": "Closed mesh, filled with particles on Start",
            "COLLIDER": "Closed static obstacle",
        }.get(role or "", "Not part of the simulation")
        layout.label(text=help_text, icon='INFO')
        if obj.warp_sph.role == "NONE" and role is not None:
            layout.label(text=f"Role {role} from the legacy 'warp_sph_role' property")
        if context.scene.warp_sph.scene_source != 'CUSTOM':
            layout.label(text="Roles are used when the WarpSPH scene source is Custom")
        session = _session(context)
        if session.running and session.active_demo == CUSTOM_SCENE:
            layout.label(text="Running: Stop and Start to apply role or geometry changes", icon='INFO')


def _draw_vram_warning(layout, particles: int):
    warning = props.vram_warning(particles, sim_session.device_total_mb())
    if warning:
        layout.label(text=warning, icon='ERROR')


def _draw_custom_scene(box, scene, session):
    settings = scene.warp_sph
    try:
        by_role = _scene_api().objects_by_role(scene)
        warnings = _scene_api().scene_warnings(scene)
    except Exception as exc:  # never break the panel; Start reports the full validation
        by_role, warnings = {}, [f"Cannot read scene roles: {exc}"]
    col = box.column(align=True)
    col.prop(settings, "particle_spacing")
    solver = session.solver
    custom_running = session.running and session.active_demo == CUSTOM_SCENE and solver is not None
    if custom_running:
        col.label(text=f"Particles: {solver.num_particles:,} (from the fill)")
        _draw_vram_warning(col, solver.num_particles)
    else:
        _draw_vram_warning(col, props.custom_particle_estimate(by_role.get("FLUID", []), settings.particle_spacing))
    box.operator("object.warp_sph_add_fluid_scene", icon='ADD')
    box.label(text="Set selected objects as:")
    row = box.row(align=True)
    for role, text, icon in (("DOMAIN", "Domain", 'MESH_CUBE'), ("FLUID", "Fluid", 'MOD_FLUIDSIM'),
                             ("COLLIDER", "Collider", 'PHYSICS'), ("NONE", "Clear", 'X')):
        row.operator("object.warp_sph_set_role", text=text, icon=icon).role = role
    col = box.column(align=True)
    for role, title in (("DOMAIN", "Domain"), ("FLUID", "Fluid"), ("COLLIDER", "Colliders")):
        objs = by_role.get(role, [])
        names = ", ".join(o.name for o in objs[:3]) + (", ..." if len(objs) > 3 else "")
        col.label(text=f"{title} ({len(objs)}): {names or '-'}")
    for warning in warnings:
        box.label(text=warning, icon='ERROR')
    if custom_running:
        box.label(text="Running: Stop and Start to apply scene changes", icon='INFO')


def _draw_demo_scene(box, settings):
    col_demo = box.column(align=True)
    op1 = col_demo.operator("wm.warp_sph_load_demo", icon='MOD_FLUIDSIM', text="Demo 1: Double Dam + Pillars")
    op1.demo_id = "DOUBLE_DAM_PILLARS"
    op2 = col_demo.operator("wm.warp_sph_load_demo", icon='FORCE_TURBULENCE', text="Demo 2: Block Flop Crown Splash")
    op2.demo_id = "BLOCK_FLOP_SPLASH"
    col = box.column(align=True)
    col.prop(settings, "demo_scene", text="Demo")
    col.prop(settings, "particle_count", slider=True)
    _draw_vram_warning(col, settings.particle_count)


class VIEW3D_PT_warp_sph(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WarpSPH"
    bl_label = "WarpSPH Realtime Fluid"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        session = _session(context)
        other = sim_session.active()
        if other is session:
            other = None
        state = warp_runtime.status()
        status_box = layout.box()
        status_box.label(text="Runtime: " + ("Installed (GPU check on Start)" if state.available else "Not ready"), icon='CHECKMARK' if state.available else 'ERROR')
        status_box.label(text=state.message)
        if _startup_error:
            status_box.label(text=_startup_error, icon='ERROR')
        if install_ui.job_active():
            status_box.label(text=install_ui.job_status_text(), icon='TIME')
        elif not warp_installer.active_source().is_folder:
            if not state.available:
                status_box.operator("wm.warp_sph_install", text="Install NVIDIA Warp", icon='IMPORT')
            op = status_box.operator("wm.warp_sph_install", text="Repair Warp", icon='FILE_REFRESH')
            op.repair = True
        if session.restart_required:
            status_box.label(text="GPU transport failed: restart Blender to run WarpSPH again", icon='ERROR')
            status_box.label(text=session.restart_required)
        if session.session is not None:
            status_box.label(text="Transport: " + session.session.label,
                             icon='CHECKMARK' if session.session.transport.capabilities().zero_copy else 'INFO')
            if session.session.reason:
                status_box.label(text=session.session.reason)
            if session.readback_active():
                status_box.operator("wm.warp_sph_use_opengl", icon='PREFERENCES')
        elif session.interop_unavailable and session.gl_buf is None:
            status_box.label(text=session.interop_unavailable + " (Readback until restart)", icon='INFO')
        if session.last_close and session.last_close != COMPLETED:
            status_box.label(text=f"Last GPU cleanup: {session.last_close} (retained until exit)", icon='ERROR')
        if session.last_error:
            status_box.label(text=session.last_error, icon='ERROR')
        if other is not None:
            status_box.label(text=f"Running in scene '{sim_session.scene_name(other.scene)}': stop it there first",
                             icon='INFO')
        status_box.operator("wm.warp_sph_copy_diagnostics", icon='COPYDOWN')

        settings = scene.warp_sph
        box_demo = layout.box()
        box_demo.label(text="Scene", icon='SCENE_DATA')
        box_demo.row().prop(settings, "scene_source", expand=True)
        if settings.scene_source == 'CUSTOM':
            _draw_custom_scene(box_demo, scene, session)
        else:
            _draw_demo_scene(box_demo, settings)
        if session.running and (session.active_demo == CUSTOM_SCENE) != (settings.scene_source == 'CUSTOM'):
            box_demo.label(text="Scene source changes apply on the next Start", icon='INFO')

        box_speed = layout.box()
        box_speed.label(text="Simulation Speed vs. Stability", icon='TIME')
        col_speed = box_speed.column(align=True)
        col_speed.prop(settings, "substeps")
        col_speed.prop(settings, "time_step_scale", slider=True)
        if session.solver:
            col_speed.label(
                text=f"dt = {session.solver.dt * 1000:.3f} ms  |  sim speed ≈ "
                     f"{session.solver.dt * session.solver.substeps * session.fps_avg:.2f}x real time"
            )

        box_vis = layout.box()
        box_vis.label(text="Viewport Display", icon='SHADING_RENDERED')
        col_vis = box_vis.column(align=True)
        col_vis.prop(settings, "show_particles")
        col_vis.prop(settings, "render_mode", text="Mode")
        col_vis.prop(settings, "particle_color", text="")
        col_vis.prop(settings, "depth_test")
        if session.session is not None:
            note = settings_from_scene(scene, session.session.backend).note
            if note:
                col_vis.label(text=note, icon='INFO')
        if settings.render_mode == "POINTS":
            col_vis.prop(settings, "point_size")
        else:
            col_vis.prop(settings, "particle_radius")

        layout.separator()
        row = layout.row(align=True)
        if not session.running:
            row.enabled = (state.available and not bool(_startup_error) and not session.restart_required
                           and other is None)
            row.operator("wm.warp_sph_start", icon='PLAY', text="Start Realtime Sim")
        else:
            if session.paused:
                row.operator("wm.warp_sph_pause", icon='PLAY', text="Resume")
            else:
                row.operator("wm.warp_sph_pause", icon='PAUSE', text="Pause")
            row.operator("wm.warp_sph_step_frame", icon='FRAME_NEXT', text="Step")
            row.operator("wm.warp_sph_stop", icon='SNAP_FACE', text="Stop (ESC)")

        # Live GPU Telemetry Readout
        box = layout.box()
        box.label(text="GPU Performance Telemetry", icon='PREFERENCES')
        col_stat = box.column(align=True)
        status = 'IDLE' if not session.running else ('PAUSED' if session.paused else 'RUNNING')
        col_stat.label(text=f"Status: {status}")
        if session.active_demo == CUSTOM_SCENE:
            col_stat.label(text="Active Scene: Custom (tagged objects)")
        else:
            col_stat.label(text=f"Active Demo: {session.active_demo}")
        col_stat.label(text=f"FPS Avg: {session.fps_avg:.1f} FPS")
        col_stat.label(text=f"Frame Time: {session.frame_time_ms:.2f} ms")
        col_stat.label(text=f"SoA VRAM Footprint: {session.vram_mb:.1f} MB")
        if session.running and session.solver:
            throughput = (session.solver.num_particles / max(session.frame_time_ms, 0.001))
            col_stat.label(text=f"Throughput: {throughput:,.0f} particles/ms")
        if session.running and session.session is not None:
            drops = getattr(session.session.transport, "drops", 0)
            col_stat.label(text=f"Transport waits: {session.session.waits}  |  drops: {drops}")


class VIEW3D_PT_warp_sph_advanced(bpy.types.Panel):
    """Solver and SSFR tuning most users never need: collapsed under the main panel."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WarpSPH"
    bl_label = "Advanced"
    bl_parent_id = "VIEW3D_PT_warp_sph"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.warp_sph
        col = layout.column(align=True)
        col.prop(settings, "stiffness")
        col.prop(settings, "use_cuda_graph")
        if settings.render_mode == "SSFR_LIQUID":
            col = layout.column(align=True)
            col.label(text="SSFR Liquid", icon='SHADING_RENDERED')
            col.prop(settings, "filter_iterations")
            col.prop(settings, "filter_sigma")
            col.prop(settings, "absorption")


_STOP_HANDLERS = ("load_pre", "exit_pre", "undo_pre", "redo_pre")


def _on_load_pre(*_args):
    """Stop the run before the session's Blender state is replaced (plan M4).

    A file load frees the old window/screen. Undo/redo also stop it: they swap in an earlier
    scene state (properties, helper objects) under a running solver and its window-bound
    session, and no ordering guarantee makes keeping the run alive provably safe.
    """
    sim_session.stop_all(getattr(bpy, "context", None))


def _migrate_open_file():
    """Move the flat ``warp_sph_*`` settings of a file saved by an older version into ``scene.warp_sph``.
    Returns None, so it also serves as a one-shot ``bpy.app.timers`` callback."""
    try:
        scenes = list(bpy.data.scenes)
    except AttributeError:  # bpy.data is restricted (add-on registration, Blender start-up)
        return None
    props.migrate_scenes(scenes)
    return None


def _on_load_post(*_args):
    _migrate_open_file()


def _schedule_open_file_migration():
    """register() runs with bpy.data restricted (addon_utils' RestrictBlend), so the file that is open
    when the add-on is enabled or updated is migrated right after registration, from a timer."""
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is not None and not timers.is_registered(_migrate_open_file):
        timers.register(_migrate_open_file, first_interval=0.0)


def _cancel_open_file_migration():
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is not None and timers.is_registered(_migrate_open_file):
        timers.unregister(_migrate_open_file)


def _app_handlers():
    return getattr(getattr(bpy, "app", None), "handlers", None)


def _hooks():
    """(handler, app.handlers list names) pairs this add-on installs."""
    return ((_on_load_pre, _STOP_HANDLERS), (_on_load_post, ("load_post",)))


def _install_handlers():
    handlers = _app_handlers()
    if handlers is None:
        return
    persistent = getattr(handlers, "persistent", None)
    for handler, names in _hooks():
        if persistent is not None and not getattr(handler, "_bpy_persistent", False):
            persistent(handler)
        for name in names:
            hooks = getattr(handlers, name, None)
            if hooks is not None and handler not in hooks:
                hooks.append(handler)


def _remove_handlers():
    handlers = _app_handlers()
    if handlers is None:
        return
    for handler, names in _hooks():
        for name in names:
            hooks = getattr(handlers, name, None)
            while hooks is not None and handler in hooks:
                hooks.remove(handler)


def register():
    global _startup_error
    _startup_error = ""
    try:
        warp_installer.apply_pending_repair_on_startup()
    except Exception as exc:
        _startup_error = str(exc)
    bpy.utils.register_class(WARPSPH_ObjectSettings)
    bpy.types.Object.warp_sph = PointerProperty(type=WARPSPH_ObjectSettings)
    bpy.utils.register_class(props.WARPSPH_SceneSettings)
    bpy.types.Scene.warp_sph = PointerProperty(type=props.WARPSPH_SceneSettings)
    bpy.utils.register_class(WM_OT_warp_sph_start)
    bpy.utils.register_class(WM_OT_warp_sph_load_demo)
    bpy.utils.register_class(WM_OT_warp_sph_stop)
    bpy.utils.register_class(WM_OT_warp_sph_pause)
    bpy.utils.register_class(WM_OT_warp_sph_step_frame)
    bpy.utils.register_class(WM_OT_warp_sph_use_opengl)
    bpy.utils.register_class(WM_OT_warp_sph_copy_diagnostics)
    bpy.utils.register_class(OBJECT_OT_warp_sph_set_role)
    bpy.utils.register_class(OBJECT_OT_warp_sph_add_fluid_scene)
    install_ui.register(
        is_running=lambda: sim_session.active() is not None, startup_error=lambda: _startup_error,
        active_warp=lambda: sim_session.runtime.wp)
    bpy.utils.register_class(VIEW3D_PT_warp_sph)
    bpy.utils.register_class(VIEW3D_PT_warp_sph_advanced)
    bpy.utils.register_class(OBJECT_PT_warp_sph)
    _install_handlers()
    _schedule_open_file_migration()  # enabling or updating the add-on with an older file already open
    try:
        diagnostics.setup_logging(_log_dir())
    except Exception:  # logging is best effort and must not block registration
        pass


def unregister():
    diagnostics.teardown_logging()
    _remove_handlers()
    _cancel_open_file_migration()
    sim_session.dispose_all(getattr(bpy, "context", None))
    bpy.utils.unregister_class(OBJECT_PT_warp_sph)
    bpy.utils.unregister_class(VIEW3D_PT_warp_sph_advanced)
    bpy.utils.unregister_class(VIEW3D_PT_warp_sph)
    bpy.utils.unregister_class(WM_OT_warp_sph_step_frame)
    bpy.utils.unregister_class(WM_OT_warp_sph_pause)
    install_ui.unregister()
    bpy.utils.unregister_class(OBJECT_OT_warp_sph_add_fluid_scene)
    bpy.utils.unregister_class(OBJECT_OT_warp_sph_set_role)
    bpy.utils.unregister_class(WM_OT_warp_sph_copy_diagnostics)
    bpy.utils.unregister_class(WM_OT_warp_sph_use_opengl)
    bpy.utils.unregister_class(WM_OT_warp_sph_stop)
    bpy.utils.unregister_class(WM_OT_warp_sph_load_demo)
    bpy.utils.unregister_class(WM_OT_warp_sph_start)
    del bpy.types.Scene.warp_sph
    bpy.utils.unregister_class(props.WARPSPH_SceneSettings)
    del bpy.types.Object.warp_sph
    bpy.utils.unregister_class(WARPSPH_ObjectSettings)
