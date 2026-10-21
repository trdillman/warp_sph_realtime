"""Record / bake UI (P1.6, P2.2): operators and the sub-panel under the main WarpSPH panel.

Thin: the bake itself is ``SimSession.begin_bake`` (driven by the session's modal timer in small chunks),
the files are ``cache.disk.DiskCache`` and playback is ``cache.blender_adapter``. This module only turns
button presses into those calls and shows their state.
"""
from __future__ import annotations

import importlib
import time

import bpy

from . import props, sim_session


def _adapter():
    return importlib.import_module(".cache.blender_adapter", __package__)


def _disk():
    return importlib.import_module(".cache.disk", __package__)


def _session(context) -> sim_session.SimSession:
    return sim_session.get(context.scene)


def _cache_dir(scene):
    return _adapter().cache_dir(scene)


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return ""


_status_memo: dict = {}


def cache_status(directory, max_age: float = 1.0):
    """(cached range or None, size in bytes) of a cache folder; memoised briefly because panels redraw often."""
    now = time.monotonic()
    memo = _status_memo.get(str(directory))
    if memo is not None and now - memo[0] < max_age:
        return memo[1]
    disk = _disk().DiskCache(directory)
    try:
        value = (disk.cached_range(), disk.size_bytes())
    except _disk().CacheError:
        value = (None, 0)
    _status_memo[str(directory)] = (now, value)
    return value


def _forget_status():
    _status_memo.clear()


def particle_estimate(scene, session) -> int:
    """Particles a bake of this scene would have: the running solver's count, else the settings' estimate."""
    if session.solver is not None:
        return int(session.solver.num_particles)
    settings = scene.warp_sph
    if settings.scene_source != 'CUSTOM':
        return int(settings.particle_count)
    try:
        api = importlib.import_module(".scene.blender_scene", __package__)
        return props.custom_particle_estimate(api.objects_by_role(scene).get("FLUID", []), settings.particle_spacing)
    except Exception:
        return 0


class WM_OT_warp_sph_bake(bpy.types.Operator):
    bl_idname = "wm.warp_sph_bake"
    bl_label = "Bake"
    bl_description = ("Simulate the frame range from the initial state and save every frame to the cache folder "
                      "(starts the simulation if needed). Resume continues after the last cached frame")

    resume: bpy.props.BoolProperty(default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return _session(context).bake is None

    def invoke(self, context, event):
        cached, _size = cache_status(_cache_dir(context.scene), max_age=0.0)
        if not self.resume and cached is not None:
            return context.window_manager.invoke_confirm(
                self, event, message=f"Replace the {cached[1] - cached[0] + 1} cached frames?")
        return self.execute(context)

    def execute(self, context):
        session = _session(context)
        settings = context.scene.warp_sph
        if not session.running:
            other = sim_session.active()
            if other is not None:
                self.report({'ERROR'}, "WarpSPH is already running in another scene; stop it first")
                return {'CANCELLED'}
            try:
                bpy.ops.wm.warp_sph_start()
            except RuntimeError as exc:
                self.report({'ERROR'}, f"WarpSPH cannot start: {exc}")
                return {'CANCELLED'}
            if not session.running:
                return {'CANCELLED'}  # Start reported why
        _forget_status()
        started = session.begin_bake(
            context, self.report, first=settings.bake_start, last=settings.bake_end, resume=self.resume)
        return {'FINISHED'} if started else {'CANCELLED'}


class WM_OT_warp_sph_bake_cancel(bpy.types.Operator):
    bl_idname = "wm.warp_sph_bake_cancel"
    bl_label = "Cancel Bake"
    bl_description = "Stop baking. The frames saved so far stay valid and Resume continues from them"

    @classmethod
    def poll(cls, context):
        return _session(context).bake is not None

    def execute(self, context):
        _session(context).cancel_bake()
        return {'FINISHED'}


class WM_OT_warp_sph_cache_invalidate(bpy.types.Operator):
    bl_idname = "wm.warp_sph_cache_invalidate"
    bl_label = "Invalidate From Current Frame"
    bl_description = ("Delete the cached frames from the current frame onward, so a Resume re-simulates them "
                      "(for example after changing a setting that applies from that frame)")

    @classmethod
    def poll(cls, context):
        return _session(context).bake is None

    def execute(self, context):
        disk = _disk().DiskCache(_cache_dir(context.scene))
        try:
            removed = disk.invalidate_from(context.scene.frame_current)
        except _disk().CacheError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        _forget_status()
        self.report({'INFO'}, f"Invalidated {removed} cached frames from frame {context.scene.frame_current}")
        return {'FINISHED'}


class WM_OT_warp_sph_cache_clear(bpy.types.Operator):
    bl_idname = "wm.warp_sph_cache_clear"
    bl_label = "Delete Cache"
    bl_description = "Delete every cached frame of this scene"

    @classmethod
    def poll(cls, context):
        return _session(context).bake is None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event, message="Delete all cached frames of this scene?")

    def execute(self, context):
        try:
            removed = _disk().DiskCache(_cache_dir(context.scene)).clear()
        except _disk().CacheError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        _forget_status()
        self.report({'INFO'}, f"Deleted {removed} cached frames")
        return {'FINISHED'}


class WM_OT_warp_sph_cache_playback(bpy.types.Operator):
    bl_idname = "wm.warp_sph_cache_playback"
    bl_label = "Show Cache"
    bl_description = ("Add a point cloud object that follows the timeline with the cached frames "
                      "(renders in Cycles; has velocity and id attributes)")

    show: bpy.props.BoolProperty(default=True, options={'SKIP_SAVE'})

    def execute(self, context):
        adapter = _adapter()
        scene = context.scene
        if not self.show:
            adapter.remove_playback_object(scene)
            return {'FINISHED'}
        cached, _size = cache_status(_cache_dir(scene), max_age=0.0)
        if cached is None:
            self.report({'ERROR'}, "Nothing is cached yet: bake first")
            return {'CANCELLED'}
        adapter.ensure_playback_object(scene)
        try:
            adapter.show_frame(scene, scene.frame_current)
        except _disk().CacheError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class VIEW3D_PT_warp_sph_cache(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WarpSPH"
    bl_label = "Record / Bake Cache"
    bl_parent_id = "VIEW3D_PT_warp_sph"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.warp_sph
        session = _session(context)
        baking = session.bake
        cache_disk = _disk()
        directory = _cache_dir(scene)

        col = layout.column(align=True)
        col.enabled = baking is None
        col.prop(settings, "cache_dir")
        col.label(text=str(directory), icon='FILE_FOLDER')
        row = col.row(align=True)
        row.prop(settings, "bake_start")
        row.prop(settings, "bake_end")
        col.prop(settings, "bake_steps")

        frames = settings.bake_end - settings.bake_start + 1
        if frames < 1:
            layout.label(text="End must not be before Start", icon='ERROR')
        else:
            particles = particle_estimate(scene, session)
            need = cache_disk.estimate_bytes(particles, frames)
            free = cache_disk.free_bytes(directory)
            text = f"Size: {'~' if session.solver is None else ''}{_human(need)} ({frames} frames"
            text += f" x {_human(cache_disk.frame_bytes(particles))})"
            layout.label(text=text, icon='DISK_DRIVE')
            if free is not None and need > free:
                layout.label(text=f"Only {_human(free)} free on this drive", icon='ERROR')

        cached, size = cache_status(directory)
        if cached is not None:
            layout.label(text=f"Cached: frames {cached[0]}-{cached[1]} ({_human(size)})")
        else:
            layout.label(text="Cached: nothing")

        if baking is not None:
            layout.progress(factor=baking.fraction, type='BAR',
                            text=f"{baking.state.title()} {baking.frames_done}/{baking.plan.total}")
            eta = baking.eta_seconds()
            if eta is not None:
                layout.label(text=f"About {eta:.0f} s left")
            layout.operator("wm.warp_sph_bake_cancel", icon='CANCEL')
        else:
            if session.bake_result:
                layout.label(text=session.bake_result, icon='INFO')
            row = layout.row(align=True)
            row.operator("wm.warp_sph_bake", icon='REC').resume = False
            resumable = cached is not None and cached[1] < settings.bake_end
            sub = row.row(align=True)
            sub.enabled = resumable
            sub.operator("wm.warp_sph_bake", text="Resume", icon='PLAY').resume = True
            row = layout.row(align=True)
            row.enabled = cached is not None
            row.operator("wm.warp_sph_cache_invalidate", icon='TRASH')
            row.operator("wm.warp_sph_cache_clear", text="Delete", icon='X')

        row = layout.row(align=True)
        if _adapter().find_playback_object(scene) is None:
            row.enabled = cached is not None
            row.operator("wm.warp_sph_cache_playback", text="Show Cache as Point Cloud", icon='OUTLINER_OB_POINTCLOUD')
        else:
            row.operator("wm.warp_sph_cache_playback", text="Hide Cache", icon='HIDE_ON').show = False


CLASSES = (
    WM_OT_warp_sph_bake,
    WM_OT_warp_sph_bake_cancel,
    WM_OT_warp_sph_cache_invalidate,
    WM_OT_warp_sph_cache_clear,
    WM_OT_warp_sph_cache_playback,
    VIEW3D_PT_warp_sph_cache,
)


def register():
    """Register after the main panel: the cache panel is its sub-panel."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _adapter().register()


def unregister():
    _adapter().unregister()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _forget_status()
