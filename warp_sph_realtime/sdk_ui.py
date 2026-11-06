"""Explicit opt-in UI for the shared SDK consumer; no CUDA imports on registration."""

import uuid

import bpy

from . import sim_session


class WM_OT_warp_sph_sdk_snapshot(bpy.types.Operator):
    bl_idname = "wm.warp_sph_sdk_snapshot"
    bl_label = "Snapshot to Geometry Nodes"
    bl_description = (
        "Capture the paused solver's exact rich frame as native attributes and an editable Geometry Nodes object"
    )

    @classmethod
    def poll(cls, context):
        session = sim_session.get(context.scene)
        output = session.session
        return (
            session.running
            and session.paused
            and session.step_count > 0
            and getattr(output, "is_shared_sdk", False)
            and getattr(output, "snapshot_status", "") not in ("QUEUED", "CAPTURED")
        )

    def execute(self, context):
        session = sim_session.get(context.scene)
        if not self.poll(context):
            self.report({"ERROR"}, "Pause a running Shared SDK simulation after at least one step")
            return {"CANCELLED"}
        settings = context.scene.warp_sph
        if not settings.sdk_snapshot_name.strip():
            self.report({"ERROR"}, "Give the snapshot object a name")
            return {"CANCELLED"}
        if not settings.sdk_snapshot_owner:
            settings.sdk_snapshot_owner = uuid.uuid4().hex
        try:
            session.session.request_snapshot(
                {"name": settings.sdk_snapshot_name, "collection": "WarpSPH SDK Output " + settings.sdk_snapshot_owner,
                 "max_bytes": int(settings.sdk_native_budget_mb) * 1024 * 1024,
                 "max_native_bytes": int(settings.sdk_native_budget_mb) * 1024 * 1024}
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Snapshot queued for the exact displayed solver frame")
        return {"FINISHED"}


class WM_OT_warp_sph_sdk_bake_animation(bpy.types.Operator):
    bl_idname = "wm.warp_sph_sdk_bake_animation"
    bl_label = "Bake Native Animation"
    bl_description = (
        "Bake an independent copy of the paused simulation; keep every frame and leave live state unchanged"
    )

    def execute(self, context):
        from .native_output import begin_bake

        session = sim_session.get(context.scene)
        settings = context.scene.warp_sph
        ok = begin_bake(session, context, self.report, first=settings.bake_start, last=settings.bake_end)
        return {"FINISHED"} if ok else {"CANCELLED"}


class WM_OT_warp_sph_sdk_cancel_animation(bpy.types.Operator):
    bl_idname = "wm.warp_sph_sdk_cancel_animation"
    bl_label = "Cancel Native Bake"

    @classmethod
    def poll(cls, context):
        return any(
            s.scene is context.scene and getattr(s, "native_bake", None) is not None for s in sim_session.sessions()
        )

    def execute(self, context):
        sim_session.get(context.scene).cancel_bake()
        return {"FINISHED"}


class WM_OT_warp_sph_sdk_load_animation(bpy.types.Operator):
    bl_idname = "wm.warp_sph_sdk_load_animation"
    bl_label = "Load Native Animation"
    bl_description = "Attach a completed or cancelled-prefix rich cache to native timeline playback"

    def execute(self, context):
        import importlib

        from .cache import native_playback

        settings = context.scene.warp_sph
        if sim_session.active() is not None:
            self.report({"ERROR"}, "Stop the simulation before loading a recorded animation")
            return {"CANCELLED"}
        try:
            if not settings.sdk_snapshot_owner:
                settings.sdk_snapshot_owner = uuid.uuid4().hex

            def create_graph(obj):
                return importlib.import_module(
                    "blender_vulkan_interop.output.geometry_nodes"
                ).create_particle_instancer(obj)

            native_playback.attach(
                context.scene, bpy.path.abspath(settings.sdk_animation_dir), settings.sdk_animation_name, create_graph,
                max_bytes=int(settings.sdk_native_budget_mb) * 1024 * 1024
            )
            settings.sdk_native_status = "Native animation loaded. Scrub the timeline; Render Lock Interface enabled."
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class WM_OT_warp_sph_sdk_embed_animation(bpy.types.Operator):
    bl_idname = "wm.warp_sph_sdk_embed_animation"
    bl_label = "Embed Standalone Animation"
    bl_description = "Build native time-driven GN playback for saving/rendering without WarpSPH, SDK or Warp"

    object_name: bpy.props.StringProperty(name="Standalone Object", default="WarpSPH Standalone Animation")

    def execute(self, context):
        from .cache import native_playback

        if sim_session.active() is not None:
            self.report({"ERROR"}, "Stop the simulation before embedding animation")
            return {"CANCELLED"}
        settings = context.scene.warp_sph
        player = bpy.data.objects.get(settings.sdk_animation_name)
        try:
            if player is None:
                raise ValueError("Load or bake a native animation first")
            obj = native_playback.make_standalone(
                context.scene, player, name=self.object_name, max_bytes=int(settings.sdk_native_budget_mb) * 1024 * 1024
            )
            settings.sdk_native_status = f"Standalone animation ready: {obj.name}. Save the .blend; no runtime needed."
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class VIEW3D_PT_warp_sph_sdk_output(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WarpSPH"
    bl_label = "Live Geometry Nodes / Animation"
    bl_parent_id = "VIEW3D_PT_warp_sph"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        session = sim_session.get(context.scene)
        settings = context.scene.warp_sph
        row = layout.row()
        row.enabled = not session.running
        row.prop(settings, "use_shared_sdk")
        if settings.use_shared_sdk:
            live = layout.box()
            live.prop(settings, "sdk_live_geometry")
            live.prop(settings, "sdk_live_name")
            live.prop(settings, "sdk_live_rate")
            live.prop(settings, "sdk_live_hide_preview")
            live.label(text="Explicit native readback; target rate, not guaranteed FPS.")
        layout.prop(settings, "sdk_native_budget_mb")
        if getattr(session.session, "is_shared_sdk", False):
            layout.prop(settings, "sdk_snapshot_name")
            layout.operator("wm.warp_sph_sdk_snapshot", icon="GEOMETRY_NODES")
        box = layout.box()
        box.label(text="Native Animation")
        box.prop(settings, "sdk_animation_dir")
        box.prop(settings, "sdk_animation_name")
        row = box.row(align=True)
        row.prop(settings, "bake_start")
        row.prop(settings, "bake_end")
        box.prop(settings, "bake_steps")
        box.label(text="Pause first: bake starts from a private copy of that state.")
        box.label(text="Animated collider replay is not yet supported.")
        if getattr(session, "native_bake", None) is not None:
            box.operator("wm.warp_sph_sdk_cancel_animation", icon="CANCEL")
        else:
            box.operator("wm.warp_sph_sdk_bake_animation", icon="REC")
        box.operator("wm.warp_sph_sdk_load_animation", icon="FILE_FOLDER")
        box.operator("wm.warp_sph_sdk_embed_animation", icon="PACKAGE")
        box.label(text="Embed uses memory for all point frames; streaming stays bounded.")
        if settings.sdk_native_status:
            layout.label(text=settings.sdk_native_status)
        if getattr(session.session, "snapshot_status", "IDLE") not in ("IDLE", "COMPLETED"):
            layout.label(text=session.session.snapshot_message)


_CLASSES = (
    WM_OT_warp_sph_sdk_snapshot,
    WM_OT_warp_sph_sdk_bake_animation,
    WM_OT_warp_sph_sdk_cancel_animation,
    WM_OT_warp_sph_sdk_load_animation,
    WM_OT_warp_sph_sdk_embed_animation,
    VIEW3D_PT_warp_sph_sdk_output,
)


def register():
    from .cache import native_playback

    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    native_playback.register()


def unregister():
    from .cache import native_playback

    native_playback.unregister()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
