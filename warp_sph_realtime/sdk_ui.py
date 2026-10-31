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
                {"name": settings.sdk_snapshot_name, "collection": "WarpSPH SDK Output " + settings.sdk_snapshot_owner}
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Snapshot queued for the exact displayed solver frame")
        return {"FINISHED"}


class VIEW3D_PT_warp_sph_sdk_output(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WarpSPH"
    bl_label = "Native Particles / Geometry Nodes"
    bl_parent_id = "VIEW3D_PT_warp_sph"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        session = sim_session.get(context.scene)
        settings = context.scene.warp_sph
        row = layout.row()
        row.enabled = not session.running
        row.prop(settings, "use_shared_sdk")
        if not getattr(session.session, "is_shared_sdk", False):
            if settings.use_shared_sdk:
                layout.label(text="Requires pinned development SDK and Vulkan", icon="INFO")
                layout.label(text="Start, then Pause to capture native attributes")
            return
        layout.prop(settings, "sdk_snapshot_name")
        layout.operator("wm.warp_sph_sdk_snapshot", icon="GEOMETRY_NODES")
        layout.label(text="Explicit snapshot readback; not live zero-copy GN")
        layout.label(text="Cache baking remains on the existing output route")
        if session.session.snapshot_status != "IDLE":
            layout.label(text=session.session.snapshot_status)
            layout.label(text=session.session.snapshot_message)


def register():
    bpy.utils.register_class(WM_OT_warp_sph_sdk_snapshot)
    bpy.utils.register_class(VIEW3D_PT_warp_sph_sdk_output)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_warp_sph_sdk_output)
    bpy.utils.unregister_class(WM_OT_warp_sph_sdk_snapshot)
