"""Thin explicit native-snapshot action; no live-step or draw readback hooks."""
import bpy

from . import sdk_snapshot, sim_session


def _session(context):
    return next((session for session in sim_session.sessions() if session.scene is context.scene), None)


class WM_OT_warp_sph_snapshot_native(bpy.types.Operator):
    bl_idname = 'wm.warp_sph_snapshot_native'
    bl_label = 'Snapshot Paused Simulation to Geometry Nodes'
    bl_description = ('Copy the paused real solver into native Blender attributes and an editable GN graph. '
                      'Explicit bounded readback; requires the separately installed qualified SDK candidate')
    bl_options = {'REGISTER', 'UNDO'}

    object_name: bpy.props.StringProperty(name='Snapshot Object', default='WarpSPH Native Snapshot')
    collection_name: bpy.props.StringProperty(name='Output Collection', default='WarpSPH Native Snapshots')
    replace_existing: bpy.props.BoolProperty(name='Update Existing Snapshot', default=False,
                                             description='Only an SDK-owned snapshot from this simulation can be updated')
    geometry_nodes: bpy.props.BoolProperty(name='Geometry Nodes Instances', default=True)

    @classmethod
    def poll(cls, context):
        reason = sdk_snapshot.eligibility(_session(context), context.scene)
        if reason:
            setter = getattr(cls, 'poll_message_set', None)
            if callable(setter):
                setter(reason)
        return not reason

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=440)

    def draw(self, context):
        self.layout.prop(self, 'object_name')
        self.layout.prop(self, 'collection_name')
        self.layout.prop(self, 'replace_existing')
        self.layout.prop(self, 'geometry_nodes')
        self.layout.label(text='Explicit snapshot, not live zero-copy Geometry Nodes.')
        self.layout.label(text='Current particle radius and colour settings are used.')

    def execute(self, context):
        session = _session(context)
        try:
            result = sdk_snapshot.snapshot_to_native(
                session, context.scene, wp_module=sim_session.runtime.wp,
                object_name=self.object_name, collection_name=self.collection_name,
                replace_existing=self.replace_existing, geometry_nodes=self.geometry_nodes)
        except Exception as exc:
            self.report({'ERROR'}, f'Native snapshot not created: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Native snapshot saved at solver step {result.stamp.frame_id}; simulation remains paused')
        return {'FINISHED'}


class VIEW3D_PT_warp_sph_native_snapshot(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'WarpSPH'
    bl_label = 'Native Snapshot (SDK Preview)'
    bl_parent_id = 'VIEW3D_PT_warp_sph_cache'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        reason = sdk_snapshot.eligibility(_session(context), context.scene)
        self.layout.label(text='Actual solver IDs, velocity and density to native GN.')
        if reason:
            self.layout.label(text=reason, icon='INFO')
        self.layout.operator('wm.warp_sph_snapshot_native', text='Snapshot Paused Simulation', icon='GEOMETRY_NODES')
        self.layout.label(text='Requires separately installed, checked SDK candidate.')


CLASSES = (WM_OT_warp_sph_snapshot_native, VIEW3D_PT_warp_sph_native_snapshot)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
