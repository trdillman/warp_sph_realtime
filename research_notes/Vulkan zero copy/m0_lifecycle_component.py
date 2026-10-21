"""Disposable legacy test module, never shipped or installed into regular preferences."""
import bpy
bl_info={'name':'M0 disposable lifecycle test','blender':(5,2,0),'category':'Development'}
cleanup_callback=None
events=[]

class WM_OT_m0_disposable_component(bpy.types.Operator):
    bl_idname='wm.m0_disposable_component'
    bl_label='M0 disposable component'
    def execute(self,context):return {'FINISHED'}

def register():
    bpy.utils.register_class(WM_OT_m0_disposable_component)
    events.append('register')

def unregister():
    events.append('unregister_enter')
    if cleanup_callback:cleanup_callback('unregister')
    bpy.utils.unregister_class(WM_OT_m0_disposable_component)
    events.append('unregister_return')
