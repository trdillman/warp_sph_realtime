"""Real final-window close, synthetic unresolved work; no internal GPU/native calls."""
import json
import os
from pathlib import Path
import time
import bpy

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/final-window'
WATCH=OUT/'watch'
expected=json.loads((OUT/'expected.json').read_text())
ready=json.loads((WATCH/'ready.json').read_text())
assert expected['pid']==os.getpid()
assert all(expected[k]==ready[k] for k in ['pid','creation_time','nonce'])
assert ready['job_member'] and ready['kill_on_close']
assert len(bpy.context.window_manager.windows)==1

def persist(name,data):
    with (OUT/name).open('w') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())

def before_exit(is_background):
    persist('exit-pre.json',{'pid':os.getpid(),'is_background':is_background,'time':time.time()})

class WM_OT_m0_final_window_guard(bpy.types.Operator):
    bl_idname='wm.m0_final_window_guard';bl_label='M0 synthetic pending-work guard'
    def invoke(self,context,event):
        self.owner=context.window.as_pointer()
        context.window_manager.modal_handler_add(self);return {'RUNNING_MODAL'}
    def modal(self,context,event):return {'PASS_THROUGH'}
    def cancel(self,context):
        try:
            persist('cancel.json',{'pid':os.getpid(),'synthetic_pending':True,
                'captured_owner':self.owner,'callback_window':context.window.as_pointer() if context.window else None,
                'exit_pre_seen':(OUT/'exit-pre.json').exists(),'native_gpu_calls':0,'time':time.time()})
            with (WATCH/'abort.json').open('w') as f:
                json.dump({'synthetic_final_window_pending':True},f);f.flush();os.fsync(f.fileno())
        finally:
            while True:
                try:time.sleep(1)
                except BaseException:pass

bpy.app.handlers.exit_pre.append(before_exit)
bpy.utils.register_class(WM_OT_m0_final_window_guard)
assert bpy.ops.wm.m0_final_window_guard('INVOKE_DEFAULT')=={'RUNNING_MODAL'}
bpy.context.preferences.view.use_save_prompt=False

def close_final():
    persist('close-request.json',{'pid':os.getpid(),'windows':len(bpy.context.window_manager.windows)})
    bpy.ops.wm.window_close()
    return None

bpy.app.timers.register(close_final,first_interval=1)
persist('armed.json',{'pid':os.getpid(),'synthetic':True})
