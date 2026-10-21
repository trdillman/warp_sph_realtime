"""Identity-only exact-build lifecycle trace. No GPU allocation or activation."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback
import bpy
import gpu
import addon_utils

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/context-identity'
sys.path.insert(0,str(Path(__file__).parent))
from m0_supervisor import save
ready=json.loads((OUT/'watch/ready.json').read_text())
expected=json.loads((OUT/'expected.json').read_text())
assert ready['job_member'] and ready['kill_on_close']
assert all(ready[k]==expected[k] for k in ['pid','creation_time','nonce'])
assert ready['pid']==os.getpid()
assert hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest()=='0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327'
assert gpu.platform.backend_type_get()=='VULKAN'
k=c.WinDLL('kernel32');k.GetModuleHandleW.argtypes=[c.c_wchar_p];k.GetModuleHandleW.restype=c.c_void_p
base=k.GetModuleHandleW(None)
active=c.CFUNCTYPE(c.c_void_p)(base+0x412580)
enabled=c.CFUNCTYPE(c.c_bool)(base+0xb45980)
report={'pid':os.getpid(),'rows':[],'status':'STARTING','gpu_allocations':0,'context_switches':0}
handler=None
drawn=False
def sample(where):
    windows=[]
    for w in bpy.context.window_manager.windows:
        ptr=w.as_pointer();runtime=c.c_void_p.from_address(ptr+0xf0).value
        assert runtime
        windows.append({'window':ptr,'runtime':runtime,'gpu_context':c.c_void_p.from_address(runtime+0x28).value})
    report['rows'].append({'where':where,'thread':threading.get_ident(),'main_thread':threading.current_thread() is threading.main_thread(),
                           'active':active(),'drw_enabled':enabled(),'windows':windows})
    save(OUT/'identity.json',report)
def draw():
    global drawn
    if not drawn:
        sample('draw');drawn=True
def tick():
    if not drawn:
        for a in bpy.context.screen.areas:a.tag_redraw()
        return .05
    try:
        sample('timer')
        addon_utils.disable('m0_lifecycle_component',default_set=False)
        assert component.events==['register','unregister_enter','unregister_return']
        sample('after_unregister')
        bpy.types.SpaceView3D.draw_handler_remove(handler,'WINDOW')
        report['status']='PASS_IDENTITY_TRACE';save(OUT/'identity.json',report)
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());save(OUT/'identity.json',report)
    bpy.ops.wm.quit_blender()
    return None
component=addon_utils.enable('m0_lifecycle_component',default_set=False,persistent=False)
component.cleanup_callback=sample
sample('install')
handler=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW','POST_VIEW')
bpy.app.timers.register(tick,first_interval=.1)
