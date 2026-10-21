"""Bounded normal lifecycle and partial-acquisition M0 harness; no product code."""
import ctypes as c
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import threading
import bpy
import gpu
import addon_utils

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/lifecycle-owned'
frozen=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/source-corrected/m0_native_probe.py'
source=frozen.read_text().split('bpy.utils.register_class(WM_OT_m0_lifetime_guard)')[0]
source=source.replace('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(ROOT)!r})')
source=source.replace("OUT = ROOT / 'evidence/vulkan-zero-copy/20261017-contained-m0/native-corrected'",f'OUT = Path({str(OUT)!r})')
n=types.ModuleType('m0_lifecycle_native'); n.__file__=str(frozen);sys.modules[n.__name__]=n
exec(compile(source,str(frozen),'exec'),n.__dict__)
report={'status':'STARTING','normal':[],'partial':[],'pid':os.getpid()}
state={'phase':'init','slot':None,'texture':None,'handler':None,'case':0,'closed':False}
cases=['unregister','producer_stop','load_pre','last_view_loss']

def receipt():n.save(OUT/'lifecycle.json',report)

def fail():
    error=traceback.format_exc()
    try:
        report.update(status='FAILED_QUARANTINED',error=error)
        receipt()
    finally:n.park(error)

def activate():
    assert threading.current_thread() is threading.main_thread()
    assert len(bpy.context.window_manager.windows)==1 and len(bpy.data.window_managers)==1
    assert n.owner in {w.as_pointer() for w in bpy.context.window_manager.windows}
    prior=n.funcs['context']()
    runtime=c.c_void_p.from_address(n.owner+0xf0).value;assert runtime
    window_context=c.c_void_p.from_address(runtime+0x28).value
    is_drw=c.CFUNCTYPE(c.c_bool)(n.base+0xb45980)
    report['activation_preflight']={'prior':prior,'captured_owner_context':n.owner_context,
                                    'owner_window':n.owner,'window_context':window_context,
                                    'drw_enabled':is_drw()}
    receipt()
    assert prior==window_context and prior!=n.owner_context and not is_drw(), 'unclassified or nested context'
    c.CFUNCTYPE(None)(n.base+0xb458b0)()  # Locked DRW scope; paired disable restores drawable.
    assert n.funcs['context']()==n.owner_context
    assert is_drw()
    report.setdefault('contexts',[]).append({'prior':prior,'window_context':window_context,
        'after_enable':n.funcs['context'](),'captured':n.owner_context})
    return prior

def cleanup(reason):
    if state['closed']:return
    if state['slot'] is None:
        state['closed']=True
        return
    start=time.monotonic()
    # No further draws/publication can occur after removing this handle.
    if state['handler'] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(state['handler'],'WINDOW');state['handler']=None
    prior=activate()
    slot=state['slot']
    # Cover every submitted texture read; all work is in one recorded context.
    if slot.state in ['FREE','IN_FLIGHT']:slot.submitted()
    while slot.state!='FREE':
        slot.poll()
        # Producer-only work has no submitted consumer; successful event permits discard.
        if slot.state=='READY':slot.state='FREE'
        if time.monotonic()-start>=.1:n.park(f'{reason}: cleanup exceeded 100ms')
        if slot.state!='FREE':time.sleep(.001)
    slot.close();state['slot']=None;state['texture']=None
    if state.get('offscreen') is not None:
        state['offscreen'].free();state['offscreen']=None
    state['closed']=True
    c.CFUNCTYPE(None)(n.base+0xb459b0)()
    assert n.funcs['context']()==prior, 'window context restoration failed'
    assert not c.CFUNCTYPE(c.c_bool)(n.base+0xb45980)()
    report['contexts'][-1]['after_restore']=n.funcs['context']()
    if time.monotonic()-start>=.1:n.park(f'{reason}: total cleanup exceeded 100ms after release')
    report['normal'].append({'case':reason,'seconds':time.monotonic()-start,
        'owned_mapping_import_event_fence_pb':0,'release':'CUDA_RELEASED_BLENDER_DEFERRED'})
    receipt()

def partial_cases():
    # No submitted producer work in these deliberately interrupted acquisitions.
    for stop in ['pixel_buffer','export_handle','import','mapping','event','fence']:
        pb=ext=ptr=event=fence=handle=None
        try:
            pb=n.funcs['create'](65536);assert pb
            if stop=='pixel_buffer':continue
            nh=n.NativeHandle();n.funcs['handle'](c.byref(nh),pb);handle=nh.handle;assert handle
            if stop=='export_handle':continue
            desc=n.MemDesc();desc.type=2;desc.handle.win32.handle=handle;desc.size=nh.size;desc.flags=1
            ext=n.P();n.checked(n.driver('cuImportExternalMemory',c.byref(ext),c.byref(desc)),'partial import')
            assert n.k32.CloseHandle(n.P(handle));handle=None
            if stop=='import':continue
            ptr=n.U();bd=n.BufferDesc(0,65536,0)
            n.checked(n.driver('cuExternalMemoryGetMappedBuffer',c.byref(ptr),ext,c.byref(bd)),'partial map')
            if stop=='mapping':continue
            event=n.P();n.checked(n.driver('cuEventCreate',c.byref(event),2),'partial event')
            if stop=='event':continue
            fence=n.funcs['fence']();assert fence
        except BaseException:
            fail()  # Nonreturning: prevents rollback after raised context push/pop errors.
        finally:
            # Real driver failures park inside driver(), so this is only healthy synthetic rollback.
            if ptr is not None:n.checked(n.driver('cuMemFree_v2',ptr.value),'partial free')
            if ext is not None:n.checked(n.driver('cuDestroyExternalMemory',ext),'partial destroy')
            if event is not None:n.checked(n.driver('cuEventDestroy_v2',event),'partial event destroy')
            if handle is not None:assert n.k32.CloseHandle(n.P(handle))
            if fence is not None:n.funcs['free'](fence)
            if pb is not None:n.funcs['free'](pb)
            report['partial'].append({'after':stop,'owned_resources':0,'synthetic':True})
            receipt()

def load_pre(unused):
    try:
        cleanup('load_pre');state['phase']='load_complete'
    except BaseException:fail()

def unregister_probe():
    addon_utils.disable('m0_lifecycle_component',default_set=False)
    assert component.events==['register','unregister_enter','unregister_return']
    report['unregister_events']=list(component.events)
    assert not addon_utils.check('m0_lifecycle_component')[1]
    cleanup('unregister')  # Idempotent close must not free twice.

def unregister_cleanup(reason):
    try:cleanup(reason)
    except BaseException:fail()  # Preserve original error before addon_utils can catch it.

def draw():
    try:
        if state['phase']=='init':
            n.initialize();n.phase('cycles');partial_cases();state['phase']='new_case'
        if bpy.context.window.as_pointer()!=n.owner:return
        if state.pop('recapture',False):
            assert c.CFUNCTYPE(c.c_bool)(n.base+0xb45980)()
            n.owner_context=n.funcs['context']()
        assert n.funcs['context']()==n.owner_context, 'owner draw context mismatch'
        if state['phase']=='new_case':
            state['closed']=False;state['slot']=n.Slot(4097)
            state['texture']=gpu.types.GPUTexture((4096,2),format='RGBA32F')
            state['slot'].launch(27);state['phase']='producing'
            if cases[state['case']]=='producer_stop':state['phase']='trigger'
        elif state['phase']=='producing' and state['slot'].state=='READY':
            state['slot'].upload(state['texture'])
            # Actual texture consumer, even in lifecycle cases.
            off=gpu.types.GPUOffScreen(4096,2,format='RGBA32F')
            with off.bind():
                gpu.state.blend_set('NONE')
                n.shader.bind();n.shader.uniform_sampler('positions',state['texture'])
                n.shader.uniform_int('count',4097)
                n.shader.uniform_int('frame',28 if state['case']==0 else 27);n.batch.draw(n.shader)
            if state['case']==0:
                pixels=off.texture_color.read();pixels.dimensions=4096*2*4
                errors=float(n.np.asarray(pixels).reshape(-1,4)[:,0].sum())
                assert errors==4097,'GPU mismatch detector failed negative control'
                report['gpu_negative_control']={'expected_mismatches':4097,'observed_mismatches':errors}
                receipt()
            # Keep framebuffer texture alive until final retirement fence.
            state['offscreen']=off
            state['slot'].submitted();state['phase']='trigger'
    except BaseException:fail()

def tick():
    try:
        if state['phase']=='producing':state['slot'].poll()
        elif state['phase']=='trigger':
            case=cases[state['case']]
            if case=='unregister':unregister_probe();state['phase']='advance'
            elif case=='producer_stop':cleanup('producer_stop');state['phase']='advance'
            elif case=='load_pre':
                bpy.app.handlers.load_pre.append(load_pre)
                bpy.ops.wm.read_factory_settings(use_empty=True)
                assert state['closed'] and state['phase']=='load_complete'
                state['phase']='advance'
            else:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type=='VIEW_3D':area.type='TEXT_EDITOR'
                assert not any(a.type=='VIEW_3D' for w in bpy.context.window_manager.windows for a in w.screen.areas)
                cleanup('last_view_loss');state['phase']='advance'
        elif state['phase']=='advance':
            state['case']+=1
            if state['case']==len(cases):
                report.update(status='PASS_BOUNDED_LIFECYCLE',final_memory=n.memory())
                receipt();n.state['mode']='done';bpy.ops.wm.quit_blender();return None
            # File replacement invalidates old window/context identity. New session captures anew.
            n.owner=bpy.context.window.as_pointer()
            state['recapture']=True  # Capture only inside next authenticated DRW draw.
            state['closed']=False
            assert bpy.ops.wm.m0_lifecycle_guard('INVOKE_DEFAULT')=={'RUNNING_MODAL'}
            state['handler']=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW','POST_VIEW')
            state['phase']='new_case'
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type=='VIEW_3D':a.tag_redraw()
        return .005
    except BaseException:fail()

class WM_OT_m0_lifecycle_guard(bpy.types.Operator):
    bl_idname='wm.m0_lifecycle_guard';bl_label='M0 lifecycle containment'
    def invoke(self,context,event):
        context.window_manager.modal_handler_add(self);return {'RUNNING_MODAL'}
    def modal(self,context,event):
        if state['closed']:return {'FINISHED'}
        return {'PASS_THROUGH'}
    def cancel(self,context):
        if not state['closed'] and state['slot'] is not None:
            n.park('unexpected window cancel during lifecycle probe')

@bpy.app.handlers.persistent
def exit_pre(unused):
    if not state['closed'] and state['slot'] is not None:
        n.park('unexpected final-window exit during lifecycle probe')

sys.path.insert(0,str(Path(__file__).parent))
component=addon_utils.enable('m0_lifecycle_component',default_set=False,persistent=False)
assert component and addon_utils.check('m0_lifecycle_component')[1]
component.cleanup_callback=unregister_cleanup
bpy.utils.register_class(WM_OT_m0_lifecycle_guard)
assert bpy.ops.wm.m0_lifecycle_guard('INVOKE_DEFAULT')=={'RUNNING_MODAL'}
bpy.app.handlers.exit_pre.append(exit_pre)
state['handler']=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW','POST_VIEW')
bpy.app.timers.register(tick,first_interval=.1,persistent=True)
receipt()
