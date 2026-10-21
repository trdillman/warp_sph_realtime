"""Bounded M0 supplement: sizing, cadence/generations, completion observation. No M1."""
import ctypes as c
import json,os,sys,time,traceback,types
from pathlib import Path
import bpy,gpu

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/qualification'
frozen=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/source-corrected/m0_native_probe.py'
source=frozen.read_text().split('bpy.utils.register_class(WM_OT_m0_lifetime_guard)')[0]
source=source.replace('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(ROOT)!r})')
source=source.replace("OUT = ROOT / 'evidence/vulkan-zero-copy/20261017-contained-m0/native-corrected'",f'OUT = Path({str(OUT)!r})')
n=types.ModuleType('m0_qualification_native');n.__file__=str(frozen);sys.modules[n.__name__]=n
exec(compile(source,str(frozen),'exec'),n.__dict__)
r={'pid':os.getpid(),'status':'STARTING','ticks':0,'draws':0,'consumed':0,'accepted':0,
   'publication_drops':0,'stale_ready_retired':0,'no_ready_draws':0,'cadence_deferred_draws':0,
   'readbacks':0,'max_cuda_queries_per_tick':0,'max_vulkan_queries_per_tick':0,'generations':[],
   'completion_samples':[],'cadence':[1,3,7],'transitions':[], 'query_results':[],
   'repeated_texture_draws':0,'generation_rejections':0,'target_consumed':120,
   'clock':vars(time.get_clock_info('perf_counter')),'cadence_consumed':{'1':0,'3':0,'7':0}}
s={'mode':'init','slots':[],'generation':0,'sequence':0,'last_draw_tick':0,'texture':None,'off':None}
def save():n.save(OUT/'qualification.json',r)
def fail():
    try:r.update(status='FAILED_QUARANTINED',error=traceback.format_exc());save()
    finally:n.park(r.get('error','qualification failure'))

def layout(count,limit,free):
    if type(count) is not int or count<0 or count>4194304:raise ValueError('count')
    if count==0:return None
    width=4096;height=(count+4095)//4096
    if width>limit or height>limit:raise ValueError('device texture extent')
    size=width*height*16
    if size*8+(1<<30)>free:raise ValueError('available VRAM reserve')
    return width,height,size

def sizes():
    limit=gpu.capabilities.max_texture_size_get();mem=n.memory();before=(n.R['imports'],n.R['native_calls'])
    r['limits']={'max_texture_size':limit,**mem,'reserve_bytes':1<<30,'budget_buffers':8,
                 'budget_explanation':'two PB + texture + scratch, plus equal retained resize overlap'}
    rows=[]
    for value in [-1,4194305,1.5,True,None]:
        try:layout(value,limit,mem['free_device_bytes'])
        except ValueError:rows.append({'count':value,'rejected':True})
        else:raise AssertionError('invalid count accepted')
    assert layout(0,limit,mem['free_device_bytes']) is None
    for count in [1,4095,4096,4097,4194304]:assert layout(count,limit,mem['free_device_bytes'])
    for test_limit,test_free in [(4095,mem['free_device_bytes']),(limit,0)]:
        try:layout(4097,test_limit,test_free)
        except ValueError:pass
        else:raise AssertionError('limit or budget accepted')
    assert before==(n.R['imports'],n.R['native_calls'])
    original=n.R['cuda_uuid'];uuid_rows=[]
    for invalid in ['00', ('00' if original[:2]!='00' else '01')+original[2:]]:
        n.R['cuda_uuid']=invalid
        try:
            try:n.Slot(4097)
            except AssertionError:uuid_rows.append({'bytes':len(invalid)//2,'rejected':True})
            else:raise AssertionError('UUID guard accepted mismatch')
        finally:n.R['cuda_uuid']=original
    assert before==(n.R['imports'],n.R['native_calls'])
    r['uuid_constructor_rejections']=uuid_rows
    r['sizing']={'invalid':rows,'zero_no_allocation':True,'device_limit_rejected_before_native':True,
                 'insufficient_budget_rejected_before_native':True,'actual_limits_checked':True,
                 'negative_limits_are_injected':True,'native_calls_during_validation':0,'imports_during_validation':0}
    save()

def consume(frame):
    with s['off'].bind():
        gpu.state.blend_set('ADDITIVE');n.shader.bind()
        n.shader.uniform_sampler('positions',s['texture']);n.shader.uniform_int('count',4097)
        n.shader.uniform_int('frame',frame);n.batch.draw(n.shader);gpu.state.blend_set('NONE')

def observe_queries():
    driver=n.driver;wait=n.funcs['wait']
    def observed_driver(name,*args):
        result=driver(name,*args)
        if name=='cuEventQuery':r['query_results'].append([r['ticks'],'CUDA',result,time.perf_counter_ns()])
        return result
    def observed_wait(*args):
        result=wait(*args);r['query_results'].append([r['ticks'],'VULKAN',result,time.perf_counter_ns()]);return result
    n.driver=observed_driver;n.funcs['wait']=observed_wait

def generation_matches(generation):return generation==s['generation']

def allocate():
    assert not s['slots']
    s['generation']+=1;s['sequence']=0;s['last_frame']=None
    layout(4097,r['limits']['max_texture_size'],n.memory()['free_device_bytes'])
    s['slots']=[n.Slot(4097),n.Slot(4097)]
    for slot in s['slots']:slot.generation=s['generation'];slot.sample=None
    s['texture']=gpu.types.GPUTexture((4096,2),format='RGBA32F')
    r['generations'].append({'id':s['generation'],'consumed_start':r['consumed'],'imports':n.R['imports']})
    assert not generation_matches(s['generation']-1);r['generation_rejections']+=1
    s['mode']='running'

def release():
    assert all(slot.state=='FREE' for slot in s['slots'])
    r['generations'][-1]['consumed_end']=r['consumed']
    for slot in s['slots']:slot.close()
    s['slots']=[];s['texture']=None

def draw():
    try:
        if s['mode']=='init':
            n.initialize();sizes();observe_queries();n.phase('stress')
            s['off']=gpu.types.GPUOffScreen(4096,2,format='RGBA32F')
            with s['off'].bind():gpu.state.active_framebuffer_get().clear(color=(0,0,0,0))
            allocate();return
        if s['mode']=='done':return
        assert bpy.context.window.as_pointer()==n.owner and n.funcs['context']()==n.owner_context
        r['draws']+=1
        if s['mode']=='drain':
            for slot in s['slots']:
                if slot.state=='READY':slot.state='FREE';r['stale_ready_retired']+=1
            if all(slot.state=='FREE' for slot in s['slots']):
                release()
                if r['consumed']==r['target_consumed']:
                    pixels=s['off'].texture_color.read();pixels.dimensions=4096*2*4;r['readbacks']+=1
                    error=float(n.np.asarray(pixels).reshape(-1,4)[:,0].sum());assert error==0
                    s['off'].free();s['off']=None
                    assert len(r['completion_samples'])==120
                    assert len(r['generations'])==2 and all(x['consumed_end']-x['consumed_start']==60 for x in r['generations'])
                    assert all(r[key]>0 for key in ['publication_drops','stale_ready_retired','repeated_texture_draws','cadence_deferred_draws'])
                    assert all(value==40 for value in r['cadence_consumed'].values())
                    assert r['accepted']==r['consumed']+r['stale_ready_retired']
                    assert r['clock']['monotonic'] and not r['clock']['adjustable']
                    assert r['generation_rejections']==2
                    for sample in r['completion_samples']:
                        assert sample['published_ns']<=sample['cuda_ready_observed_ns']<=sample['upload_returned_ns']<=sample['submitted_ns']<=sample['completed_observed_ns']
                    r.update(status='PASS_BOUNDED_QUALIFICATION',gpu_error_sum=error,imports=n.R['imports'],
                             cuda_queries=n.R['cuda_queries'],vulkan_queries=n.R['vulkan_queries'],not_ready=n.R['not_ready'])
                    s['mode']='done';n.state['mode']='done';save()
                else:allocate()
            return
        if s['last_frame'] is not None:
            consume(s['last_frame']);r['repeated_texture_draws']+=1
        cadence=r['cadence'][min(r['consumed']//40,2)]
        if r['draws']%3==0 or r['ticks']-s['last_draw_tick']<cadence:r['cadence_deferred_draws']+=1;return
        ready=[slot for slot in s['slots'] if slot.state=='READY']
        if not ready:r['no_ready_draws']+=1;return
        slot=max(ready,key=lambda item:item.frame)
        for old in ready:
            if old is not slot:old.state='FREE';r['stale_ready_retired']+=1
        assert generation_matches(slot.generation)
        slot.upload(s['texture'])
        slot.sample['upload_returned_ns']=time.perf_counter_ns()
        consume(slot.frame);s['last_frame']=slot.frame
        slot.submitted();slot.sample['submitted_ns']=time.perf_counter_ns()
        r['consumed']+=1;r['cadence_consumed'][str(cadence)]+=1;s['last_draw_tick']=r['ticks']
        if r['consumed'] in (60,120):s['mode']='drain'
        if r['consumed']%40==0:save()
    except BaseException:fail()

def tick():
    try:
        if s['mode']=='done':
            bpy.types.SpaceView3D.draw_handler_remove(handler,'WINDOW');bpy.ops.wm.quit_blender();return None
        r['ticks']+=1
        if s['mode']!='init':
            assert n.owner in {w.as_pointer() for w in bpy.context.window_manager.windows}
            before=(n.R['cuda_queries'],n.R['vulkan_queries'])
            for slot in s['slots']:
                old=slot.state;query_begin=time.perf_counter_ns();slot.poll();query_end=time.perf_counter_ns()
                if old=='WRITING' and slot.state=='READY':slot.sample['cuda_ready_observed_ns']=query_end
                if old=='IN_FLIGHT' and slot.state=='FREE':
                    sample=slot.sample;sample.update(completed_observed_ns=query_end,
                        upper_bound_ns=query_end-sample['published_ns'],generation=slot.generation,frame=slot.frame)
                    r['completion_samples'].append(sample);slot.sample=None
                elif old=='IN_FLIGHT':slot.sample['last_incomplete_query_begin_ns']=query_begin
            delta=(n.R['cuda_queries']-before[0],n.R['vulkan_queries']-before[1])
            assert delta[0]<=2 and delta[1]<=2
            r['max_cuda_queries_per_tick']=max(r['max_cuda_queries_per_tick'],delta[0])
            r['max_vulkan_queries_per_tick']=max(r['max_vulkan_queries_per_tick'],delta[1])
            if s['mode']=='running':
                available=next((slot for slot in s['slots'] if slot.state=='FREE'),None)
                if available is None:r['publication_drops']+=1
                else:
                    s['sequence']+=1;encoded=s['generation']*1000000+s['sequence']
                    available.sample={'published_ns':time.perf_counter_ns(),'tick':r['ticks']}
                    available.launch(encoded);r['accepted']+=1
                    r['transitions'].append([s['generation'],encoded,r['ticks']])
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type=='VIEW_3D':area.tag_redraw()
        return .001
    except BaseException:fail()

bpy.utils.register_class(n.WM_OT_m0_lifetime_guard)
assert bpy.ops.wm.m0_lifetime_guard('INVOKE_DEFAULT')=={'RUNNING_MODAL'}
bpy.app.handlers.load_pre.append(n.before_load)
@bpy.app.handlers.persistent
def exit_pre(unused):
    if s['mode']!='done':n.park('unexpected qualification exit')
bpy.app.handlers.exit_pre.append(exit_pre)
handler=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW','POST_VIEW')
bpy.app.timers.register(tick,first_interval=.1)
save()
