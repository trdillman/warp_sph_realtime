"""Disposable M0 research harness, exact build only. NOT production transport/M1."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import bpy
import gpu
import numpy as np
import warp as wp
from gpu_extras.batch import batch_for_shader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/vulkan-zero-copy/20261017-contained-m0/native-corrected'
WATCH = OUT / 'watch'
OUT.mkdir(parents=True, exist_ok=True)
R = {'pid': os.getpid(), 'status': 'STARTING', 'imports': 0, 'stages': [], 'rows': [],
     'native_calls': 0, 'cuda_queries': 0, 'vulkan_queries': 0, 'not_ready': 0,
     'stress_frames': 0, 'cycles': [], 'readbacks_stress': 0, 'M0_pass': False}

def save(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def stage(name, **kw):
    R['stages'].append({'name': name, 'time': time.time(), **kw})
    save(OUT / 'receipt.json', R)

def phase(name):
    save(WATCH / 'phase.json', {'phase': name})
    stage(name)

def park(error):
    # Do not return/unwind into Blender destruction even if disk persistence fails.
    try:
        R.update(status='FAILED_QUARANTINED', error=error)
        save(OUT / 'receipt.json', R)
        save(WATCH / 'abort.json', {'error': error, 'pid': os.getpid()})
    finally:
        while True:
            try: time.sleep(1)
            except BaseException: pass

def checked(result, name):
    if result: raise RuntimeError(f'{name}: CUresult={result}')

P = c.c_void_p
U = c.c_uint64
class Span(c.Structure):
    _fields_ = [('ptr', P), ('count', c.c_int64)]
class NativeHandle(c.Structure):
    _fields_ = [('handle', c.c_int64), ('size', c.c_size_t)]
class WinHandle(c.Structure):
    _fields_ = [('handle', P), ('name', P)]
class HandleUnion(c.Union):
    _fields_ = [('fd', c.c_int), ('win32', WinHandle), ('sci', P)]
class MemDesc(c.Structure):
    _fields_ = [('type', c.c_int), ('handle', HandleUnion), ('size', U), ('flags', c.c_uint),
                ('reserved', c.c_uint*16)]
class BufferDesc(c.Structure):
    _fields_ = [('offset', U), ('size', U), ('flags', c.c_uint), ('reserved', c.c_uint*16)]
class WaitInfo(c.Structure):
    _fields_ = [('type', c.c_uint), ('next', P), ('flags', c.c_uint), ('count', c.c_uint),
                ('semaphores', c.POINTER(U)), ('values', c.POINTER(U))]

RVA = {'uuid':0x41b040, 'create':0xd96ce0, 'free':0xad2100, 'size':0xd96d10,
       'handle':0xd96d20, 'update':0xd95b30, 'fence':0xd8ef80, 'signal':0x43ff50,
       'backend':0x4123a0, 'context':0x412580, 'wait':0x4c0da00}

@wp.kernel
def fill(a: wp.array(dtype=wp.vec4), n: int, frame: int):
    i = wp.tid()
    if i < n:
        a[i] = wp.vec4(float(i)*0.5, float(frame), float(i%1000), 1.0)
    else:
        a[i] = wp.vec4(0.0)

def initialize():
    global k32, base, funcs, cu, dev, stream, owner, owner_context, shader, batch
    ready = json.loads((WATCH / 'ready.json').read_text())
    assert ready['pid'] == os.getpid() and ready['job_member'] and ready['kill_on_close']
    expected=json.loads((OUT/'expected.json').read_text())
    assert all(ready[key]==expected[key] for key in ['pid','creation_time','nonce'])
    exe = Path(bpy.app.binary_path)
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    assert digest == '0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327'
    assert bpy.app.build_hash.decode() == 'fbe6228777e7'
    assert gpu.platform.backend_type_get() == 'VULKAN' and wp.__version__ == '1.17.0'
    assert c.sizeof(MemDesc)==104 and c.sizeof(BufferDesc)==88 and c.sizeof(WaitInfo)==40
    assert MemDesc.reserved.offset==36 and BufferDesc.reserved.offset==20
    R.update(exe_sha256=digest, blender=bpy.app.version_string, warp=wp.__version__,
             python=__import__('sys').version, backend=gpu.platform.backend_type_get())
    k32 = c.WinDLL('kernel32', use_last_error=True)
    k32.GetModuleHandleW.argtypes=[c.c_wchar_p]; k32.GetModuleHandleW.restype=P
    k32.CloseHandle.argtypes=[P]; k32.CloseHandle.restype=c.c_int
    base = k32.GetModuleHandleW(None)
    def native(key, ret, *args):
        raw = c.CFUNCTYPE(ret, *args)(base+RVA[key])
        def call(*values):
            R['native_calls'] += 1
            return raw(*values)
        return call
    funcs = {'uuid':native('uuid',P,c.POINTER(Span)), 'create':native('create',P,c.c_size_t),
             'free':native('free',None,P), 'size':native('size',c.c_size_t,P),
             'handle':native('handle',P,c.POINTER(NativeHandle),P),
             'update':native('update',None,P,c.c_int,P,*([c.c_int]*6)),
             'fence':native('fence',P), 'signal':native('signal',None,P),
             'backend':native('backend',P), 'context':native('context',P),
             'wait':native('wait',c.c_int,P,c.POINTER(WaitInfo),U)}
    phase('identity')
    span=Span(); assert funcs['uuid'](c.byref(span))==c.addressof(span)
    assert span.ptr and span.count==16
    vk_uuid = c.string_at(span.ptr,16)
    wp.config.quiet=True; wp.init(); dev=wp.get_device('cuda:0'); stream=wp.Stream(dev)
    cu = c.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/nvcuda.dll'))
    signatures={'cuDeviceGetUuid_v2':(P,c.c_int), 'cuCtxPushCurrent_v2':(P,),
        'cuCtxPopCurrent_v2':(c.POINTER(P),), 'cuImportExternalMemory':(c.POINTER(P),c.POINTER(MemDesc)),
        'cuExternalMemoryGetMappedBuffer':(c.POINTER(U),P,c.POINTER(BufferDesc)),
        'cuMemFree_v2':(U,), 'cuDestroyExternalMemory':(P,), 'cuEventCreate':(c.POINTER(P),c.c_uint),
        'cuEventRecord':(P,P), 'cuEventQuery':(P,), 'cuEventDestroy_v2':(P,),
        'cuMemGetInfo_v2':(c.POINTER(c.c_size_t),c.POINTER(c.c_size_t))}
    for name,args in signatures.items():
        f=getattr(cu,name); f.argtypes=args; f.restype=c.c_int
    uuid=(c.c_ubyte*16)(); checked(cu.cuDeviceGetUuid_v2(uuid,0),'UUID')
    R.update(vulkan_uuid=vk_uuid.hex(),cuda_uuid=bytes(uuid).hex(),uuid_equal=vk_uuid==bytes(uuid))
    assert vk_uuid == bytes(uuid), 'UUID mismatch: zero imports'
    assert R['imports']==0
    # Same guard used before every allocation; negative input cannot reach import.
    R['uuid_negative_rejected'] = not identity_matches(vk_uuid, bytes([vk_uuid[0]^1])+vk_uuid[1:])
    owner=bpy.context.window.as_pointer(); owner_context=funcs['context'](); assert owner_context
    backend=funcs['backend'](); assert P.from_address(backend).value==base+0x4c38088
    R.update(owner_window=owner, owner_context=owner_context)
    info=gpu.types.GPUShaderCreateInfo(); info.vertex_in(0,'VEC2','p')
    info.sampler(0,'FLOAT_2D','positions'); info.push_constant('INT','count'); info.push_constant('INT','frame')
    info.fragment_out(0,'VEC4','color')
    info.vertex_source('void main(){gl_Position=vec4(p,0,1);}')
    info.fragment_source('void main(){ ivec2 p=ivec2(gl_FragCoord.xy); int i=p.y*4096+p.x; vec4 expected=i<count?vec4(float(i)*0.5,float(frame),float(i%1000),1):vec4(0); color=vec4(any(notEqual(texelFetch(positions,p,0),expected))?1.0:0.0,0,0,0); }')
    shader=gpu.shader.create_from_info(info)
    batch=batch_for_shader(shader,'TRIS',{'p':((-1,-1),(3,-1),(-1,3))})
    stage('identity_pass', zero_import_negative=R['uuid_negative_rejected'])

def identity_matches(a,b):
    return len(a)==16 and len(b)==16 and a==b

def driver(name,*args):
    checked(cu.cuCtxPushCurrent_v2(dev.context),'push')
    try:
        result = getattr(cu,name)(*args)
    except BaseException:
        park(traceback.format_exc())
    if result != 0 and not (name=='cuEventQuery' and result==600):
        park(f'{name}: original CUresult={result}; no pop or further CUDA calls')
    prior=P(); checked(cu.cuCtxPopCurrent_v2(c.byref(prior)),'pop')
    return result

def memory():
    free=c.c_size_t(); total=c.c_size_t()
    checked(driver('cuMemGetInfo_v2',c.byref(free),c.byref(total)),'memory')
    handles=c.c_uint(); k32.GetProcessHandleCount.argtypes=[P,c.POINTER(c.c_uint)]
    k32.GetProcessHandleCount.restype=c.c_int
    assert k32.GetProcessHandleCount(P(-1),c.byref(handles))
    return {'free_device_bytes':free.value,'total_device_bytes':total.value,'process_handles':handles.value}

def dimensions(n):
    if not isinstance(n,int) or n<0 or n>4194304: raise ValueError('count outside bounded probe capacity')
    return (4096,(n+4095)//4096) if n else (0,0)

class Slot:
    def __init__(self,n):
        assert identity_matches(bytes.fromhex(R['vulkan_uuid']),bytes.fromhex(R['cuda_uuid']))
        self.n=n; self.w,self.h=dimensions(n); self.size=self.w*self.h*16
        self.pb=self.ext=self.ptr=self.event=self.fence=None; self.state='FREE'; self.frame=-1
        self.pb=funcs['create'](self.size); assert self.pb and funcs['size'](self.pb)>=self.size
        nh=NativeHandle(); funcs['handle'](c.byref(nh),self.pb); assert nh.handle and nh.size>=self.size
        desc=MemDesc(); desc.type=2;desc.handle.win32.handle=nh.handle;desc.size=nh.size;desc.flags=1
        if R['imports']==0:
            stage('first_export',requested_bytes=self.size,export_bytes=nh.size,handle=nh.handle,
                  descriptor_bytes=c.sizeof(desc),descriptor_hex=bytes(desc).hex())
        self.ext=P()
        try:
            R['imports']+=1; checked(driver('cuImportExternalMemory',c.byref(self.ext),c.byref(desc)),'import')
        finally:
            assert k32.CloseHandle(P(nh.handle)), 'CloseHandle'
        ptr=U(); bd=BufferDesc(0,self.size,0)
        checked(driver('cuExternalMemoryGetMappedBuffer',c.byref(ptr),self.ext,c.byref(bd)),'map')
        self.ptr=ptr.value; self.arr=wp.array(ptr=self.ptr,dtype=wp.vec4,shape=self.w*self.h,device=dev,copy=False)
        self.event=P(); checked(driver('cuEventCreate',c.byref(self.event),2),'event create')
        self.fence=funcs['fence'](); assert self.fence

    def launch(self,frame):
        assert self.state=='FREE'; self.frame=frame
        wp.launch(fill,dim=self.w*self.h,inputs=[self.arr,self.n,frame],device=dev,stream=stream)
        checked(driver('cuEventRecord',self.event,stream.cuda_stream),'event record')
        self.state='WRITING'; self.started=time.monotonic()

    def poll(self):
        if self.state=='WRITING':
            R['cuda_queries']+=1; result=driver('cuEventQuery',self.event)
            if result==0:self.state='READY'
            elif result==600:R['not_ready']+=1
            else:raise RuntimeError(f'event query CUresult={result}')
        elif self.state=='IN_FLIGHT':
            R['vulkan_queries']+=1
            backend=funcs['backend'](); assert P.from_address(backend).value==base+0x4c38088
            device=backend+16; native=P.from_address(device+16).value
            sem=U.from_address(device+144); value=U.from_address(self.fence+16)
            assert native and sem.value and value.value and c.c_ubyte.from_address(self.fence+8).value
            wi=WaitInfo(1000207004,None,0,1,c.pointer(sem),c.pointer(value))
            result=funcs['wait'](native,c.byref(wi),0)
            if result==0:self.state='FREE'
            elif result==2:R['not_ready']+=1
            else:raise RuntimeError(f'Vulkan query VkResult={result}')
        if self.state in ('WRITING','IN_FLIGHT') and time.monotonic()-self.started>.5:
            raise TimeoutError(f'{self.state} exceeded 500ms')

    def upload(self,texture):
        assert self.state=='READY'
        assert P.from_address(id(texture)+8).value==base+0x66ac390
        tp=P.from_address(id(texture)+16).value; assert tp
        funcs['update'](tp,0,self.pb,0,0,0,self.w,self.h,1)

    def submitted(self):
        funcs['signal'](self.fence); self.state='IN_FLIGHT';self.started=time.monotonic()

    def close(self):
        assert self.state=='FREE', 'unresolved resource cannot enter Blender discard pool'
        self.arr=None
        checked(driver('cuMemFree_v2',self.ptr),'free mapping');self.ptr=None
        checked(driver('cuDestroyExternalMemory',self.ext),'destroy import');self.ext=None
        checked(driver('cuEventDestroy_v2',self.event),'destroy event');self.event=None
        funcs['free'](self.fence);self.fence=None
        funcs['free'](self.pb);self.pb=None

state={'mode':'init','slots':[],'texture':None,'offscreen':None,'index':0,'frame':0,'draws':0}
counts=[1,4095,4096,4097,4194304]

def new_case(n,mode):
    w,h=dimensions(n)
    state.update(mode=mode,slots=[Slot(n),Slot(n)],texture=gpu.types.GPUTexture((w,h),format='RGBA32F'),frame=0)
    if mode=='stress':
        state['offscreen']=gpu.types.GPUOffScreen(w,h,format='RGBA32F')
        with state['offscreen'].bind(): gpu.state.active_framebuffer_get().clear(color=(0,0,0,0))
    state['slots'][0].launch(0)

def draw():
    try:
        if state['mode']=='init':
            initialize(); phase('correctness')
            for n in [-1,4194305,1.5]:
                try: dimensions(n)
                except ValueError: continue
                raise AssertionError('invalid count accepted')
            assert dimensions(0)==(0,0)
            R['count_rejection_pass']=True
            new_case(counts[0],'correctness')
            return
        if state['mode']=='done':return
        if not bpy.context.window or bpy.context.window.as_pointer()!=owner:return
        assert funcs['context']()==owner_context
        state['draws']+=1
        slots=state['slots']
        ready=[s for s in slots if s.state=='READY']
        if ready:
            slot=max(ready,key=lambda s:s.frame)
            slot.upload(state['texture'])
            if state['mode']=='stress':
                with state['offscreen'].bind():
                    gpu.state.blend_set('ADDITIVE');shader.bind()
                    shader.uniform_sampler('positions',state['texture'])
                    shader.uniform_int('count',slot.n);shader.uniform_int('frame',slot.frame)
                    batch.draw(shader);gpu.state.blend_set('NONE')
                R['stress_frames']+=1
            slot.submitted()  # Covers the upload AND final texture read above.
            if state['mode']=='stress' and R['stress_frames']<10000:
                available=next((s for s in slots if s.state=='FREE'),None)
                if available:
                    state['frame']+=1;available.launch(state['frame'])
        if all(s.state=='FREE' for s in slots):
            if state['mode']=='correctness':
                n=counts[state['index']]; w,h=dimensions(n)
                data=state['texture'].read();data.dimensions=w*h*4
                actual=np.asarray(data).reshape(-1,4); ids=np.arange(w*h,dtype=np.float32)
                expected=np.stack((ids*.5,ids*0,ids%1000,np.ones_like(ids)),axis=1);expected[n:]=0
                bad=int(np.count_nonzero(actual!=expected));assert bad==0,f'{bad} components differ'
                R['rows'].append({'n':n,'components':w*h*4,'mismatches':bad})
                for s in slots:s.close()
                state['texture']=None;state['slots']=[];state['index']+=1
                stage('correctness_row',n=n)
                if state['index']<len(counts):new_case(counts[state['index']],'correctness')
                else:
                    phase('cycles'); R['cycle_baseline']=memory();new_case(4097,'cycles')
            elif state['mode']=='cycles':
                for s in slots:s.close()
                state['texture']=None;state['slots']=[]
                R['cycles'].append(memory())
                if len(R['cycles'])<50:new_case(4097,'cycles')
                else:
                    stage('cycles_complete',count=50);phase('stress');new_case(4097,'stress')
            elif R['stress_frames']>=10000:
                px=state['offscreen'].texture_color.read();px.dimensions=4096*2*4
                R['readbacks_stress']+=1
                bad=float(np.asarray(px).reshape(-1,4)[:,0].sum());assert bad==0,f'stress mismatch {bad}'
                R['stress_error_sum']=bad;phase('cleanup')
                for s in slots:s.close()
                state['texture']=None;state['slots']=[];state['offscreen'].free();state['offscreen']=None
                R.update(status='PASS_BOUNDED_NATIVE_CORE',cleanup='CUDA_RELEASED_BLENDER_DEFERRED',final_memory=memory(),draws=state['draws'])
                state['mode']='done';stage('core_complete')
            else:
                state['frame']+=1;slots[0].launch(state['frame'])
    except BaseException:
        park(traceback.format_exc())

def tick():
    try:
        if state['mode']=='done':
            bpy.types.SpaceView3D.draw_handler_remove(handler,'WINDOW')
            save(OUT/'finished.json',{'status':R['status'],'pid':os.getpid()})
            bpy.ops.wm.quit_blender()
            return None
        # Timer queries require no current graphics context; submission/free stays in draw.
        if state['mode']!='init':
            assert owner in {w.as_pointer() for w in bpy.context.window_manager.windows}
            for slot in state['slots']: slot.poll()
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type=='VIEW_3D':area.tag_redraw()
        return .001
    except BaseException:
        park(traceback.format_exc())

class WM_OT_m0_lifetime_guard(bpy.types.Operator):
    bl_idname='wm.m0_lifetime_guard'
    bl_label='Disposable M0 lifetime guard'

    def invoke(self, context, event):
        self.owner=context.window.as_pointer()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if state['mode']=='done':return {'FINISHED'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        if state['mode']!='done':
            park('owner cancellation before normal drain; no return into context destruction')

def before_load(unused):
    if state['mode']!='done':park('file load before normal drain; nonreturning containment')

bpy.utils.register_class(WM_OT_m0_lifetime_guard)
assert bpy.ops.wm.m0_lifetime_guard('INVOKE_DEFAULT')=={'RUNNING_MODAL'}
bpy.app.handlers.load_pre.append(before_load)
handler=bpy.types.SpaceView3D.draw_handler_add(draw,(), 'WINDOW','POST_VIEW')
bpy.app.timers.register(tick,first_interval=.1)
stage('registered')
