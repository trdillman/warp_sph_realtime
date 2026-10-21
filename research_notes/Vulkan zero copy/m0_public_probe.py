"""Public-API prerequisite probe only; no native pointers, external memory or M1 code."""
import bpy
import gpu
import json
import os
import pathlib
import sys
import time
import traceback
import numpy as np
import warp as wp
from gpu_extras.batch import batch_for_shader

OUT = pathlib.Path(__file__).resolve().parents[2] / 'evidence/vulkan-zero-copy/20261017-m0'
RESULT = {'scope':'public API prerequisite only; not zero-copy transport', 'pid':os.getpid(),
          'backend':gpu.platform.backend_type_get(), 'blender':bpy.app.version_string,
          'build_hash':bpy.app.build_hash.decode(), 'python':sys.version,
          'warp_version':wp.__version__, 'warp_path':wp.__file__,
          'user_paths':{k:v for k,v in os.environ.items() if k.startswith('BLENDER_USER_')},
          'gpu_renderer':gpu.platform.renderer_get(),
          'uuid_public_attributes':[n for n in dir(gpu.platform) if 'uuid' in n.lower()],
          'rows':[], 'native_calls':0, 'external_imports':0}
TARGET=OUT/('public-'+RESULT['backend'].lower()+'.json')

def save():
    TARGET.write_text(json.dumps(RESULT,indent=2),encoding='utf8')

def run():
    try:
        assert wp.__version__=='1.17.0'
        wp.config.kernel_cache_dir=str(OUT/'warp-cache')
        wp.init()
        device=wp.get_device('cuda:0')
        RESULT['cuda_device']=str(device)
        RESULT['cuda_name']=device.name
        RESULT['cuda_uuid']=str(getattr(device,'uuid',None))
        stream=wp.Stream(device)
        with wp.ScopedStream(stream):
            arr=wp.ones(4097,dtype=wp.float32,device=device)
        wp.synchronize_stream(stream)
        RESULT['warp_stream_known_answer']=bool(np.all(arr.numpy()==1))
        assert RESULT['warp_stream_known_answer']
        info=gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0,'VEC2','corner')
        info.sampler(0,'FLOAT_2D','positions')
        info.push_constant('VEC2','dims')
        iface=gpu.types.GPUStageInterfaceInfo('m0_values')
        iface.flat('VEC4','value')
        info.vertex_out(iface)
        info.fragment_out(0,'VEC4','fragColor')
        info.vertex_source('void main(){ int i=gpu_InstanceIndex; ivec2 cell=ivec2(i%4096,i/4096); value=texelFetch(positions,cell,0); vec2 xy=(vec2(cell)+corner)/dims*2.0-1.0; gl_Position=vec4(xy,0.0,1.0); }')
        info.fragment_source('void main(){ fragColor=value; }')
        shader=gpu.shader.create_from_info(info)
        batch=batch_for_shader(shader,'TRIS',{'corner':[(0,0),(1,0),(1,1),(0,0),(1,1),(0,1)]})
        for n in [0,1,4095,4096,4097,4194304]:
            if n==0:
                RESULT['rows'].append({'n':0,'status':'PASS','allocated':False,'drawn':False})
                save()
                continue
            h=(n+4095)//4096
            values=np.zeros((h*4096,4),dtype=np.float32)
            values[:n,0]=np.arange(n,dtype=np.float32)+1
            values[:n,1]=3
            values[:n,2]=7
            values[:n,3]=11
            tex=gpu.types.GPUTexture((4096,h),format='RGBA32F',data=gpu.types.Buffer('FLOAT',values.size,values.ravel()))
            off=gpu.types.GPUOffScreen(4096,h,format='RGBA32F')
            try:
                begin=time.perf_counter()
                with off.bind():
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    gpu.state.face_culling_set('NONE')
                    gpu.state.viewport_set(0,0,4096,h)
                    gpu.state.active_framebuffer_get().clear(color=(0,0,0,0))
                    shader.bind()
                    shader.uniform_sampler('positions',tex)
                    shader.uniform_float('dims',(4096,h))
                    batch.draw_instanced(shader,instance_count=n)
                    read=off.texture_color.read()
                    read.dimensions=values.size
                    observed=np.asarray(read,dtype=np.float32).reshape(values.shape)
                    bad=int(np.count_nonzero(observed!=values))
                row={'n':n,'padded_count':len(values),'components_compared':values.size,
                     'mismatches':bad,'status':'PASS' if bad==0 else 'FAIL',
                     'draw_plus_readback_ms':(time.perf_counter()-begin)*1000,
                     'timing_scope':'diagnostic with full readback; not transport performance'}
                RESULT['rows'].append(row)
                save()
                if bad: raise AssertionError(row)
            finally:
                off.free()
            del tex,values,observed,read
        RESULT['status']='PASS_PUBLIC_PREREQUISITES_ONLY'
    except Exception:
        RESULT['status']='FAIL'
        RESULT['error']=traceback.format_exc()
    finally:
        save()
        bpy.types.SpaceView3D.draw_handler_remove(HANDLE,'WINDOW')
    return None

def once():
    global FIRED
    if FIRED: return
    FIRED=True
    run()

FIRED=False
save()
HANDLE=bpy.types.SpaceView3D.draw_handler_add(once,(),'WINDOW','POST_PIXEL')
for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type=='VIEW_3D': area.tag_redraw()
