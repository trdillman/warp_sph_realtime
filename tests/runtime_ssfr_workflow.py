"""GUI acceptance through public operators at the unmodified default workload.

Run in a disposable factory profile with the candidate enabled. Deadlines and
sample counts are verification settings; solver/physics properties stay default.
"""
import bpy
import gpu
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from mathutils import Vector
_module = os.environ.get('SSFR_MODULE', 'warp_sph_realtime')
a = importlib.import_module(_module + '.addon')
v = importlib.import_module(_module + '.render.viewport_draw')
g = importlib.import_module(_module + '.render.interop_bridge')

OUT=Path(os.environ.get('SSFR_EVIDENCE',str(Path.cwd() / 'evidence')))
R={'status':'RUNNING','pid':os.getpid(),'cases':[],'errors':[], 'blender':bpy.app.version_string,
   'backend':gpu.platform.backend_type_get(),'gpu':gpu.platform.renderer_get(),
   'hashes':{n:hashlib.sha256((Path(a.__file__).parent/n).read_bytes()).hexdigest()
             for n in ('render/interop_bridge.py', 'render/viewport_draw.py', 'render/ssfr_shaders.py', 'sim/sim_core.py')}}
start=time.monotonic()
scene=bpy.context.scene
area=next(x for x in bpy.context.screen.areas if x.type=='VIEW_3D')
region=next(x for x in area.regions if x.type=='WINDOW')
rv=area.spaces.active.region_3d
rv.view_location=(0,0,1.3);rv.view_distance=9
rv.view_rotation=Vector((.8,-1,.65)).to_track_quat('Z','Y')
area.spaces.active.shading.type='SOLID'
area.spaces.active.overlay.show_overlays=False
R['workload']={k:getattr(scene,k) for k in ('warp_sph_particle_count','warp_sph_substeps',
                  'warp_sph_use_cuda_graph','warp_sph_stiffness','warp_sph_time_step_scale')}
assert scene.warp_sph_particle_count==1_000_000
CASES=[('SSFR_LIQUID','PERSP'),('SSFR_LIQUID','ORTHO'),('SPHERES','ORTHO'),
       ('POINTS','PERSP'),('SSFR_LIQUID','PERSP')]*3
idx=-1
draws=[]
phase='start'
deadline=time.monotonic()+30
a._activate_runtime()  # bind the lazy draw callback before wrapping it
original=a.draw_particles_viewport


def save(): (OUT/'gui-workflow.json').write_text(json.dumps(R,indent=2))


def observed(buf,running):
    try:
        t=time.perf_counter()
        if 'depth_attachment' not in R:
            gl=g._GLInterop;gl.ensure_initialized()
            def proc(name,args):return ctypes.WINFUNCTYPE(None,*args)(gl._get_proc_addr(name.encode()))
            query=proc('glGetFramebufferAttachmentParameteriv',[ctypes.c_uint]*3+[ctypes.POINTER(ctypes.c_int)])
            values=[]
            for param in (0x8CD0,0x8CD1):
                out=ctypes.c_int();query(g.GL_DRAW_FRAMEBUFFER,g.GL_DEPTH_ATTACHMENT,param,ctypes.byref(out));values.append(out.value)
            fmt=ctypes.c_int()
            if values[0]==0x1702:proc('glGetTextureLevelParameteriv',[ctypes.c_uint,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(ctypes.c_int)])(values[1],0,0x1003,ctypes.byref(fmt))
            elif values[0]==0x8D41:proc('glGetNamedRenderbufferParameteriv',[ctypes.c_uint,ctypes.c_uint,ctypes.POINTER(ctypes.c_int)])(values[1],0x8D44,ctypes.byref(fmt))
            R['depth_attachment']={'type':hex(values[0]),'name':values[1],'format':hex(fmt.value),'error':gl.glGetError()};save()
        original(buf,running)
        if running and idx>=0:
            error=g._GLInterop.glGetError()
            if error:raise RuntimeError(f'GL error 0x{error:x}')
            draws.append((time.perf_counter()-t)*1000)
    except Exception:
        R['errors'].append(traceback.format_exc());save()
a.draw_particles_viewport=observed


def tick():
    global idx,phase,deadline,draws,region
    try:
        if time.monotonic()-start>180:raise TimeoutError('GUI workflow deadline')
        if R['errors']:raise RuntimeError('draw callback failed')
        with bpy.context.temp_override(area=area,region=region):
            if phase=='start':
                result=bpy.ops.wm.warp_sph_start()
                assert 'RUNNING_MODAL' in result
                phase='case';idx+=1;draws=[];deadline=time.monotonic()+20
                scene.warp_sph_render_mode,rv.view_perspective=CASES[idx]
                area.tag_redraw()
                return .3
            if phase=='stopping':
                if a.RT.gl_buf is not None:
                    if time.monotonic()>deadline:raise TimeoutError('stop did not release CUDA buffer')
                    return .2
                R['cases'].append({'cycle_stop':idx//5+1,'cuda_buffer_released':True,
                                   'solver_released':a.RT.solver is None})
                if idx==len(CASES)-1:
                    R['status']='PASS';save();bpy.ops.wm.quit_blender();return None
                phase='start';return .3
            if len(draws)<8 or a.RT.step_count<8:
                if time.monotonic()>deadline:raise TimeoutError('no fresh rendered simulation steps')
                area.tag_redraw();return .2
            data=v._SSFR_RENDERER._ctx_resources
            R['cases'].append({'cycle':idx//5+1,'mode':CASES[idx][0],'camera':CASES[idx][1],
                               'rendered_frames':len(draws),'step_count':a.RT.step_count,
                               'submission_median_ms':statistics.median(draws[2:]),
                               'solver_step_ms':a.RT.frame_time_ms,'vram_solver_mb':a.RT.vram_mb,
                               'contexts':len(data),'fbo_sizes':[[d['width'],d['height']] for d in data.values()]})
            # Capture the actual viewport after a final liquid case; not an offline render.
            if idx in (0,1,4):
                bpy.ops.screen.screenshot(filepath=str(OUT/f'viewport-{idx}-{CASES[idx][1]}.png'))
            if idx%5==4:
                bpy.ops.wm.warp_sph_stop();phase='stopping';deadline=time.monotonic()+20
                save();return .3
            idx+=1;draws=[];deadline=time.monotonic()+20
            scene.warp_sph_render_mode,rv.view_perspective=CASES[idx]
            # Resize the actual VIEW_3D by changing the header layout via area split.
            if idx==1:
                bpy.ops.screen.area_split(direction='VERTICAL',factor=.7)
                region=next(x for x in area.regions if x.type=='WINDOW')
            area.tag_redraw();save();return .3
    except Exception:
        R['status']='FAIL';R['errors'].append(traceback.format_exc());save()
        try:bpy.ops.wm.warp_sph_stop()
        except Exception:pass
        bpy.app.timers.register(lambda:bpy.ops.wm.quit_blender() and None,first_interval=1)
        return None
bpy.app.timers.register(tick,first_interval=2)
save()
