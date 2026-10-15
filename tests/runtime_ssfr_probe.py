"""Execute via Blender MCP in an isolated GUI profile with the candidate installed.

Controlled GL pixels test rendering independently of the solver. Readbacks occur
only in this verifier, never in the production simulation/draw path.
"""
import ctypes as c
import hashlib
import importlib
import json
import itertools
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

import bpy
import gpu
from mathutils import Matrix
import warp as wp
_module = os.environ.get('SSFR_MODULE', 'warp_sph_realtime')
addon = importlib.import_module(_module)
g = importlib.import_module(_module + '.render.interop_bridge')
v = importlib.import_module(_module + '.render.viewport_draw')

OUT = Path(os.environ.get('SSFR_EVIDENCE', str(Path.cwd() / 'evidence')))
OUT.mkdir(parents=True, exist_ok=True)
RESULT = {'status': 'RUNNING', 'cases': [], 'errors': [], 'pid': os.getpid(),
          'blender': bpy.app.version_string, 'build': bpy.app.build_hash.decode(),
          'backend': gpu.platform.backend_type_get(), 'gpu': gpu.platform.renderer_get(),
          'warp': wp.__version__, 'addon': addon.__file__, 'hashes': {}}
for name in ('render/interop_bridge.py', 'render/viewport_draw.py', 'render/ssfr_shaders.py', 'sim/sim_core.py'):
    RESULT['hashes'][name] = hashlib.sha256((Path(addon.__file__).parent/name).read_bytes()).hexdigest()


def save():
    (OUT/'gpu-controls.json').write_text(json.dumps(RESULT, indent=2))


def fn(name, restype, args):
    exported = getattr(g._GLInterop.gl, name, None)
    if exported is not None:
        exported.restype, exported.argtypes = restype, args
        return exported
    addr = g._GLInterop._get_proc_addr(name.encode())
    return c.WINFUNCTYPE(restype, *args)(addr)


def run():
    gl = g._GLInterop
    gl.ensure_initialized()
    read_pixels = fn('glReadPixels', None, [c.c_int]*4+[c.c_uint,c.c_uint,c.c_void_p])
    gen_rb = fn('glGenRenderbuffers', None, [c.c_int,c.POINTER(c.c_uint)])
    bind_rb = fn('glBindRenderbuffer', None, [c.c_uint,c.c_uint])
    store_rb = fn('glRenderbufferStorageMultisample', None, [c.c_uint,c.c_int,c.c_uint,c.c_int,c.c_int])
    attach_rb = fn('glFramebufferRenderbuffer', None, [c.c_uint]*4)
    delete_rb = fn('glDeleteRenderbuffers', None, [c.c_int,c.POINTER(c.c_uint)])
    check_fb = fn('glCheckFramebufferStatus', c.c_uint, [c.c_uint])
    gl_finish = fn('glFinish', None, [])
    renderer = v.SSFRViewportRenderer()
    scene = SimpleNamespace(warp_sph_particle_radius=.2, warp_sph_particle_color=(.12,.68,1,1),
                            warp_sph_depth_test=True, warp_sph_filter_iterations=2)
    buf = None
    @v._preserve_gl_state
    def controlled():
        nonlocal buf
        wp.init()
        assert wp.get_device('cuda:0').is_cuda
        buf = g.ZeroCopyGLBuffer52(1)
        positions = wp.array([[0.,0.,-.5]], dtype=wp.vec3, device='cuda:0')
        wp.copy(buf.map_to_warp(), positions)
        buf.unmap()
        RESULT['cuda_vbo_alias'] = True
        for samples, (width,height), camera in itertools.product((0,4), ((64,48),(65,49)), ('ORTHO','PERSP')):
            fbo = c.c_uint()
            rbs = (c.c_uint*2)()
            gl.glGenFramebuffers(1,c.byref(fbo))
            gen_rb(2,rbs)
            gl.glBindFramebuffer(g.GL_FRAMEBUFFER,fbo.value)
            for rb,fmt,attachment in ((rbs[0],g.GL_RGBA8,g.GL_COLOR_ATTACHMENT0),
                                     (rbs[1],g.GL_DEPTH24_STENCIL8,g.GL_DEPTH_STENCIL_ATTACHMENT)):
                bind_rb(0x8D41,rb)
                store_rb(0x8D41,samples,fmt,width,height)
                attach_rb(g.GL_FRAMEBUFFER,attachment,0x8D41,rb)
            assert check_fb(g.GL_FRAMEBUFFER)==0x8CD5, 'incomplete control FBO'
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4) if camera == 'ORTHO' else
                        Matrix(((2.,0.,0.,0.),(0.,2.,0.,0.),(0.,0.,-1.020202,-.2020202),(0.,0.,-1.,0.))))
                measurements=[]
                for blocked in (False,True):
                    gl.glBindFramebuffer(g.GL_FRAMEBUFFER,fbo.value)
                    gl.glViewport(0,0,width,height)
                    gl.glDisable(g.GL_SCISSOR_TEST)
                    gl.glColorMask(1,1,1,1)
                    gl.glDepthMask(1)
                    gl.glClearDepth(.1 if blocked else 1.)
                    gl.glClearColor(.18,.22,.28,1.)
                    gl.glClear(g.GL_COLOR_BUFFER_BIT|g.GL_DEPTH_BUFFER_BIT)
                    def capture_state():
                        state=[]
                        for pname,count in ((g.GL_DEPTH_WRITEMASK,1),(g.GL_DEPTH_FUNC,1),
                                (g.GL_BLEND_SRC_RGB,1),(g.GL_BLEND_DST_RGB,1),(g.GL_BLEND_SRC_ALPHA,1),
                                (g.GL_BLEND_DST_ALPHA,1),(g.GL_BLEND_EQUATION_RGB,1),(g.GL_BLEND_EQUATION_ALPHA,1),
                                (g.GL_CURRENT_PROGRAM,1),(g.GL_VERTEX_ARRAY_BINDING,1),(g.GL_ACTIVE_TEXTURE,1),
                                (g.GL_VIEWPORT,4),(g.GL_DRAW_FRAMEBUFFER_BINDING,1),(g.GL_READ_FRAMEBUFFER_BINDING,1)):
                            value=(c.c_int*count)();gl.glGetIntegerv(pname,value);state.append(list(value))
                        return state
                    gl.glDepthMask(0)
                    gl.glBlendFuncSeparate(g.GL_ONE,0,g.GL_SRC_ALPHA,g.GL_ONE)
                    before=capture_state()
                    renderer.draw_ssfr_liquid(buf,scene,0,0,width,height)
                    assert capture_state()==before, 'real GL state changed'
                    RESULT['graphics_state_roundtrip']='PASS'
                    data=renderer._ctx_resources[gl.current_context()]
                    gl.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,data['fbo_thick'].value)
                    pixels=(c.c_float*(data['half_w']*data['half_h']))()
                    read_pixels(0,0,data['half_w'],data['half_h'],g.GL_RED,g.GL_FLOAT,pixels)
                    thickness=sum(pixels)
                    gl.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,fbo.value)
                    rgba=(c.c_float*(width*height*4))()
                    # Multisampled readbacks must be resolved before readPixels.
                    gl.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER,data['fbo_scene_col'].value)
                    gl.glBlitFramebuffer(0,0,width,height,0,0,width,height,g.GL_COLOR_BUFFER_BIT,g.GL_NEAREST)
                    gl.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,data['fbo_scene_col'].value)
                    read_pixels(0,0,width,height,g.GL_RGBA,g.GL_FLOAT,rgba)
                    error=gl.glGetError()
                    assert error==0, f'GL error {error:x}'
                    measurements.append(thickness)
                    if not blocked and samples==4 and width==65:
                        image=bpy.data.images.new('SSFR GPU control',width=width,height=height,alpha=True)
                        image.pixels.foreach_set(list(rgba))
                        image.filepath_raw=str(OUT/'liquid-control.png');image.file_format='PNG';image.save()
                assert measurements[0]>.01, f'no visible thickness: {measurements}'
                assert measurements[1]<1e-6, f'hidden thickness leaked: {measurements}'
                # Negative control: the exact forbidden MSAA scaling must fail.
                if samples:
                    gl.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,fbo.value)
                    gl.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER,data['fbo_thick'].value)
                    gl.glBlitFramebuffer(0,0,width,height,0,0,data['half_w'],data['half_h'],g.GL_DEPTH_BUFFER_BIT,g.GL_NEAREST)
                    assert gl.glGetError()==g.GL_INVALID_OPERATION, 'negative MSAA control not reached'
                RESULT['cases'].append({'samples':samples,'camera':camera,'size':[width,height],
                                        'visible_thickness':measurements[0], 'occluded_thickness':measurements[1]})
            gl.glBindFramebuffer(g.GL_FRAMEBUFFER,0)
            gl.glDeleteFramebuffers(1,c.byref(fbo));delete_rb(2,rbs)
        # Real invalid fragment compilation and deletion are driver tested.
        shader_ids=[]
        create_shader=gl.glCreateShader
        def tracked_create(stage):
            shader=create_shader(stage);shader_ids.append(shader);return shader
        is_shader=fn('glIsShader',c.c_ubyte,[c.c_uint])
        gl.glCreateShader=staticmethod(tracked_create)
        for _ in range(3):
            try: gl.compile_program('#version 450\nvoid main(){gl_Position=vec4(0);}', 'broken shader')
            except RuntimeError as exc: assert 'compile error' in str(exc)
            else: raise AssertionError('invalid shader accepted')
        gl.glCreateShader=create_shader
        assert shader_ids and not any(is_shader(shader) for shader in shader_ids), 'shader objects leaked'
        RESULT['failed_shader_objects_deleted']=len(shader_ids)
        gl_finish()
        RESULT['shader_failure_control']='PASS'
        renderer.destroy()
        assert not renderer._ctx_resources
        buf.destroy();buf=None
    try:
        controlled()
        RESULT['status']='PASS'
    except Exception:
        RESULT['status']='FAIL';RESULT['errors'].append(traceback.format_exc())
        if buf: buf.destroy()
    save()


_done=False
def on_draw():
    global _done
    if _done:return
    _done=True
    try:
        run()
    except Exception:
        RESULT['status']='FAIL';RESULT['errors'].append(traceback.format_exc());save()
    def remove():
        bpy.types.SpaceView3D.draw_handler_remove(_handle,'WINDOW')
    bpy.app.timers.register(remove,first_interval=.1)

_handle=bpy.types.SpaceView3D.draw_handler_add(on_draw,(),'WINDOW','POST_VIEW')
for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type=='VIEW_3D':area.tag_redraw()
save()
