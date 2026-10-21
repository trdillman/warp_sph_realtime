"""Probe: CUDA -> Blender Vulkan zero-copy via Blender's internal GPU_pixel_buffer API.

Path: GPU_pixel_buffer_create -> get_native_handle (exportable VkDeviceMemory, Win32 handle)
-> cudaImportExternalMemory -> CUDA writes -> GPU_texture_update_sub_from_pixel_buffer
(GPU->GPU copy into a Python GPUTexture) -> texture.read() + gpu shader sample verification.

Run: blender --factory-startup --gpu-backend vulkan --python this.py -- out.json
RVAs are for Blender 5.2.0 windows-x64 (resolved from the shipped blender.pdb).
"""
import ctypes, json, os, sys, time, traceback
from ctypes import wintypes
import bpy, gpu

out_path = sys.argv[sys.argv.index("--") + 1]
R = {"blender": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash,
     "backend": gpu.platform.backend_type_get(), "stages": []}


def stage(name, **kw):
    R["stages"].append({"stage": name, **kw})
    with open(out_path, "w") as f:  # write after every stage so a crash shows where
        json.dump(R, f, indent=2)


RVA = {
    "GPU_pixel_buffer_create": 0x00d96ce0,
    "GPU_pixel_buffer_free": 0x00ad2100,
    "GPU_pixel_buffer_get_native_handle": 0x00d96d20,
    "GPU_pixel_buffer_size": 0x00d96d10,
    "GPU_texture_update_sub_from_pixel_buffer": 0x00d95b30,
    "GPU_finish": 0x00d8ef10,
    "GPU_texture_width": 0x0046f910,
    "GPU_texture_height": 0x00d96830,
}
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetModuleHandleW.restype = ctypes.c_void_p
k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
k32.CloseHandle.argtypes = [wintypes.HANDLE]
BASE = k32.GetModuleHandleW(None)


def fn(name, restype, *argtypes):
    return ctypes.CFUNCTYPE(restype, *argtypes)(BASE + RVA[name])


class NativeHandle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_int64), ("size", ctypes.c_size_t)]


pb_create = fn("GPU_pixel_buffer_create", ctypes.c_void_p, ctypes.c_size_t)
pb_free = fn("GPU_pixel_buffer_free", None, ctypes.c_void_p)
pb_size = fn("GPU_pixel_buffer_size", ctypes.c_size_t, ctypes.c_void_p)
pb_native = fn("GPU_pixel_buffer_get_native_handle", NativeHandle, ctypes.c_void_p)
tex_update_from_pb = fn("GPU_texture_update_sub_from_pixel_buffer", None, ctypes.c_void_p, ctypes.c_int,
                        ctypes.c_void_p, *([ctypes.c_int] * 6))
gpu_finish = fn("GPU_finish", None)
tex_w = fn("GPU_texture_width", ctypes.c_int, ctypes.c_void_p)
tex_h = fn("GPU_texture_height", ctypes.c_int, ctypes.c_void_p)
GPU_DATA_FLOAT = 0

# ---- CUDA runtime ----
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")
cu = ctypes.CDLL("cudart64_12.dll")


class ExtMemHandleDesc(ctypes.Structure):
    class _H(ctypes.Union):
        class _W(ctypes.Structure):
            _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]
        _fields_ = [("fd", ctypes.c_int), ("win32", _W), ("nvSciBufObject", ctypes.c_void_p)]
    _fields_ = [("type", ctypes.c_int), ("handle", _H), ("size", ctypes.c_ulonglong),
                ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class ExtMemBufferDesc(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


def ck(err, what):
    if err != 0:
        cu.cudaGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(f"{what} failed: {err} {cu.cudaGetErrorString(err).decode()}")


def tex_ptr(pytex):
    return ctypes.c_void_p.from_address(id(pytex) + 16).value  # BPyGPUTexture{PyObject_HEAD; Texture *tex}


W, H = 4096, 1024  # 4,194,304 vec4 particles = 64 MiB
N = W * H
NBYTES = N * 16


def run():
    stage("start", base=hex(BASE))
    tex = gpu.types.GPUTexture((W, H), format="RGBA32F")
    tp = tex_ptr(tex)
    stage("texture", ptr=hex(tp), w=tex_w(tp), h=tex_h(tp))
    assert (tex_w(tp), tex_h(tp)) == (W, H), "texture pointer offset wrong"

    pb = pb_create(NBYTES)
    stage("pixel_buffer_create", ptr=hex(pb or 0), size=pb_size(pb))
    nh = pb_native(pb)
    stage("native_handle", handle=hex(nh.handle), size=nh.size)
    assert nh.handle, "no native handle (external_memory unsupported?)"

    d = ExtMemHandleDesc()
    d.type = 2  # cudaExternalMemoryHandleTypeOpaqueWin32
    d.handle.win32.handle = nh.handle
    d.size = nh.size
    d.flags = 1  # cudaExternalMemoryDedicated (VMA dedicated pool)
    ext = ctypes.c_void_p()
    ck(cu.cudaImportExternalMemory(ctypes.byref(ext), ctypes.byref(d)), "cudaImportExternalMemory")
    k32.CloseHandle(wintypes.HANDLE(nh.handle))  # CUDA does not take ownership
    bd = ExtMemBufferDesc(offset=0, size=NBYTES, flags=0)
    dptr = ctypes.c_void_p()
    ck(cu.cudaExternalMemoryGetMappedBuffer(ctypes.byref(dptr), ext, ctypes.byref(bd)), "GetMappedBuffer")
    stage("cuda_import", devptr=hex(dptr.value))

    # Pattern 1: cudaMemcpy host->device into the Vulkan-owned memory.
    host = (ctypes.c_float * (N * 4))()
    for i in range(0, N, 977):  # sparse pattern, cheap to build
        host[i * 4 + 0] = float(i); host[i * 4 + 1] = 1.5; host[i * 4 + 2] = -2.0; host[i * 4 + 3] = 7.0
    ck(cu.cudaMemcpy(dptr, host, ctypes.c_size_t(NBYTES), 1), "cudaMemcpy")
    ck(cu.cudaDeviceSynchronize(), "sync")
    tex_update_from_pb(tp, GPU_DATA_FLOAT, pb, 0, 0, 0, W, H, 1)
    gpu_finish()
    buf = tex.read()
    buf.dimensions = N * 4
    bad = [i for i in range(0, N, 977)
           if (buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]) != (float(i), 1.5, -2.0, 7.0)]
    stage("verify_cudamemcpy", checked=len(range(0, N, 977)), mismatches=len(bad), first_bad=bad[:5])

    # Pattern 2: a Warp kernel writes into the same memory (the real sim path).
    try:
        sys.path.insert(0, os.path.join(bpy.utils.resource_path("USER"), "scripts", "warp_foundry_runtime"))
        import warp as wp
        wp.config.quiet = True
        wp.init()

        @wp.kernel
        def fill(a: wp.array(dtype=wp.vec4)):
            i = wp.tid()
            a[i] = wp.vec4(float(i) * 0.5, 3.0, float(i % 1000), 1.0)

        arr = wp.array(ptr=dptr.value, dtype=wp.vec4, shape=N, device="cuda:0", copy=False)
        wp.launch(fill, dim=N, inputs=[arr], device="cuda:0")
        wp.synchronize_device("cuda:0")
        tex_update_from_pb(tp, GPU_DATA_FLOAT, pb, 0, 0, 0, W, H, 1)
        gpu_finish()
        buf = tex.read(); buf.dimensions = N * 4
        bad = [i for i in range(0, N, 1009)
               if (buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2]) != (float(i) * 0.5, 3.0, float(i % 1000))]
        stage("verify_warp_kernel", warp=wp.__version__, mismatches=len(bad), first_bad=bad[:5])

        # Timing: per-frame GPU->GPU copy cost (pixel buffer -> texture), 64 MiB.
        ts = []
        for _ in range(30):
            wp.launch(fill, dim=N, inputs=[arr], device="cuda:0"); wp.synchronize_device("cuda:0")
            t0 = time.perf_counter()
            tex_update_from_pb(tp, GPU_DATA_FLOAT, pb, 0, 0, 0, W, H, 1)
            gpu_finish()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        stage("timing_update_plus_finish_ms", median=ts[len(ts) // 2], min=ts[0], max=ts[-1])
    except Exception:
        stage("warp_error", tb=traceback.format_exc())

    # Shader sampling check: draw texture into offscreen with a gpu-module shader, read back.
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.sampler(0, "FLOAT_2D", "pos_tex")
        info.vertex_in(0, "VEC2", "p")
        info.fragment_out(0, "VEC4", "color")
        info.vertex_source("void main(){ gl_Position = vec4(p, 0.0, 1.0); }")
        info.fragment_source("void main(){ color = texelFetch(pos_tex, ivec2(gl_FragCoord.xy), 0); }")
        sh = gpu.shader.create_from_info(info)
        from gpu_extras.batch import batch_for_shader
        batch = batch_for_shader(sh, "TRIS", {"p": ((-1, -1), (3, -1), (-1, 3))})
        off = gpu.types.GPUOffScreen(256, 64, format="RGBA32F")
        with off.bind():
            sh.bind(); sh.uniform_sampler("pos_tex", tex); batch.draw(sh)
        px = off.texture_color.read(); px.dimensions = 256 * 64 * 4
        bad = [x for x in range(256) if (px[x * 4], px[x * 4 + 1]) != (float(x) * 0.5, 3.0)]
        stage("verify_shader_sample", mismatches=len(bad), first_bad=bad[:5])
        off.free()
    except Exception:
        stage("shader_error", tb=traceback.format_exc())

    ck(cu.cudaFree(dptr), "cudaFree")
    ck(cu.cudaDestroyExternalMemory(ext), "cudaDestroyExternalMemory")
    pb_free(pb)
    stage("cleanup_ok")


_done = []


def draw_cb():
    if _done:
        return
    _done.append(1)
    try:
        run()
    except Exception:
        stage("error", tb=traceback.format_exc())


def finish():
    if not _done:
        return 1.0
    bpy.ops.wm.quit_blender()
    return None


def start():
    bpy.types.SpaceView3D.draw_handler_add(draw_cb, (), "WINDOW", "POST_VIEW")
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            area.tag_redraw()
    bpy.app.timers.register(finish, first_interval=3.0)


stage("loaded")
bpy.app.timers.register(start, first_interval=2.0)
