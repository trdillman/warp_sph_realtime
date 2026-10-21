"""Probe: is an OpenGL context current for Python code when Blender runs on Vulkan?

Run: blender --factory-startup --gpu-backend {vulkan|opengl} --python this.py -- <out.json>
"""
import ctypes, json, sys
import bpy, gpu

out_path = sys.argv[sys.argv.index("--") + 1]
k32 = ctypes.WinDLL("kernel32")
k32.GetModuleHandleW.restype = ctypes.c_void_p
k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]

result = {
    "blender": bpy.app.version_string,
    "backend_type_get": gpu.platform.backend_type_get(),
    "pref_gpu_backend": getattr(bpy.context.preferences.system, "gpu_backend", None),
    "pref_gpu_backend_items": [i.identifier for i in bpy.types.PreferencesSystem.bl_rna.properties["gpu_backend"].enum_items],
    "gpu_vendor": gpu.platform.vendor_get(),
    "gpu_renderer": gpu.platform.renderer_get(),
    "gpu_version": gpu.platform.version_get(),
    "opengl32_loaded_at_start": bool(k32.GetModuleHandleW("opengl32.dll")),
    "contexts": {},
}


def gl_state(label):
    loaded = bool(k32.GetModuleHandleW("opengl32.dll"))
    entry = {"opengl32_loaded": loaded}
    gl = ctypes.WinDLL("opengl32")  # loads it if absent, like the add-on does
    gl.wglGetCurrentContext.restype = ctypes.c_void_p
    ctx = gl.wglGetCurrentContext()
    entry["wglGetCurrentContext"] = hex(ctx) if ctx else None
    if ctx:
        gl.glGetString.restype = ctypes.c_char_p
        gl.glGetString.argtypes = [ctypes.c_uint]
        v = gl.glGetString(0x1F02)
        entry["GL_VERSION"] = v.decode() if v else None
    result["contexts"][label] = entry


def draw_cb():
    if "draw_handler" not in result["contexts"]:
        gl_state("draw_handler")


def finish():
    gl_state("timer")
    try:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    finally:
        bpy.ops.wm.quit_blender()
    return None


def start():
    gl_state("startup_timer")
    bpy.types.SpaceView3D.draw_handler_add(draw_cb, (), "WINDOW", "POST_VIEW")
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            area.tag_redraw()
    bpy.app.timers.register(finish, first_interval=3.0)
    return None


bpy.app.timers.register(start, first_interval=2.0)
