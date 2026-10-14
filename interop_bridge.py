import ctypes
import sys
import gpu
import warp as wp

def _load_gl_and_cudart():
    if sys.platform == "win32":
        gl = ctypes.windll.opengl32
        wglGetProcAddress = gl.wglGetProcAddress
        wglGetProcAddress.restype = ctypes.c_void_p
        wglGetProcAddress.argtypes = [ctypes.c_char_p]
        cudart = None
        for dll in ("cudart64_13.dll", "cudart64_12.dll"):
            try:
                cudart = ctypes.WinDLL(dll)
                break
            except OSError:
                continue
        if cudart is None:
            raise RuntimeError("Could not find cudart64_12.dll or cudart64_13.dll in PATH.")
        return gl, wglGetProcAddress, cudart
    else:
        gl = ctypes.CDLL("libGL.so.1")
        glXGetProcAddress = gl.glXGetProcAddress
        glXGetProcAddress.restype = ctypes.c_void_p
        glXGetProcAddress.argtypes = [ctypes.c_char_p]
        cudart = None
        for so in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            try:
                cudart = ctypes.CDLL(so)
                break
            except OSError:
                continue
        if cudart is None:
            raise RuntimeError("Could not find libcudart.so in LD_LIBRARY_PATH.")
        return gl, glXGetProcAddress, cudart

gl, _get_proc_addr, cudart = _load_gl_and_cudart()

def _gl_fn(name, restype, argtypes):
    fn = getattr(gl, name, None)
    if fn is None:
        addr = _get_proc_addr(name.encode("ascii"))
        if not addr:
            raise RuntimeError(f"OpenGL symbol not found: {name}")
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn

GL_ARRAY_BUFFER = 0x8892
GL_DYNAMIC_DRAW = 0x88E8
GL_FLOAT = 0x1406
GL_POINTS = 0x0000

glGenVertexArrays = _gl_fn("glGenVertexArrays", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glBindVertexArray = _gl_fn("glBindVertexArray", None, [ctypes.c_uint])
glDeleteVertexArrays = _gl_fn("glDeleteVertexArrays", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glGenBuffers = _gl_fn("glGenBuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glBindBuffer = _gl_fn("glBindBuffer", None, [ctypes.c_uint, ctypes.c_uint])
glBufferData = _gl_fn("glBufferData", None, [ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint])
glDeleteBuffers = _gl_fn("glDeleteBuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glEnableVertexAttribArray = _gl_fn("glEnableVertexAttribArray", None, [ctypes.c_uint])
glVertexAttribPointer = _gl_fn("glVertexAttribPointer", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_int, ctypes.c_void_p])
glDrawArrays = _gl_fn("glDrawArrays", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int])


class ZeroCopyGLBuffer52:
    def __init__(self, num_particles: int):
        if gpu.platform.backend_type_get() != 'OPENGL':
            raise RuntimeError(
                "Blender is currently running on Vulkan/Metal. "
                "Switch to OpenGL in Preferences > System > Display Graphics for GL-CUDA interop."
            )

        self.num_particles = num_particles
        self.byte_size = num_particles * 3 * 4  # float3 SoA position buffer (12B/particle)
        self.vao = ctypes.c_uint(0)
        self.vbo = ctypes.c_uint(0)
        self.cuda_resource = ctypes.c_void_p(0)

        # 1. Create VAO + VBO in Blender 5.2 Core Profile context
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glGenBuffers(1, ctypes.byref(self.vbo))

        glBindVertexArray(self.vao.value)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo.value)
        glBufferData(GL_ARRAY_BUFFER, self.byte_size, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)  # Attribute 0 = 'pos' in Blender 5.2 UNIFORM_COLOR
        glVertexAttribPointer(0, 3, GL_FLOAT, 0, 12, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # 2. Register VBO with CUDA Runtime
        err = cudart.cudaGraphicsGLRegisterBuffer(
            ctypes.byref(self.cuda_resource), self.vbo.value, 0
        )
        if err != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed with CUDA error {err}")

    def map_to_warp(self) -> wp.array:
        """Locks VBO for CUDA and wraps raw device pointer via Warp 1.16+ wp.from_ptr."""
        cudart.cudaGraphicsMapResources(1, ctypes.byref(self.cuda_resource), ctypes.c_void_p(0))
        dev_ptr = ctypes.c_void_p(0)
        size = ctypes.c_size_t(0)
        cudart.cudaGraphicsResourceGetMappedPointer(
            ctypes.byref(dev_ptr), ctypes.byref(size), self.cuda_resource
        )
        return wp.from_ptr(dev_ptr.value, dtype=wp.vec3, shape=(self.num_particles,), device="cuda:0")

    def unmap(self):
        cudart.cudaGraphicsUnmapResources(1, ctypes.byref(self.cuda_resource), ctypes.c_void_p(0))

    def draw_points(self):
        glBindVertexArray(self.vao.value)
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glBindVertexArray(0)

    def destroy(self):
        if self.cuda_resource.value:
            cudart.cudaGraphicsUnregisterResource(self.cuda_resource)
            self.cuda_resource = ctypes.c_void_p(0)
        if self.vbo.value:
            glDeleteBuffers(1, ctypes.byref(self.vbo))
            self.vbo = ctypes.c_uint(0)
        if self.vao.value:
            glDeleteVertexArrays(1, ctypes.byref(self.vao))
            self.vao = ctypes.c_uint(0)
