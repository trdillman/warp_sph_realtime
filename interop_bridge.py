import ctypes
import os
import sys
import gpu
import warp as wp

GL_ARRAY_BUFFER = 0x8892
GL_DYNAMIC_DRAW = 0x88E8
GL_FLOAT = 0x1406
GL_POINTS = 0x0000


class _GLInterop:
    _initialized = False
    gl = None
    _get_proc_addr = None
    cudart = None

    glGenVertexArrays = None
    glBindVertexArray = None
    glDeleteVertexArrays = None
    glGenBuffers = None
    glBindBuffer = None
    glBufferData = None
    glDeleteBuffers = None
    glEnableVertexAttribArray = None
    glVertexAttribPointer = None
    glDrawArrays = None

    @classmethod
    def _find_and_load_cudart(cls):
        if sys.platform == "win32":
            candidates = ["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64_11.dll"]
            for name in candidates:
                try:
                    return ctypes.WinDLL(name)
                except OSError:
                    pass

            search_dirs = []
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                search_dirs.append(os.path.join(cuda_path, "bin"))
            for k, v in os.environ.items():
                if k.startswith("CUDA_PATH_"):
                    search_dirs.append(os.path.join(v, "bin"))
            for p in os.environ.get("PATH", "").split(";"):
                if p and os.path.isdir(p):
                    search_dirs.append(p)

            for d in search_dirs:
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
                for name in candidates:
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        try:
                            return ctypes.WinDLL(full)
                        except OSError:
                            pass
            raise RuntimeError("Could not find cudart64_12.dll or cudart64_13.dll in PATH or CUDA_PATH.")
        else:
            candidates = ["libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]
            for so in candidates:
                try:
                    return ctypes.CDLL(so)
                except OSError:
                    continue
            raise RuntimeError("Could not find libcudart.so in LD_LIBRARY_PATH.")

    @classmethod
    def ensure_initialized(cls):
        if cls._initialized:
            return

        cls.cudart = cls._find_and_load_cudart()

        if sys.platform == "win32":
            cls.gl = ctypes.windll.opengl32
            cls._get_proc_addr = cls.gl.wglGetProcAddress
            cls._get_proc_addr.restype = ctypes.c_void_p
            cls._get_proc_addr.argtypes = [ctypes.c_char_p]
            fn_type = ctypes.WINFUNCTYPE
        else:
            cls.gl = ctypes.CDLL("libGL.so.1")
            cls._get_proc_addr = cls.gl.glXGetProcAddress
            cls._get_proc_addr.restype = ctypes.c_void_p
            cls._get_proc_addr.argtypes = [ctypes.c_char_p]
            fn_type = ctypes.CFUNCTYPE

        def _gl_fn(name, restype, argtypes):
            fn = getattr(cls.gl, name, None)
            if fn is None:
                addr = cls._get_proc_addr(name.encode("ascii"))
                if not addr:
                    raise RuntimeError(
                        f"OpenGL symbol '{name}' not found. "
                        "Ensure an active OpenGL Core Profile context is current."
                    )
                return fn_type(restype, *argtypes)(addr)
            fn.restype = restype
            fn.argtypes = argtypes
            return fn

        cls.glGenVertexArrays = _gl_fn("glGenVertexArrays", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glBindVertexArray = _gl_fn("glBindVertexArray", None, [ctypes.c_uint])
        cls.glDeleteVertexArrays = _gl_fn("glDeleteVertexArrays", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glGenBuffers = _gl_fn("glGenBuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glBindBuffer = _gl_fn("glBindBuffer", None, [ctypes.c_uint, ctypes.c_uint])
        cls.glBufferData = _gl_fn("glBufferData", None, [ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint])
        cls.glDeleteBuffers = _gl_fn("glDeleteBuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glEnableVertexAttribArray = _gl_fn("glEnableVertexAttribArray", None, [ctypes.c_uint])
        cls.glVertexAttribPointer = _gl_fn("glVertexAttribPointer", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_int, ctypes.c_void_p])
        cls.glDrawArrays = _gl_fn("glDrawArrays", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int])

        cls._initialized = True


class ZeroCopyGLBuffer52:
    def __init__(self, num_particles: int):
        if gpu.platform.backend_type_get() != 'OPENGL':
            raise RuntimeError(
                "Blender is currently running on Vulkan/Metal. "
                "Switch to OpenGL in Preferences > System > Display Graphics for GL-CUDA interop."
            )

        _GLInterop.ensure_initialized()

        self.num_particles = num_particles
        self.byte_size = num_particles * 3 * 4  # float3 SoA position buffer (12B/particle)
        self.vao = ctypes.c_uint(0)
        self.vbo = ctypes.c_uint(0)
        self.cuda_resource = ctypes.c_void_p(0)

        # 1. Create VAO + VBO in Blender 5.2 Core Profile context
        _GLInterop.glGenVertexArrays(1, ctypes.byref(self.vao))
        _GLInterop.glGenBuffers(1, ctypes.byref(self.vbo))

        _GLInterop.glBindVertexArray(self.vao.value)
        _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, self.vbo.value)
        _GLInterop.glBufferData(GL_ARRAY_BUFFER, self.byte_size, None, GL_DYNAMIC_DRAW)
        _GLInterop.glEnableVertexAttribArray(0)  # Attribute 0 = 'pos' in Blender 5.2 UNIFORM_COLOR
        _GLInterop.glVertexAttribPointer(0, 3, GL_FLOAT, 0, 12, ctypes.c_void_p(0))
        _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, 0)
        _GLInterop.glBindVertexArray(0)

        # 2. Register VBO with CUDA Runtime
        err = _GLInterop.cudart.cudaGraphicsGLRegisterBuffer(
            ctypes.byref(self.cuda_resource), self.vbo.value, 0
        )
        if err != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed with CUDA error {err}")

    def map_to_warp(self) -> wp.array:
        """Locks VBO for CUDA and wraps raw device pointer via Warp 1.16+ wp.from_ptr."""
        _GLInterop.cudart.cudaGraphicsMapResources(1, ctypes.byref(self.cuda_resource), ctypes.c_void_p(0))
        dev_ptr = ctypes.c_void_p(0)
        size = ctypes.c_size_t(0)
        _GLInterop.cudart.cudaGraphicsResourceGetMappedPointer(
            ctypes.byref(dev_ptr), ctypes.byref(size), self.cuda_resource
        )
        return wp.from_ptr(dev_ptr.value, dtype=wp.vec3, shape=(self.num_particles,), device="cuda:0")

    def unmap(self):
        _GLInterop.cudart.cudaGraphicsUnmapResources(1, ctypes.byref(self.cuda_resource), ctypes.c_void_p(0))

    def draw_points(self):
        _GLInterop.glBindVertexArray(self.vao.value)
        _GLInterop.glDrawArrays(GL_POINTS, 0, self.num_particles)
        _GLInterop.glBindVertexArray(0)

    def destroy(self):
        if self.cuda_resource.value:
            _GLInterop.cudart.cudaGraphicsUnregisterResource(self.cuda_resource)
            self.cuda_resource = ctypes.c_void_p(0)
        if self.vbo.value:
            _GLInterop.glDeleteBuffers(1, ctypes.byref(self.vbo))
            self.vbo = ctypes.c_uint(0)
        if self.vao.value:
            _GLInterop.glDeleteVertexArrays(1, ctypes.byref(self.vao))
            self.vao = ctypes.c_uint(0)
