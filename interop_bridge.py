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
    _get_current_context = None
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
            cls._get_current_context = cls.gl.wglGetCurrentContext
            fn_type = ctypes.WINFUNCTYPE
        else:
            cls.gl = ctypes.CDLL("libGL.so.1")
            cls._get_proc_addr = cls.gl.glXGetProcAddress
            cls._get_proc_addr.restype = ctypes.c_void_p
            cls._get_proc_addr.argtypes = [ctypes.c_char_p]
            cls._get_current_context = cls.gl.glXGetCurrentContext
            fn_type = ctypes.CFUNCTYPE
        cls._get_current_context.restype = ctypes.c_void_p
        cls._get_current_context.argtypes = []

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

    @classmethod
    def current_context(cls) -> int:
        return cls._get_current_context() or 0


def wrap_device_vec3(ptr: int, num_particles: int, device: str = "cuda:0") -> wp.array:
    """Wraps an externally owned CUDA pointer as a non-owning vec3 Warp array (no copy)."""
    return wp.array(
        ptr=ptr,
        dtype=wp.vec3,
        shape=(num_particles,),
        capacity=num_particles * 3 * 4,
        device=device,
    )


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
        # VAOs are per-GL-context (VBOs are shared), and operators run in a different context
        # than viewport draw callbacks, so VAOs are created lazily per drawing context.
        self.vaos = {}
        self.vbo = ctypes.c_uint(0)
        self.cuda_resource = ctypes.c_void_p(0)

        # 1. Create the shared VBO
        _GLInterop.glGenBuffers(1, ctypes.byref(self.vbo))
        _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, self.vbo.value)
        _GLInterop.glBufferData(GL_ARRAY_BUFFER, self.byte_size, None, GL_DYNAMIC_DRAW)
        _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, 0)

        # 2. Register VBO with CUDA Runtime
        err = _GLInterop.cudart.cudaGraphicsGLRegisterBuffer(
            ctypes.byref(self.cuda_resource), self.vbo.value, 0
        )
        if err != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed with CUDA error {err}")

    def map_to_warp(self) -> wp.array:
        """Locks VBO for CUDA and wraps raw device pointer via Warp 1.16+ wp.from_ptr."""
        err = _GLInterop.cudart.cudaGraphicsMapResources(1, ctypes.byref(self.cuda_resource), ctypes.c_void_p(0))
        if err != 0:
            raise RuntimeError(f"cudaGraphicsMapResources failed with CUDA error {err}")
        dev_ptr = ctypes.c_void_p(0)
        size = ctypes.c_size_t(0)
        err = _GLInterop.cudart.cudaGraphicsResourceGetMappedPointer(
            ctypes.byref(dev_ptr), ctypes.byref(size), self.cuda_resource
        )
        if err != 0 or not dev_ptr.value or size.value < self.byte_size:
            self.unmap()
            raise RuntimeError(
                f"cudaGraphicsResourceGetMappedPointer failed (CUDA error {err}, "
                f"{size.value} of {self.byte_size} bytes mapped)"
            )
        return wrap_device_vec3(dev_ptr.value, self.num_particles)

    def unmap(self):
        err = _GLInterop.cudart.cudaGraphicsUnmapResources(1, ctypes.byref(self.cuda_resource), ctypes.c_void_p(0))
        if err != 0:
            raise RuntimeError(f"cudaGraphicsUnmapResources failed with CUDA error {err}")

    def _vao_for_current_context(self) -> int:
        ctx = _GLInterop.current_context()
        vao = self.vaos.get(ctx)
        if vao is None:
            vao = ctypes.c_uint(0)
            _GLInterop.glGenVertexArrays(1, ctypes.byref(vao))
            _GLInterop.glBindVertexArray(vao.value)
            _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, self.vbo.value)
            _GLInterop.glEnableVertexAttribArray(0)  # Attribute 0 = 'pos' in UNIFORM_COLOR
            _GLInterop.glVertexAttribPointer(0, 3, GL_FLOAT, 0, 12, ctypes.c_void_p(0))
            _GLInterop.glBindVertexArray(0)
            _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.vaos[ctx] = vao
        return vao.value

    def draw_points(self):
        _GLInterop.glBindVertexArray(self._vao_for_current_context())
        _GLInterop.glDrawArrays(GL_POINTS, 0, self.num_particles)
        _GLInterop.glBindVertexArray(0)

    def destroy(self):
        if self.cuda_resource.value:
            _GLInterop.cudart.cudaGraphicsUnregisterResource(self.cuda_resource)
            self.cuda_resource = ctypes.c_void_p(0)
        if self.vbo.value:
            _GLInterop.glDeleteBuffers(1, ctypes.byref(self.vbo))
            self.vbo = ctypes.c_uint(0)
        # A VAO can only be deleted from its own context; others are freed with that context.
        vao = self.vaos.pop(_GLInterop.current_context(), None)
        if vao is not None:
            _GLInterop.glDeleteVertexArrays(1, ctypes.byref(vao))
        self.vaos.clear()
