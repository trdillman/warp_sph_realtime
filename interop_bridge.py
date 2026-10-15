import ctypes
import os
import sys
import gpu
import warp as wp

GL_ARRAY_BUFFER = 0x8892
GL_ARRAY_BUFFER_BINDING = 0x8894
GL_DYNAMIC_DRAW = 0x88E8
GL_FLOAT = 0x1406
GL_UNSIGNED_BYTE = 0x1401
GL_POINTS = 0x0000
GL_TRIANGLES = 0x0004
GL_PROGRAM_POINT_SIZE = 0x8642
GL_DEPTH_TEST = 0x0B71
GL_LEQUAL = 0x0203
GL_BLEND = 0x0BE2
GL_ONE = 0x0001
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303

# Framebuffer & Texture Constants for Offscreen SSFR Passes
GL_FRAMEBUFFER = 0x8D40
GL_READ_FRAMEBUFFER = 0x8CA8
GL_DRAW_FRAMEBUFFER = 0x8CA9
GL_DRAW_FRAMEBUFFER_BINDING = 0x8CA6
GL_READ_FRAMEBUFFER_BINDING = 0x8CAA
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_DEPTH_ATTACHMENT = 0x8D00
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_NEAREST = 0x2600
GL_LINEAR = 0x2601
GL_CLAMP_TO_EDGE = 0x812F
GL_R32F = 0x822E
GL_R16F = 0x822D
GL_RED = 0x1903
GL_RGBA8 = 0x8058
GL_RGBA = 0x1908
GL_DEPTH_COMPONENT32F = 0x8CAC
GL_DEPTH_COMPONENT = 0x1902
GL_DEPTH24_STENCIL8 = 0x88F0
GL_DEPTH32F_STENCIL8 = 0x8CAD
GL_FLOAT_32_UNSIGNED_INT_24_8_REV = 0x8DAD
GL_DEPTH_STENCIL = 0x84F9
GL_UNSIGNED_INT_24_8 = 0x84FA
GL_DEPTH_STENCIL_ATTACHMENT = 0x821A
GL_NO_ERROR = 0
GL_INVALID_OPERATION = 0x0502
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_VIEWPORT = 0x0BA2
GL_DEPTH_WRITEMASK = 0x0B72
GL_DEPTH_FUNC = 0x0B74
GL_BLEND_SRC_RGB = 0x80C9
GL_BLEND_DST_RGB = 0x80C8
GL_BLEND_SRC_ALPHA = 0x80CB
GL_BLEND_DST_ALPHA = 0x80CA
GL_BLEND_EQUATION_RGB = 0x8009
GL_BLEND_EQUATION_ALPHA = 0x883D
GL_FUNC_ADD = 0x8006
GL_CURRENT_PROGRAM = 0x8B8D
GL_VERTEX_ARRAY_BINDING = 0x85B5
GL_ACTIVE_TEXTURE = 0x84E0
GL_TEXTURE_BINDING_2D = 0x8069
GL_COLOR_CLEAR_VALUE = 0x0C22
GL_DEPTH_CLEAR_VALUE = 0x0B73
GL_COLOR_WRITEMASK = 0x0C23
GL_SCISSOR_TEST = 0x0C11


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
    glPointSize = None
    glEnable = None
    glDisable = None
    glIsEnabled = None
    glDepthFunc = None
    glDepthMask = None
    glBlendFuncSeparate = None
    glBlendEquationSeparate = None
    glGetFramebufferAttachmentParameteriv = None
    glGetTextureLevelParameteriv = None
    glGetNamedRenderbufferParameteriv = None
    glGetFloatv = None
    glClearDepth = None
    glColorMask = None
    glBlendFunc = None
    glClearColor = None
    glClear = None
    glGetIntegerv = None
    glViewport = None
    glGetError = None

    # FBO, Blit, Texture & Shader functions for SSFR
    glGenFramebuffers = None
    glBindFramebuffer = None
    glDeleteFramebuffers = None
    glFramebufferTexture2D = None
    glBlitFramebuffer = None
    glGenTextures = None
    glBindTexture = None
    glDeleteTextures = None
    glTexImage2D = None
    glTexParameteri = None
    glActiveTexture = None
    glCreateShader = None
    glShaderSource = None
    glCompileShader = None
    glGetShaderiv = None
    glGetShaderInfoLog = None
    glDeleteShader = None
    glCreateProgram = None
    glAttachShader = None
    glLinkProgram = None
    glGetProgramiv = None
    glGetProgramInfoLog = None
    glUseProgram = None
    glDeleteProgram = None
    glGetUniformLocation = None
    glUniform1i = None
    glUniform1f = None
    glUniform2f = None
    glUniform3f = None
    glUniform4f = None
    glUniformMatrix4fv = None

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
        cls.glPointSize = _gl_fn("glPointSize", None, [ctypes.c_float])
        cls.glEnable = _gl_fn("glEnable", None, [ctypes.c_uint])
        cls.glDisable = _gl_fn("glDisable", None, [ctypes.c_uint])
        cls.glIsEnabled = _gl_fn("glIsEnabled", ctypes.c_ubyte, [ctypes.c_uint])
        cls.glDepthFunc = _gl_fn("glDepthFunc", None, [ctypes.c_uint])
        cls.glDepthMask = _gl_fn("glDepthMask", None, [ctypes.c_ubyte])
        cls.glBlendFuncSeparate = _gl_fn("glBlendFuncSeparate", None, [ctypes.c_uint] * 4)
        cls.glBlendEquationSeparate = _gl_fn("glBlendEquationSeparate", None, [ctypes.c_uint] * 2)
        cls.glGetFramebufferAttachmentParameteriv = _gl_fn("glGetFramebufferAttachmentParameteriv", None, [ctypes.c_uint] * 3 + [ctypes.POINTER(ctypes.c_int)])
        cls.glGetTextureLevelParameteriv = _gl_fn("glGetTextureLevelParameteriv", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
        cls.glGetNamedRenderbufferParameteriv = _gl_fn("glGetNamedRenderbufferParameteriv", None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
        cls.glGetFloatv = _gl_fn("glGetFloatv", None, [ctypes.c_uint, ctypes.POINTER(ctypes.c_float)])
        cls.glClearDepth = _gl_fn("glClearDepth", None, [ctypes.c_double])
        cls.glColorMask = _gl_fn("glColorMask", None, [ctypes.c_ubyte] * 4)
        cls.glBlendFunc = _gl_fn("glBlendFunc", None, [ctypes.c_uint, ctypes.c_uint])
        cls.glClearColor = _gl_fn("glClearColor", None, [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float])
        cls.glClear = _gl_fn("glClear", None, [ctypes.c_uint])
        cls.glGetIntegerv = _gl_fn("glGetIntegerv", None, [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
        cls.glViewport = _gl_fn("glViewport", None, [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int])

        cls.glGenFramebuffers = _gl_fn("glGenFramebuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glBindFramebuffer = _gl_fn("glBindFramebuffer", None, [ctypes.c_uint, ctypes.c_uint])
        cls.glDeleteFramebuffers = _gl_fn("glDeleteFramebuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glFramebufferTexture2D = _gl_fn("glFramebufferTexture2D", None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int])
        cls.glBlitFramebuffer = _gl_fn(
            "glBlitFramebuffer",
            None,
            [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint, ctypes.c_uint,
            ],
        )
        cls.glGenTextures = _gl_fn("glGenTextures", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glBindTexture = _gl_fn("glBindTexture", None, [ctypes.c_uint, ctypes.c_uint])
        cls.glDeleteTextures = _gl_fn("glDeleteTextures", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        cls.glTexImage2D = _gl_fn("glTexImage2D", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p])
        cls.glTexParameteri = _gl_fn("glTexParameteri", None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_int])
        cls.glActiveTexture = _gl_fn("glActiveTexture", None, [ctypes.c_uint])

        cls.glCreateShader = _gl_fn("glCreateShader", ctypes.c_uint, [ctypes.c_uint])
        cls.glShaderSource = _gl_fn("glShaderSource", None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)])
        cls.glCompileShader = _gl_fn("glCompileShader", None, [ctypes.c_uint])
        cls.glGetShaderiv = _gl_fn("glGetShaderiv", None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
        cls.glGetShaderInfoLog = _gl_fn("glGetShaderInfoLog", None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p])
        cls.glDeleteShader = _gl_fn("glDeleteShader", None, [ctypes.c_uint])
        cls.glCreateProgram = _gl_fn("glCreateProgram", ctypes.c_uint, [])
        cls.glAttachShader = _gl_fn("glAttachShader", None, [ctypes.c_uint, ctypes.c_uint])
        cls.glLinkProgram = _gl_fn("glLinkProgram", None, [ctypes.c_uint])
        cls.glGetProgramiv = _gl_fn("glGetProgramiv", None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
        cls.glGetProgramInfoLog = _gl_fn("glGetProgramInfoLog", None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p])
        cls.glUseProgram = _gl_fn("glUseProgram", None, [ctypes.c_uint])
        cls.glDeleteProgram = _gl_fn("glDeleteProgram", None, [ctypes.c_uint])
        cls.glGetUniformLocation = _gl_fn("glGetUniformLocation", ctypes.c_int, [ctypes.c_uint, ctypes.c_char_p])
        cls.glUniform1i = _gl_fn("glUniform1i", None, [ctypes.c_int, ctypes.c_int])
        cls.glUniform1f = _gl_fn("glUniform1f", None, [ctypes.c_int, ctypes.c_float])
        cls.glUniform2f = _gl_fn("glUniform2f", None, [ctypes.c_int, ctypes.c_float, ctypes.c_float])
        cls.glUniform3f = _gl_fn("glUniform3f", None, [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float])
        cls.glUniform4f = _gl_fn("glUniform4f", None, [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float])
        cls.glUniformMatrix4fv = _gl_fn("glUniformMatrix4fv", None, [ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_float)])
        cls.glGetError = _gl_fn("glGetError", ctypes.c_uint, [])

        cls._initialized = True

    @classmethod
    def current_context(cls) -> int:
        return cls._get_current_context() or 0

    @classmethod
    def framebuffer_depth_format(cls) -> int:
        """Inspect the actual draw target; Blender viewport depth can be D32F/S8.

        Depth blits require identical source/destination formats. DSA queries
        avoid disturbing texture/renderbuffer bindings in Blender's state cache.
        """
        binding = ctypes.c_int()
        cls.glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, ctypes.byref(binding))
        attachment = GL_DEPTH_ATTACHMENT if binding.value else 0x1801  # GL_DEPTH
        def query(pname, target_attachment=attachment):
            result = ctypes.c_int()
            cls.glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, target_attachment, pname, ctypes.byref(result))
            return result.value
        kind = query(0x8CD0)  # OBJECT_TYPE
        name = query(0x8CD1) if kind in (0x1702, 0x8D41) else 0  # OBJECT_NAME
        result = ctypes.c_int()
        if kind == 0x1702:  # GL_TEXTURE
            level = query(0x8CD2)  # TEXTURE_LEVEL
            cls.glGetTextureLevelParameteriv(name, level, 0x1003, ctypes.byref(result))
        elif kind == 0x8D41:  # GL_RENDERBUFFER
            cls.glGetNamedRenderbufferParameteriv(name, 0x8D44, ctypes.byref(result))
        elif not binding.value and kind == 0x8218:  # GL_FRAMEBUFFER_DEFAULT
            bits = query(0x8216)  # DEPTH_SIZE
            floating = query(0x8211) == GL_FLOAT  # COMPONENT_TYPE
            stencil = query(0x8217, 0x1802)  # STENCIL_SIZE, GL_STENCIL
            if stencil:
                result.value = GL_DEPTH32F_STENCIL8 if floating else GL_DEPTH24_STENCIL8
            else:
                result.value = GL_DEPTH_COMPONENT32F if floating else {16: 0x81A5, 24: 0x81A6, 32: 0x81A7}.get(bits, 0)
        if result.value not in (GL_DEPTH24_STENCIL8, GL_DEPTH32F_STENCIL8,
                                GL_DEPTH_COMPONENT32F, 0x81A5, 0x81A6, 0x81A7):
            raise RuntimeError(f"SSFR unsupported scene depth format 0x{result.value:04x}")
        return result.value

    @classmethod
    def compile_program(cls, vert_src: str, frag_src: str) -> int:
        """Compiles and links a GLSL 450 Core shader program."""
        cls.ensure_initialized()

        def _compile(stage: int, src: str) -> int:
            sh = cls.glCreateShader(stage)
            b_src = src.encode("utf-8")
            c_str = ctypes.c_char_p(b_src)
            length = ctypes.c_int(len(b_src))
            cls.glShaderSource(sh, 1, ctypes.byref(c_str), ctypes.byref(length))
            cls.glCompileShader(sh)
            status = ctypes.c_int(0)
            cls.glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(status))
            if not status.value:
                log = ctypes.create_string_buffer(2048)
                cls.glGetShaderInfoLog(sh, 2048, None, log)
                cls.glDeleteShader(sh)
                raise RuntimeError(f"GLSL compile error: {log.value.decode('utf-8', errors='replace')}")
            return sh

        vs = fs = prog = 0
        try:
            vs = _compile(GL_VERTEX_SHADER, vert_src)
            fs = _compile(GL_FRAGMENT_SHADER, frag_src)
            prog = cls.glCreateProgram()
            cls.glAttachShader(prog, vs)
            cls.glAttachShader(prog, fs)
            cls.glLinkProgram(prog)
            status = ctypes.c_int(0)
            cls.glGetProgramiv(prog, GL_LINK_STATUS, ctypes.byref(status))
            if not status.value:
                log = ctypes.create_string_buffer(2048)
                cls.glGetProgramInfoLog(prog, 2048, None, log)
                raise RuntimeError(f"GLSL link error: {log.value.decode('utf-8', errors='replace')}")
            return prog
        except Exception:
            if prog:
                cls.glDeleteProgram(prog)
            raise
        finally:
            if vs:
                cls.glDeleteShader(vs)
            if fs:
                cls.glDeleteShader(fs)



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
            _GLInterop.glEnableVertexAttribArray(0)  # Attribute 0 = 'pos' in UNIFORM_COLOR / SSFR
            _GLInterop.glVertexAttribPointer(0, 3, GL_FLOAT, 0, 12, ctypes.c_void_p(0))
            _GLInterop.glBindVertexArray(0)
            _GLInterop.glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.vaos[ctx] = vao
        return vao.value

    def bind_vao(self):
        _GLInterop.glBindVertexArray(self._vao_for_current_context())

    def unbind_vao(self):
        _GLInterop.glBindVertexArray(0)

    def draw_points(self, point_size: float = 1.0, depth_test: bool = True, blend: bool = False):
        # Raw GL draws bypass Blender's lazily-applied gpu.state, so set point size, depth and blend
        # directly and restore Blender's GL state afterwards. GL_PROGRAM_POINT_SIZE (shader-driven
        # size) must be off for glPointSize to apply.
        gl = _GLInterop
        saved = {cap: gl.glIsEnabled(cap) for cap in (GL_PROGRAM_POINT_SIZE, GL_DEPTH_TEST, GL_BLEND)}
        gl.glDisable(GL_PROGRAM_POINT_SIZE)
        gl.glPointSize(float(point_size))
        if depth_test:
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(GL_LEQUAL)
        else:
            gl.glDisable(GL_DEPTH_TEST)
        if blend:
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        else:
            gl.glDisable(GL_BLEND)

        gl.glBindVertexArray(self._vao_for_current_context())
        gl.glDrawArrays(GL_POINTS, 0, self.num_particles)
        gl.glBindVertexArray(0)

        gl.glPointSize(1.0)
        for cap, was_on in saved.items():
            (gl.glEnable if was_on else gl.glDisable)(cap)

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
