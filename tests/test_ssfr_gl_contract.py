"""Driver-free fault controls; real framebuffer checks live in the GUI probe."""
import ctypes
import unittest
from unittest.mock import MagicMock, Mock, patch
from types import SimpleNamespace

from test_ssfr_renderer import _load_viewport_draw, _StatefulGL

v = _load_viewport_draw()


class DefaultDepthFormat(unittest.TestCase):
    def detect(self, bits, floating=False, stencil=0):
        gl = v._ig._GLInterop
        def integer(pname, output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 0
        def attachment(target, which, pname, output):
            values = {0x8CD0: 0x8218, 0x8216: bits, 0x8211: 0x1406 if floating else 0x8C17,
                      0x8217: stencil}
            # OBJECT_NAME is invalid for a default framebuffer; do not query it.
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = values[pname]
        with patch.multiple(gl, glGetIntegerv=integer,
                            glGetFramebufferAttachmentParameteriv=attachment):
            return gl.framebuffer_depth_format()

    def test_default_without_supported_depth_fails_before_allocation(self):
        for bits in (0, 8):
            with self.subTest(bits=bits), self.assertRaisesRegex(RuntimeError, 'unsupported.*depth'):
                self.detect(bits)

    def test_supported_default_depth_formats_are_preserved(self):
        for bits, fmt in ((16, 0x81A5), (24, 0x81A6), (32, 0x81A7)):
            self.assertEqual(self.detect(bits), fmt)
        self.assertEqual(self.detect(32, floating=True), 0x8CAC)
        self.assertEqual(self.detect(24, stencil=8), 0x88F0)
        self.assertEqual(self.detect(32, floating=True, stencil=8), 0x8CAD)


class ShaderCleanup(unittest.TestCase):
    def test_partial_renderer_initialization_releases_programs_and_vao(self):
        gl = _StatefulGL()
        gl.compile_program = Mock(side_effect=[21, 22, RuntimeError('compile error')])
        gl.glDeleteProgram = Mock()
        gl.glDeleteVertexArrays = Mock()
        renderer = v.SSFRViewportRenderer()
        with patch.object(v, '_GLInterop', gl):
            with self.assertRaisesRegex(RuntimeError, 'compile error'):
                renderer._get_or_create_ctx_data(64, 48)
        self.assertEqual(gl.glDeleteProgram.call_args_list,
                         [unittest.mock.call(21), unittest.mock.call(22)])
        gl.glDeleteVertexArrays.assert_called_once()
        self.assertEqual(renderer._ctx_resources, {})

    def test_failed_fragment_releases_both_shaders(self):
        gl = v._GLInterop
        deleted = []
        def status(shader, pname, output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = int(shader == 11)
        with patch.multiple(gl, ensure_initialized=Mock(), glCreateShader=Mock(side_effect=[11, 12]),
                            glShaderSource=Mock(), glCompileShader=Mock(), glGetShaderiv=status,
                            glGetShaderInfoLog=Mock(), glDeleteShader=deleted.append):
            with self.assertRaisesRegex(RuntimeError, "compile error"):
                gl.compile_program("vertex", "broken fragment")
        self.assertCountEqual(deleted, [11, 12])


class PipelineGL(_StatefulGL):
    """Track actual mutations and enforce the multisample scaling restriction."""
    def __init__(self):
        super().__init__()
        self.ints = {0x0BA2: [9, 7, 65, 49], 0x0B72: [0], 0x0B74: [0x0201],
                     0x80C9: [1], 0x80C8: [0], 0x80CB: [0x0302], 0x80CA: [1],
                     0x8009: [0x800A], 0x883D: [0x800B], 0x8B8D: [88],
                     0x85B5: [89], 0x84E0: [0x84C5], 0x0C23: [0, 1, 0, 1], 0x8894: [71]}
        self.floats = {0x0C22: [.1, .2, .3, .4], 0x0B73: [0.7]}
        self.enabled = {0x0BE2, 0x0C11}
        self.textures = {0x84C0+i: 50+i for i in range(6)}
        self.blit_calls = []
        self.draw_calls = 0
        self.error = 0
        self.fail_blit = False
        self.framebuffer_depth_format = Mock(return_value=0x88F0)
        self.glBindTexture = lambda target, tex: self.textures.__setitem__(self.ints[0x84E0][0], tex)
    def __getattr__(self, name):
        if name.startswith('gl'):
            return Mock(return_value=0)
        raise AttributeError(name)
    def glGetIntegerv(self, pname, output):
        if pname in (self.GL_DRAW_FRAMEBUFFER_BINDING, self.GL_READ_FRAMEBUFFER_BINDING):
            return super().glGetIntegerv(pname, output)
        values = [self.textures[self.ints[0x84E0][0]]] if pname == 0x8069 else self.ints[pname]
        ptr = ctypes.cast(output, ctypes.POINTER(ctypes.c_int))
        for i, value in enumerate(values): ptr[i] = value
    def glGetFloatv(self, pname, output):
        ptr = ctypes.cast(output, ctypes.POINTER(ctypes.c_float))
        for i, value in enumerate(self.floats[pname]): ptr[i] = value
    def glIsEnabled(self, cap): return cap in self.enabled
    def glEnable(self, cap): self.enabled.add(cap)
    def glDisable(self, cap): self.enabled.discard(cap)
    def glDepthMask(self, value): self.ints[0x0B72] = [value]
    def glDepthFunc(self, value): self.ints[0x0B74] = [value]
    def glUseProgram(self, value): self.ints[0x8B8D] = [value]
    def glBindVertexArray(self, value): self.ints[0x85B5] = [value]
    def glBindBuffer(self, target, value): self.ints[0x8894] = [value]
    def glActiveTexture(self, value): self.ints[0x84E0] = [value]
    def glViewport(self, *values): self.ints[0x0BA2] = list(values)
    def glClearColor(self, *values): self.floats[0x0C22] = list(values)
    def glClearDepth(self, value): self.floats[0x0B73] = [value]
    def glColorMask(self, *values): self.ints[0x0C23] = list(values)
    def glBlendFunc(self, src, dst): self.glBlendFuncSeparate(src, dst, src, dst)
    def glBlendFuncSeparate(self, *values):
        for key, val in zip((0x80C9, 0x80C8, 0x80CB, 0x80CA), values): self.ints[key] = [val]
    def glBlendEquationSeparate(self, rgb, alpha):
        self.ints[0x8009], self.ints[0x883D] = [rgb], [alpha]
    def glGetError(self):
        value, self.error = self.error, 0
        return value
    def glBlitFramebuffer(self, *args):
        self.blit_calls.append((self.read_binding, self.draw_binding, args, self.draw_calls))
        if self.fail_blit: self.error = 0x0502
        if self.read_binding == 100 and (args[2]-args[0], args[3]-args[1]) != (args[6]-args[4], args[7]-args[5]):
            self.error = 0x0502
    def glDrawArrays(self, *args): self.draw_calls += 1


class DrawContracts(unittest.TestCase):
    def setUp(self):
        self.gl = PipelineGL()
        self.gl.framebuffers.update((100, 101))
        self.gl.draw_binding, self.gl.read_binding = 100, 101
        self.renderer = v.SSFRViewportRenderer()
        self.scene = SimpleNamespace(warp_sph_particle_color=(.1,.3,.8,1), warp_sph_depth_test=True)
        self.buf = SimpleNamespace(bind_vao=lambda:self.gl.glBindVertexArray(90),
                                   unbind_vao=lambda:self.gl.glBindVertexArray(0), num_particles=5)
        matrix = [[int(i==j) for j in range(4)] for i in range(4)]
        self.patches = [patch.object(v, '_GLInterop', self.gl),
                        patch.object(v.gpu, 'matrix', SimpleNamespace(get_model_view_matrix=lambda:matrix,
                                     get_projection_matrix=lambda:matrix), create=True)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
    def draw(self): self.renderer.draw_ssfr_liquid(self.buf, self.scene, 9, 7, 65, 49)
    def snapshot(self):
        return repr((self.gl.ints, self.gl.enabled, self.gl.textures, self.gl.draw_binding, self.gl.read_binding))
    def test_liquid_restores_nondefault_state_including_separate_alpha_blend(self):
        before = self.snapshot()
        self.draw()
        self.assertEqual(self.snapshot(), before)
        self.assertAlmostEqual(self.gl.floats[0x0C22][0], .1, places=6)
    def test_exception_restores_state(self):
        before = self.snapshot()
        with patch.object(self.gl, 'glDrawArrays', side_effect=RuntimeError('draw fault')):
            with self.assertRaisesRegex(RuntimeError, 'draw fault'): self.draw()
        self.assertEqual(self.snapshot(), before)
    def test_raw_points_draw_failure_restores_gl_state(self):
        buffer = v._ig.ZeroCopyGLBuffer52.__new__(v._ig.ZeroCopyGLBuffer52)
        buffer.vbo, buffer.vaos, buffer.num_particles = ctypes.c_uint(72), {}, 5
        shader = Mock()
        scene = SimpleNamespace(warp_sph_render_mode='POINTS', warp_sph_show_particles=True, warp_sph_depth_test=False,
                                warp_sph_particle_color=(.1, .3, .8, .5), warp_sph_point_size=3.0)
        before = self.snapshot()
        with patch.object(v._ig, '_GLInterop', self.gl), \
                patch.object(v.bpy, 'context', SimpleNamespace(scene=scene), create=True), \
                patch.object(v.gpu, 'shader', SimpleNamespace(from_builtin=lambda name: shader), create=True), \
                patch.object(v.gpu, 'matrix', SimpleNamespace(get_model_view_matrix=MagicMock,
                                                              get_projection_matrix=MagicMock), create=True), \
                patch.object(self.gl, 'glDrawArrays', side_effect=RuntimeError('draw fault')):
            with self.assertRaisesRegex(RuntimeError, 'draw fault'):
                v.draw_particles_viewport(buffer, True)
        self.assertEqual(self.snapshot(), before)

    def test_lazy_real_vao_creation_restores_incoming_array_buffer(self):
        buffer = v._ig.ZeroCopyGLBuffer52.__new__(v._ig.ZeroCopyGLBuffer52)
        buffer.vbo, buffer.vaos, buffer.num_particles = ctypes.c_uint(72), {}, 5
        self.buf = buffer
        before = self.snapshot()
        with patch.object(v._ig, '_GLInterop', self.gl):
            self.draw()
        self.assertEqual(self.snapshot(), before)
    def test_multisample_source_is_resolved_before_scaling_scene_only_depth(self):
        self.draw()
        data = next(iter(self.renderer._ctx_resources.values()))
        calls = [c for c in self.gl.blit_calls if c[2][8] & v.GL_DEPTH_BUFFER_BIT]
        half = [c for c in calls if c[1] == data['fbo_thick'].value]
        self.assertEqual(len(half), 1)
        self.assertNotEqual(half[0][0], 100, 'cannot scale a multisampled source')
        self.assertEqual(half[0][3], 0, 'must copy scene depth before particle writes')
        self.assertEqual(self.gl.glGetError(), 0)
    def test_matches_blender_float_depth_stencil_format_and_reallocates_on_change(self):
        self.draw()
        self.gl.glTexImage2D.reset_mock()
        self.gl.framebuffer_depth_format.return_value = 0x8CAD
        self.draw()
        formats = [call.args[2] for call in self.gl.glTexImage2D.call_args_list]
        self.assertEqual(formats.count(0x8CAD), 3,
                         'scene resolve, particle depth and thickness must match viewport format')
    def test_failed_depth_transfer_rejects_liquid_instead_of_drawing_unoccluded(self):
        self.gl.fail_blit = True
        with self.assertRaisesRegex(RuntimeError, 'depth'):
            self.draw()
        self.assertEqual(self.gl.draw_calls, 0)


if __name__ == '__main__': unittest.main()


class MappedVboScope(unittest.TestCase):
    """P0.4: the legacy VBO is unmapped even when the solver step raises inside the mapping."""

    def buffer(self, unmap_error=0):
        buffer = v._ig.ZeroCopyGLBuffer52.__new__(v._ig.ZeroCopyGLBuffer52)
        buffer.cuda_resource, buffer.byte_size, buffer.num_particles = ctypes.c_void_p(1), 12, 1
        cudart = Mock()
        cudart.cudaGraphicsMapResources.return_value = 0
        cudart.cudaGraphicsUnmapResources.return_value = unmap_error

        def mapped_pointer(ptr, size, resource):
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 12
            return 0
        cudart.cudaGraphicsResourceGetMappedPointer.side_effect = mapped_pointer
        return buffer, cudart

    def test_step_failure_still_unmaps_and_propagates(self):
        buffer, cudart = self.buffer()
        with patch.object(v._ig._GLInterop, 'cudart', cudart, create=True), \
                patch.object(v._ig, 'wrap_device_vec3', lambda ptr, n: ('array', ptr, n)):
            with self.assertRaisesRegex(RuntimeError, 'step fault'):
                with buffer.mapped() as positions:
                    self.assertEqual(positions, ('array', 0x1000, 1))
                    raise RuntimeError('step fault')
        cudart.cudaGraphicsUnmapResources.assert_called_once()

    def test_unmap_failure_after_step_failure_keeps_the_original_error(self):
        buffer, cudart = self.buffer(unmap_error=700)  # sticky CUDA error: unmap fails too
        with patch.object(v._ig._GLInterop, 'cudart', cudart, create=True), \
                patch.object(v._ig, 'wrap_device_vec3', lambda ptr, n: None):
            with self.assertRaisesRegex(RuntimeError, 'step fault') as caught:
                with buffer.mapped():
                    raise RuntimeError('step fault')
        self.assertIn('700', str(caught.exception.__notes__))

    def test_unmap_failure_on_success_is_raised(self):
        buffer, cudart = self.buffer(unmap_error=700)
        with patch.object(v._ig._GLInterop, 'cudart', cudart, create=True), \
                patch.object(v._ig, 'wrap_device_vec3', lambda ptr, n: None):
            with self.assertRaisesRegex(RuntimeError, 'cudaGraphicsUnmapResources failed with CUDA error 700'):
                with buffer.mapped():
                    pass
