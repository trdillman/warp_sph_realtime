"""GLInterop texture adapter tests with a fake cudart/GL seam (no GPU, no bpy).

These prove call order, stream use, validation, rollback, fallback and fatal handling only;
the real native path is qualified in Blender (see evidence/gl-texture), never by these fakes.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from warp_sph_realtime.render import session as ts
from warp_sph_realtime.render import transport as tr
from warp_sph_realtime.render import transport_gl as tg

STREAM = 0x5757
STAGING_PTR = 0x4000


class FakeNative:
    def __init__(self, log):
        self.log = log
        self.opengl = True
        self.device_reason = ''
        self.texture_limit = 16384
        self.free_vram, self.total_vram = 8 << 30, 16 << 30
        self.fail = {}             # method name -> cudart code
        self.query_results = []    # scripted cudaEventQuery results, default 0
        self.unidentifiable = False
        self.corrupt_readback = False
        self.live = set()          # registered resources / events
        self.textures = []
        self._next = 100

    def _id(self):
        self._next += 1
        return self._next

    def _code(self, name):
        self.log.append(name)
        return self.fail.get(name, 0)

    def backend_is_opengl(self):
        return self.opengl

    def check_gl_device(self):
        return self.device_reason

    def max_texture_size(self):
        return self.texture_limit

    def mem_info(self):
        return self._code('mem_info'), self.free_vram, self.total_vram

    def create_texture(self, width, height):
        self.log.append('create_texture')
        texture = SimpleNamespace(size=(width, height))
        self.textures.append(texture)
        return texture

    def identify(self, texture, width, height):
        self.log.append('identify')
        return 0 if self.unidentifiable else self._id()

    def read_texture(self, texture):
        self.log.append('read_texture')
        w, h = texture.size
        data = tg.known_answer(w * h)
        return data + 1 if self.corrupt_readback else data

    def register_image(self, name):
        code = self._code('register_image')
        if code:
            return code, 0
        resource = self._id()
        self.live.add(('res', resource))
        return 0, resource

    def map(self, resource, stream):
        self.log.append(('map', resource, stream))
        return self.fail.get('map', 0)

    def mapped_array(self, resource):
        return self._code('mapped_array'), 0xA77A

    def copy_to_array(self, array, src, pitch, width_bytes, rows, stream):
        self.log.append(('copy', array, src, pitch, width_bytes, rows, stream))
        return self.fail.get('copy', 0)

    def unmap(self, resource, stream):
        self.log.append(('unmap', resource, stream))
        return self.fail.get('unmap', 0)

    def unregister(self, resource):
        self.log.append(('unregister', resource))
        self.live.discard(('res', resource))
        return self.fail.get('unregister', 0)

    def event_create(self):
        code = self._code('event_create')
        if code:
            return code, 0
        event = self._id()
        self.live.add(('event', event))
        return 0, event

    def event_record(self, event, stream):
        self.log.append(('event_record', event, stream))
        return self.fail.get('event_record', 0)

    def event_query(self, event):
        self.log.append('event_query')
        return self.query_results.pop(0) if self.query_results else 0

    def event_destroy(self, event):
        self.log.append(('event_destroy', event))
        self.live.discard(('event', event))
        return self.fail.get('event_destroy', 0)


class FakeOps:
    def __init__(self, log):
        self.log = log
        self.sync_error = None

    def alloc(self, texels):
        self.log.append('alloc')
        return SimpleNamespace(texels=texels)

    def pointer(self, array):
        return STAGING_PTR

    def stream_handle(self):
        return STREAM

    def pack(self, positions, dst, count):
        self.log.append(('pack', count))

    def fill_known(self, dst, count):
        self.log.append(('fill_known', count))

    def synchronize(self):
        self.log.append('sync')
        if self.sync_error:
            raise self.sync_error


@pytest.fixture(autouse=True)
def quarantine_isolated():
    before = len(tg._QUARANTINE)
    yield
    del tg._QUARANTINE[before:]


def make(count=5000, **native_attrs):
    log = []
    native, ops = FakeNative(log), FakeOps(log)
    for key, value in native_attrs.items():
        setattr(native, key, value)
    return tg.GLInteropTransport(native, ops, count), native, ops, log


def positions(n):
    return np.zeros((n, 3), dtype=np.float32)


def names(log):
    return [e if isinstance(e, str) else e[0] for e in log]


# -- capabilities / label -----------------------------------------------------------------
def test_capabilities_and_label():
    t, *_ = make()
    caps = t.capabilities()
    assert caps == tr.Capabilities('GL_INTEROP', True, tr.MAX_COUNT, True, '')
    session = ts.TransportSession(t, [], SimpleNamespace(free=lambda: None), 'OPENGL')
    assert session.label == 'OpenGL interop (GPU copy, no CPU readback)' == ts.GL_LABEL
    assert 'no CPU readback' in session.label and 'zero' not in session.label.lower()
    assert (t.width, t.height) == (4096, 2)


def test_not_ready_before_first_publish_and_frame_contract():
    t, native, _, _ = make()
    assert t.positions_texture() is None and t.newest_frame() == -1
    assert t.publish(positions(5000)) is True
    frame = t.positions_texture()
    assert (frame.count, frame.width, frame.height, frame.frame) == (5000, 4096, 2, 0)
    assert frame.generation == t.generation
    assert frame.texture is native.textures[-1] and t.newest_frame() == 0


# -- ordering -----------------------------------------------------------------------------
def test_publish_orders_pack_map_copy_unmap_record_on_one_stream():
    t, native, _, log = make()
    resource = next(r for kind, r in native.live if kind == 'res')
    event = next(e for kind, e in native.live if kind == 'event')
    del log[:]
    assert t.publish(positions(5000)) is True
    assert log == [
        ('pack', 5000),
        ('map', resource, STREAM),
        'mapped_array',
        ('copy', 0xA77A, STAGING_PTR, 4096 * 16, 4096 * 16, 2, STREAM),
        ('unmap', resource, STREAM),
        ('event_record', event, STREAM),
    ]


def test_backpressure_while_previous_publish_runs():
    t, native, _, log = make()
    assert t.can_publish() and t.publish(positions(5000))
    native.query_results = [tg.CUDA_NOT_READY, tg.CUDA_NOT_READY, 0]
    del log[:]
    assert t.can_publish() is False
    assert t.publish(positions(5000)) is False          # skipped: nothing packed, nothing mapped
    assert names(log) == ['event_query', 'event_query']
    assert t.can_publish() is True                       # event completed
    assert t.publish(positions(5000)) is True and t.newest_frame() == 1


# -- validation ---------------------------------------------------------------------------
@pytest.mark.parametrize('count', [-1, tr.MAX_COUNT + 1, True, 1.5])
def test_invalid_counts_raise_before_any_native_call(count):
    log = []
    with pytest.raises(ValueError):
        tg.GLInteropTransport(FakeNative(log), FakeOps(log), count)
    assert log == []


def test_wrong_position_length_is_rejected():
    t, _, _, log = make()
    del log[:]
    with pytest.raises(ValueError, match='expected 5000 positions, got 4999'):
        t.publish(positions(4999))
    assert log == [] and t.capabilities().ready


def test_dimensions_are_padded_to_the_shared_width():
    for count, dims in ((1, (4096, 1)), (4096, (4096, 1)), (4097, (4096, 2)), (tr.MAX_COUNT, (4096, 1024))):
        t, native, *_ = make(count)
        assert (t.width, t.height) == dims
        assert native.textures[-1].size == dims
        t.close()


def test_zero_particles_allocate_and_draw_nothing():
    t, native, ops, log = make(0)
    assert 'alloc' not in log and not native.textures and not native.live
    assert t.can_publish() and t.publish(positions(0)) is True
    assert t.positions_texture() is None
    assert t.close() == tr.COMPLETED


# -- activation gates / rollback ----------------------------------------------------------
def test_self_test_uses_a_4x4_texture_then_releases_it():
    _, native, _, log = make()
    assert [t.size for t in native.textures] == [(4, 4), (4096, 2)]
    first = log.index(('fill_known', 16))
    assert names(log)[first:first + 2] == ['fill_known', 'create_texture']
    assert names(log).count('read_texture') == 1
    # the self-test registration was unregistered; only the real target and the event stay live
    assert sorted(kind for kind, _ in native.live) == ['event', 'res']


@pytest.mark.parametrize('attrs,message', [
    ({'opengl': False}, 'not OpenGL'),
    ({'device_reason': 'different GPU'}, 'different GPU'),
    ({'texture_limit': 2048}, 'exceeds texture limit'),
    ({'free_vram': 1 << 20}, 'free VRAM'),
    ({'unidentifiable': True}, 'could not identify'),
    ({'corrupt_readback': True}, 'self-test mismatch'),
    ({'fail': {'register_image': 1}}, 'cudaGraphicsGLRegisterImage failed with CUDA error 1'),
    ({'fail': {'event_create': 2}}, 'cudaEventCreateWithFlags'),
])
def test_recoverable_setup_failure_releases_everything(attrs, message):
    log = []
    native, ops = FakeNative(log), FakeOps(log)
    for key, value in attrs.items():
        setattr(native, key, value)
    with pytest.raises(tr.RecoverableTransportError, match=message):
        tg.GLInteropTransport(native, ops, 5000)
    assert native.live == set()
    assert not tg._QUARANTINE


def test_unknown_setup_error_is_fatal_and_quarantined():
    log = []
    native = FakeNative(log)
    native.fail['register_image'] = 700  # cudaErrorIllegalAddress: not on the allowlist
    with pytest.raises(tr.FatalTransportError, match='700'):
        tg.GLInteropTransport(native, FakeOps(log), 5000)
    assert len(tg._QUARANTINE) == 1 and tg._QUARANTINE[0].fatal_reason.endswith('700')


# -- fatal handling -----------------------------------------------------------------------
@pytest.mark.parametrize('step', ['map', 'mapped_array', 'copy', 'unmap', 'event_record'])
def test_publish_error_is_fatal_and_stops_all_cuda_calls(step):
    t, native, _, log = make()
    native.fail[step] = 700
    with pytest.raises(tr.FatalTransportError, match='700'):
        t.publish(positions(5000))
    assert t in tg._QUARANTINE and '700' in t.fatal_reason
    assert not t.capabilities().ready and t.capabilities().reason == t.fatal_reason
    del log[:]
    assert (t.can_publish(), t.publish(positions(5000)), t.positions_texture()) == (False, False, None)
    assert t.close() == tr.FAILED and t.close() == tr.FAILED
    assert log == []  # no further CUDA calls after a fatal error


def test_event_query_error_is_fatal():
    t, native, *_ = make()
    t.publish(positions(5000))
    native.query_results = [716]
    with pytest.raises(tr.FatalTransportError, match='716'):
        t.can_publish()


def test_non_cuda_exception_in_publish_is_wrapped_as_fatal():
    t, _, ops, _ = make()
    ops.pack = lambda *a: (_ for _ in ()).throw(RuntimeError('launch failed'))
    with pytest.raises(tr.FatalTransportError, match='launch failed') as info:
        t.publish(positions(5000))
    assert isinstance(info.value.__cause__, RuntimeError)


# -- close --------------------------------------------------------------------------------
def test_close_synchronizes_then_unregisters_once_and_is_idempotent():
    t, native, _, log = make()
    resource = next(r for kind, r in native.live if kind == 'res')
    event = next(e for kind, e in native.live if kind == 'event')
    t.publish(positions(5000))
    del log[:]
    assert t.close() == tr.COMPLETED
    assert log == ['sync', ('unregister', resource), ('event_destroy', event)]
    assert native.live == set()
    del log[:]
    assert t.close() == tr.COMPLETED and log == []
    assert (t.can_publish(), t.publish(positions(5000)), t.positions_texture()) == (False, False, None)
    assert 'closed' in t.capabilities().reason and not t.capabilities().ready


def test_close_failure_keeps_ownership_and_reports_failed():
    t, native, *_ = make()
    native.fail['unregister'] = 700
    assert t.close() == tr.FAILED
    assert t in tg._QUARANTINE and t.close() == tr.FAILED


def test_close_sync_failure_reports_failed_without_unregistering():
    t, native, ops, log = make()
    ops.sync_error = RuntimeError('context lost')
    del log[:]
    assert t.close() == tr.FAILED
    assert log == ['sync'] and native.live


# -- session selection / fallback ---------------------------------------------------------
def opener(**over):
    calls = SimpleNamespace(gl=[], readback=[], vulkan=0)

    def create_gl(count):
        calls.gl.append(count)
        if 'gl_error' in over:
            raise over['gl_error']
        return SimpleNamespace(
            capabilities=lambda: tr.Capabilities(tg.NAME, True, 1, True), close=lambda: tr.COMPLETED)

    def create_readback(count, *, reason=''):
        t = SimpleNamespace(capabilities=lambda: tr.Capabilities('READBACK', False, 1, True, reason),
                            close=lambda: tr.COMPLETED, reason=reason)
        calls.readback.append(t)
        return t

    def open_(backend='OPENGL', unavailable=''):
        return ts.open_session(backend, 8, 1, interop_unavailable=unavailable, create_gl=create_gl,
                               create_vulkan=lambda *a, **k: calls.__setattr__('vulkan', calls.vulkan + 1),
                               create_readback=create_readback, resolve=lambda: None,
                               alloc=lambda n: ['pos'] * n,
                               renderer_factory=lambda: SimpleNamespace(free=lambda: None))
    return open_, calls


def test_opengl_backend_selects_gl_interop():
    open_, calls = opener()
    s = open_()
    assert calls.gl == [8] and calls.readback == [] and calls.vulkan == 0
    assert s.label == ts.GL_LABEL and s.reason == ''


def test_recoverable_gl_failure_falls_back_once_to_labelled_readback():
    open_, calls = opener(gl_error=tr.RecoverableTransportError('CUDA 12.9 runtime not found'))
    s = open_()
    assert s.label == ts.READBACK_LABEL and not s.transport.capabilities().zero_copy
    assert s.reason == 'OpenGL interop unavailable: CUDA 12.9 runtime not found'
    assert calls.readback[0].reason == s.reason
    s2 = open_(unavailable=s.reason)   # the caller passes the sticky reason: no retry
    assert calls.gl == [8] and s2.reason == s.reason and s2.label == ts.READBACK_LABEL


def test_fatal_gl_failure_propagates_without_fallback():
    open_, calls = opener(gl_error=tr.FatalTransportError('device lost'))
    with pytest.raises(tr.FatalTransportError):
        open_()
    assert calls.readback == []


def test_vulkan_and_other_backends_are_not_routed_to_gl():
    open_, calls = opener()
    assert open_('METAL').reason == 'no interop adapter for the METAL backend'
    assert calls.gl == []


# -- numerics (real Warp kernels on the CPU device) ---------------------------------------
def test_pack_kernel_pads_with_w_one_and_zero_tail():
    src = wp.array(np.arange(15, dtype=np.float32).reshape(5, 3), dtype=wp.vec3, device='cpu')
    dst = wp.zeros(8, dtype=wp.vec4, device='cpu')
    wp.launch(tg._pack, dim=8, inputs=[src, dst, 5], device='cpu')
    out = dst.numpy()
    np.testing.assert_array_equal(out[:5, :3], np.arange(15, dtype=np.float32).reshape(5, 3))
    np.testing.assert_array_equal(out[:5, 3], np.ones(5))
    np.testing.assert_array_equal(out[5:], np.zeros((3, 4)))


def test_known_kernel_matches_known_answer():
    dst = wp.zeros(32, dtype=wp.vec4, device='cpu')
    wp.launch(tg._known, dim=32, inputs=[dst, 16], device='cpu')
    np.testing.assert_array_equal(dst.numpy()[:16], tg.known_answer(16))
    np.testing.assert_array_equal(dst.numpy()[16:], np.zeros((16, 4)))
