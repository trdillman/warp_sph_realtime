"""M2 transport tests with a fake Blender/CUDA seam (no GPU, no bpy).

These prove state-machine, ordering, rollback and quarantine behaviour only; the real
native path is qualified in Blender (M5), never by these fakes.
"""
import numpy as np
import pytest
import warp as wp

from warp_sph_realtime.render import transport as tr
from warp_sph_realtime.render import transport_vulkan as tv

WINDOW, WINDOW_CTX, DRW_CTX = 1, 100, 200
UUID = bytes(range(16))


class FakeNative:
    def __init__(self):
        self.calls = []
        self.fail = {}          # method name -> CUresult (or False for Blender-side failure)
        self.event_results = []  # scripted cuEventQuery results, default 0
        self.fence_results = []  # scripted vkWaitSemaphores results, default 0
        self.active, self.enabled = WINDOW_CTX, False
        self.window, self.windows = WINDOW, {WINDOW}
        self.main_thread = True
        self.vulkan = True
        self.uuid = UUID
        self.free_vram, self.total_vram = 8 << 30, 16 << 30
        self.restore_ok = True
        self._next = 1000
        self.pbs, self.handles, self.exts, self.ptrs, self.events, self.fences = {}, set(), {}, {}, set(), set()
        self.textures = []
        self.corrupt_upload = False

    def _id(self):
        self._next += 1
        return self._next

    def _log(self, name, *args):
        self.calls.append((name, *args))

    def live(self):
        return len(self.pbs) + len(self.handles) + len(self.exts) + len(self.ptrs) + len(self.events) + len(self.fences)

    # host
    def is_main_thread(self):
        return self.main_thread

    def live_windows(self):
        return set(self.windows)

    def current_window(self):
        return self.window

    def max_texture_size(self):
        return 16384

    # Blender
    def backend_is_vulkan(self):
        return self.vulkan

    def blender_uuid(self):
        return self.uuid

    def active_context(self):
        return self.active

    def window_context(self, window):
        return WINDOW_CTX if window == WINDOW else 0

    def drw_enable(self):
        self._log('drw_enable')
        self.enabled, self.active = True, DRW_CTX

    def drw_disable(self):
        self._log('drw_disable')
        self.enabled = False
        self.active = WINDOW_CTX if self.restore_ok else 999

    def drw_enabled(self):
        return self.enabled

    def create_texture(self, w, h):
        tex = {'data': np.zeros((w * h, 4), np.float32), 'size': (w, h)}
        self.textures.append(tex)
        return tex

    def read_texture(self, tex, w, h):
        return tex['data'].copy()

    def texture_pointer(self, tex):
        if self.fail.get('texture_pointer') is False:
            raise tr.RecoverableTransportError('wrapper type mismatch')
        return id(tex)

    def pb_create(self, size):
        if self.fail.get('pb_create') is False:
            return 0
        pb = self._id()
        self.pbs[pb] = np.zeros((size // 16, 4), np.float32)
        return pb

    def pb_size(self, pb):
        return self.pbs[pb].size * 4

    def pb_export(self, pb):
        handle = self._id()
        self.handles.add(handle)
        self._handle_pb = getattr(self, '_handle_pb', {})
        self._handle_pb[handle] = pb
        return handle, self.pbs[pb].size * 4

    def pb_free(self, pb):
        self._log('pb_free', pb)
        assert self.enabled, 'Blender object freed outside the GPU scope'
        del self.pbs[pb]

    def texture_update(self, tp, pb, w, h):
        self._log('texture_update', pb)
        assert self.active == DRW_CTX
        tex = next(t for t in self.textures if id(t) == tp)
        tex['data'][:] = self.pbs[pb][:w * h]
        if self.corrupt_upload:
            tex['data'][0, 0] += 1

    def fence_create(self):
        if self.fail.get('fence_create') is False:
            return 0
        fence = self._id()
        self.fences.add(fence)
        return fence

    def fence_signal(self, fence):
        self._log('fence_signal', fence)
        assert self.active == DRW_CTX

    def fence_free(self, fence):
        self._log('fence_free', fence)
        assert self.enabled
        self.fences.remove(fence)

    def fence_query(self, fence):
        self._log('fence_query', fence)
        return self.fence_results.pop(0) if self.fence_results else 0

    def close_handle(self, handle):
        self.handles.remove(handle)
        return True

    # CUDA
    def _cu(self, name):
        self._log(name)
        return self.fail.get(name, 0)

    def cuda_uuid(self):
        return self._cu('cuda_uuid'), UUID

    def mem_info(self):
        return self._cu('mem_info'), self.free_vram, self.total_vram

    def import_memory(self, handle, size):
        code = self._cu('import_memory')
        if code:
            return code, 0
        ext = self._id()
        self.exts[ext] = self._handle_pb[handle]
        return 0, ext

    def map_buffer(self, ext, size):
        code = self._cu('map_buffer')
        if code:
            return code, 0
        ptr = self._id()
        self.ptrs[ptr] = self.exts[ext]
        return 0, ptr

    def mem_free(self, ptr):
        code = self._cu('mem_free')
        del self.ptrs[ptr]
        return code

    def destroy_external(self, ext):
        code = self._cu('destroy_external')
        del self.exts[ext]
        return code

    def event_create(self):
        code = self._cu('event_create')
        if code:
            return code, 0
        event = self._id()
        self.events.add(event)
        return 0, event

    def event_record(self, event, stream):
        assert stream == 7, 'event must be recorded on the Warp producer stream'
        return self._cu('event_record')

    def event_query(self, event):
        self._log('event_query', event)
        return self.event_results.pop(0) if self.event_results else 0

    def event_destroy(self, event):
        code = self._cu('event_destroy')
        self.events.remove(event)
        return code


class FakeOps:
    def __init__(self, native):
        self.n = native

    def wrap(self, ptr, texels):
        return self.n.pbs[self.n.ptrs[ptr]]  # aliases the pixel buffer like the real import

    def stream_handle(self):
        return 7

    def pack(self, positions, dst, count):
        dst[:] = 0
        dst[:count, :3] = np.asarray(positions, np.float32)
        dst[:count, 3] = 1

    def fill_known(self, dst, count):
        dst[:] = tv.known_answer(len(dst))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt or tv.PRODUCER_WAIT  # sleep(0) is the producer-wait yield: one poll, then give up


@pytest.fixture(autouse=True)
def _reset_quarantine():
    tv._QUARANTINE.clear()
    yield
    tv._QUARANTINE.clear()


def make(count=5, native=None, clock=None):
    native = native or FakeNative()
    clock = clock or Clock()
    t = tv.VulkanInteropTransport(native, FakeOps(native), count, window=WINDOW, clock=clock, sleep=clock.sleep)
    return t, native, clock


def draw(t, native):
    native.active = DRW_CTX
    try:
        return t.positions_texture()
    finally:
        native.active = WINDOW_CTX


def positions(n, offset=0.0):
    return np.arange(n * 3, dtype=np.float32).reshape(n, 3) + offset


def cuda_names(native):
    names = {'cuda_uuid', 'mem_info', 'import_memory', 'map_buffer', 'mem_free', 'destroy_external',
             'event_create', 'event_record', 'event_query', 'event_destroy'}
    return [c[0] for c in native.calls if c[0] in names]


# -- dimensions -----------------------------------------------------------------------------
@pytest.mark.parametrize('count,expected', [(0, (0, 0)), (1, (4096, 1)), (4096, (4096, 1)), (4097, (4096, 2)),
                                            (tr.MAX_COUNT, (4096, 1024))])
def test_dimensions(count, expected):
    assert tr.dimensions(count) == expected


@pytest.mark.parametrize('bad', [-1, tr.MAX_COUNT + 1, 1.5, True, '3'])
def test_dimensions_rejects(bad):
    with pytest.raises(ValueError):
        tr.dimensions(bad)


def test_invalid_count_makes_no_native_call():
    native = FakeNative()
    with pytest.raises(ValueError):
        make(tr.MAX_COUNT + 1, native)
    assert native.calls == []


# -- activation -----------------------------------------------------------------------------
def test_activation_self_test_and_resources():
    t, native, _ = make(5)
    assert t.capabilities() == tr.Capabilities('VULKAN_INTEROP', True, tr.MAX_COUNT, True, '')
    assert len(native.pbs) == 2 and len(native.fences) == 2 and len(native.events) == 2
    assert not native.handles  # every exported handle closed after import
    assert native.active == WINDOW_CTX and not native.enabled
    assert native.textures[0]['size'] == tv.SELF_TEST  # 256-byte known-answer texture first
    assert t.positions_texture() is None  # nothing uploaded yet


def test_uuid_mismatch_imports_nothing():
    native = FakeNative()
    native.uuid = bytes(16)
    with pytest.raises(tr.RecoverableTransportError, match='UUID'):
        make(5, native)
    assert 'import_memory' not in cuda_names(native) and native.live() == 0


def test_backend_mismatch_is_recoverable():
    native = FakeNative()
    native.vulkan = False
    with pytest.raises(tr.RecoverableTransportError):
        make(5, native)
    assert native.calls == []


def test_vram_budget_rejects_before_allocation():
    native = FakeNative()
    native.free_vram = 100 << 20
    with pytest.raises(tr.RecoverableTransportError, match='VRAM'):
        make(5, native)
    assert native.live() == 0 and not native.textures


def test_self_test_mismatch_rolls_back():
    native = FakeNative()
    native.corrupt_upload = True
    with pytest.raises(tr.RecoverableTransportError, match='known-answer'):
        make(5, native)
    assert native.live() == 0 and native.active == WINDOW_CTX and not native.enabled


@pytest.mark.parametrize('method,code', [('import_memory', 2), ('map_buffer', 1), ('event_create', 2)])
def test_recoverable_setup_error_rolls_back(method, code):
    native = FakeNative()
    native.fail[method] = code
    with pytest.raises(tr.RecoverableTransportError):
        make(5, native)
    assert native.live() == 0 and not tv._QUARANTINE


@pytest.mark.parametrize('method', ['pb_create', 'fence_create', 'texture_pointer'])
def test_blender_setup_failure_is_recoverable(method):
    native = FakeNative()
    native.fail[method] = False
    with pytest.raises(tr.RecoverableTransportError):
        make(5, native)
    assert native.live() == 0


def test_fatal_setup_error_quarantines_without_rollback():
    native = FakeNative()
    native.fail['map_buffer'] = 700  # CUDA_ERROR_ILLEGAL_ADDRESS: context is unusable
    with pytest.raises(tr.FatalTransportError):
        make(5, native)
    assert native.live() > 0  # nothing released after a fatal error
    assert any(isinstance(q, tv.VulkanInteropTransport) for q in tv._QUARANTINE)
    assert native.active == WINDOW_CTX  # GPU scope still restored


@pytest.mark.parametrize('results', ['event_results', 'fence_results'])
def test_self_test_timeout_quarantines_its_texture(results):
    native = FakeNative()
    setattr(native, results, [600 if results == 'event_results' else 2] * 1000)  # never completes
    with pytest.raises(tr.FatalTransportError, match='incomplete'):
        make(5, native)
    (quarantined,) = tv._QUARANTINE
    test_texture = native.textures[0]  # the upload may still be reading or writing it
    assert any(value is test_texture for value in vars(quarantined).values())
    assert native.live() > 0 and native.active == WINDOW_CTX


def test_unclassified_context_falls_back():
    native = FakeNative()
    native.active = 555
    with pytest.raises(tr.RecoverableTransportError, match='context'):
        make(5, native)
    assert 'drw_enable' not in [c[0] for c in native.calls]


def test_context_restore_failure_is_fatal():
    native = FakeNative()
    native.restore_ok = False
    with pytest.raises(tr.FatalTransportError, match='restoration'):
        make(5, native)


def test_zero_count_allocates_nothing():
    t, native, _ = make(0)
    assert len(native.pbs) == 0 and not native.events
    assert t.publish(positions(0)) is True and t.positions_texture() is None
    assert t.close() == tr.COMPLETED


# -- runtime state machine ------------------------------------------------------------------
def test_publish_then_draw_uploads_positions():
    t, native, _ = make(5)
    native.calls.clear()
    assert t.publish(positions(5))
    names = [c[0] for c in native.calls]
    # the timer submits the upload on its own: its fence covers this copy, not a viewport frame
    assert names[-4:] == ['drw_enable', 'texture_update', 'fence_signal', 'drw_disable']
    assert t.timer_uploads == t.uploads == 1 and native.active == WINDOW_CTX
    native.calls.clear()
    frame = draw(t, native)
    assert not [c for c in native.calls if c[0] in ('texture_update', 'fence_signal')]
    assert frame.count == 5 and (frame.width, frame.height) == (4096, 1) and frame.frame == 0
    data = frame.texture['data']
    np.testing.assert_array_equal(data[:5, :3], positions(5))
    assert (data[:5, 3] == 1).all() and not data[5:].any()  # padding stays zero


def test_draw_before_publish_returns_none():
    t, native, _ = make(5)
    assert draw(t, native) is None


def test_multiple_ticks_keep_newest_and_draw_once():
    t, native, _ = make(5)
    native.event_results = [600] * 5  # producers slower than PRODUCER_WAIT for every query
    assert t.publish(positions(5, 0))
    assert t.publish(positions(5, 10))  # second slot
    assert not t.publish(positions(5, 20))  # both WRITING: skip, never block
    assert t.drops == 1
    native.event_results = [600, 600]
    assert not t.can_publish()  # timer backpressure while both producers run
    frame = draw(t, native)  # producers done: the draw uploads the newest, in its frame
    np.testing.assert_array_equal(frame.texture['data'][:5, :3], positions(5, 10))
    assert t.uploads == 1 and t.timer_uploads == 0
    native.calls.clear()
    again = draw(t, native)  # repeated draw without new data: no re-upload / re-signal
    assert again.frame == frame.frame
    assert not [c for c in native.calls if c[0] in ('texture_update', 'fence_signal')]


def test_ready_slot_is_replaced_by_newer_tick():
    t, native, _ = make(5)
    native.windows = {WINDOW, 2}  # no timer scope: uploads are left to the draw, so READY slots queue
    t.publish(positions(5, 0))
    t.publish(positions(5, 10))   # both READY after polling
    t.publish(positions(5, 20))   # replaces oldest READY
    frame = draw(t, native)
    np.testing.assert_array_equal(frame.texture['data'][:5, :3], positions(5, 20))
    assert frame.frame == 2 and t.timer_uploads == 0


def test_in_flight_slot_not_rewritten_until_fence_completes():
    t, native, _ = make(5)
    native.windows = {WINDOW, 2}          # uploads left to the draw
    t.publish(positions(5))
    draw(t, native)                       # slot 0 IN_FLIGHT
    shown = next(s for s in t._slots if s.state == tv.IN_FLIGHT)
    native.fence_results = [2] * 10       # slot 0 fence stays pending
    t.publish(positions(5, 1))            # slot 1 READY
    other = next(s for s in t._slots if s is not shown)
    assert other.state == tv.READY and other.frame == 1
    assert t.publish(positions(5, 2))     # slot 1 READY -> replaced; slot 0 untouched
    assert other.state == tv.READY and other.frame == 2
    assert shown.state == tv.IN_FLIGHT and shown.frame == 0


def test_foreign_window_or_context_touches_nothing():
    t, native, _ = make(5)
    t.publish(positions(5))
    native.calls.clear()
    assert t.positions_texture() is None  # window context, not DRW
    native.window = 2
    assert draw(t, native) is None
    assert native.calls == []


def test_runtime_query_error_is_fatal_and_stops_calls():
    t, native, _ = make(5)
    t.publish(positions(5))
    native.event_results = [719]  # CUDA_ERROR_LAUNCH_FAILED
    with pytest.raises(tr.FatalTransportError):
        t.publish(positions(5))
    native.calls.clear()
    assert t.publish(positions(5)) is False and draw(t, native) is None
    assert t.close() == tr.FAILED and native.calls == []
    assert not t.capabilities().ready and native.live() > 0


def test_vulkan_query_error_is_fatal():
    t, native, _ = make(5)
    t.publish(positions(5))
    draw(t, native)
    native.fence_results = [-4]  # VK_ERROR_DEVICE_LOST
    with pytest.raises(tr.FatalTransportError, match='VkResult'):
        t.publish(positions(5))


def test_stalled_work_is_fatal():
    t, native, clock = make(5)
    native.event_results = [600] * 10
    t.publish(positions(5))
    clock.t += tv.STALL_DEADLINE + 0.1
    with pytest.raises(tr.FatalTransportError, match='stalled'):
        t.publish(positions(5))


def test_slow_healthy_step_holds_the_frame_instead_of_failing():
    # Regression: a 0.5 s pending deadline made any step slower than 500 ms fatal (restart).
    t, native, clock = make(5)
    t.publish(positions(5))                       # frame 0 shown
    assert draw(t, native).frame == 0
    native.event_results = [600] * 50
    assert t.publish(positions(5, 1.0))           # frame 1: a slow step, still being written
    for _ in range(20):                           # 2 s of timer ticks while the GPU works
        clock.t += 0.1
        assert not t.can_publish()                # skip publication, keep the slot owned
        assert draw(t, native).frame == 0         # hold the last shown frame
    native.event_results = []                     # the step lands
    assert t.can_publish() and t.capabilities().ready and not t.fatal_reason
    assert draw(t, native).frame == 1


def test_close_waits_for_a_slow_step_instead_of_failing():
    t, native, clock = make(5)
    t.publish(positions(5))
    draw(t, native)
    native.event_results = [600] * 5
    assert t.publish(positions(5))
    native.event_results = [600] * 800           # ~0.8 s more of GPU work at 1 ms per poll
    assert t.close() == tr.COMPLETED and native.live() == 0


def test_event_record_error_is_fatal():
    t, native, _ = make(5)
    native.fail['event_record'] = 1
    with pytest.raises(tr.FatalTransportError):
        t.publish(positions(5))


def test_length_mismatch_raises():
    t, _, _ = make(5)
    with pytest.raises(ValueError):
        t.publish(positions(4))


def test_generation_increments_per_allocation():
    a, na, _ = make(5)
    b, nb, _ = make(5)
    a.publish(positions(5))
    b.publish(positions(5))
    assert draw(b, nb).generation == draw(a, na).generation + 1


# -- close ----------------------------------------------------------------------------------
def test_close_releases_everything_in_m0_order():
    t, native, _ = make(5)
    t.publish(positions(5))
    draw(t, native)
    native.calls.clear()
    assert t.close() == tr.COMPLETED
    assert native.live() == 0 and native.active == WINDOW_CTX and not native.enabled
    names = [c[0] for c in native.calls]
    assert names[0] == 'drw_enable' and names[-1] == 'drw_disable'
    assert names.count('fence_signal') == 1  # one timeline fence after the last texture read
    first = names.index('mem_free')
    assert names[first:first + 5] == ['mem_free', 'destroy_external', 'event_destroy', 'fence_free', 'pb_free']
    assert t.close() == tr.COMPLETED and not t.capabilities().ready
    assert t.publish(positions(5)) is False


def test_close_waits_for_writing_slot():
    t, native, _ = make(5)
    native.event_results = [600, 600, 0]
    t.publish(positions(5))
    assert t.close() == tr.COMPLETED and native.live() == 0


def test_close_timeout_fails_and_retains():
    t, native, _ = make(5)
    t.publish(positions(5))
    draw(t, native)
    native.fence_results = [2] * 100_000  # stuck GPU: never completes within the budget
    assert t.close() == tr.FAILED
    assert native.live() > 0 and t in tv._QUARANTINE and native.active == WINDOW_CTX


def test_close_without_route_releases_cuda_only():
    t, native, _ = make(5)
    t.publish(positions(5))
    native.windows = {WINDOW, 2}  # owner no longer the single window
    native.calls.clear()
    assert t.close() == tr.QUARANTINED
    assert not native.ptrs and not native.exts and not native.events  # CUDA side released
    assert len(native.pbs) == 2 and len(native.fences) == 2           # Blender side retained
    assert 'drw_enable' not in [c[0] for c in native.calls[-10:]]


def test_close_with_both_slots_writing_finishes_producers_then_fences():
    t, native, _ = make(5)
    t.publish(positions(5))
    draw(t, native)
    native.event_results = [600] * 4
    assert t.publish(positions(5)) and t.publish(positions(5))
    assert all(s.state == tv.WRITING for s in t._slots)
    native.calls.clear()
    assert t.close() == tr.COMPLETED and native.live() == 0
    names = [c[0] for c in native.calls]
    assert names.count('fence_signal') == 1
    assert names.index('fence_signal') > names.index('event_query')


def test_scope_restoration_failure_keeps_original_error():
    t, native, _ = make(5)
    t.publish(positions(5))
    draw(t, native)
    native.fence_results = [2] * 100_000  # stuck GPU: never completes within the budget
    native.restore_ok = False
    assert t.close() == tr.FAILED
    assert 'restoration failed' in t.fatal_reason and 'while handling FatalTransportError' in t.fatal_reason


# -- kernels and readback -------------------------------------------------------------------
def test_pack_kernel_pads_with_zero():
    src = wp.array(positions(3), dtype=wp.vec3, device='cpu')
    dst = wp.array(np.full((8, 4), 9, np.float32), dtype=wp.vec4, device='cpu')
    wp.launch(tv._pack, dim=8, inputs=[src, dst, 3], device='cpu')
    out = dst.numpy()
    np.testing.assert_array_equal(out[:3, :3], positions(3))
    assert (out[:3, 3] == 1).all() and not out[3:].any()


def test_known_kernel_matches_reference():
    dst = wp.zeros(20, dtype=wp.vec4, device='cpu')
    wp.launch(tv._known, dim=20, inputs=[dst, 16], device='cpu')
    out = dst.numpy()
    np.testing.assert_array_equal(out[:16], tv.known_answer(16))
    assert not out[16:].any()


class FakeGpu:
    class types:
        @staticmethod
        def Buffer(fmt, size, data):
            assert fmt == 'FLOAT' and size == data.size
            return np.array(data, np.float32)

        @staticmethod
        def GPUTexture(size, format, data):
            assert format == 'RGBA32F'
            return {'size': size, 'data': data.reshape(-1, 4)}


_needs_cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason='Readback stages through pinned host memory, which needs a CUDA driver')


@_needs_cuda
def test_readback_round_trip_and_label():
    rb = tr.ReadbackTransport(5, reason='interop unavailable', gpu_module=FakeGpu)
    assert rb.capabilities() == tr.Capabilities('READBACK', False, tr.MAX_COUNT, True, 'interop unavailable')
    assert rb.positions_texture() is None and rb.newest_frame() == -1
    assert rb.publish(wp.array(positions(5), dtype=wp.vec3, device='cpu'))
    frame = rb.positions_texture()
    assert frame.texture['size'] == (4096, 1) and frame.frame == 0
    np.testing.assert_array_equal(frame.texture['data'][:5, :3], positions(5))
    assert not frame.texture['data'][5:].any()
    assert rb.can_publish() and rb.newest_frame() == 0
    assert rb.close() == tr.COMPLETED and rb.publish(None) is False and not rb.can_publish()


def test_readback_zero_count():
    rb = tr.ReadbackTransport(0, gpu_module=FakeGpu)
    assert rb.publish(wp.empty(0, dtype=wp.vec3, device='cpu')) is True
    assert rb.positions_texture() is None


# -- review regressions ---------------------------------------------------------------------
def test_queued_work_deadline_restarts_when_predecessor_completes():
    t, native, clock = make(5)
    d = tv.STALL_DEADLINE
    native.event_results = [600, 600, 600, 0, 600, 600, 600]
    t.publish(positions(5))            # A at t=0
    clock.t = 0.6 * d
    assert t.publish(positions(5))     # A pending; B queued behind it
    clock.t = 0.9 * d
    t.can_publish()                    # A completes (and uploads); B's deadline restarts at 0.9 d
    clock.t = 1.6 * d
    assert not t.can_publish()         # B pending 0.7 d: in flight, not overdue despite 1.0 d since record
    clock.t = 2.0 * d
    with pytest.raises(tr.FatalTransportError, match='stalled'):
        t.can_publish()


def test_publish_reuses_can_publish_observation():
    t, native, _ = make(5)
    t.publish(positions(5))            # A uploaded, its fence still pending
    a = t._slots[0]
    native.fence_results = [2] * 10
    native.calls.clear()
    native.event_results = [600, 600]
    assert t.can_publish()
    assert t.publish(positions(5))     # packs B; only its own producer wait queries B
    fences = [c[1] for c in native.calls if c[0] == 'fence_query']
    assert fences.count(a.fence) == 1  # C2: <= 1 query per other slot per tick
    assert {c[1] for c in native.calls if c[0] == 'event_query'} == {t._slots[1].event}


def test_one_step_in_flight():
    t, native, _ = make(5)
    native.event_results = [600] * 5
    assert t.publish(positions(5))     # A still WRITING after the producer wait
    native.event_results = [600]
    assert not t.can_publish()         # B is FREE, but a queued step would make the shown one older
    assert t.newest_frame() == -1      # nothing drawable yet
    assert t.can_publish()             # A done: uploaded from the timer, B may be produced
    assert t.newest_frame() == 0


def test_newest_frame_counts_ready_and_shown_frames():
    t, native, _ = make(5)
    assert t.newest_frame() == -1
    native.windows = {WINDOW, 2}       # no timer scope: READY frames wait for the draw
    t.publish(positions(5))
    assert t.newest_frame() == 0
    t.publish(positions(5, 1.0))
    assert t.newest_frame() == 1
    assert draw(t, native).frame == 1 and t.newest_frame() == 1


def test_draw_between_can_publish_and_publish_forces_a_fresh_poll():
    t, native, _ = make(5)
    t.publish(positions(5))
    shown = next(s for s in t._slots if s.state == tv.IN_FLIGHT)
    native.fence_results = [2] * 10       # keep the shown slot IN_FLIGHT
    assert t.can_publish()
    draw(t, native)
    native.calls.clear()
    t.publish(positions(5, 1.0))
    assert ('fence_query', shown.fence) in native.calls


def test_scope_does_not_blame_an_outer_handled_error():
    t, native, _ = make(5)
    native.restore_ok = False
    try:
        raise KeyError('outer')
    except KeyError:
        assert t.close() == tr.FAILED
    assert 'restoration failed' in t.fatal_reason and 'outer' not in t.fatal_reason


@pytest.mark.parametrize('error', [OSError('access violation'), RuntimeError('sticky'), KeyError('x')])
def test_unexpected_native_error_is_fatal_and_stops_calls(error):
    t, native, _ = make(5)
    native.event_results = [600]
    t.publish(positions(5))  # producer still WRITING, so can_publish queries its event

    def boom(event):
        raise error
    native.event_query = boom
    with pytest.raises(tr.FatalTransportError) as info:
        t.can_publish()
    assert info.value.__cause__ is error and t in tv._QUARANTINE
    assert type(error).__name__ in t.fatal_reason
    native.calls.clear()
    assert t.publish(positions(5)) is False and draw(t, native) is None and t.close() == tr.FAILED
    assert native.calls == []


def test_unexpected_activation_error_is_fatal():
    native = FakeNative()

    def boom(size):
        raise OSError('native crash')
    native.pb_create = boom
    with pytest.raises(tr.FatalTransportError, match='OSError'):
        make(5, native)


def test_native_binding_errors_are_recoverable(monkeypatch):
    import sys
    import types
    for name in ('bpy', 'gpu'):  # get past the imports so the profile lookup itself fails
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class Resolved:
        callables, data, enums, layouts = {}, {}, {}, {}
    with pytest.raises(tr.RecoverableTransportError, match='native binding unavailable: KeyError'):
        tv.BlenderVulkanNative(Resolved())


class FakeCu:
    def __init__(self, result, pop_result=0):
        self.calls, self.result, self.pop_result = [], result, pop_result

    def cuCtxPushCurrent_v2(self, context):
        self.calls.append('push')
        return 0

    def cuCtxPopCurrent_v2(self, prior):
        self.calls.append('pop')
        return self.pop_result

    def cuEventQuery(self, event):
        self.calls.append('query')
        return self.result


def bound_native(cu):
    native = object.__new__(tv.BlenderVulkanNative)  # the CUDA seam only; no Blender profile
    native._cu, native._cuda_context = cu, 1
    return native


@pytest.mark.parametrize('code', [0, 600, 2])
def test_cuda_call_pops_its_context(code):
    cu = FakeCu(code)
    assert bound_native(cu).event_query(5) == code
    assert cu.calls == ['push', 'query', 'pop']


def test_fatal_curesult_makes_no_further_cuda_call():
    cu = FakeCu(700)  # CUDA_ERROR_ILLEGAL_ADDRESS: sticky, the context is unusable
    assert bound_native(cu).event_query(5) == 700  # the caller's check reports this code
    assert cu.calls == ['push', 'query']


def test_failed_pop_names_the_call_result():
    cu = FakeCu(0, pop_result=201)
    with pytest.raises(tr.FatalTransportError, match='after cuEventQuery returned CUresult 0'):
        bound_native(cu).event_query(5)


def test_quarantine_is_process_state():
    assert tv._QUARANTINE is tr.process_state().quarantine


@_needs_cuda
def test_readback_device_error_is_fatal(monkeypatch):
    rb = tr.ReadbackTransport(5, gpu_module=FakeGpu)
    src = wp.array(positions(5), dtype=wp.vec3, device='cpu')

    def boom(dst, src):
        raise RuntimeError('CUDA error 700')
    monkeypatch.setattr(rb._wp, 'copy', boom)
    with pytest.raises(tr.FatalTransportError, match='Readback device copy failed'):
        rb.publish(src)


# -- dedicated upload submission (M5 option 1) ----------------------------------------------
def test_producer_wait_is_bounded():
    t, native, clock = make(5)
    native.event_results = [600] * 100
    assert t.publish(positions(5))
    assert clock.t <= tv.PRODUCER_WAIT and len(native.event_results) == 99
    assert t._slots[0].state == tv.WRITING and t.timer_uploads == 0


def test_can_publish_submits_a_late_producer():
    t, native, _ = make(5)
    native.event_results = [600]
    t.publish(positions(5))
    assert t.timer_uploads == 0
    assert t.can_publish()  # producer done now: upload submitted from the timer
    assert t.timer_uploads == 1 and t._slots[0].state == tv.IN_FLIGHT
    np.testing.assert_array_equal(draw(t, native).texture['data'][:5, :3], positions(5))


def test_unavailable_route_leaves_upload_to_the_draw():
    t, native, _ = make(5)
    native.windows = {WINDOW, 2}  # e.g. Preferences open: no qualified timer scope
    native.calls.clear()
    assert t.publish(positions(5))
    assert 'drw_enable' not in [c[0] for c in native.calls] and t._slots[0].state == tv.READY
    frame = draw(t, native)
    np.testing.assert_array_equal(frame.texture['data'][:5, :3], positions(5))
    assert t.uploads == 1 and t.timer_uploads == 0


def test_timer_upload_restoration_failure_is_fatal():
    t, native, _ = make(5)
    native.restore_ok = False
    with pytest.raises(tr.FatalTransportError, match='restoration'):
        t.publish(positions(5))
    assert t in tv._QUARANTINE and t.publish(positions(5)) is False


def test_can_publish_restoration_failure_is_fatal():
    t, native, _ = make(5)
    native.event_results = [600]
    t.publish(positions(5))               # producer late: upload deferred
    native.restore_ok = False
    with pytest.raises(tr.FatalTransportError, match='restoration'):
        t.can_publish()
    assert t in tv._QUARANTINE


def test_drw_context_mismatch_in_timer_scope_leaves_upload_to_the_draw():
    t, native, _ = make(5)
    enable = native.drw_enable

    def wrong_context():
        enable()
        native.active = 777               # DRW scope activated some other context
    native.drw_enable = wrong_context
    assert t.publish(positions(5))
    assert t._slots[0].state == tv.READY and t.timer_uploads == 0 and native.active == WINDOW_CTX
    native.drw_enable = enable
    np.testing.assert_array_equal(draw(t, native).texture['data'][:5, :3], positions(5))
    assert t.uploads == 1
