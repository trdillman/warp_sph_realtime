"""M4 session-owner and add-on lifecycle tests with fake transports.

Fakes prove ownership and ordering only; real Vulkan/CUDA behaviour is proven in Blender (M5).
"""
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from warp_sph_realtime.render import session as ts
from warp_sph_realtime.render.transport import (
    COMPLETED,
    Capabilities,
    FatalTransportError,
    RecoverableTransportError,
)

ROOT = Path(__file__).resolve().parents[1]


FRAME = SimpleNamespace(frame=-1)


class FakeTransport:
    def __init__(self, zero_copy=True, reason=''):
        self.zero_copy, self.reason = zero_copy, reason
        self.ready, self.published, self.closes = True, 0, 0
        self.close_result = COMPLETED

    def can_publish(self):
        return self.ready

    def publish(self, positions):
        self.published += 1
        return True

    def positions_texture(self):
        return FRAME

    def newest_frame(self):
        return self.published - 1

    def capabilities(self):
        return Capabilities('VULKAN' if self.zero_copy else 'READBACK', self.zero_copy, 16, True, self.reason)

    @property
    def fatal_reason(self):
        return ''

    def close(self):
        self.closes += 1
        return self.close_result


class FakeRenderer:
    def __init__(self):
        self.frees, self.drawn = 0, []

    def draw(self, frame, settings):
        self.drawn.append((frame, settings))
        return True

    def free(self):
        self.frees += 1


def opener(**over):
    calls = SimpleNamespace(resolve=0, vulkan=[], readback=[])

    def resolve():
        calls.resolve += 1
        return over.get('resolution', SimpleNamespace(available=True, reason='', detail='', resolved='profile'))

    def create_vulkan(resolved, count, *, window):
        calls.vulkan.append((resolved, count, window))
        if 'vulkan_error' in over:
            raise over['vulkan_error']
        return FakeTransport()

    def create_readback(count, *, reason=''):
        t = FakeTransport(zero_copy=False, reason=reason)
        calls.readback.append(t)
        return t

    def open_(backend, count=8, window=1, unavailable=''):
        return ts.open_session(backend, count, window, interop_unavailable=unavailable, resolve=resolve,
                               create_vulkan=create_vulkan, create_readback=create_readback,
                               alloc=over.get('alloc', lambda n: ['pos'] * n),
                               renderer_factory=over.get('renderer', FakeRenderer))
    return open_, calls


class SessionSelectionTest(unittest.TestCase):
    def test_vulkan_selects_interop_with_label(self):
        open_, calls = opener()
        s = open_('VULKAN', 8, 42)
        self.assertEqual(calls.vulkan, [('profile', 8, 42)])
        self.assertEqual(s.label, ts.VULKAN_LABEL)
        self.assertEqual(s.reason, '')

    def test_unsupported_build_falls_back_to_labelled_readback(self):
        res = SimpleNamespace(available=False, reason='unsupported build', detail='sha mismatch', resolved=None)
        open_, calls = opener(resolution=res)
        s = open_('VULKAN')
        self.assertEqual(calls.vulkan, [])
        self.assertEqual(s.label, ts.READBACK_LABEL)
        self.assertEqual(s.reason, 'Vulkan interop unavailable: unsupported build (sha mismatch)')
        self.assertEqual(calls.readback[0].reason, s.reason)

    def test_recoverable_setup_error_falls_back_once(self):
        open_, calls = opener(vulkan_error=RecoverableTransportError('external memory rejected'))
        s = open_('VULKAN')
        self.assertFalse(s.transport.capabilities().zero_copy)
        self.assertIn('external memory rejected', s.reason)
        s2 = open_('VULKAN', unavailable=s.reason)  # caller passes the sticky reason: no retry
        self.assertEqual((calls.resolve, len(calls.vulkan)), (1, 1))
        self.assertEqual(s2.reason, s.reason)

    def test_fatal_setup_error_propagates_without_fallback(self):
        open_, calls = opener(vulkan_error=FatalTransportError('device lost'))
        with self.assertRaises(FatalTransportError):
            open_('VULKAN')
        self.assertEqual(calls.readback, [])

    def test_other_backends_use_readback(self):
        open_, calls = opener()
        s = open_('METAL')
        self.assertEqual(calls.resolve, 0)
        self.assertEqual(s.reason, 'no interop adapter for the METAL backend')

    def test_failure_after_transport_closes_it(self):
        def boom(*_):
            raise MemoryError('scratch')
        for key in ('alloc', 'renderer'):
            open_, calls = opener(**{key: boom})
            with self.subTest(key=key), self.assertRaises(MemoryError):
                open_('METAL')
            self.assertEqual(calls.readback[0].closes, 1)


class SessionTickCloseTest(unittest.TestCase):
    def test_tick_skips_without_producing_while_busy(self):
        s = ts.TransportSession(FakeTransport(), ['p'], FakeRenderer(), 'VULKAN')
        produced = []
        s.transport.ready = False
        self.assertFalse(s.tick(produced.append))
        self.assertEqual((produced, s.waits, s.transport.published), ([], 1, 0))
        s.transport.ready = True
        self.assertTrue(s.tick(produced.append))
        self.assertEqual((produced, s.transport.published), ([['p']], 1))

    def test_produce_error_is_fatal(self):
        s = ts.TransportSession(FakeTransport(), ['p'], FakeRenderer(), 'VULKAN')
        def produce(out):
            raise RuntimeError('CUDA error 700')
        with self.assertRaises(FatalTransportError) as info:
            s.tick(produce)
        self.assertIsInstance(info.exception.__cause__, RuntimeError)
        self.assertEqual(s.transport.published, 0)

    def test_take_new_frame_once_per_drawable_frame(self):
        s = ts.TransportSession(FakeTransport(), ['p'], FakeRenderer(), 'VULKAN')
        self.assertFalse(s.take_new_frame())  # nothing published
        s.tick(lambda out: None)
        self.assertEqual((s.take_new_frame(), s.take_new_frame()), (True, False))
        s.transport.ready = False
        s.tick(lambda out: None)               # skipped: nothing new to redraw
        self.assertFalse(s.take_new_frame())

    def test_frame_shown_by_another_redraw_is_not_new(self):
        s = ts.TransportSession(FakeTransport(), ['p'], FakeRenderer(), 'VULKAN')
        s.tick(lambda out: None)
        s.transport.positions_texture = lambda: SimpleNamespace(frame=0)
        s.draw('settings')                     # e.g. a mouse-move redraw uploaded and showed it
        self.assertFalse(s.take_new_frame())

    def test_draw_passes_adapter_frame(self):
        s = ts.TransportSession(FakeTransport(), [], FakeRenderer(), 'VULKAN')
        self.assertTrue(s.draw('settings'))
        self.assertEqual(s.renderer.drawn, [(FRAME, 'settings')])

    def test_close_is_idempotent_and_keeps_quarantined_scratch(self):
        s = ts.TransportSession(FakeTransport(), ['p'], FakeRenderer(), 'VULKAN')
        self.assertEqual((s.close(), s.close()), (COMPLETED, COMPLETED))
        self.assertEqual((s.transport.closes, s.renderer.frees, s.positions), (1, 1, None))
        q = ts.TransportSession(FakeTransport(), ['p'], FakeRenderer(), 'VULKAN')
        q.transport.close_result = 'quarantined'
        self.assertEqual(q.close(), 'quarantined')
        self.assertEqual(q.positions, ['p'])  # an in-flight copy may still read it


# Add-on lifecycle runs in a separate interpreter: it needs its own fake bpy (see
# test_missing_warp_registration.py) and must not share module state with test_blender_mock.
LIFECYCLE = r'''
import contextlib, sys, types
from types import SimpleNamespace as NS

class Operator:
    def report(self, level, message):
        self.reports = getattr(self, 'reports', []) + [(next(iter(level)), message)]
class Area:
    def __init__(self, type): self.type, self.redraws = type, 0
    def tag_redraw(self): self.redraws += 1
class WM:
    def __init__(self): self.timers, self.modals, self.removed, self.steps, self.confirms = [], [], [], [], []
    def event_timer_add(self, step, window=None):
        self.timers.append(window); self.steps.append(step); return ('timer', len(self.timers))
    def invoke_confirm(self, op, event, **kw): self.confirms.append((op, event)); return {'RUNNING_MODAL'}
    def event_timer_remove(self, timer): self.removed.append(timer)
    def modal_handler_add(self, op): self.modals.append(op)
class Window:
    def as_pointer(self): return 0xABC
class Scene: pass
class Object: pass

handles = []
bpy = types.ModuleType('bpy')
bpy.types = NS(Operator=Operator, Panel=object, AddonPreferences=object, Scene=Scene,
               PropertyGroup=object, Object=Object,
               SpaceView3D=NS(draw_handler_add=lambda *a: handles.append(a) or 'draw',
                              draw_handler_remove=lambda h, r: handles.append(('removed', h))))
bpy.props = types.ModuleType('bpy.props')
for name in ('BoolProperty', 'EnumProperty', 'FloatProperty', 'FloatVectorProperty', 'IntProperty', 'StringProperty',
             'PointerProperty'):
    setattr(bpy.props, name, lambda **kw: kw.get('default'))
bpy.utils = NS(register_class=lambda c: None, unregister_class=lambda c: None)
def persistent(f): f._bpy_persistent = True; return f
bpy.app = NS(online_access=False, online_access_override=True,
             handlers=NS(load_pre=[], exit_pre=[], undo_pre=[], redo_pre=[], persistent=persistent))
PRE_HANDLERS = ('load_pre', 'exit_pre', 'undo_pre', 'redo_pre')
sys.modules['bpy'] = bpy
sys.modules['bpy.props'] = bpy.props
gpu = types.ModuleType('gpu')
gpu.platform = NS(backend_type_get=lambda: 'VULKAN')
sys.modules['gpu'] = gpu

import warp_sph_realtime as ext
from warp_sph_realtime import addon, sim_session
from warp_sph_realtime.render import session as ts
from warp_sph_realtime.render.transport import (Capabilities, FatalTransportError, RecoverableTransportError,
                                                process_state)
from warp_sph_realtime.scene.description import SceneFillError, SceneValidationError

class Solver:
    dt = 0.001
    fail = None
    def __init__(self, num_particles, h, substeps, enable_obstacles):
        if Solver.fail: raise Solver.fail
        self.num_particles, self.substeps, self.pos_sim, self.steps = num_particles, substeps, 'pos', 0
        self.stiffness = self.cfl_factor = 0.0
    rebuild_error = None
    def rebuild_cuda_graph(self):
        if Solver.rebuild_error: raise Solver.rebuild_error
    def update_time_step(self): pass
    from_scene_calls, fill_count, fill_error = [], 777, None
    @classmethod
    def from_scene(cls, desc, spacing, substeps=2, stiffness=None, cfl_factor=None):
        Solver.from_scene_calls.append((desc, spacing, substeps, stiffness, cfl_factor))
        if Solver.fill_error: raise Solver.fill_error
        solver = cls(Solver.fill_count, 2.0 * spacing, substeps, 0)
        solver.stiffness, solver.cfl_factor = stiffness, cfl_factor
        return solver
    def get_vram_usage_mb(self): return 1.0
    step_error = None
    def step(self, out, use_graph=True):
        if Solver.step_error: raise Solver.step_error
        self.steps += 1

class Transport:
    log = []
    def __init__(self, zero_copy, reason=''):
        self.zero_copy, self.reason, self.ready, self.closes = zero_copy, reason, True, 0
        self.close_result, self.publish_error, self.drops = 'completed', None, 0
        self.frame = -1
        Transport.log.append(self)
    def can_publish(self): return self.ready
    def publish(self, positions):
        if self.publish_error: raise self.publish_error
        self.frame += 1
        return True
    def newest_frame(self): return self.frame
    def positions_texture(self): return None
    def capabilities(self): return Capabilities('X', self.zero_copy, 1 << 22, True, self.reason)
    @property
    def fatal_reason(self): return 'fatal' if self.close_result == 'failed' else ''
    def close(self): self.closes += 1; return self.close_result

class Renderer:
    def draw(self, frame, settings): return False
    def free(self): pass

FACTORY = NS(vulkan_error=None, gl_error=None, resolves=0, open_error=None, counts=[])
def resolve():
    FACTORY.resolves += 1
    return NS(available=True, reason='', detail='', resolved='profile')
def create_vulkan(resolved, count, *, window):
    if FACTORY.vulkan_error: raise FACTORY.vulkan_error
    return Transport(True)
def create_gl(count):
    if FACTORY.gl_error: raise FACTORY.gl_error
    return Transport(True)
real_open = ts.open_session
def open_session(backend, count, window, *, interop_unavailable=''):
    if FACTORY.open_error: raise FACTORY.open_error
    FACTORY.counts.append(count)
    return real_open(backend, count, window, interop_unavailable=interop_unavailable, resolve=resolve,
                     create_vulkan=create_vulkan, create_gl=create_gl, create_readback=lambda n, reason='': Transport(False, reason),
                     alloc=lambda n: 'scratch', renderer_factory=Renderer)
sim_session.transport_session.open_session = open_session
RUNTIME = sim_session.runtime
RUNTIME.wp = NS(copy=lambda dst, src: None)  # marks the runtime active; no Warp import
RUNTIME.RealtimeWCSPHSolver = Solver
RUNTIME.launch_demo_scene_init = lambda solver, demo: None
HELPERS = []
RUNTIME.build_blender_demo_helpers = lambda demo: HELPERS.append('built:' + demo)
RUNTIME.remove_blender_demo_helpers = lambda: HELPERS.append('removed')
DESC = NS(calls=[], error=None)
def description_from_blender(scene, depsgraph):
    DESC.calls.append((scene, depsgraph))
    if DESC.error: raise DESC.error
    return 'desc'
from warp_sph_realtime.scene import blender_scene
blender_scene.description_from_blender = description_from_blender
CUR = [None]
class _Current:
    """The SimSession under test (the scene last started or reset): reads and writes go to it."""
    def __getattr__(self, name): return getattr(CUR[0], name)
    def __setattr__(self, name, value): setattr(CUR[0], name, value)
RT = _Current()

def make_context(areas=('VIEW_3D',)):
    scene = Scene()
    scene.warp_sph = NS(particle_count=1000, demo_scene='DOUBLE_DAM_PILLARS', substeps=2, stiffness=1.0,
                        time_step_scale=0.4, use_cuda_graph=True, show_particles=True, scene_source='DEMO',
                        particle_spacing=0.02, render_mode='SSFR_LIQUID')  # the real defaults
    return NS(scene=scene, window=Window(), window_manager=WM(), screen=NS(areas=[Area(a) for a in areas]),
              evaluated_depsgraph_get=lambda: 'depsgraph')
bpy.context = make_context()

def reset():
    sim_session.dispose_all(bpy.context)
    state = process_state()
    state.restart_required = state.interop_unavailable = ''
    sim_session._QUARANTINE.clear()
    Transport.log.clear()
    Solver.fail = Solver.step_error = Solver.rebuild_error = None
    FACTORY.vulkan_error = FACTORY.gl_error = FACTORY.open_error = None
    FACTORY.resolves = 0
    FACTORY.counts.clear()
    Solver.from_scene_calls.clear()
    Solver.fill_error = DESC.error = None
    DESC.calls.clear()
    HELPERS.clear()
    handles.clear()
    bpy.context = make_context()
    CUR[0] = sim_session.get(bpy.context.scene)
    return bpy.context

def start(ctx):
    CUR[0] = sim_session.get(ctx.scene)
    op = addon.WM_OT_warp_sph_start()
    return op, op.execute(ctx)

TIMER = NS(type='TIMER')

def case(fn):
    ctx = reset()
    fn(ctx)
    print('ok', fn.__name__)

ext.register()
assert all(addon._on_load_pre in getattr(bpy.app.handlers, n) for n in PRE_HANDLERS)
addon._install_handlers()  # idempotent: never installs a handler twice
assert all(getattr(bpy.app.handlers, n).count(addon._on_load_pre) == 1 for n in PRE_HANDLERS)
assert addon._on_load_pre._bpy_persistent

@case
def start_step_stop(ctx):
    op, result = start(ctx)
    assert result == {'RUNNING_MODAL'} and RT.running and RT.session.label == ts.VULKAN_LABEL, result
    assert ctx.window_manager.timers == [ctx.window] and RT.timer is not None
    assert ctx.window_manager.steps == [0.001]  # P0.5: 1 ms timer chosen by measured latency; do not change
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1 and RT.step_count == 1
    session = RT.session
    RT.session.transport.ready = False  # both slots busy: skip, never block or step
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1 and session.waits == 1
    addon.WM_OT_warp_sph_stop().execute(ctx)
    assert op.modal(ctx, TIMER) == {'CANCELLED'}
    assert session.transport.closes == 1 and RT.session is None and RT.timer is None and RT.last_close == 'completed'
    assert ctx.window_manager.removed and ('removed', 'draw') in handles
    assert op.modal(ctx, TIMER) == {'CANCELLED'}  # orphaned modal exits by token

@case
def immediate_disable(ctx):
    op, _ = start(ctx)
    session = RT.session
    ext.unregister()
    assert session.transport.closes == 1 and RT.session is None and RT.timer is None
    assert all(addon._on_load_pre not in getattr(bpy.app.handlers, n) for n in PRE_HANDLERS)
    assert op.modal(ctx, TIMER) == {'CANCELLED'}
    ext.register()

@case
def in_flight_disable_quarantines(ctx):
    start(ctx)
    session = RT.session
    session.transport.close_result = 'quarantined'
    ext.unregister()
    assert RT.last_close == 'quarantined' and session in sim_session._QUARANTINE and not RT.restart_required
    assert 'quarantined' in RT.interop_unavailable  # retained Blender objects: Readback until restart
    ext.register()

@case
def failed_close_requires_restart(ctx):
    start(ctx)
    RT.session.transport.close_result = 'failed'
    RT.stop(ctx)
    assert RT.restart_required == 'fatal'
    op, result = start(make_context())
    assert result == {'CANCELLED'} and 'restart' in op.reports[-1][1]

@case
def repeated_close(ctx):
    start(ctx)
    session = RT.session
    for _ in range(3):
        RT.stop(ctx)
    op = addon.WM_OT_warp_sph_start()
    op.cancel(ctx)
    assert session.transport.closes == 1

@case
def no_view_startup_and_last_view_closure(ctx):
    for areas in ((), ('PROPERTIES',)):
        ctx = reset()
        ctx.screen = NS(areas=[Area(a) for a in areas])
        op, _ = start(ctx)
        session = RT.session
        assert op.modal(ctx, TIMER) == {'CANCELLED'} and session.transport.closes == 1
        assert 'no 3D viewport' in op.reports[-1][1]
    ctx = reset()
    op, _ = start(ctx)
    ctx.screen = None  # window closed with its screen
    assert op.modal(ctx, TIMER) == {'CANCELLED'} and RT.session is None

@case
def file_load(ctx):
    op, _ = start(ctx)
    session = RT.session
    for handler in bpy.app.handlers.load_pre:
        handler(None)
    assert session.transport.closes == 1 and RT.session is None and not RT.running
    assert op.modal(ctx, TIMER) == {'CANCELLED'}

@case
def failure_after_every_setup_acquisition(ctx):
    Solver.fail = RuntimeError('cuda oom')                    # 1. solver
    op, result = start(ctx)
    assert result == {'CANCELLED'} and not Transport.log and RT.session is None and not RT.restart_required
    ctx = reset()
    FACTORY.open_error = RuntimeError('scratch oom')          # 2. session (non-fatal)
    assert start(ctx)[1] == {'CANCELLED'} and RT.solver is None and not sim_session._QUARANTINE
    ctx = reset()
    FACTORY.open_error = FatalTransportError('device lost')    # 3. session (fatal)
    assert start(ctx)[1] == {'CANCELLED'} and RT.restart_required == 'device lost'
    assert any(isinstance(o, Solver) for o in sim_session._QUARANTINE) and not Transport.log
    ctx = reset()
    def bad_publish(self, positions): raise RuntimeError('prime failed')
    Transport.publish, good = bad_publish, Transport.publish   # 4. priming publish
    try:
        assert start(ctx)[1] == {'CANCELLED'}
    finally:
        Transport.publish = good
    assert Transport.log[0].closes == 1 and RT.session is None and not sim_session._QUARANTINE
    ctx = reset()
    def fatal_publish(self, positions): raise FatalTransportError('prime fatal')
    Transport.publish = fatal_publish                          # 4b. priming publish (fatal)
    try:
        assert start(ctx)[1] == {'CANCELLED'}
    finally:
        Transport.publish = good
    assert Transport.log[0].closes == 0 and Transport.log[0] in [getattr(o, 'transport', None) for o in sim_session._QUARANTINE]
    ctx = reset()
    ctx.window_manager.event_timer_add = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no timer'))
    assert start(ctx)[1] == {'CANCELLED'}                      # 5. timer/modal registration
    assert Transport.log[0].closes == 1 and RT.session is None and ('removed', 'draw') in handles

@case
def recoverable_interop_falls_back_once(ctx):
    FACTORY.vulkan_error = RecoverableTransportError('external memory rejected')
    start(ctx)
    assert RT.session.label == ts.READBACK_LABEL and 'external memory rejected' in RT.interop_unavailable
    RT.stop(ctx)
    FACTORY.vulkan_error = None
    start(make_context())
    assert RT.session.label == ts.READBACK_LABEL and FACTORY.resolves == 1  # sticky until restart

@case
def fatal_tick_stops_without_cleanup(ctx):
    op, _ = start(ctx)
    session = RT.session
    session.transport.publish_error = FatalTransportError('CUDA_ERROR_LAUNCH_FAILED')
    assert op.modal(ctx, TIMER) == {'CANCELLED'}
    assert RT.restart_required == 'CUDA_ERROR_LAUNCH_FAILED' and session.transport.closes == 0
    assert session in sim_session._QUARANTINE and any(isinstance(o, Solver) for o in sim_session._QUARANTINE)

@case
def transactional_resize(ctx):
    op, _ = start(ctx)
    old = RT.session
    ctx.scene.warp_sph.particle_count = 2000
    FACTORY.open_error = RuntimeError('oom')
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'}  # keeps running on the old session
    assert RT.session is old and old.transport.closes == 0 and ctx.scene.warp_sph.particle_count == 1000
    assert RT.solver.steps == 0 and RT.step_count == 0  # no solver step in the failed tick
    FACTORY.open_error = None
    ctx.scene.warp_sph.particle_count = 2000
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'}
    assert RT.session is not old and old.transport.closes == 1 and RT.solver.num_particles == 2000

@case
def load_demo_while_running_uses_transaction(ctx):
    start(ctx)
    FACTORY.open_error = RuntimeError('oom')
    op = addon.WM_OT_warp_sph_load_demo()
    op.demo_id = 'BLOCK_FLOP_SPLASH'
    assert op.execute(ctx) == {'CANCELLED'} and ctx.scene.warp_sph.demo_scene == 'DOUBLE_DAM_PILLARS'

@case
def solver_step_error_is_fatal(ctx):
    op, _ = start(ctx)
    session = RT.session
    Solver.step_error = RuntimeError('CUDA error 700')  # Warp reports sticky errors as plain exceptions
    assert op.modal(ctx, TIMER) == {'CANCELLED'}
    assert 'solver step failed' in RT.restart_required and session in sim_session._QUARANTINE
    assert session.transport.closes == 0

@case
def graph_rebuild_error_is_fatal(ctx):
    op, _ = start(ctx)
    session, solver = RT.session, RT.solver
    Solver.rebuild_error = RuntimeError('graph capture failed')
    ctx.scene.warp_sph.stiffness = 2.0
    assert op.modal(ctx, TIMER) == {'CANCELLED'} and not RT.running and RT.session is None
    assert 'graph rebuild failed' in RT.restart_required and solver.steps == 0
    assert session in sim_session._QUARANTINE and solver in sim_session._QUARANTINE and session.transport.closes == 0

@case
def step_once_survives_backpressure(ctx):
    op, _ = start(ctx)
    RT.paused = RT.step_once = True
    RT.session.transport.ready = False  # both slots busy: the request waits for a free slot
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 0 and RT.step_once
    RT.session.transport.ready = True
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1 and not RT.step_once
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1  # paused again

@case
def redraw_only_on_new_frame(ctx):
    op, _ = start(ctx)
    area = ctx.screen.areas[0]
    before = area.redraws
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and area.redraws == before + 1  # new step
    RT.session.transport.ready = False  # step in flight: no new frame, no redraw contention
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and area.redraws == before + 1
    RT.session.transport.frame += 1     # the in-flight step became drawable on this tick's poll
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and area.redraws == before + 2

@case
def priming_failure_with_failed_close_quarantines_solver(ctx):
    def bad_publish(self, positions): raise RuntimeError('prime failed')
    def failed_close(self): self.closes += 1; self.close_result = 'failed'; return 'failed'
    publish, close = Transport.publish, Transport.close
    Transport.publish, Transport.close = bad_publish, failed_close
    try:
        assert start(ctx)[1] == {'CANCELLED'}
    finally:
        Transport.publish, Transport.close = publish, close
    assert RT.restart_required == 'fatal' and any(isinstance(o, Solver) for o in sim_session._QUARANTINE)

@case
def gl_resize_commits_solver_with_buffer(ctx):
    class GLBuffer:
        def __init__(self, n): self.n, self.destroys = n, 0
        def map_to_warp(self): return 'vbo'
        def unmap(self): pass
        @contextlib.contextmanager
        def mapped(self): yield 'vbo'
        def destroy(self):
            self.destroys += 1
            if self.n == 1000 and self.destroys == 1: raise RuntimeError('old VBO release failed')
    backend, gl_buffer = gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52
    gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52 = (lambda: 'OPENGL'), GLBuffer
    RUNTIME.wp.synchronize_device = lambda device: None
    try:
        op, _ = start(ctx)
        ctx.scene.warp_sph.particle_count = 2000
        assert op.modal(ctx, TIMER) == {'PASS_THROUGH'}  # failed after the commit point
        assert RT.gl_buf.n == RT.solver.num_particles == ctx.scene.warp_sph.particle_count == 2000
        assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1
    finally:
        RT.stop(ctx)
        gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52 = backend, gl_buffer

class GLBackend:
    """Run a case on the OpenGL backend with a fake legacy VBO (the real one needs a GL context)."""
    class Buffer:
        def __init__(self, n): self.n, self.destroys = n, 0
        def map_to_warp(self): return 'vbo'
        def unmap(self): pass
        @contextlib.contextmanager
        def mapped(self): yield 'vbo'
        def destroy(self): self.destroys += 1
    def __enter__(self):
        self.saved = gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52
        gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52 = (lambda: 'OPENGL'), self.Buffer
        RUNTIME.wp.synchronize_device = lambda device: None
    def __exit__(self, *exc):
        sim_session.stop_all(bpy.context)
        gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52 = self.saved

@case
def opengl_points_runs_through_gl_texture_session(ctx):
    ctx.scene.warp_sph.render_mode = 'POINTS'
    with GLBackend():
        op, result = start(ctx)
        assert result == {'RUNNING_MODAL'} and RT.gl_buf is None and RT.session.backend == 'OPENGL'
        assert RT.session.label == ts.GL_LABEL and FACTORY.resolves == 0 and not RT.interop_unavailable
        assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1
        # the route is fixed for the run: a mode flip plus a resize stays on the session
        ctx.scene.warp_sph.render_mode = 'SSFR_LIQUID'
        ctx.scene.warp_sph.particle_count = 2000
        old = RT.session
        assert op.modal(ctx, TIMER) == {'PASS_THROUGH'}
        assert RT.gl_buf is None and RT.session is not old and old.transport.closes == 1
        RT.stop(ctx)
        assert Transport.log[-1].closes == 1 and RT.session is None

@case
def opengl_ssfr_and_spheres_keep_the_legacy_vbo_path(ctx):
    with GLBackend():
        for mode in ('SSFR_LIQUID', 'SPHERES'):
            ctx = reset()
            ctx.scene.warp_sph.render_mode = mode
            op, result = start(ctx)
            assert result == {'RUNNING_MODAL'} and RT.session is None and RT.gl_buf is not None, mode
            assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1
            assert not Transport.log
            buf = RT.gl_buf
            RT.stop(ctx)
            assert buf.destroys == 1 and RT.gl_buf is None

@case
def opengl_texture_failure_falls_back_once_then_uses_legacy_vbo(ctx):
    FACTORY.gl_error = RecoverableTransportError('cudaGraphicsGLRegisterImage failed')
    ctx.scene.warp_sph.render_mode = 'POINTS'
    with GLBackend():
        start(ctx)
        assert RT.session.label == ts.READBACK_LABEL and RT.gl_buf is None
        assert 'OpenGL interop unavailable' in RT.interop_unavailable
        assert not RT.readback_active()                   # already on OpenGL: no "switch to OpenGL" offer
        RT.stop(ctx)
        FACTORY.gl_error = None
        ctx = bpy.context = make_context()
        ctx.scene.warp_sph.render_mode = 'POINTS'
        start(ctx)
        # no texture retry until restart; later Points runs use the legacy VBO interop, not Readback
        assert RT.session is None and RT.gl_buf is not None
        RT.stop(ctx)
        ctx.scene.warp_sph.render_mode = 'SSFR_LIQUID'        # legacy modes ignore the interop flag
        start(ctx)
        assert RT.session is None and RT.gl_buf is not None

@case
def opengl_texture_fatal_failure_requires_restart(ctx):
    FACTORY.open_error = FatalTransportError('cuda context lost')
    ctx.scene.warp_sph.render_mode = 'POINTS'
    with GLBackend():
        assert start(ctx)[1] == {'CANCELLED'} and RT.restart_required == 'cuda context lost'
        assert RT.gl_buf is None and RT.session is None

@case
def resize_never_downgrades_interop(ctx):
    op, _ = start(ctx)
    old = RT.session
    ctx.scene.warp_sph.particle_count = 2000
    FACTORY.vulkan_error = RecoverableTransportError('needs more VRAM')
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'}
    assert RT.session is old and old.transport.closes == 0 and ctx.scene.warp_sph.particle_count == 1000
    assert not RT.interop_unavailable and Transport.log[-1].closes == 1  # new Readback released
    assert 'Vulkan interop' in op.reports[-1][1]

@case
def resize_failed_old_close_quarantines_everything(ctx):
    op, _ = start(ctx)
    old, old_solver = RT.session, RT.solver
    old.transport.close_result = 'failed'
    ctx.scene.warp_sph.particle_count = 2000
    assert op.modal(ctx, TIMER) == {'CANCELLED'}
    assert RT.restart_required == 'fatal' and RT.session is None and RT.solver is None
    new = Transport.log[-1]
    assert new.closes == 0 and new in [getattr(o, 'transport', None) for o in sim_session._QUARANTINE]
    solvers = [o for o in sim_session._QUARANTINE if isinstance(o, Solver)]
    assert old_solver in solvers and any(o.num_particles == 2000 for o in solvers)

@case
def undo_and_redo_stop_the_run(ctx):
    for name in ('undo_pre', 'redo_pre'):
        ctx = reset()
        op, _ = start(ctx)
        session = RT.session
        for handler in getattr(bpy.app.handlers, name):
            handler(None)
        assert session.transport.closes == 1 and RT.session is None and RT.solver is None and not RT.running, name
        assert op.modal(ctx, TIMER) == {'CANCELLED'}
        for handler in getattr(bpy.app.handlers, name):  # idempotent when nothing is running
            handler(None)
        assert session.transport.closes == 1

def _capture_draw_log():
    import logging
    records = []
    class Capture(logging.Handler):
        def emit(self, record): records.append(record)
    logger = logging.getLogger(sim_session.__name__)
    handler = Capture()
    logger.addHandler(handler)
    return records, lambda: logger.removeHandler(handler)

@case
def draw_exception_reported_once_per_distinct_error(ctx):
    records, detach = _capture_draw_log()
    try:
        op, _ = start(ctx)
        calls = []
        def broken(settings): calls.append(1); raise RuntimeError(broken.message)
        broken.message = 'shader failed'
        RT.session.draw = broken
        for _ in range(5):                       # every redraw of every viewport
            RT.running = True
            RT.draw()     # must never raise into Blender
        assert len(calls) == 5 and len(records) == 1, len(records)
        assert not RT.running and RT.last_error == 'Viewport draw failed: shader failed'
        broken.message = 'different failure'
        RT.running = True
        RT.draw()
        assert len(records) == 2 and 'different failure' in records[1].getMessage()
        assert op.modal(ctx, TIMER) == {'CANCELLED'} and RT.session is None  # timer tears the session down
        start(make_context())                    # a new run reports the same error again
        RT.session.draw = broken
        broken.message = 'shader failed'
        RT.draw()
        assert len(records) == 3
    finally:
        detach()

@case
def legacy_gl_draw_exception_is_contained_and_reported_once(ctx):
    records, detach = _capture_draw_log()
    real_draw = RUNTIME.draw_particles_viewport
    try:
        RT.session, RT.gl_buf, RT.running = None, object(), True
        def broken(gl_buf, running): raise RuntimeError('GL draw failed')
        RUNTIME.draw_particles_viewport = broken
        for _ in range(3):
            RT.running = True
            RT.draw()
        assert len(records) == 1 and not RT.running and RT.last_error == 'Viewport draw failed: GL draw failed'
    finally:
        RUNTIME.draw_particles_viewport = real_draw
        RT.gl_buf = None
        detach()

@case
def fatal_draw_error_is_not_a_draw_report(ctx):
    records, detach = _capture_draw_log()
    try:
        start(ctx)
        def fatal(settings): raise FatalTransportError('device lost in draw')
        RT.session.draw = fatal
        RT.draw()
        assert RT.restart_required == 'device lost in draw' and not RT.running and not records
    finally:
        detach()

@case
def use_opengl_operator_is_confirmed_and_decided_by_what_runs(ctx):
    prefs = NS(system=NS(gpu_backend='VULKAN'))
    ctx.preferences = prefs
    saved = []
    save_result = [{'FINISHED'}]
    bpy.ops = NS(wm=NS(save_userpref=lambda: saved.append(prefs.system.gpu_backend) or save_result[0]))
    cls = addon.WM_OT_warp_sph_use_opengl
    assert cls.bl_idname == 'wm.warp_sph_use_opengl' and cls.bl_label == 'Use OpenGL fast mode'
    assert not cls.poll(ctx)                                  # nothing running
    start(ctx)
    assert RT.session.transport.capabilities().zero_copy and not cls.poll(ctx)   # zero-copy: no offer
    RT.stop(ctx)
    FACTORY.vulkan_error = RecoverableTransportError('external memory rejected')
    ctx = bpy.context = make_context(); ctx.preferences = prefs
    start(ctx)
    prefs.system.gpu_backend = 'OPENGL'                       # preference says OpenGL, Vulkan is running
    assert not RT.session.transport.capabilities().zero_copy and cls.poll(ctx)
    prefs.system.gpu_backend = 'VULKAN'
    op = cls()
    assert op.invoke(ctx, 'evt') == {'RUNNING_MODAL'} and ctx.window_manager.confirms == [(op, 'evt')]
    assert prefs.system.gpu_backend == 'VULKAN' and not saved  # nothing changes before confirmation
    assert op.execute(ctx) == {'FINISHED'}
    assert prefs.system.gpu_backend == 'OPENGL' and saved == ['OPENGL']
    assert 'restart' in op.reports[-1][1].lower()
    save_result[0] = {'CANCELLED'}                            # save refused: preference is put back
    prefs.system.gpu_backend = 'VULKAN'
    op = cls()
    assert op.execute(ctx) == {'CANCELLED'} and prefs.system.gpu_backend == 'VULKAN'
    assert op.reports[-1][0] == 'ERROR'
    def boom(): raise RuntimeError('disk full')
    bpy.ops.wm.save_userpref = boom
    op = cls()
    assert op.execute(ctx) == {'CANCELLED'} and prefs.system.gpu_backend == 'VULKAN'
    assert 'disk full' in op.reports[-1][1]

@case
def panel_offers_opengl_only_for_running_readback(ctx):
    class Layout:
        def __init__(self, ops): self.ops = ops
        def operator(self, idname, **kw): self.ops.append(idname); return NS()
        def __getattr__(self, name): return lambda *a, **k: self
    def panel_ops():
        ops = []
        panel = addon.VIEW3D_PT_warp_sph()
        panel.layout = Layout(ops)
        for k, v in dict(render_mode='POINTS', depth_test=True, point_size=2.0,
                         particle_radius=0.02, particle_color=(1, 1, 1, 1)).items():
            setattr(ctx.scene.warp_sph, k, v)
        panel.draw(ctx)
        return ops
    assert 'wm.warp_sph_use_opengl' not in panel_ops()        # idle
    start(ctx)
    assert 'wm.warp_sph_use_opengl' not in panel_ops()        # zero-copy
    RT.stop(ctx)
    FACTORY.vulkan_error = RecoverableTransportError('external memory rejected')
    start(ctx)
    assert 'wm.warp_sph_use_opengl' in panel_ops()            # Readback running

@case
def custom_start_builds_solver_from_description(ctx):
    ctx.scene.warp_sph.scene_source = 'CUSTOM'
    ctx.scene.warp_sph.particle_spacing = 0.025
    op, result = start(ctx)
    assert result == {'RUNNING_MODAL'} and RT.running and RT.active_demo == addon.CUSTOM_SCENE, result
    assert DESC.calls == [(ctx.scene, 'depsgraph')]                     # evaluated depsgraph
    assert Solver.from_scene_calls == [('desc', 0.025, 2, 1000.0, 0.4)]  # spacing + speed; stiffness x1000
    assert RT.solver.num_particles == 777 and FACTORY.counts == [777]  # output sized from the fill
    assert HELPERS == ['removed']                                       # no demo wireframes
    session = RT.session
    assert ctx.scene.warp_sph.particle_count == 1000                    # demo slider is not the count
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1 and RT.session is session
    assert ctx.window_manager.steps == [0.001]
    ctx.scene.warp_sph.scene_source = 'DEMO'   # a source change while running applies on the next Start
    assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 2 and RT.session is session

@case
def custom_scene_errors_are_reported_and_leave_nothing_behind(ctx):
    for kind, message in (('validation', 'exactly one DOMAIN object is required; found 0 (none)'),
                          ('fill', 'fluid volumes contain no particles at spacing 0.02')):
        ctx = reset()
        ctx.scene.warp_sph.scene_source = 'CUSTOM'
        if kind == 'validation':
            DESC.error = SceneValidationError(message)
        else:
            Solver.fill_error = SceneFillError(message)
        op, result = start(ctx)
        assert result == {'CANCELLED'} and op.reports[-1][0] == 'ERROR', (kind, result)
        assert 'cannot start this scene' in op.reports[-1][1] and message in op.reports[-1][1]
        assert RT.last_error == 'Scene not valid: ' + message
        assert RT.session is None and RT.solver is None and not RT.running and RT.timer is None
        assert not Transport.log and not sim_session._QUARANTINE and not RT.restart_required and not handles
        assert not ctx.window_manager.timers and not ctx.window_manager.modals
    ctx = reset()
    ctx.scene.warp_sph.scene_source = 'CUSTOM'
    op, result = start(ctx)                    # the next valid Start clears the message
    assert result == {'RUNNING_MODAL'} and RT.last_error == ''

@case
def custom_start_on_legacy_gl_sizes_vbo_from_fill(ctx):
    made = []
    class GLBuffer:
        def __init__(self, n): self.n, self.destroys = n, 0; made.append(self)
        @contextlib.contextmanager
        def mapped(self): yield 'vbo'
        def destroy(self): self.destroys += 1
    backend, gl_buffer = gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52
    gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52 = (lambda: 'OPENGL'), GLBuffer
    RUNTIME.wp.synchronize_device = lambda device: None
    try:
        ctx.scene.warp_sph.scene_source = 'CUSTOM'
        op, result = start(ctx)
        assert result == {'RUNNING_MODAL'} and RT.session is None
        assert RT.gl_buf.n == RT.solver.num_particles == 777
        assert op.modal(ctx, TIMER) == {'PASS_THROUGH'} and RT.solver.steps == 1 and len(made) == 1
        RT.stop(ctx)
        assert made[0].destroys == 1
        DESC.error = SceneValidationError('fluid volume extends outside the domain')
        ctx = make_context()
        ctx.scene.warp_sph.scene_source = 'CUSTOM'
        op, result = start(ctx)
        assert result == {'CANCELLED'} and len(made) == 1 and RT.gl_buf is None  # no VBO for a bad scene
    finally:
        RT.stop(ctx)
        gpu.platform.backend_type_get, RUNTIME.ZeroCopyGLBuffer52 = backend, gl_buffer

@case
def demo_start_never_reads_tagged_objects(ctx):
    op, result = start(ctx)
    assert result == {'RUNNING_MODAL'} and RT.active_demo == 'DOUBLE_DAM_PILLARS'
    assert not DESC.calls and not Solver.from_scene_calls and FACTORY.counts == [1000]
    assert HELPERS == ['built:DOUBLE_DAM_PILLARS']

@case
def load_demo_while_custom_runs(ctx):
    ctx.scene.warp_sph.scene_source = 'CUSTOM'
    start(ctx)
    FACTORY.open_error = RuntimeError('oom')
    op = addon.WM_OT_warp_sph_load_demo()
    op.demo_id = 'BLOCK_FLOP_SPLASH'
    assert op.execute(ctx) == {'CANCELLED'}
    assert ctx.scene.warp_sph.scene_source == 'CUSTOM' and RT.active_demo == addon.CUSTOM_SCENE  # kept running
    assert ctx.scene.warp_sph.particle_count == 1000
    FACTORY.open_error = None
    assert op.execute(ctx) == {'FINISHED'}
    assert ctx.scene.warp_sph.scene_source == 'DEMO' and RT.active_demo == 'BLOCK_FLOP_SPLASH'
    assert RT.solver.num_particles == 1000

@case
def sessions_are_per_scene_and_one_runs_at_a_time(ctx):
    op, result = start(ctx)
    first = CUR[0]
    assert result == {'RUNNING_MODAL'} and sim_session.active() is first and sim_session.get(ctx.scene) is first
    other_ctx = make_context()
    other = sim_session.get(other_ctx.scene)
    assert other is not first and not other.running and other.solver is None and other.scene is other_ctx.scene
    assert sim_session.get(other_ctx.scene) is other                       # one session per scene
    second = addon.WM_OT_warp_sph_start()
    assert second.execute(other_ctx) == {'CANCELLED'} and 'already running' in second.reports[-1][1]
    assert not other.running and len(Transport.log) == 1 and not other_ctx.window_manager.timers
    assert addon.WM_OT_warp_sph_pause.poll(ctx) and not addon.WM_OT_warp_sph_pause.poll(other_ctx)
    addon.WM_OT_warp_sph_stop().execute(other_ctx)                       # stopping another scene: no effect
    assert first.running and op.modal(ctx, TIMER) == {'PASS_THROUGH'}
    addon.WM_OT_warp_sph_stop().execute(ctx)
    assert op.modal(ctx, TIMER) == {'CANCELLED'} and sim_session.active() is None
    second, result = start(other_ctx)
    assert result == {'RUNNING_MODAL'} and sim_session.active() is other and other.solver.steps == 0
    assert other_ctx.window_manager.steps == [0.001]
    assert op.modal(ctx, TIMER) == {'CANCELLED'}                          # the first run's modal stays orphaned
    assert second.modal(other_ctx, TIMER) == {'PASS_THROUGH'} and other.solver.steps == 1

@case
def dispose_releases_and_forgets_the_session(ctx):
    op, _ = start(ctx)
    session = CUR[0]
    transport = session.session.transport
    session.dispose(ctx)
    assert transport.closes == 1 and not session.running and session.session is None and session.solver is None
    assert session.scene is None and session not in sim_session.sessions() and session.timer is None
    assert ('removed', 'draw') in handles and op.modal(ctx, TIMER) == {'CANCELLED'}
    session.dispose(ctx)                                                  # idempotent
    assert transport.closes == 1
    fresh = sim_session.get(ctx.scene)
    assert fresh is not session and not fresh.running and fresh.scene is ctx.scene

@case
def removed_scene_stops_the_run(ctx):
    op, _ = start(ctx)
    transport = RT.session.transport
    class Freed:
        def __getattr__(self, name): raise ReferenceError('StructRNA of type Scene has been removed')
    CUR[0].scene = Freed()
    assert op.modal(ctx, TIMER) == {'CANCELLED'} and not RT.running and RT.session is None
    assert transport.closes == 1 and 'scene was removed' in op.reports[-1][1]
    assert sim_session.get(ctx.scene) is not CUR[0]                       # the stale session is replaced
    freed = sim_session.get(make_context().scene)                         # e.g. a scene freed by a file load
    freed.scene = Freed()
    sim_session.get(make_context().scene)
    assert freed not in sim_session.sessions() and CUR[0] not in sim_session.sessions()  # idle dead ones pruned

class FakeJob:
    """Stands in for cache.bake.BakeJob: the session only drives run_chunk and reads progress (P2.2)."""
    def __init__(self, finish_after=3):
        self.state, self.chunks, self.frames_done, self.error = 'RUNNING', 0, 0, ''
        self.finish_after, self.closes = finish_after, []
        self.plan = NS(first=1, total=10)
        self.store = NS(close=lambda: self.closes.append(1))
    @property
    def active(self): return self.state in ('RUNNING', 'CANCELLING', 'FINISHING')
    def run_chunk(self, budget):
        self.chunks += 1
        if self.state == 'CANCELLING':
            self.state = 'CANCELLED'
            return False
        self.frames_done += 1
        if self.frames_done == self.finish_after: self.state = 'DONE'
        return True
    def cancel(self):
        if self.state == 'RUNNING': self.state = 'CANCELLING'

@case
def bake_ticks_drive_the_job_not_the_solver(ctx):
    op, _ = start(ctx)
    job, steps, shown = FakeJob(), RT.solver.steps, RT.session.transport.frame
    RT.bake = job
    for _ in range(3):
        assert op.modal(ctx, TIMER) == {'PASS_THROUGH'}
    assert job.chunks == 3 and RT.solver.steps == steps          # the live step is replaced, never added
    assert RT.session.transport.frame > shown                    # the viewport follows the bake
    assert RT.bake is None and RT.paused and RT.bake_result == 'Baked frames 1-3 (3 frames)', RT.bake_result
    assert job.closes == [1] and 'Baked' in op.reports[-1][1]    # writer drained before the job is dropped

@case
def bake_cancel_and_stop_leave_no_job(ctx):
    op, _ = start(ctx)
    job = FakeJob(finish_after=99)
    RT.bake = job
    op.modal(ctx, TIMER)
    RT.cancel_bake()
    assert job.state == 'CANCELLING' and RT.bake is job          # cancel is a request; the tick finishes it
    op.modal(ctx, TIMER)
    assert RT.bake is None and 'cancelled' in RT.bake_result and job.closes == [1]
    again = FakeJob(finish_after=99)
    RT.bake = again
    RT.stop(ctx)                                                 # stop mid-bake: drained, dropped
    assert RT.bake is None and again.closes == [1] and not RT.running and 'stopped' in RT.bake_result

@case
def bake_step_failure_is_fatal_and_stops(ctx):
    op, _ = start(ctx)
    job = FakeJob()
    def boom(budget): raise RuntimeError('cuda error')
    job.run_chunk = boom
    RT.bake = job
    assert op.modal(ctx, TIMER) == {'CANCELLED'} and not RT.running and RT.bake is None
    assert 'cuda error' in RT.restart_required and job in sim_session._QUARANTINE and job.closes == [1]

@case
def bake_needs_a_running_session(ctx):
    reports = []
    ok = sim_session.get(ctx.scene).begin_bake(ctx, lambda kind, msg: reports.append(msg), first=1, last=4, resume=False)
    assert ok is False and reports == ['Start the simulation before baking']

reset()
ext.unregister()

import importlib
from warp_sph_realtime.render import transport
RT.restart_required = 'persist'
sentinel = object()
sim_session._QUARANTINE.append(sentinel)
importlib.reload(transport)
reloaded_session = importlib.reload(sim_session)  # add-on disable/enable reloads the modules
reloaded = importlib.reload(addon)
assert reloaded.sim_session is reloaded_session and not reloaded_session.sessions()
assert reloaded_session.SimSession().restart_required == 'persist' and sentinel in reloaded_session._QUARANTINE
assert reloaded_session._QUARANTINE is transport.process_state().quarantine
print('LIFECYCLE_OK')
'''


class AddonLifecycleTest(unittest.TestCase):
    def test_lifecycle_with_fake_transport(self):
        # Script on stdin: it outgrew the Windows command-line limit for `-c`.
        result = subprocess.run([sys.executable, '-'], input=LIFECYCLE, cwd=str(ROOT),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('LIFECYCLE_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
