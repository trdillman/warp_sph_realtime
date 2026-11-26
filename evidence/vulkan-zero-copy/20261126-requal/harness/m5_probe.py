"""M5 in-Blender probe: one generator driven by a persistent bpy.app.timers callback.

Runs against the installed, frozen extension package only (bl_ext.user_default.warp_sph_realtime).
Vulkan lane: install, parity, 4x50 lifecycle cycles, fault matrix, benchmarks, mocked fatal.
OpenGL lane: install, Readback parity images, Readback benchmarks. Results -> $M5_OUT.
Instrumentation wraps the real native seam/ops; it never replaces product logic. Mocked
faults prove classification only (real fatal-GPU injection is not authorized).
"""
import ctypes as c
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import addon_utils
import bpy
import gpu
import numpy as np
from mathutils import Matrix, Quaternion

LANE = os.environ['M5_LANE']
OUT = Path(os.environ['M5_OUT'])
HERE = Path(__file__).resolve().parent
PKG = 'bl_ext.user_default.warp_sph_realtime'
WHEEL = Path(r'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\snapshot_pre_clean_20261003'
             r'\fluidSim_warp_blender51\wheels\warp_lang-1.17.0-py3-none-win_amd64.whl')
PARITY_COUNTS = (1, 4095, 4096, 4097, 4194304)
BENCH_COUNTS = (500_000, 1_000_000, 2_000_000, 4_194_304)
WARMUP, SAMPLES, CASE_SECONDS = 300, 3000, 240
CYCLES = 50
if os.environ.get('M5_SMOKE'):  # pre-freeze smoke of the call pattern only; counted runs never set it
    PARITY_COUNTS, BENCH_COUNTS = (1, 4097), (500_000,)
    WARMUP, SAMPLES, CASE_SECONDS, CYCLES = 30, 120, 60, 3
OFF = 512
OWNED = ('VulkanInteropTransport', 'ReadbackTransport', 'TransportSession', 'RealtimeWCSPHSolver', 'PointRenderer')
LOG = {'lane': LANE, 'status': 'RUNNING', 'steps': []}
DRAW = {'fn': None, 'error': None}


def mod(name=''):
    return importlib.import_module(PKG + name)


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1, default=str), encoding='utf-8')


def progress(step, **detail):
    save('progress.json', {'step': step, 't': time.time(), **detail})


def window():
    return bpy.context.window_manager.windows[0]


def view3d():
    win = window()
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return win, area, region


def override():
    win, area, region = view3d()
    return bpy.context.temp_override(window=win, area=area, region=region)


def redraw():
    for area in window().screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def wait_for(cond, timeout, what):
    start = time.monotonic()
    while not cond():
        if time.monotonic() - start > timeout:
            raise AssertionError(f'timeout after {timeout}s: {what}')
        redraw()
        yield 0.005


def in_draw(fn, timeout=30.0, what='draw'):
    """Run fn() inside the next POST_VIEW draws until it returns a non-None value."""
    box = {}

    def hook():
        value = fn()
        if value is not None:
            box['value'] = value
            DRAW['fn'] = None
    DRAW['fn'] = hook
    yield from wait_for(lambda: 'value' in box or DRAW['error'], timeout, what)
    if DRAW['error']:
        error, DRAW['error'] = DRAW['error'], None
        raise RuntimeError('draw hook failed:\n' + error)
    return box['value']


def draw_handler():
    fn = DRAW['fn']
    if fn is None:
        return
    try:
        fn()
    except Exception:
        DRAW['fn'], DRAW['error'] = None, traceback.format_exc()


def golden(n):
    i = np.arange(n, dtype=np.float64)
    pts = np.stack([(i * a) % 1.0 for a in (0.6180339887498949, 0.7548776662466927, 0.5698402909980532)], 1)
    return (pts * 2.0 - 1.0).astype(np.float32)


def padded(points, width, height):
    out = np.zeros((width * height, 4), dtype=np.float32)
    out[:len(points), :3] = points
    out[:len(points), 3] = 1.0
    return out


def read_rgba(texture, texels):
    data = texture.read()
    data.dimensions = texels * 4
    return np.asarray(data, dtype=np.float32).reshape(-1, 4)


def fixed_matrices():
    f = 1.0 / np.tan(np.radians(50.0) / 2)
    near, far = 0.1, 20.0
    proj = Matrix(((f, 0, 0, 0), (0, f, 0, 0), (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
                   (0, 0, -1, 0)))
    view = Matrix.Translation((0, 0, -4.0)) @ Quaternion((0.88, 0.32, 0.25, 0.24)).normalized().to_matrix().to_4x4()
    return proj, view


def offscreen_image(renderer, frame, settings):
    off = gpu.types.GPUOffScreen(OFF, OFF)
    try:
        proj, view = fixed_matrices()
        with off.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(view)
                gpu.matrix.load_projection_matrix(proj)
                renderer.draw(frame, settings)
        data = off.texture_color.read()
        data.dimensions = OFF * OFF * 4
        pixels = np.asarray(data)
        if pixels.dtype.kind == 'f':
            pixels = np.clip(np.rint(pixels * 255.0), 0, 255)
        return pixels.astype(np.uint8).reshape(OFF, OFF, 4)
    finally:
        off.free()


def gpu_mem():
    import warp as wp
    device = wp.get_device('cuda:0')
    free = device.free_memory() if callable(device.free_memory) else device.free_memory
    total = device.total_memory() if callable(device.total_memory) else device.total_memory
    return int(total - free), int(total)


def handle_count():
    k32 = c.WinDLL('kernel32')
    count = c.c_ulong()
    k32.GetProcessHandleCount(c.c_void_p(k32.GetCurrentProcess()), c.byref(count))
    return count.value


class Counting:
    """Real native seam behind a call/ownership ledger with optional fault injection."""

    REAL = object()

    def __init__(self, real, faults=None):
        self._real, self.faults, self.calls, self.ledger = real, dict(faults or {}), Counter(), Counter()
        self.armed = True

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr) or name.startswith('_'):
            return attr

        def call(*args):
            self.calls[name] += 1
            fault = self.faults.get(name) if self.armed else None
            result = fault(self.calls[name], *args) if fault else self.REAL
            if result is self.REAL:
                result = attr(*args)
            self._account(name, args, result)
            return result
        return call

    def _account(self, name, args, result):
        led = self.ledger
        # Evaluated only for the acquiring call: other results (e.g. read_texture arrays) have no truth value.
        ok = {'pb_create': lambda: bool(result), 'fence_create': lambda: bool(result),
              'import_memory': lambda: isinstance(result, tuple) and result[0] == 0 and result[1],
              'map_buffer': lambda: isinstance(result, tuple) and result[0] == 0,
              'event_create': lambda: isinstance(result, tuple) and result[0] == 0 and result[1],
              'pb_export': lambda: isinstance(result, tuple) and bool(result[0])}
        if name in ok and ok[name]():
            led[name] += 1
        if name in ('pb_free', 'fence_free', 'close_handle') or \
                name in ('destroy_external', 'mem_free', 'event_destroy') and result == 0:
            led[name] += 1

    def balanced(self):
        led = self.ledger
        pairs = (('pb_create', 'pb_free'), ('fence_create', 'fence_free'), ('import_memory', 'destroy_external'),
                 ('map_buffer', 'mem_free'), ('event_create', 'event_destroy'), ('pb_export', 'close_handle'))
        return all(led[a] == led[b] for a, b in pairs), dict(led)


def resolve(**overrides):
    bs = mod('.render.blender_symbols')
    k32 = c.WinDLL('kernel32')
    k32.GetModuleHandleW.restype = c.c_void_p
    buffer = c.create_unicode_buffer(32768)
    k32.GetModuleFileNameW(None, buffer, len(buffer))
    exe = Path(buffer.value)
    kwargs = dict(module_base=k32.GetModuleHandleW(None), build_hash=bpy.app.build_hash.decode('ascii'),
                  python_abi=f'{sys.version_info.major}.{sys.version_info.minor}')
    pdb = overrides.pop('pdb', exe.with_suffix('.pdb'))
    kwargs.update(overrides)
    return bs.resolve_symbols(exe, pdb, **kwargs)


# -- steps -------------------------------------------------------------------------------------
def step_setup():
    progress('setup')
    if not addon_utils.check(PKG)[1]:
        addon_utils.enable(PKG, default_set=True, handle_error=None)
    installer = mod('.warp_installer')
    info = installer.get_wheel_info_for_platform()
    assert WHEEL.stat().st_size == info.size, 'local wheel differs from lock size'
    requested = []

    class Response:
        def __init__(self, url):
            self.url, self.file = url, WHEEL.open('rb')

        def geturl(self):
            return self.url  # same locked HTTPS URL: the product host/size/sha checks still run

        def read(self, n=-1):
            return self.file.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.file.close()

    def fake_urlopen(url, timeout=None):
        requested.append(url)
        return Response(url)
    installer.urlopen = fake_urlopen
    with override():
        result = bpy.ops.wm.warp_sph_install()
    assert result == {'FINISHED'}, result
    # Today's operator installs on a worker thread (v4's was synchronous): wait for it, then require the owned copy.
    yield from wait_for(lambda: not installer.install_running(), 900, 'warp install worker')
    owned = installer.owned_install_status()
    assert owned, 'Warp install finished without an owned install'
    addon = mod('.addon')
    addon._activate_runtime()
    import warp as wp
    spec = importlib.util.spec_from_file_location('m5_kernels', HERE / 'm5_kernels.py')
    m5k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5k)
    sys.modules['m5_kernels'] = m5k
    win, area, region = view3d()
    r3d = area.spaces.active.region_3d
    r3d.view_perspective, r3d.view_location, r3d.view_distance = 'PERSP', (0, 0, 0), 4.0
    r3d.view_rotation = Quaternion((0.88, 0.32, 0.25, 0.24)).normalized()
    bpy.context.scene.warp_sph.particle_count = 100_000  # today's settings group (v4: flat scene props)
    bpy.context.scene.warp_sph.render_mode = 'POINTS'
    assert not bpy.context.scene.warp_sph.use_shared_sdk  # the product transport path, as v4
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / 'cycle.blend'), copy=True)
    identity = {'blender': bpy.app.version_string, 'build_hash': bpy.app.build_hash.decode(),
                'gpu_backend': gpu.platform.backend_type_get(), 'gpu_renderer': gpu.platform.renderer_get(),
                'gpu_version': gpu.platform.version_get(), 'python': sys.version, 'warp': wp.__version__,
                'warp_file': wp.__file__, 'device': wp.get_device('cuda:0').name,
                'install_message': 'FINISHED', 'owned_install': owned, 'wheel_requests': requested, 'wheel_lock': vars(info),
                'addon_file': mod('.addon').__file__, 'window': [win.width, win.height],
                'region': [region.width, region.height]}
    save('identity.json', identity)
    LOG['steps'].append('setup')
    yield 0.1


def step_parity():
    transport = mod('.render.transport')
    pts = mod('.render.gpu_points')
    import warp as wp
    renderer = pts.PointRenderer()
    rows, images = [], {}
    modes = {'points': pts.DrawSettings(pts.POINTS, 3.0, (0.12, 0.68, 1.0, 1.0), True),
             'spheres': pts.DrawSettings(pts.SPHERES, 0.02, (0.12, 0.68, 1.0, 1.0), True)}
    kinds = ['readback'] + (['interop'] if LANE == 'vulkan' else [])
    for n in PARITY_COUNTS:
        points = golden(n)
        source = wp.array(points, dtype=wp.vec3, device='cuda:0')
        w, h = transport.dimensions(n)
        expected = padded(points, w, h)
        for kind in kinds:
            progress('parity', count=n, kind=kind)
            if kind == 'interop':
                tv = mod('.render.transport_vulkan')
                adapter = tv.create(mod('.render.blender_symbols').resolve_current_blender().resolved, n,
                                    window=window().as_pointer())
            else:
                adapter = transport.ReadbackTransport(n)
            assert adapter.publish(source)

            def check():
                frame = adapter.positions_texture()
                if frame is None or frame.frame != 0:
                    return None
                observed = read_rgba(frame.texture, w * h)
                result = {'count': n, 'kind': kind, 'width': w, 'height': h, 'frame_count': frame.count,
                          'exact': bool(np.array_equal(observed, expected)),
                          'max_abs': float(np.abs(observed - expected).max()),
                          'padding_zero': bool(not observed[n:].any()),
                          'sha256': hashlib.sha256(observed.tobytes()).hexdigest()}
                for mode, settings in modes.items():
                    images[f'{kind}_{mode}_{n}'] = offscreen_image(renderer, frame, settings)
                return result
            row = yield from in_draw(check, 60, f'parity {kind} {n}')
            row['close'] = adapter.close()
            row['capabilities'] = vars(adapter.capabilities())
            rows.append(row)
            assert row['exact'] and row['padding_zero'] and row['frame_count'] == n and row['close'] == 'completed', row
        del source
    renderer.free()
    np.savez_compressed(OUT / 'parity-images.npz', **images)
    save('parity.json', rows)
    LOG['steps'].append('parity')


# Today's add-on keeps one sim_session.SimSession per scene (v4: one global addon.RT). RAN is the session the
# latest cycle started; its last_close is the close result of that cycle, as RT.last_close was in v4.
RAN = {'s': None}


def sim():
    return mod('.sim_session').get(bpy.context.scene)


def ran():
    return RAN['s'] if RAN['s'] is not None else sim()


def snapshot(kind, index, extra=None):
    gc.collect()
    counts = Counter(type(o).__name__ for o in gc.get_objects())
    import warp as wp
    rt, live = ran(), sim()
    state = mod('.render.transport').process_state()
    used, total = gpu_mem()
    row = {'kind': kind, 'cycle': index, 't': time.time(), 'owned': {k: counts.get(k, 0) for k in OWNED},
           'warp_arrays': sum(v for k, v in counts.items() if k == 'array'),
           'quarantine': len(mod('.render.transport').process_state().quarantine),
           'last_close': rt.last_close, 'restart_required': state.restart_required,
           'interop_unavailable': state.interop_unavailable, 'handles': handle_count(),
           'vram_used': used, 'vram_total': total, 'timer': live.timer is not None,
           'draw_handle': bool(live.draw_handle), 'session': live.session is not None,
           'warp_device': wp.get_device('cuda:0').name}
    row.update(extra or {})
    return row


def running_interop(min_uploads=3, previous=None):
    rt = sim()
    s = rt.session
    return (s is not None and s is not previous and rt.running and s.transport.capabilities().zero_copy
            and getattr(s.transport, 'uploads', 0) >= min_uploads)


def stopped():
    ss = mod('.sim_session')
    return ss.active() is None and all(s.session is None and s.timer is None and not s.draw_handle and not s.running
                                       for s in ss.sessions())


def start_sim():
    with override():
        result = bpy.ops.wm.warp_sph_start()
    assert result == {'RUNNING_MODAL'}, (result, sim().last_error, mod('.render.transport').process_state().restart_required)
    RAN['s'] = sim()
    # Today's live run is paced by the timeline and starts paused (v4 stepped on every timer event): press Play
    # through the product's Pause/Resume operator, as a user does after Start.
    with override():
        if not bpy.context.screen.is_animation_playing:
            assert bpy.ops.wm.warp_sph_pause() == {'FINISHED'}
    try:
        yield from wait_for(running_interop, 60, 'interop session with 3 uploads')
    except AssertionError as exc:  # diagnostic detail only; the wait condition is unchanged
        raise AssertionError(f'{exc}; state={sim_state()}') from None


def sim_state():
    s = sim()
    t = s.session.transport if s.session is not None else None
    return {'running': s.running, 'paused': s.paused, 'timer': s.timer is not None, 'solver': getattr(s.solver, 'num_particles', None),
            'session': None if s.session is None else {'label': s.session.label, 'reason': s.session.reason,
                                                       'transport': type(t).__name__, 'caps': vars(t.capabilities()),
                                                       'uploads': getattr(t, 'uploads', None)},
            'gl_buf': s.gl_buf is not None, 'last_error': s.last_error, 'step_count': s.step_count,
            'interop_unavailable': mod('.render.transport').process_state().interop_unavailable,
            'restart_required': mod('.render.transport').process_state().restart_required}


def stop_sim():
    with override():
        bpy.ops.wm.warp_sph_stop()
        if bpy.context.screen.is_animation_playing:  # leave the timeline as Start found it
            bpy.ops.screen.animation_cancel(restore_frame=False)
    yield from wait_for(stopped, 30, 'session teardown')


def slot_states():
    s = sim().session
    return [slot.state for slot in getattr(s.transport, '_slots', [])] if s else []


def step_cycles():
    rows = [snapshot('baseline', 0)]
    scene = bpy.context.scene
    for kind in ('start_stop', 'resize', 'disable', 'file_load'):
        if kind == 'resize':
            yield from start_sim()
        for i in range(1, CYCLES + 1):
            progress('cycles', kind=kind, cycle=i)
            extra = {}
            if kind == 'start_stop':
                yield from start_sim()
                yield from stop_sim()
            elif kind == 'resize':
                previous = sim().session
                target = 200_000 if scene.warp_sph.particle_count == 100_000 else 100_000
                scene.warp_sph.particle_count = target
                yield from wait_for(lambda: running_interop(3, previous) and
                                    sim().solver.num_particles == target, 60, f'resize to {target}')
                previous = None  # the harness must not keep the closed session alive
                extra = {'target': target, 'old_close': sim().last_close}
                assert extra['old_close'] == 'completed', extra
            elif kind == 'disable':
                yield from start_sim()
                extra = {'slots_before': slot_states()}
                addon_utils.disable(PKG, default_set=True)
                extra['close_after_disable'] = ran().last_close
                assert stopped(), 'disable left a live session'
                addon_utils.enable(PKG, default_set=True, handle_error=None)
                assert addon_utils.check(PKG)[1], 're-enable failed'
            else:
                yield from start_sim()
                extra = {'slots_before': slot_states()}
                with override():
                    bpy.ops.wm.open_mainfile(filepath=str(OUT / 'cycle.blend'), load_ui=False)
                yield 0.05
                yield from wait_for(stopped, 30, 'file-load teardown')
                pump()
                extra['close_after_load'] = ran().last_close
            yield 0.05
            rows.append(snapshot(kind, i, extra))
            save('cycles.json', rows)
        if kind == 'resize':
            yield from stop_sim()
            rows.append(snapshot('resize_stop', CYCLES + 1))
    rows.append(snapshot('final', 0))
    save('cycles.json', rows)
    LOG['steps'].append('cycles')


def step_faults():
    import warp as wp
    S = mod('.render.session')
    T = mod('.render.transport')
    tv = mod('.render.transport_vulkan')
    bs = mod('.render.blender_symbols')
    profile = bs.load_profile()
    names = {key: s['name'] for key, s in profile['symbols'].items()}
    win = window().as_pointer()
    rows = []
    cache = OUT / 'symbol-cache.json'

    def run_case(name, resolution_fn, faults=None, expect_interop=False, expect=''):
        progress('faults', case=name)
        made = {'vulkan': 0, 'readback': 0}
        proxies = []

        def create_vulkan(resolved, count, window):
            made['vulkan'] += 1
            proxy = Counting(tv.BlenderVulkanNative(resolved), faults)
            proxies.append(proxy)
            return tv.VulkanInteropTransport(proxy, tv.WarpOps(), count, window=window)

        def create_readback(count, reason=''):
            made['readback'] += 1
            return T.ReadbackTransport(count, reason=reason)
        quarantine = len(T.process_state().quarantine)
        resolution = resolution_fn()
        session = S.open_session('VULKAN', 4097, win, resolve=lambda: resolution, create_vulkan=create_vulkan,
                                 create_readback=create_readback)
        session.tick(lambda out: wp.copy(out, wp.array(golden(4097), dtype=wp.vec3, device='cuda:0')))
        row = {'case': name, 'resolution': [resolution.available, resolution.reason, resolution.detail],
               'made': made, 'label': session.label, 'reason': session.reason}
        row['close'] = session.close()  # the ledger must balance after close, also for live interop cases
        row.update(calls=[dict(p.calls) for p in proxies], ledger=[p.balanced() for p in proxies])
        row['quarantine_delta'] = len(T.process_state().quarantine) - quarantine
        row['balanced'] = all(b for b, _ in row['ledger'])
        if expect_interop:
            row['pass'] = made == {'vulkan': 1, 'readback': 0} and session.label == S.VULKAN_LABEL
        else:
            row['pass'] = made['readback'] == 1 and session.label == S.READBACK_LABEL and bool(session.reason) \
                and expect in session.reason
        row['pass'] = bool(row['pass'] and row['close'] == 'completed' and row['quarantine_delta'] == 0
                           and row['balanced'])
        rows.append(row)
        save('faults.json', rows)
        yield 0.05

    wrong = {s['name']: s['rva'] for s in profile['symbols'].values()}
    first = next(iter(wrong))
    wrong[first] += 16

    def wrong_helper(request, timeout):
        return {'schema_version': 1, 'symbols': dict(wrong)}

    yield from run_case('unknown_abi', lambda: resolve(python_abi='3.12'), expect='unknown_profile')
    yield from run_case('missing_pdb', lambda: resolve(verify_debug_symbols=True, pdb=OUT / 'absent.pdb'),
                        expect='pdb_missing')
    yield from run_case('wrong_rva_helper', lambda: resolve(verify_debug_symbols=True, helper_runner=wrong_helper,
                                                            cache_path=OUT / 'unused-cache.json'),
                        expect='helper_result')
    # Cache faults recover through exactly one real isolated helper resolution (no fallback).
    yield from run_case('cache_create', lambda: resolve(verify_debug_symbols=True, cache_path=cache),
                        expect_interop=True)
    valid = json.loads(cache.read_text())
    mutations = {'stale_cache': {**valid, 'identity': {**valid['identity'], 'build_hash': '000000000000'}},
                 'wrong_rva_cache': {**valid, 'rvas': {**valid['rvas'], next(iter(names)):
                                                       valid['rvas'][next(iter(names))] + 16}}}
    sources = {}
    for name, payload in mutations.items():
        cache.write_text(json.dumps(payload))
        yield from run_case(name, lambda: resolve(verify_debug_symbols=True, cache_path=cache), expect_interop=True)
        sources[name] = rows[-1]['resolution'][2]
    cache.write_bytes(b'\x00garbage{')
    yield from run_case('corrupt_cache', lambda: resolve(verify_debug_symbols=True, cache_path=cache),
                        expect_interop=True)
    sources['corrupt_cache'] = rows[-1]['resolution'][2]
    yield from run_case('valid_cache', lambda: resolve(verify_debug_symbols=True, cache_path=cache),
                        expect_interop=True)
    sources['valid_cache'] = rows[-1]['resolution'][2]
    real = resolve()
    yield from run_case('uuid_mismatch', lambda: real, {'cuda_uuid': lambda k, *a: (0, bytes(16))},
                        expect='device UUID')
    for k in (1, 2, 3):
        yield from run_case(f'pb_create_fail_{k}', lambda: real,
                            {'pb_create': lambda i, *a, k=k: 0 if i == k else Counting.REAL},
                            expect='pixel buffer allocation failed')
    yield from run_case('import_fail_2', lambda: real,
                        {'import_memory': lambda i, *a: (2, 0) if i == 2 else Counting.REAL}, expect='CUresult 2')
    yield from run_case('self_test_mismatch', lambda: real,
                        {'read_texture': lambda i, *a: np.zeros(64, dtype=np.float32)}, expect='self-test')
    checks = {
        'uuid_no_allocation': rows[-6]['calls'][0].get('pb_create', 0) == 0 and
        rows[-6]['calls'][0].get('import_memory', 0) == 0,
        'cache_sources': sources == {'stale_cache': 'helper', 'wrong_rva_cache': 'helper',
                                     'corrupt_cache': 'helper', 'valid_cache': 'cache'},
        'resolver_no_vulkan': all(r['made']['vulkan'] == 0 for r in rows[:3]),
    }
    save('faults.json', {'cases': rows, 'checks': checks, 'all_pass': all(r['pass'] for r in rows) and all(checks.values())})
    LOG['steps'].append('faults')


def stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(a):
        return None
    return {'n': int(len(a)), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'max': float(a.max()), 'mean': float(a.mean())}


def bench_case(kind, n):
    import warp as wp
    T = mod('.render.transport')
    tv = mod('.render.transport_vulkan')
    pts = mod('.render.gpu_points')
    m5k = sys.modules['m5_kernels']
    base = wp.array(golden(n), dtype=wp.vec3, device='cuda:0')
    out = wp.empty(n, dtype=wp.vec3, device='cuda:0')
    renderer = pts.PointRenderer()
    settings = pts.DrawSettings(pts.POINTS, 2.0, (0.12, 0.68, 1.0, 1.0), True)
    rec = {'publish_ns': {}, 'shown_ns': {}, 'complete_ns': {}, 'poll_free_ns': {}, 'submit_ns': [],
           'ptex_ns': [], 'draw_ns': [], 'waits': 0, 'publish_false': 0}
    observed = queue.Queue()
    stop = threading.Event()
    state = {'transport': None, 'closing': False}
    pack_events = []

    if kind == 'interop':
        real = tv.BlenderVulkanNative(mod('.render.blender_symbols').resolve_current_blender().resolved)

        class Instrumented(Counting):
            def fence_signal(self, fence):
                self._real.fence_signal(fence)
                t = state['transport']
                if t is None or state['closing']:
                    return
                slot = next(s for s in t._slots if s.fence == fence)
                r = self._real
                device = r._backend() + r._backend_device
                observed.put((slot.frame, c.c_void_p.from_address(device + r._vk_device).value,
                              c.c_uint64.from_address(device + r._vk_semaphore).value,
                              c.c_uint64.from_address(fence + r._fence_value).value))

        class TimedOps(tv.WarpOps):
            def pack(self, positions, dst, count):
                e0, e1 = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(e0)
                super().pack(positions, dst, count)
                wp.record_event(e1)
                pack_events.append((e0, e1))

        native = Instrumented(real)

        def observer():
            while not stop.is_set() or not observed.empty():
                try:
                    frame, vk_device, semaphore, value = observed.get(timeout=0.05)
                except queue.Empty:
                    continue
                sem, val = c.c_uint64(semaphore), c.c_uint64(value)
                info = tv._WaitInfo(real._enums['VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO'], None, 0, 1,
                                    c.pointer(sem), c.pointer(val))
                code = real._wait(vk_device, c.byref(info), 2_000_000_000)
                rec['complete_ns'][frame] = (time.perf_counter_ns(), code)
        thread = threading.Thread(target=observer, daemon=True)
        thread.start()
        transport = tv.VulkanInteropTransport(native, TimedOps(), n, window=window().as_pointer())
        leave = transport._leave

        def observed_leave(slot, new):
            if slot.state == 'IN_FLIGHT' and new == 'FREE':
                rec['poll_free_ns'].setdefault(slot.frame, time.perf_counter_ns())
            leave(slot, new)
        transport._leave = observed_leave
        state['transport'] = transport
    else:
        transport = T.ReadbackTransport(n)

    last_draw = [None]

    def hook():
        t0 = time.perf_counter_ns()
        frame = transport.positions_texture()
        t1 = time.perf_counter_ns()
        renderer.draw(frame, settings)
        rec['ptex_ns'].append(t1 - t0)
        if last_draw[0] is not None:
            rec['draw_ns'].append(t0 - last_draw[0])
        last_draw[0] = t0
        if frame is not None and frame.frame not in rec['shown_ns']:
            rec['shown_ns'][frame.frame] = t0
    DRAW['fn'] = hook
    published = -1
    start = time.monotonic()
    tick = 0
    try:
        while len(rec['shown_ns']) < WARMUP + SAMPLES and time.monotonic() - start < CASE_SECONDS:
            if DRAW['error']:
                raise RuntimeError(DRAW['error'])
            if published < 0 or published in rec['shown_ns']:
                if transport.can_publish():
                    wp.launch(m5k.animate, dim=n, inputs=[base, out, float(published + 1) * 0.01], device='cuda:0')
                    t0 = time.perf_counter_ns()
                    ok = transport.publish(out)
                    rec['submit_ns'].append(time.perf_counter_ns() - t0)
                    if ok:
                        published += 1
                        rec['publish_ns'][published] = t0
                    else:
                        rec['publish_false'] += 1
                else:
                    rec['waits'] += 1
            tick += 1
            if tick % 200 == 0:
                progress('bench', kind=kind, count=n, shown=len(rec['shown_ns']))
            redraw()
            yield 0.0
        DRAW['fn'] = None
        yield 0.05
        state['closing'] = True
        close = transport.close()
    finally:
        DRAW['fn'] = None
        stop.set()
    if kind == 'interop':
        thread.join(5)
        pack_ms = [wp.get_event_elapsed_time(a, b) for a, b in pack_events]
    else:
        pack_ms = []
    frames = sorted(f for f in rec['publish_ns'] if f >= WARMUP)
    pub = rec['publish_ns']

    def lat(table):
        return [(table[f][0] if isinstance(table[f], tuple) else table[f]) - pub[f] if f in table else None
                for f in frames]
    completed = lat(rec['complete_ns'])
    missing = sum(v is None for v in completed) if kind == 'interop' else None
    never_shown = sum(f not in rec['shown_ns'] for f in pub if f < published)
    samples = len([f for f in frames if f in rec['shown_ns']])
    result = {
        'kind': kind, 'count': n, 'backend': gpu.platform.backend_type_get(), 'close': close,
        'status': 'COMPLETE' if samples >= SAMPLES else 'INCOMPLETE_SAMPLES', 'samples': samples,
        'warmup': WARMUP, 'elapsed_s': round(time.monotonic() - start, 2), 'published': published + 1,
        'cadence': 'one publish attempt per timer tick after the previous frame was drawn',
        'slot_waits': rec['waits'], 'publish_false': rec['publish_false'],
        'uploads': getattr(transport, 'uploads', None), 'timer_uploads': getattr(transport, 'timer_uploads', None),
        'drops': getattr(transport, 'drops', 0) + rec['publish_false'], 'never_shown': never_shown,
        'cpu_submit_ms': stats(np.asarray(rec['submit_ns'][WARMUP:]) / 1e6),
        'positions_texture_cpu_ms': stats(np.asarray(rec['ptex_ns'][WARMUP:]) / 1e6),
        'frame_time_ms': stats(np.asarray(rec['draw_ns'][WARMUP:]) / 1e6),
        'publish_to_draw_ms': stats([None if v is None else v / 1e6 for v in lat(rec['shown_ns'])]),
        'pack_gpu_ms': stats(pack_ms[WARMUP:]) if pack_ms else None,
        'upload_gpu_ms': 'NOT MEASURED (no GPU timestamp on the Blender upload)',
        'completed_latency_ms': stats([None if v is None else v / 1e6 for v in completed])
        if kind == 'interop' else 'NOT MEASURED (Readback has no completion fence)',
        'completed_missing': missing,
        'completion_codes': dict(Counter(v[1] for v in rec['complete_ns'].values())),
        'poll_observed_free_ms': stats([None if v is None else v / 1e6 for v in lat(rec['poll_free_ns'])])
        if kind == 'interop' else None,
    }
    np.savez_compressed(OUT / f'bench-{kind}-{n}.npz',
                        frames=np.asarray(frames), publish=np.asarray([pub[f] for f in frames]),
                        shown=np.asarray([rec['shown_ns'].get(f, -1) for f in frames]),
                        complete=np.asarray([rec['complete_ns'].get(f, (-1, 0))[0] for f in frames]),
                        submit=np.asarray(rec['submit_ns']), ptex=np.asarray(rec['ptex_ns']),
                        draw=np.asarray(rec['draw_ns']), pack_ms=np.asarray(pack_ms))
    renderer.free()
    return result


def step_bench():
    rows = []
    kinds = ('interop', 'readback') if LANE == 'vulkan' else ('readback',)
    for kind in kinds:
        for n in BENCH_COUNTS:
            progress('bench', kind=kind, count=n)
            rows.append((yield from bench_case(kind, n)))
            save('bench.json', rows)
            gc.collect()
            yield 0.2
    LOG['steps'].append('bench')


def step_fatal():
    """Mocked CUresult 700 after activation: classification only, not a real GPU fault."""
    import warp as wp
    progress('fatal')
    S, T, tv = mod('.render.session'), mod('.render.transport'), mod('.render.transport_vulkan')
    made = {'readback': 0}
    holder = []

    def create_vulkan(resolved, count, window):
        proxy = Counting(tv.BlenderVulkanNative(resolved), {'event_query': lambda i, *a: 700})
        proxy.armed = False
        holder.append(proxy)
        t = tv.VulkanInteropTransport(proxy, tv.WarpOps(), count, window=window)
        proxy.armed = True
        return t

    def create_readback(count, reason=''):
        made['readback'] += 1
        return T.ReadbackTransport(count, reason=reason)
    session = S.open_session('VULKAN', 4097, window().as_pointer(), resolve=resolve,
                             create_vulkan=create_vulkan, create_readback=create_readback)
    src = wp.array(golden(4097), dtype=wp.vec3, device='cuda:0')
    first, error, fatal_tick = None, '', None
    try:  # publish polls its own pack (v3), so the mocked 700 may surface on the first tick
        fatal_tick = 1
        first = session.tick(lambda out: wp.copy(out, src))
        yield 0.05
        fatal_tick = 2
        session.tick(lambda out: wp.copy(out, src))
        fatal_tick = None
    except T.FatalTransportError as exc:
        error = str(exc)
    proxy = holder[0]
    calls_at_fatal = sum(proxy.calls.values())
    after = [session.tick(lambda out: None), session.transport.positions_texture()]
    close = session.close()
    row = {'first_publish': first, 'fatal_tick': fatal_tick, 'error': error, 'fatal_reason': session.fatal_reason, 'close': close,
           'readback_created': made['readback'], 'calls_at_fatal': calls_at_fatal,
           'calls_after': sum(proxy.calls.values()) - calls_at_fatal, 'after': [after[0], after[1] is None],
           'in_quarantine': any(o is session.transport for o in T.process_state().quarantine)}
    row['pass'] = bool('700' in error and close == 'failed' and made['readback'] == 0 and row['calls_after'] == 0
                       and row['in_quarantine'] and after == [False, None])
    save('fatal.json', row)
    LOG['steps'].append('fatal')


def main():
    yield from step_setup()
    yield from step_parity()
    if LANE == 'vulkan':
        yield from step_cycles()
        yield from step_faults()
    yield from step_bench()
    if LANE == 'vulkan':
        yield from step_fatal()


GEN = main()
PUMP = {'timer': None}


def pump():
    """One window event timer keeps the event loop (and redraws) running; re-added after loads."""
    wm = bpy.context.window_manager
    if PUMP['timer'] is not None:
        try:
            wm.event_timer_remove(PUMP['timer'])
        except (ReferenceError, RuntimeError, ValueError):
            pass
    PUMP['timer'] = wm.event_timer_add(0.004, window=window())


def driver():
    try:
        return next(GEN)
    except StopIteration:
        LOG['status'] = 'DONE'
    except Exception:
        LOG.update(status='FAILED', error=traceback.format_exc())
    save('probe.json', LOG)
    progress('quit', status=LOG['status'])
    try:
        with override():
            bpy.ops.wm.quit_blender()
    except Exception:
        os._exit(3)
    return None


bpy.types.SpaceView3D.draw_handler_add(draw_handler, (), 'WINDOW', 'POST_VIEW')
bpy.app.timers.register(driver, first_interval=1.0, persistent=True)
bpy.app.timers.register(pump, first_interval=0.5)
progress('registered')
