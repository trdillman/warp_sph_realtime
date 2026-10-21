"""P0.13 gpu-module SSFR tests with a recording fake gpu module; image parity is proven live."""
from types import SimpleNamespace

import numpy as np
import pytest

from warp_sph_realtime.render import gpu_points as gp
from warp_sph_realtime.render import gpu_ssfr as gs
from warp_sph_realtime.render import session as ts
from warp_sph_realtime.render.transport import Frame

PUSH_SIZES = {'MAT4': 64, 'VEC4': 16, 'VEC3': 12, 'VEC2': 8, 'FLOAT': 4, 'INT': 4}


class Recorder:
    def __init__(self, viewport=(10, 20, 800, 600)):
        self.calls = []
        self.depth, self.mask, self.blend = 'LESS', True, 'ALPHA'
        self.viewport = viewport
        self.offscreens = []

    def rec(self, *call):
        self.calls.append(call)

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


def fake_gpu(rec):
    class Iface:
        def __init__(self, name):
            self.name, self.items = name, []

        def smooth(self, *a):
            self.items.append(('smooth', *a))

        def flat(self, *a):
            self.items.append(('flat', *a))

    class Info:
        def __init__(self):
            self.calls, self.tag = [], None

        def vertex_out(self, iface):
            self.tag = iface.name
            self.calls.append(('vertex_out', iface))

        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            return lambda *a: self.calls.append((name, *a))

    class Shader:
        def __init__(self, info):
            self.info, self.tag = info, info.tag

        def bind(self):
            rec.rec('bind', self.tag)

        def uniform_sampler(self, name, tex):
            rec.rec('sampler', self.tag, name, tex)

        def uniform_float(self, name, value):
            rec.rec('uniform', self.tag, name, value)

    class Texture:
        def __init__(self, label):
            self.label = label

        def __repr__(self):
            return self.label

    class OffScreen:
        def __init__(self, w, h, format='RGBA8'):
            self.width, self.height, self.format, self.freed = w, h, format, False
            self.texture_color = Texture(f'tex{len(rec.offscreens)}')
            rec.offscreens.append(self)
            rec.rec('alloc', w, h, format)

        def bind(self):
            os = self

            class Ctx:
                def __enter__(self):
                    rec.rec('enter', os.texture_color.label)

                def __exit__(self, *exc):
                    rec.rec('exit', os.texture_color.label)
            return Ctx()

        def free(self):
            self.freed = True
            rec.rec('free', self.texture_color.label)

    class Framebuffer:
        def clear(self, **kw):
            rec.rec('clear', tuple(sorted(kw)))

    def set_state(attr):
        def setter(*value):
            v = value[0] if len(value) == 1 else tuple(value)
            setattr(rec, attr, v)
            rec.rec('state.' + attr, v)
        return setter

    proj = np.array([[2.0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, -1.02, -0.2], [0, 0, -1, 0]])
    return SimpleNamespace(
        types=SimpleNamespace(GPUShaderCreateInfo=Info, GPUStageInterfaceInfo=Iface, GPUOffScreen=OffScreen),
        shader=SimpleNamespace(create_from_info=lambda info: (rec.rec('compile', info.tag), Shader(info))[1]),
        matrix=SimpleNamespace(get_model_view_matrix=lambda: np.eye(4), get_projection_matrix=lambda: proj),
        state=SimpleNamespace(
            viewport_get=lambda: rec.viewport, viewport_set=set_state('viewport'),
            depth_test_get=lambda: rec.depth, depth_mask_get=lambda: rec.mask, blend_get=lambda: rec.blend,
            depth_test_set=set_state('depth'), depth_mask_set=set_state('mask'), blend_set=set_state('blend'),
            active_framebuffer_get=Framebuffer),
    )


class Batch:
    def __init__(self, rec, shader, kind, attrs, fail=False):
        self.rec, self.kind, self.n, self.fail = rec, kind, len(next(iter(attrs.values()))), fail
        rec.rec('batch', kind, tuple(attrs), self.n)

    def draw_instanced(self, shader, instance_count):
        if self.fail:
            raise RuntimeError('draw failed')
        self.rec.rec('draw', shader.tag, instance_count)

    def draw(self, shader):
        self.rec.rec('draw', shader.tag, 1)


def renderer(viewport=(10, 20, 800, 600), fail=False):
    rec = Recorder(viewport)
    r = gs.SSFRRenderer(fake_gpu(rec), lambda s, k, a: Batch(rec, s, k, a, fail))
    return r, rec


def frame(count=10):
    return Frame(texture='positions_tex', count=count, width=4096, height=1, generation=1, frame=0)


SETTINGS = gp.DrawSettings(gp.SSFR, 0.02, (0.1, 0.4, 0.9, 1.0), True, filter_sigma=4.0, filter_iterations=2,
                           absorption=(0.85, 0.22, 0.05))


def push_bytes(info):
    return sum(PUSH_SIZES[c[1]] for c in info.calls if c[0] == 'push_constant')


def test_shader_create_infos_are_portable_and_complete():
    r, _ = renderer()
    r.draw(frame(), SETTINGS)
    shaders = r._shaders
    assert set(shaders) == {'depth', 'thickness', 'filter', 'composite'}
    for name, shader in shaders.items():
        calls = shader.info.calls
        assert push_bytes(shader.info) <= 128, name          # Vulkan minimum push-constant budget
        assert any(c[0] == 'vertex_source' for c in calls) and any(c[0] == 'fragment_source' for c in calls)
        assert ('fragment_out', 0, 'VEC4', 'fragColor') in calls
    for name in ('depth', 'thickness'):
        calls = shaders[name].info.calls
        assert ('vertex_in', 0, 'VEC2', 'corner') in calls
        assert ('sampler', 0, 'FLOAT_2D', 'positions') in calls
    assert ('depth_write', 'ANY') in shaders['depth'].info.calls
    assert ('depth_write', 'ANY') in shaders['composite'].info.calls
    assert ('sampler', 0, 'FLOAT_2D', 'src') in shaders['filter'].info.calls
    assert ('sampler', 0, 'FLOAT_2D', 'depth_tex') in shaders['composite'].info.calls
    assert ('sampler', 1, 'FLOAT_2D', 'thick_tex') in shaders['composite'].info.calls
    for name in ('filter', 'composite'):
        assert ('vertex_in', 0, 'VEC2', 'pos') in shaders[name].info.calls


def test_shader_sources_fetch_by_instance_and_keep_legacy_math():
    assert 'gpu_InstanceIndex' in gs.SPRITE_VERTEX and 'texelFetch(positions' in gs.SPRITE_VERTEX
    assert 'i % 4096' in gs.SPRITE_VERTEX and 'clamp(size_px, 1.5, 96.0)' in gs.SPRITE_VERTEX
    assert 'gl_FragDepth' in gs.DEPTH_FRAGMENT and 'discard' in gs.THICKNESS_FRAGMENT
    assert 'gl_PointSize' not in gs.SPRITE_VERTEX  # no point sprites: instanced quads only
    for needle in ('10.0 * radius', 'clamp(3.0 / max(-zi, 0.25), 0.35, 3.0)', 'clamp(sigma * depth_scale, 1.0, 18.0)'):
        assert needle in gs.FILTER_FRAGMENT
    for needle in ('0.02037', '96.0', 'smoothstep(0.35, 0.95, thick * 2.2) * 0.18', 'thick * 4.5'):
        assert needle in gs.COMPOSITE_FRAGMENT


def test_pass_order_targets_and_blend_modes():
    r, rec = renderer()
    assert r.draw(frame(4097), SETTINGS)
    depth, thick, ping, pong = (os.texture_color.label for os in r._targets.offscreens())
    seq = [c for c in rec.calls if c[0] in ('enter', 'draw', 'clear')]
    assert seq == [
        ('enter', depth), ('clear', ('color', 'depth')), ('draw', 'warp_sph_ssfr_depth', 4097),
        ('enter', thick), ('clear', ('color',)), ('draw', 'warp_sph_ssfr_thickness', 4097),
        ('enter', ping), ('draw', 'warp_sph_ssfr_filter', 1), ('enter', pong), ('draw', 'warp_sph_ssfr_filter', 1),
        ('enter', ping), ('draw', 'warp_sph_ssfr_filter', 1), ('enter', pong), ('draw', 'warp_sph_ssfr_filter', 1),
        ('draw', 'warp_sph_ssfr_composite', 1), ('draw', 'warp_sph_ssfr_composite', 1),
    ]
    sources = [c[3] for c in rec.of('sampler') if c[2] == 'src']
    assert [s.label for s in sources] == [depth, ping, pong, ping]
    comp = {c[2]: c[3].label for c in rec.of('sampler') if c[1] == 'warp_sph_ssfr_composite'}
    assert comp == {'depth_tex': pong, 'thick_tex': thick}
    assert ('sampler', 'warp_sph_ssfr_depth', 'positions', 'positions_tex') in rec.calls
    # Blend per pass: depth opaque, thickness additive, filter opaque, composite multiply then add.
    blends = [c[1] for c in rec.of('state.blend')]
    assert blends[:4] == ['NONE', 'ADDITIVE_PREMULT', 'NONE', 'MULTIPLY'] and blends[4] == 'ADDITIVE_PREMULT'
    # The composite runs in the caller's viewport with the scene depth test.
    i = rec.calls.index(('draw', 'warp_sph_ssfr_composite', 1))
    before = rec.calls[:i]
    assert [c for c in before if c[0] == 'state.viewport'][-1] == ('state.viewport', (10, 20, 800, 600))
    assert [c for c in before if c[0] == 'state.depth'][-1] == ('state.depth', 'LESS_EQUAL')
    # Thickness runs at half resolution with the legacy 1.35x radius.
    params = [c[3] for c in rec.of('uniform') if c[1] == 'warp_sph_ssfr_thickness' and c[2] == 'u_params']
    assert params[0][:3] == pytest.approx((0.02 * 1.35, 300.0, 400.0))


def test_offscreens_allocated_once_and_reallocated_on_resize():
    r, rec = renderer()
    r.draw(frame(), SETTINGS)
    assert rec.of('alloc') == [('alloc', 800, 600, 'RGBA32F'), ('alloc', 400, 300, 'RGBA16F'),
                               ('alloc', 800, 600, 'RGBA32F'), ('alloc', 800, 600, 'RGBA32F')]
    r.draw(frame(), SETTINGS)
    assert len(rec.of('alloc')) == 4 and rec.of('free') == []
    first = list(rec.offscreens)
    rec.viewport = (0, 0, 1025, 767)
    r.draw(frame(), SETTINGS)
    assert all(os.freed for os in first)
    assert rec.of('alloc')[4:] == [('alloc', 1025, 767, 'RGBA32F'), ('alloc', 512, 383, 'RGBA16F'),
                                   ('alloc', 1025, 767, 'RGBA32F'), ('alloc', 1025, 767, 'RGBA32F')]
    assert [c[1] for c in rec.of('compile')].count('warp_sph_ssfr_depth') == 1  # shaders survive resize


def test_free_releases_offscreens_and_shaders_then_rebuilds():
    r, rec = renderer()
    r.draw(frame(), SETTINGS)
    r.free()
    assert all(os.freed for os in rec.offscreens) and len(rec.of('free')) == 4
    assert r._shaders == {} and r._batches == {} and r._targets is None
    r.free()  # idempotent
    assert len(rec.of('free')) == 4
    r.draw(frame(), SETTINGS)
    assert len(rec.of('alloc')) == 8 and [c[1] for c in rec.of('compile')].count('warp_sph_ssfr_depth') == 2


def test_no_frame_or_zero_count_draws_nothing():
    r, rec = renderer()
    assert not r.draw(None, SETTINGS) and not r.draw(frame(0), SETTINGS)
    assert rec.calls == []


def test_degenerate_viewport_draws_nothing():
    r, rec = renderer(viewport=(0, 0, 0, 600))
    assert not r.draw(frame(), SETTINGS)
    assert rec.of('draw') == [] and rec.of('alloc') == []


def test_state_restored_even_when_a_pass_fails():
    r, rec = renderer(fail=True)
    with pytest.raises(RuntimeError):
        r.draw(frame(), SETTINGS)
    assert (rec.depth, rec.mask, rec.blend, rec.viewport) == ('LESS', True, 'ALPHA', (10, 20, 800, 600))
    assert rec.calls[-1][0] == 'state.viewport'
    r2, rec2 = renderer()
    r2.draw(frame(), SETTINGS)
    assert (rec2.depth, rec2.mask, rec2.blend, rec2.viewport) == ('LESS', True, 'ALPHA', (10, 20, 800, 600))


def test_settings_map_to_push_constants():
    r, rec = renderer()
    s = gp.DrawSettings(gp.SSFR, 0.03, (0.2, 0.3, 0.4, 1.0), False, filter_sigma=6.0, filter_iterations=3,
                        absorption=(0.5, 0.25, 0.125))
    r.draw(frame(), s)
    filt = [c[3] for c in rec.of('uniform') if c[1] == 'warp_sph_ssfr_filter']
    assert len(filt) == 6                                   # 3 iterations x (horizontal, vertical)
    assert filt[0] == pytest.approx((0.03, 6.0, 1.0, 0.0)) and filt[1] == pytest.approx((0.03, 6.0, 0.0, 1.0))
    comp = [(c[2], c[3]) for c in rec.of('uniform') if c[1] == 'warp_sph_ssfr_composite']
    absorb = [v for n, v in comp if n == 'u_absorb']
    assert absorb == [pytest.approx((0.5, 0.25, 0.125, 0.0)), pytest.approx((0.5, 0.25, 0.125, 1.0))]
    assert [v for n, v in comp if n == 'u_base'][0] == pytest.approx((0.2, 0.3, 0.4, 1.0))
    assert [v for n, v in comp if n == 'u_focal'][0] == pytest.approx((0.5, 0.4, 0.0, 0.0))
    depth_params = [c[3] for c in rec.of('uniform') if c[1] == 'warp_sph_ssfr_depth' and c[2] == 'u_params']
    assert depth_params[0][:3] == pytest.approx((0.03, 600.0, 800.0))
    # Depth test off: the composite must not test against scene depth.
    i = rec.calls.index(('draw', 'warp_sph_ssfr_composite', 1))
    assert [c for c in rec.calls[:i] if c[0] == 'state.depth'][-1] == ('state.depth', 'NONE')
    assert gs.filter_iterations(s) == 3 and gs.filter_iterations(gp.DrawSettings(gp.SSFR, 0.1, (1, 1, 1, 1), True,
                                                                                  filter_iterations=0)) == 1


def test_view_rows_feed_eye_space_centres():
    view = np.arange(16, dtype=float).reshape(4, 4)
    rows = gs.view_rows(view)
    assert rows == [tuple(view[0]), tuple(view[1]), tuple(view[2])]


class Stub:
    def __init__(self):
        self.drawn, self.frees = [], 0

    def draw(self, frame, settings):
        self.drawn.append(settings.mode)
        return True

    def free(self):
        self.frees += 1


def test_viewport_renderer_dispatches_on_mode_and_frees_both():
    points, ssfr = Stub(), Stub()
    v = gs.ViewportRenderer(points=points, ssfr=ssfr)
    v.draw(frame(), SETTINGS)
    v.draw(frame(), gp.DrawSettings(gp.SPHERES, 0.02, (1, 1, 1, 1), True))
    v.draw(frame(), gp.DrawSettings(gp.POINTS, 3.0, (1, 1, 1, 1), True))
    assert ssfr.drawn == [gp.SSFR] and points.drawn == [gp.SPHERES, gp.POINTS]
    v.free()
    assert points.frees == ssfr.frees == 1


def test_session_defaults_to_the_dispatching_renderer():
    transport = SimpleNamespace(close=lambda: 'completed')
    session = ts.open_session('OPENGL', 8, 1, create_readback=lambda count, reason='': transport,
                              alloc=lambda n: None, resolve=lambda: None, create_vulkan=lambda *a, **k: None)
    assert isinstance(session.renderer, gs.ViewportRenderer)
