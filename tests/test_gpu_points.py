"""M3 renderer tests with a recording fake gpu module; visual parity is proven in Blender (M5)."""
from types import SimpleNamespace

import numpy as np

from warp_sph_realtime.render import gpu_points as gp
from warp_sph_realtime.render.transport import Frame


class Recorder:
    def __init__(self):
        self.calls = []
        self.depth, self.mask, self.blend = 'NONE', False, 'NONE'

    def rec(self, *call):
        self.calls.append(call)


def fake_gpu(rec, view):
    class Info:
        def __getattr__(self, name):
            return lambda *a: rec.rec('info.' + name, *a)

    class Iface:
        def __init__(self, name):
            rec.rec('iface', name)

        def smooth(self, *a):
            rec.rec('smooth', *a)

    class Shader:
        def bind(self):
            rec.rec('bind')

        def uniform_sampler(self, *a):
            rec.rec('sampler', *a)

        def uniform_float(self, name, value):
            rec.rec('uniform', name, value)

    class Matrix:
        def __init__(self, m):
            self.m = np.asarray(m, float)

        def __getitem__(self, r):
            return self.m[r]

        def __matmul__(self, other):
            return Matrix(self.m @ other.m)

    def set_state(attr):
        def setter(value):
            setattr(rec, attr, value)
            rec.rec('state.' + attr, value)
        return setter

    return SimpleNamespace(
        types=SimpleNamespace(GPUShaderCreateInfo=Info, GPUStageInterfaceInfo=Iface),
        shader=SimpleNamespace(create_from_info=lambda info: (rec.rec('compile'), Shader())[1]),
        matrix=SimpleNamespace(get_model_view_matrix=lambda: Matrix(view),
                               get_projection_matrix=lambda: Matrix(np.eye(4))),
        state=SimpleNamespace(viewport_get=lambda: (0, 0, 800, 600),
                              depth_test_get=lambda: rec.depth, depth_mask_get=lambda: rec.mask,
                              blend_get=lambda: rec.blend, depth_test_set=set_state('depth'),
                              depth_mask_set=set_state('mask'), blend_set=set_state('blend')),
    )


class Batch:
    def __init__(self, rec, shader, kind, attrs):
        self.rec = rec
        rec.rec('batch', kind, len(attrs['corner']))

    def draw_instanced(self, shader, instance_count):
        self.rec.rec('draw', instance_count)


def renderer(view=None):
    rec = Recorder()
    view = np.eye(4) if view is None else view
    r = gp.PointRenderer(fake_gpu(rec, view), lambda s, k, a: Batch(rec, s, k, a))
    return r, rec


def frame(count=10):
    return Frame(texture='tex', count=count, width=4096, height=1, generation=1, frame=0)


SETTINGS = gp.DrawSettings(gp.SPHERES, 0.05, (1.0, 0.5, 0.2, 1.0), True)


def test_push_constants_fit_vulkan_minimum():
    assert gp.PUSH_CONSTANT_BYTES <= 128


def test_draw_is_one_instanced_six_vertex_batch():
    r, rec = renderer()
    assert r.draw(frame(4097), SETTINGS)
    assert ('batch', 'TRIS', 6) in rec.calls
    assert [c for c in rec.calls if c[0] == 'draw'] == [('draw', 4097)]
    assert ('sampler', 'positions', 'tex') in rec.calls
    r.draw(frame(5), SETTINGS)
    assert [c[0] for c in rec.calls].count('compile') == 1  # compiled once, lazily


def test_no_frame_or_zero_count_draws_nothing():
    r, rec = renderer()
    assert not r.draw(None, SETTINGS) and not r.draw(frame(0), SETTINGS)
    assert rec.calls == []


def test_state_restored_and_depth_blend_mapping():
    r, rec = renderer()
    rec.depth, rec.mask, rec.blend = 'LESS', True, 'ALPHA'
    r.draw(frame(), gp.DrawSettings(gp.POINTS, 3.0, (1, 1, 1, 0.5), True))
    sets = [c for c in rec.calls if c[0].startswith('state.')]
    assert sets[:3] == [('state.depth', 'LESS_EQUAL'), ('state.mask', True), ('state.blend', 'ALPHA')]
    assert (rec.depth, rec.mask, rec.blend) == ('LESS', True, 'ALPHA')
    r.draw(frame(), gp.DrawSettings(gp.POINTS, 3.0, (1, 1, 1, 1.0), False))
    assert ('state.depth', 'NONE') in rec.calls and ('state.blend', 'NONE') in rec.calls


def test_uniforms_use_camera_axes_from_view_rows():
    theta = np.pi / 3
    view = np.eye(4)
    view[:3, :3] = [[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]]
    u = gp.uniforms(view, (800, 600), SETTINGS)
    np.testing.assert_allclose(u['u_right'][:3], view[0, :3])
    np.testing.assert_allclose(u['u_up'][:3], view[1, :3])
    assert u['u_right'][3] == 0.05 and u['u_up'][3] == gp.SPHERES
    assert u['u_viewport'] == (800.0, 600.0) and u['u_color'] == SETTINGS.color
    # The billboard spans exactly one radius along camera right in view space.
    world = np.array([0.3, -0.2, 1.0]) + np.asarray(u['u_right'][:3]) * u['u_right'][3]
    offset = view[:3, :3] @ world - view[:3, :3] @ np.array([0.3, -0.2, 1.0])
    np.testing.assert_allclose(offset, [0.05, 0, 0], atol=1e-12)


def test_scene_mapping_ssfr_is_portable():
    scene = SimpleNamespace(warp_sph_render_mode='SSFR_LIQUID', warp_sph_particle_radius=0.03,
                            warp_sph_particle_color=(0.1, 0.2, 0.3, 1.0), warp_sph_depth_test=False,
                            warp_sph_filter_sigma=5.5, warp_sph_filter_iterations=3,
                            warp_sph_absorption=(0.5, 0.25, 0.125))
    s = gp.settings_from_scene(scene, 'VULKAN')
    assert s.mode == gp.SSFR and s.size == 0.03 and not s.depth_test and s.note == ''
    assert (s.filter_sigma, s.filter_iterations, s.absorption) == (5.5, 3, (0.5, 0.25, 0.125))
    scene.warp_sph_render_mode = 'SPHERES'
    s = gp.settings_from_scene(scene, 'VULKAN')
    assert s.mode == gp.SPHERES and s.size == 0.03 and s.note == ''
    scene.warp_sph_render_mode = 'POINTS'
    scene.warp_sph_point_size = 6.0
    s = gp.settings_from_scene(scene, 'VULKAN')
    assert s.mode == gp.POINTS and s.size == 6.0 and s.note == ''


def test_shader_fetches_by_instance_index():
    assert 'gpu_InstanceIndex' in gp.VERTEX and 'texelFetch(positions' in gp.VERTEX
    assert 'i % 4096' in gp.VERTEX and 'discard' in gp.FRAGMENT
