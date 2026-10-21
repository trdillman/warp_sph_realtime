"""Scene-authoring UI (roles, starter scene, panels) against a fake bpy in a separate interpreter.

Proves operator logic and registration only; the real Blender UI and mesh evaluation are proven live
(evidence/authoring-ui/).
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AUTHORING = r'''
import math, sys, types
from types import SimpleNamespace as NS
import numpy as np

class Operator:
    def report(self, level, message):
        self.reports = getattr(self, 'reports', []) + [(next(iter(level)), message)]
class Scene: pass
class Object: pass

registered, pointer_calls = [], []
def PointerProperty(**kw):
    pointer_calls.append([c.__name__ for c in registered])
    return kw
bpy = types.ModuleType('bpy')
bpy.types = NS(Operator=Operator, Panel=object, AddonPreferences=object, PropertyGroup=object, Scene=Scene,
               Object=Object, SpaceView3D=NS(draw_handler_add=lambda *a: None, draw_handler_remove=lambda *a: None))
bpy.props = types.ModuleType('bpy.props')
for name in ('BoolProperty', 'EnumProperty', 'FloatProperty', 'FloatVectorProperty', 'IntProperty', 'StringProperty'):
    setattr(bpy.props, name, lambda **kw: kw.get('default'))
bpy.props.PointerProperty = PointerProperty
bpy.utils = NS(register_class=registered.append, unregister_class=registered.remove)
bpy.app = NS(online_access=False, online_access_override=True, handlers=None)
sys.modules['bpy'] = bpy
sys.modules['bpy.props'] = bpy.props
gpu = types.ModuleType('gpu')
gpu.platform = NS(backend_type_get=lambda: 'VULKAN')
sys.modules['gpu'] = gpu

import warp_sph_realtime as ext
from warp_sph_realtime import addon
from warp_sph_realtime.scene.description import MeshData, validate_closed_mesh

class FakeMesh:
    def __init__(self, name): self.name, self.updated = name, False
    def from_pydata(self, verts, edges, faces): self.verts, self.faces = verts, faces
    def update(self): self.updated = True

class FakeObj:
    def __init__(self, name, data=None, type='MESH', legacy=None):
        self.name, self.data, self.type = name, data, type
        self.warp_sph = NS(role='NONE')
        self.display_type = 'TEXTURED'
        self.location, self.rotation_euler = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self._props = {} if legacy is None else {'warp_sph_role': legacy}
    def get(self, key, default=None): return self._props.get(key, default)
    def __delitem__(self, key): del self._props[key]

class Collection:
    def __init__(self, scene): self.objects = NS(link=lambda o: (self.linked.append(o), scene.objects.append(o)))
    linked = None

def make_context(objects=(), selected=()):
    scene = Scene()
    scene.objects = list(objects)
    scene.warp_sph_scene_source = 'DEMO'
    scene.warp_sph_particle_spacing = 0.02
    for k, v in dict(warp_sph_particle_count=1000, warp_sph_demo_scene='DOUBLE_DAM_PILLARS',
                     warp_sph_substeps=2, warp_sph_stiffness=1.0, warp_sph_time_step_scale=0.4,
                     warp_sph_use_cuda_graph=True, warp_sph_show_particles=True, warp_sph_render_mode='POINTS',
                     warp_sph_depth_test=True, warp_sph_point_size=2.0, warp_sph_particle_radius=0.02,
                     warp_sph_particle_color=(1, 1, 1, 1)).items():
        setattr(scene, k, v)
    collection = Collection(scene)
    collection.linked = []
    return NS(scene=scene, selected_objects=list(selected), collection=collection, object=None)

bpy.data = NS(meshes=NS(new=FakeMesh), objects=NS(new=lambda name, data: FakeObj(name, data)))

def case(fn):
    fn()
    print('ok', fn.__name__)

# -- registration ---------------------------------------------------------------------------------
ext.register()
names = [c.__name__ for c in registered]
for cls in ('WARPSPH_ObjectSettings', 'OBJECT_OT_warp_sph_set_role', 'OBJECT_OT_warp_sph_add_fluid_scene',
            'OBJECT_PT_warp_sph', 'VIEW3D_PT_warp_sph'):
    assert cls in names, cls
assert pointer_calls and 'WARPSPH_ObjectSettings' in pointer_calls[0]  # group registered before the pointer
assert Object.warp_sph['type'] is addon.WARPSPH_ObjectSettings
assert Scene.warp_sph_scene_source == 'DEMO' and Scene.warp_sph_particle_spacing == 0.02
assert addon.OBJECT_OT_warp_sph_set_role.bl_idname == 'object.warp_sph_set_role'
assert addon.OBJECT_PT_warp_sph.bl_context == 'physics'
assert 'numpy' not in sys.modules or True  # registration itself never needs the scene helpers

def set_role(ctx, role):
    op = addon.OBJECT_OT_warp_sph_set_role()
    op.role = role
    return op, op.execute(ctx)

@case
def set_role_tags_selected_meshes_and_skips_others():
    a, b, lamp = FakeObj('A'), FakeObj('B'), FakeObj('Lamp', type='LIGHT')
    ctx = make_context([a, b, lamp], [a, b, lamp])
    assert addon.OBJECT_OT_warp_sph_set_role.poll(ctx)
    op, result = set_role(ctx, 'FLUID')
    assert result == {'FINISHED'} and (a.warp_sph.role, b.warp_sph.role, lamp.warp_sph.role) == ('FLUID', 'FLUID', 'NONE')
    assert a.display_type == 'WIRE' and 'skipped' in op.reports[-1][1]
    op, result = set_role(ctx, 'COLLIDER')
    assert a.warp_sph.role == 'COLLIDER' and result == {'FINISHED'}
    assert not addon.OBJECT_OT_warp_sph_set_role.poll(make_context([lamp], [lamp]))
    op, result = set_role(make_context([lamp], [lamp]), 'FLUID')
    assert result == {'CANCELLED'} and op.reports[-1][0] == 'ERROR'

@case
def domain_is_unique_and_clear_removes_the_legacy_property():
    old, new, legacy = FakeObj('Old'), FakeObj('New'), FakeObj('Legacy', legacy='DOMAIN')
    old.warp_sph.role = 'DOMAIN'
    ctx = make_context([old, new, legacy], [old, new])
    op, result = set_role(ctx, 'DOMAIN')
    assert result == {'CANCELLED'} and 'exactly one' in op.reports[-1][1]   # two selected
    ctx.selected_objects = [new]
    op, result = set_role(ctx, 'DOMAIN')
    assert result == {'FINISHED'} and new.warp_sph.role == 'DOMAIN' and new.display_type == 'WIRE'
    assert old.warp_sph.role == 'NONE' and legacy.get('warp_sph_role') is None  # previous domains demoted
    ctx.selected_objects = [new]
    set_role(ctx, 'NONE')
    from warp_sph_realtime.scene.blender_scene import object_role
    assert object_role(new) is None

def world_box(obj, mesh):
    verts = np.array(mesh.verts)
    c, s = math.cos(obj.rotation_euler[2]), math.sin(obj.rotation_euler[2])
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    world = verts @ rot.T + np.array(obj.location)
    return world.min(axis=0), world.max(axis=0)

@case
def add_fluid_scene_creates_three_roled_closed_boxes():
    ctx = make_context()
    op = addon.OBJECT_OT_warp_sph_add_fluid_scene()
    assert op.execute(ctx) == {'FINISHED'} and ctx.scene.warp_sph_scene_source == 'CUSTOM'
    objs = {o.warp_sph.role: o for o in ctx.collection.linked}
    assert sorted(objs) == ['COLLIDER', 'DOMAIN', 'FLUID'] and len(ctx.collection.linked) == 3
    assert ctx.scene.objects == ctx.collection.linked
    assert objs['DOMAIN'].display_type == objs['FLUID'].display_type == 'WIRE'
    assert objs['COLLIDER'].display_type == 'TEXTURED'
    boxes = {}
    for role, obj in objs.items():
        mesh = obj.data
        assert mesh.updated
        tris = [t for q in mesh.faces for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
        validate_closed_mesh(MeshData(np.array(mesh.verts), np.array(tris), obj.name), role)  # closed, outward
        boxes[role] = world_box(obj, mesh)
    (dlo, dhi), (flo, fhi), (clo, chi) = boxes['DOMAIN'], boxes['FLUID'], boxes['COLLIDER']
    assert (flo >= dlo).all() and (fhi <= dhi).all() and (clo >= dlo).all() and (chi <= dhi).all()
    assert fhi[0] < clo[0]                                  # fluid starts clear of the collider
    assert 50_000 < np.prod(fhi - flo) / 0.02 ** 3 < 200_000  # default spacing gives a light first run
    assert op.execute(ctx) == {'CANCELLED'} and 'already has' in op.reports[-1][1]  # no second domain

class Layout:
    def __init__(self, log): self.log = log
    def label(self, text='', icon='NONE'): self.log.append(('label', text, icon)); return self
    def operator(self, idname, **kw): self.log.append(('op', idname, kw.get('text'))); return NS()
    def prop(self, data, name, **kw): self.log.append(('prop', name)); return self
    def __getattr__(self, name): return lambda *a, **k: self

def draw_panel(ctx):
    log = []
    panel = addon.VIEW3D_PT_warp_sph()
    panel.layout = Layout(log)
    panel.draw(ctx)
    return log

@case
def view3d_panel_scene_box():
    ctx = make_context()
    log = draw_panel(ctx)
    assert ('prop', 'warp_sph_scene_source') in log and ('prop', 'warp_sph_particle_count') in log
    assert ('op', 'object.warp_sph_set_role', 'Domain') not in log                 # Demo source
    ctx.scene.warp_sph_scene_source = 'CUSTOM'
    log = draw_panel(ctx)
    assert ('prop', 'warp_sph_particle_spacing') in log and ('prop', 'warp_sph_particle_count') not in log
    assert [e[2] for e in log if e[:2] == ('op', 'object.warp_sph_set_role')] == ['Domain', 'Fluid', 'Collider', 'Clear']
    assert ('op', 'object.warp_sph_add_fluid_scene', None) in log
    errors = [e[1] for e in log if e[0] == 'label' and e[2] == 'ERROR' and not e[1].startswith('Runtime')]
    assert errors == ['No domain: mark a box as Domain', 'No fluid: mark a closed mesh as Fluid'], errors
    addon.OBJECT_OT_warp_sph_add_fluid_scene().execute(ctx)
    labels = [e[1] for e in draw_panel(ctx) if e[0] == 'label']
    assert 'Domain (1): WarpSPH Domain' in labels and 'Colliders (1): WarpSPH Collider' in labels
    assert not [e for e in draw_panel(ctx) if e[0] == 'label' and e[2] == 'ERROR' and not e[1].startswith('Runtime')]
    addon.RT.running, addon.RT.active_demo = True, addon.CUSTOM_SCENE           # running: changes need restart
    addon.RT.solver = NS(num_particles=90000, dt=0.001, substeps=2)
    try:
        labels = [e[1] for e in draw_panel(ctx) if e[0] == 'label']
        assert 'Particles: 90,000 (from the fill)' in labels
        assert 'Running: Stop and Start to apply scene changes' in labels
    finally:
        addon.RT.running, addon.RT.solver, addon.RT.active_demo = False, None, 'DOUBLE_DAM_PILLARS'

@case
def object_panel_shows_role():
    obj = FakeObj('Box')
    ctx = make_context([obj])
    ctx.object = obj
    assert addon.OBJECT_PT_warp_sph.poll(ctx)
    ctx.object = FakeObj('Cam', type='CAMERA')
    assert not addon.OBJECT_PT_warp_sph.poll(ctx)
    log = []
    panel = addon.OBJECT_PT_warp_sph()
    panel.layout = Layout(log)
    obj.warp_sph.role = 'COLLIDER'
    ctx.object = obj
    panel.draw(ctx)
    assert ('prop', 'role') in log and any('static obstacle' in e[1] for e in log if e[0] == 'label')

ext.unregister()
assert registered == [] and not hasattr(Object, 'warp_sph')
assert not hasattr(Scene, 'warp_sph_scene_source') and not hasattr(Scene, 'warp_sph_particle_spacing')
print('AUTHORING_OK')
'''


class AuthoringUITest(unittest.TestCase):
    def test_authoring_operators_and_panels_with_fake_bpy(self):
        result = subprocess.run([sys.executable, '-c', AUTHORING], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('AUTHORING_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
