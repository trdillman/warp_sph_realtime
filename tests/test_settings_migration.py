"""Settings PropertyGroup (P1.3): one-time migration of the old flat ``scene.warp_sph_*`` values, the
load_post handler, and the GPU memory estimate.

The fake bpy mimics Blender 5: a file's saved add-on values are readable only while a property of that
name is registered, ``is_property_set`` tells saved from default, and ``property_unset`` drops the value.
The real round trip (save in the old layout, open with this version, save, reopen) is proven in Blender by
``tests/runtime_settings_migration.py``.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MIGRATION = r'''
import logging, math, sys, types
from types import SimpleNamespace as NS

class Prop:
    """A registered property definition: reads the saved value, or the default when none was saved."""
    def __init__(self, kind, kw): self.kind, self.kw = kind, kw
    def __get__(self, scene, owner=None):
        if scene is None:
            return self
        key = next(k for k, v in vars(type(scene)).items() if v is self)
        if key not in scene.saved:
            return self.kw.get('default', {'enum': self.kw.get('items', [('',)])[0][0], 'bool': False,
                                           'vector': (0.0,) * self.kw.get('size', 3)}.get(self.kind, 0))
        raw = scene.saved[key]
        if self.kind == 'enum':                     # enums are saved as the item index
            items = self.kw['items']
            return items[raw][0] if 0 <= raw < len(items) else ''
        if self.kind == 'vector':
            return tuple(raw)
        if self.kind == 'bool':
            return bool(raw)
        return raw

def factory(kind):
    return lambda **kw: Prop(kind, kw)

class Settings:
    """scene.warp_sph: the defaults, and is_property_set for values assigned since."""
    def __init__(self):
        self.__dict__.update(DEFAULTS, _set=set())
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        self._set.add(name)
    def is_property_set(self, name):
        return name in self._set
    def values(self):
        return {k: getattr(self, k) for k in DEFAULTS}

class Scene:
    def __init__(self, name='Scene', saved=None, settings=True):
        self.name, self.saved = name, dict(saved or {})
        if settings:
            self.warp_sph = Settings()
    def is_property_set(self, key):
        if not isinstance(getattr(type(self), key, None), Prop):
            raise TypeError(f'property "{key}" not found')
        return key in self.saved
    def property_unset(self, key):
        self.saved.pop(key, None)

registered = []
pending = []                                     # bpy.app.timers
bpy = types.ModuleType('bpy')
bpy.types = NS(Operator=object, Panel=object, AddonPreferences=object, PropertyGroup=object, Scene=Scene,
               Object=type('Object', (), {}),
               SpaceView3D=NS(draw_handler_add=lambda *a: None, draw_handler_remove=lambda *a: None))
bpy.props = types.ModuleType('bpy.props')
for name, kind in (('BoolProperty', 'bool'), ('EnumProperty', 'enum'), ('FloatProperty', 'float'),
                   ('FloatVectorProperty', 'vector'), ('IntProperty', 'int'), ('StringProperty', 'str'),
                   ('PointerProperty', 'pointer')):
    setattr(bpy.props, name, factory(kind))
bpy.utils = NS(register_class=registered.append, unregister_class=registered.remove)
bpy.app = NS(online_access=False, online_access_override=True,
             timers=NS(register=lambda fn, first_interval=0.0: pending.append(fn), unregister=pending.remove,
                       is_registered=lambda fn: fn in pending),
             handlers=NS(load_pre=[], load_post=[], exit_pre=[], undo_pre=[], redo_pre=[],
                         persistent=lambda f: setattr(f, '_bpy_persistent', True) or f))
sys.modules['bpy'] = bpy
sys.modules['bpy.props'] = bpy.props
gpu = types.ModuleType('gpu')
gpu.platform = NS(backend_type_get=lambda: 'VULKAN')
sys.modules['gpu'] = gpu

from warp_sph_realtime import props
DEFAULTS = {name: prop.kw['default'] for name, prop in props.WARPSPH_SceneSettings.__annotations__.items()}

def legacy_attrs():
    return [k for k, v in vars(Scene).items() if isinstance(v, Prop) and k.startswith('warp_sph_')]

def case(fn):
    fn()
    assert legacy_attrs() == [], legacy_attrs()     # the legacy definitions never outlive a migration
    print('ok', fn.__name__)

OLD_FILE = {  # what Blender 5 saved for the flat properties: enum indexes, int bools, float arrays
    'warp_sph_scene_source': 1, 'warp_sph_demo_scene': 1, 'warp_sph_render_mode': 2,
    'warp_sph_particle_count': 2_000_000, 'warp_sph_substeps': 4, 'warp_sph_filter_iterations': 3,
    'warp_sph_show_particles': 0, 'warp_sph_depth_test': 0, 'warp_sph_use_cuda_graph': 0,
    'warp_sph_absorption': [1.0, 2.0, 3.0], 'warp_sph_particle_color': [0.5, 0.25, 0.125, 0.75],
    'warp_sph_particle_spacing': 0.01, 'warp_sph_time_step_scale': 0.3, 'warp_sph_stiffness': 123000.0,
    'warp_sph_point_size': 4.5, 'warp_sph_particle_radius': 0.03, 'warp_sph_filter_sigma': 6.0,
}

@case
def every_saved_value_is_copied_and_converted():
    scene = Scene(saved=OLD_FILE)
    assert props.migrate_scenes([scene]) == {'Scene': list(props.LEGACY_NAMES)}
    s = scene.warp_sph
    assert (s.scene_source, s.demo_scene, s.render_mode) == ('CUSTOM', 'BLOCK_FLOP_SPLASH', 'POINTS')
    assert (s.particle_count, s.substeps, s.filter_iterations) == (2_000_000, 4, 3)
    assert (s.show_particles, s.depth_test, s.use_cuda_graph) == (False, False, False)
    assert s.absorption == (1.0, 2.0, 3.0) and s.particle_color == (0.5, 0.25, 0.125, 0.75)
    assert (s.particle_spacing, s.time_step_scale, s.point_size, s.particle_radius, s.filter_sigma) == \
        (0.01, 0.3, 4.5, 0.03, 6.0)
    assert s.stiffness == 123.0 and props.stiffness_pa(s) == 123000.0     # stored in thousands now
    assert scene.saved == {}                                              # every legacy value unset

@case
def missing_values_keep_the_defaults():
    scene = Scene()
    assert props.migrate_scenes([scene]) == {}
    assert scene.warp_sph.values() == DEFAULTS and not scene.warp_sph._set
    partial = Scene(saved={'warp_sph_substeps': 5})
    assert props.migrate_scenes([partial]) == {'Scene': ['substeps']}
    assert partial.warp_sph.substeps == 5 and partial.warp_sph.stiffness == 90.0 and partial.saved == {}

@case
def migration_runs_once():
    scene = Scene(saved=OLD_FILE)
    props.migrate_scenes([scene])
    scene.warp_sph.stiffness = 200.0                    # edited after the upgrade, then the file is reloaded
    scene.warp_sph.render_mode = 'SPHERES'
    assert props.migrate_scenes([scene]) == {}
    assert scene.warp_sph.stiffness == 200.0 and scene.warp_sph.render_mode == 'SPHERES'

@case
def stale_legacy_values_never_overwrite_newer_settings():
    # A file that kept its old values (e.g. saved before they were migrated) and was then edited and saved
    # by this version: the new setting wins and the stale old value is dropped.
    scene = Scene(saved={'warp_sph_stiffness': 123000.0, 'warp_sph_substeps': 4})
    scene.warp_sph.stiffness = 150.0
    assert props.migrate_scenes([scene]) == {'Scene': ['substeps']}
    assert scene.warp_sph.stiffness == 150.0 and scene.warp_sph.substeps == 4 and scene.saved == {}

@case
def unusable_values_keep_the_default_and_are_dropped():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger(props.__name__).addHandler(handler)
    scene = Scene(saved={'warp_sph_render_mode': 7, 'warp_sph_stiffness': math.inf, 'warp_sph_substeps': 3})
    assert props.migrate_scenes([scene]) == {'Scene': ['substeps']}
    assert scene.warp_sph.render_mode == 'SSFR_LIQUID' and scene.warp_sph.stiffness == 90.0
    assert scene.saved == {} and len([r for r in records if r.levelno == logging.WARNING]) == 2

@case
def several_scenes_and_a_scene_without_settings():
    a, b, bare = Scene('A', {'warp_sph_substeps': 7}), Scene('B'), Scene('Bare', {'warp_sph_substeps': 9}, False)
    assert props.migrate_scenes([a, b, bare]) == {'A': ['substeps']}
    assert a.warp_sph.substeps == 7 and b.warp_sph.substeps == 2 and bare.saved == {'warp_sph_substeps': 9}

@case
def a_failure_never_breaks_the_file_load():
    class Broken(Scene):
        def is_property_set(self, key): raise RuntimeError('unreadable scene')
    assert props.migrate_scenes([Broken(saved=OLD_FILE)]) == {}

@case
def an_existing_definition_is_left_alone():
    Scene.warp_sph_substeps = existing = Prop('int', {'default': 2})
    try:
        scene = Scene(saved={'warp_sph_substeps': 6})
        assert props.migrate_scenes([scene]) == {'Scene': ['substeps']} and scene.warp_sph.substeps == 6
        assert vars(Scene)['warp_sph_substeps'] is existing
    finally:
        del Scene.warp_sph_substeps

# -- the add-on runs it on load_post and when enabled with a file open -----------------------------
import warp_sph_realtime as ext
from warp_sph_realtime import addon
handlers = bpy.app.handlers

class Restricted:                                         # bpy.data inside register() and while Blender starts
    def __getattr__(self, name): raise AttributeError(name)

# Enabled with an old file open: addon_utils runs register() with bpy.data restricted, so the open file
# is migrated by a one-shot timer right after registration.
opened = Scene('Opened', {'warp_sph_stiffness': 50000.0})
bpy.data = Restricted()
ext.register()
assert opened.saved and pending == [addon._migrate_open_file]
bpy.data = NS(scenes=[opened])
assert pending.pop()() is None                             # the timer runs once
assert opened.warp_sph.stiffness == 50.0 and opened.saved == {}
assert addon._on_load_post in handlers.load_post and addon._on_load_post._bpy_persistent
assert addon._on_load_pre in handlers.load_pre and addon._on_load_post not in handlers.load_pre
loaded = Scene('Loaded', OLD_FILE)
bpy.data.scenes = [loaded]
for hook in handlers.load_post:
    hook(None)
assert loaded.warp_sph.render_mode == 'POINTS' and loaded.saved == {} and legacy_attrs() == []
bpy.data = Restricted()
addon._on_load_post(None)
ext.unregister()
ext.register()                                            # re-enable: handlers and timer are not duplicated
addon._schedule_open_file_migration()
assert handlers.load_post.count(addon._on_load_post) == 1 and pending == [addon._migrate_open_file]
ext.unregister()
assert addon._on_load_post not in handlers.load_post and addon._on_load_pre not in handlers.load_pre
assert pending == []                                      # a pending migration is cancelled
print('MIGRATION_OK')
'''


class SettingsMigrationTest(unittest.TestCase):
    def test_flat_settings_migrate_once_with_fake_bpy(self):
        result = subprocess.run([sys.executable, '-c', MIGRATION], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('MIGRATION_OK', result.stdout)


class VramEstimateTest(unittest.TestCase):
    """Pure arithmetic; props imports bpy, so it runs with a stub module."""

    def test_estimate_and_warning(self):
        code = r'''
import sys, types
bpy = types.ModuleType('bpy')
bpy.types = types.SimpleNamespace(PropertyGroup=object)
bpy.props = types.ModuleType('bpy.props')
for n in ('BoolProperty', 'EnumProperty', 'FloatProperty', 'FloatVectorProperty', 'IntProperty',
          'StringProperty'):
    setattr(bpy.props, n, lambda **kw: kw)
sys.modules['bpy'], sys.modules['bpy.props'] = bpy, bpy.props
from types import SimpleNamespace as NS
from warp_sph_realtime import props
mb = props.estimate_vram_mb
assert abs(mb(1_000_000) - (32 + 150e6 / 2**20)) < 1e-9          # ~175 MB per million particles
assert props.vram_warning(4_000_000, None) == ''                 # demo maximum fits the unknown-GPU limit
assert props.vram_warning(40_000_000, None).startswith('Needs about 5.6 GB of GPU memory')
assert props.vram_warning(4_000_000, 16376.0) == ''
assert props.vram_warning(4_000_000, 1024.0) == "Needs about 0.6 GB of this GPU's 1 GB: may not fit"
fluids = [NS(dimensions=(1.0, 1.0, 1.0)), NS(dimensions=(2.0, 0.5, 1.0)), NS()]   # no dimensions: skipped
assert props.custom_particle_estimate(fluids, 0.5) == 16                   # 2 m^3 / 0.125 m^3
assert props.custom_particle_estimate(fluids, 0.0) == 0
print('VRAM_OK')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('VRAM_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
