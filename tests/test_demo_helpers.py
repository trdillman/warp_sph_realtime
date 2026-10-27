"""Demo helper objects are built with bpy.data and bmesh, never operators: the user's selection and active object
are untouched and removed helpers free their meshes. Fake bpy in a separate interpreter (real Blender is checked
by building the helpers headless; see the commit message)."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import sys, types
from types import SimpleNamespace as NS

class Objects(list):
    def link(self, obj): self.append(obj); obj.users_collection.append(self)
    def unlink(self, obj): self.remove(obj)

class Mesh:
    def __init__(self, name): self.name, self.users, self.built = name, 0, None

class Obj:
    def __init__(self, name, data):
        self.name, self.data, self.users_collection = name, data, []
        data.users += 1

class Children(list):
    def link(self, col): self.append(col)

class Collection:
    def __init__(self, name): self.name, self.objects = name, Objects()

meshes, objects, collections = [], [], {}
def remove_object(obj, do_unlink=False):
    for linked in obj.users_collection: linked.remove(obj)
    obj.data.users -= 1
    objects.remove(obj)
def new_collection(name):
    collections[name] = Collection(name); return collections[name]

bpy = types.ModuleType('bpy')     # no bpy.ops: an operator call fails the test
bpy.data = NS(
    meshes=NS(new=lambda name: meshes.append(Mesh(name)) or meshes[-1], remove=meshes.remove),
    objects=NS(new=lambda name, data: objects.append(Obj(name, data)) or objects[-1], remove=remove_object),
    collections=NS(get=collections.get, new=new_collection))
scene = NS(collection=NS(children=Children()))
bpy.context = NS(scene=scene)
sys.modules['bpy'] = bpy

calls = []
class BM:
    def to_mesh(self, mesh): mesh.built = calls[-1]
    def free(self): pass
def op(name):
    return lambda bm, **kw: calls.append((name, kw))
bmesh = types.ModuleType('bmesh')
bmesh.new = BM
bmesh.ops = NS(create_cube=op('cube'), create_uvsphere=op('uvsphere'), create_cone=op('cone'))
sys.modules['bmesh'] = bmesh

from warp_sph_realtime.scene import demo_scenes as ds

ds.build_blender_demo_helpers('DOUBLE_DAM_PILLARS')
col = collections[ds.DEMO_HELPERS_COLLECTION]
assert scene.collection.children == [col]
by_name = {o.name: o for o in col.objects}
assert sorted(by_name) == ['WarpSPH_Center_Collider', 'WarpSPH_Domain_Bounds'] + [f'WarpSPH_Pillar_{i}' for i in (1, 2, 3, 4)]
domain = by_name['WarpSPH_Domain_Bounds']
assert (domain.location, domain.scale, domain.display_type) == ((0.0, 0.0, 2.5), (5.0, 5.0, 5.0), 'WIRE')
assert domain.data.built == ('cube', {'size': 1.0})
assert by_name['WarpSPH_Center_Collider'].data.built == ('uvsphere', {'u_segments': 32, 'v_segments': 16, 'radius': 0.55})
assert by_name['WarpSPH_Pillar_3'].location == (0.85, -0.55, 1.1)
assert by_name['WarpSPH_Pillar_3'].data.built[1]['depth'] == 2.2

ds.build_blender_demo_helpers('BLOCK_FLOP')       # rebuild: old helpers and their meshes are gone
assert [o.name for o in col.objects] == ['WarpSPH_Domain_Bounds'] and len(meshes) == 1 and len(objects) == 1
ds.remove_blender_demo_helpers()
assert list(col.objects) == [] and meshes == [] and objects == []
print('DEMO_HELPERS_OK')
'''


class DemoHelpersTest(unittest.TestCase):
    def test_helpers_use_data_api_and_free_their_meshes(self):
        result = subprocess.run([sys.executable, '-'], input=SCRIPT, cwd=str(ROOT),
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('DEMO_HELPERS_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
