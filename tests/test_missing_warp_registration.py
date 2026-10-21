"""Separate Python-process verification: module registration with no Warp installed."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SNIPPET=r'''
import importlib.abc
import sys, types

class BlockWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ImportError("Warp package deliberately absent")
sys.meta_path.insert(0, BlockWarp())

class Scene: pass
class Object: pass
class Operator:
    def report(self, level, message):
        self.last_report = message
class Panel: pass
class AddonPreferences: pass
bpy=types.ModuleType("bpy")
bpy.types=types.SimpleNamespace(
    Operator=Operator,Panel=Panel,AddonPreferences=AddonPreferences,Scene=Scene,
    PropertyGroup=object,Object=Object,
    SpaceView3D=types.SimpleNamespace(draw_handler_add=lambda *a, **k:None,
                                     draw_handler_remove=lambda *a, **k:None))
bpy.props=types.ModuleType("bpy.props")
for name in ("BoolProperty","EnumProperty","FloatProperty","FloatVectorProperty","IntProperty","StringProperty","PointerProperty"):
    setattr(bpy.props,name,lambda **kwargs: kwargs)
registered=[]
bpy.utils=types.SimpleNamespace(register_class=lambda cls:registered.append(cls),
    unregister_class=lambda cls:registered.remove(cls))
bpy.app=types.SimpleNamespace(online_access=False,online_access_override=True)
sys.modules["bpy"]=bpy
sys.modules["bpy.props"]=bpy.props
gpu=types.ModuleType("gpu")
gpu.platform=types.SimpleNamespace(backend_type_get=lambda:"OPENGL")
sys.modules["gpu"]=gpu
import warp_sph_realtime as addon
from warp_sph_realtime import warp_runtime
assert "warp" not in sys.modules
addon.register()
assert "warp" not in sys.modules
assert any(c.__name__ == "WARPSPH_Preferences" for c in registered)
assert any(c.__name__ == "WM_OT_warp_sph_install" for c in registered)
assert hasattr(bpy.types.Scene,"warp_sph")
state=warp_runtime.status()
assert state.available is False
addon.unregister()
assert "warp" not in sys.modules
assert len(registered) == 0
assert not hasattr(bpy.types.Scene,"warp_sph")
assert not hasattr(bpy.types.Object,"warp_sph")
print("MISSING_WARP_REGISTRATION_OK")
'''

class MissingWarpRegistrationTest(unittest.TestCase):
    def test_register_unregister_without_warp_or_gpu(self):
        result=subprocess.run(
            [sys.executable, "-S", "-c", SNIPPET],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("MISSING_WARP_REGISTRATION_OK",result.stdout)

if __name__=="__main__":
    unittest.main()
