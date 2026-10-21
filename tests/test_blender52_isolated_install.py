"""Optional real Blender 5.2 process acceptance for dependency-free packaging."""
from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
_MANIFEST=tomllib.loads((ROOT/"warp_sph_realtime"/"blender_manifest.toml").read_text(encoding="utf8"))
PACKAGE_NAME=f"{_MANIFEST['id']}-{_MANIFEST['version']}.zip"  # version single-sourced from the manifest
BLENDER=os.environ.get("WARPSPH_BLENDER_EXE","")

SMOKE=textwrap.dedent("""
import bpy
import importlib
import sys
from pathlib import Path
name='bl_ext.user_default.warp_sph_realtime'
assert bpy.app.version >= (5,2,0), bpy.app.version
mod=importlib.import_module(name)
assert 'warp' not in sys.modules
bpy.ops.preferences.addon_enable(module=name)
assert name in bpy.context.preferences.addons
assert 'warp' not in sys.modules
status=mod.warp_runtime.status()
assert not status.available, status
assert 'not installed' in status.message
assert bpy.app.online_access is False
result=mod.warp_installer.install_warp()
assert not result.success and 'online access' in result.message.lower(), result
assert 'warp' not in sys.modules
target=Path(bpy.utils.extension_path_user(name,path='.local',create=False))
assert not (target/'warp').exists()
bpy.ops.preferences.addon_disable(module=name)
assert name not in bpy.context.preferences.addons
print('ISOLATED_BLENDER52_MISSING_WARP_OFFLINE_PASS')
""")

@unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), "Set WARPSPH_BLENDER_EXE to Blender 5.2")
class Blender52IsolatedInstallTest(unittest.TestCase):
    def test_real_extension_build_install_register_and_offline(self):
        with tempfile.TemporaryDirectory(prefix="warpsph-blender-") as tmp:
            base=Path(tmp)
            config=base/"user"
            env=os.environ.copy()
            env.update({
                "BLENDER_USER_RESOURCES":str(config),
                "BLENDER_USER_EXTENSIONS":str(config/"extensions"),
                "BLENDER_USER_CONFIG":str(config/"config"),
                "BLENDER_USER_SCRIPTS":str(config/"scripts"),
                "BLENDER_USER_DATAFILES":str(config/"datafiles"),
                "PYTHONNOUSERSITE":"1",
            })
            def run(*args):
                result=subprocess.run([BLENDER,*map(str,args)],cwd=str(ROOT),
                                      env=env,text=True,errors="replace",capture_output=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+"\n"+result.stderr)
                return result.stdout+"\n"+result.stderr
            output=base/"dist"
            output.mkdir()
            build=run("--factory-startup","--command","extension","build","--source-dir",ROOT/"warp_sph_realtime","--output-dir",output)
            package=output/PACKAGE_NAME
            self.assertTrue(package.is_file(),build)
            installed=run("--factory-startup","--command","extension","install-file","-r","user_default","-e",package)
            self.assertIn('Installed "warp_sph_realtime"',installed)
            smoke=base/"test_blender52.py"
            smoke.write_text(SMOKE,encoding="utf8")
            log=run("--background","--factory-startup","--python",smoke)
            self.assertIn("ISOLATED_BLENDER52_MISSING_WARP_OFFLINE_PASS",log)

if __name__=="__main__":
    unittest.main()
