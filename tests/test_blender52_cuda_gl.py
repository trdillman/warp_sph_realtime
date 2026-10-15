"""Opt-in Blender 5.2 CUDA and OpenGL interoperability release qualification.

Set WARPSPH_BLENDER_EXE and WARPSPH_CERTIFIED_WHEEL to run. The wheel
must be the exact locked SHA-256/size. No user profile or global site is
changed. This is independent of the normal fast, network-free unit suite.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BLENDER=os.environ.get("WARPSPH_BLENDER_EXE", "")
WHEEL=os.environ.get("WARPSPH_CERTIFIED_WHEEL", "")

SMOKE=textwrap.dedent(r"""
import bpy
import gpu
import importlib
import shutil
import sys
from pathlib import Path

gpu.init()
name="bl_ext.user_default.warp_sph_realtime"
addon=importlib.import_module(name)
bpy.ops.preferences.addon_enable(module=name)
installer=addon.warp_installer
wheel=Path(__WHEEL_REPR__)
info=installer.get_wheel_info_for_platform()
assert "warp" not in sys.modules
assert installer._sha256_path(wheel)==info.sha256
assert wheel.stat().st_size==info.size
installer._online_access_block_reason=lambda:None
def transfer(url,destination,**kwargs):
    assert url==info.url
    assert kwargs["expected_sha256"]==info.sha256
    shutil.copyfile(wheel,destination)
installer._download_wheel=transfer
res=installer.install_warp()
assert res.success,res
assert addon.warp_runtime.status().available
wp=addon.warp_runtime.require_cuda_runtime()
assert wp.__version__=="1.17.0" and wp.is_cuda_available()
assert gpu.platform.backend_type_get()=="OPENGL"
bridge=importlib.import_module(name+".render.interop_bridge")
assert bridge._GLInterop.cudart is None  # test fresh bridge
source=wp.zeros(64,dtype=wp.vec3,device="cuda:0")
alias=bridge.wrap_device_vec3(source.ptr,64)
assert alias.ptr==source.ptr
solver=importlib.import_module(name+".sim.sim_core").RealtimeWCSPHSolver(
    num_particles=256,h=0.04,substeps=1)
demo=importlib.import_module(name+".scene.demo_scenes")
demo.launch_demo_scene_init(solver,"DOUBLE_DAM_PILLARS")
solver.rebuild_cuda_graph()
output=wp.zeros_like(solver.pos_sim)
solver.step(output,use_graph=True)
solver.step(output,use_graph=False)
assert bool((output.numpy()==output.numpy()).all())
gl_buffer=bridge.ZeroCopyGLBuffer52(64)
try:
    mapped=gl_buffer.map_to_warp()
    wp.copy(mapped,source)
    wp.synchronize_device("cuda:0")
    gl_buffer.unmap()
finally:
    gl_buffer.destroy()
print("WARP_LOCKED_CUDA_GL_QUALIFICATION_PASS")
""")

@unittest.skipUnless(
    BLENDER and Path(BLENDER).is_file() and WHEEL and Path(WHEEL).is_file(),
    "Set WARPSPH_BLENDER_EXE and WARPSPH_CERTIFIED_WHEEL",
)
class CertifiedBlender52CUDAInteropTest(unittest.TestCase):
    def test_fresh_install_cuda_solver_and_zero_copy_gl(self):
        with tempfile.TemporaryDirectory(prefix="warpsph-certified-") as tmp:
            root=Path(tmp)
            user=root/"user"
            env=os.environ.copy()
            env.update({
                "BLENDER_USER_RESOURCES":str(user),
                "BLENDER_USER_EXTENSIONS":str(user/"extensions"),
                "BLENDER_USER_CONFIG":str(user/"config"),
                "BLENDER_USER_SCRIPTS":str(user/"scripts"),
                "BLENDER_USER_DATAFILES":str(user/"datafiles"),
                "PYTHONNOUSERSITE":"1",
            })
            def run(*args, timeout=180):
                proc=subprocess.run([BLENDER,*map(str,args)],cwd=ROOT,env=env,
                                    capture_output=True,text=True,errors="replace",timeout=timeout)
                self.assertEqual(proc.returncode,0,proc.stdout+"\n"+proc.stderr)
                return proc.stdout+"\n"+proc.stderr
            dist=root/"dist"
            dist.mkdir()
            run("--factory-startup","--command","extension","build","--source-dir",ROOT/"warp_sph_realtime","--output-dir",dist)
            package=dist/"warp_sph_realtime-0.1.0.zip"
            self.assertTrue(package.is_file())
            run("--factory-startup","--command","extension","install-file","-r","user_default","-e",package)
            smoke=root/"cuda_gl.py"
            smoke.write_text(SMOKE.replace("__WHEEL_REPR__",repr(str(Path(WHEEL).resolve()))),encoding="utf8")
            output=run("--background","--factory-startup","--python",smoke,timeout=300)
            self.assertIn("WARP_LOCKED_CUDA_GL_QUALIFICATION_PASS",output)

if __name__=="__main__":
    unittest.main()
