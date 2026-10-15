"""No-Warp and no-OpenGL-device guard tests, independent of native libraries."""
from __future__ import annotations
import subprocess
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHECK=r'''
import sys, types, ctypes
gpu=types.ModuleType("gpu")
gpu.platform=types.SimpleNamespace(backend_type_get=lambda:"OPENGL")
sys.modules["gpu"]=gpu
warp=types.ModuleType("warp")
warp.vec3=type("vec3",(),{})
warp.array=lambda *a, **kw:None
sys.modules["warp"]=warp
import interop_bridge as bridge
Interop=bridge._GLInterop
class FakeCudart:
    def __init__(self, devices):
        def get_devices(length, out, capacity, selection):
            assert capacity == 16 and selection == 1
            ctypes.cast(length,ctypes.POINTER(ctypes.c_uint))[0]=len(devices)
            for i,device in enumerate(devices):
                out[i]=device
            return 0
        self.cudaGLGetDevices=get_devices

Interop.cudart=FakeCudart([1])
Interop._get_current_context=lambda:123
try:
    Interop.validate_gl_cuda_device()
    raise AssertionError("Mismatched CUDA GPU passed")
except RuntimeError as exc:
    assert "different GPU" in str(exc),str(exc)
Interop.cudart=FakeCudart([0])
Interop.validate_gl_cuda_device()
Interop._get_current_context=lambda:0
try:
    Interop.validate_gl_cuda_device()
    raise AssertionError("Missing GL context passed")
except RuntimeError as exc:
    assert "No active OpenGL" in str(exc),str(exc)
print("GL_CONTEXT_AND_DEVICE_GATES_OK")
'''

class GLInteropGateTest(unittest.TestCase):
    def test_cuda_gl_device_match_guard(self):
        proc=subprocess.run([sys.executable,"-S","-c",CHECK],cwd=str(ROOT),
                            capture_output=True,text=True,timeout=20)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertIn("GL_CONTEXT_AND_DEVICE_GATES_OK",proc.stdout)

if __name__=="__main__":
    unittest.main()
