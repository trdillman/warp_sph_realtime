"""Preferences UI and operators (source choice, progress, cancel, hardware panel) against a fake bpy.

Proves operator/timer/draw logic only; the real Blender UI is proven live (see CHANGELOG evidence notes).
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SNIPPET = r'''
import sys, threading, types
from types import SimpleNamespace as NS
from pathlib import Path

class Operator:
    def report(self, level, message):
        self.reports = getattr(self, "reports", []) + [(next(iter(level)), message)]
class Scene: pass
class Object: pass

registered, timers = [], {}
bpy = types.ModuleType("bpy")
bpy.types = NS(Operator=Operator, Panel=object, AddonPreferences=object, PropertyGroup=object, Scene=Scene,
               Object=Object, SpaceView3D=NS(draw_handler_add=lambda *a, **k: None,
                                             draw_handler_remove=lambda *a, **k: None))
bpy.props = types.ModuleType("bpy.props")
for name in ("BoolProperty", "EnumProperty", "FloatProperty", "FloatVectorProperty", "IntProperty",
             "StringProperty", "PointerProperty"):
    setattr(bpy.props, name, lambda **kw: kw.get("default"))
bpy.utils = NS(register_class=registered.append, unregister_class=registered.remove)
bpy.app = NS(online_access=True, online_access_override=False, handlers=None,
             timers=NS(register=lambda fn, first_interval=0, persistent=False: timers.__setitem__(fn, persistent),
                       unregister=lambda fn: timers.pop(fn),
                       is_registered=lambda fn: fn in timers))
prefs = NS(warp_source="DOWNLOAD", warp_wheel_path="", warp_folder_path="", warp_allow_uncertified=False)
bpy.context = NS(preferences=NS(addons={"warp_sph_realtime": NS(preferences=prefs)}),
                 window_manager=NS(windows=[]))
bpy.path = NS(abspath=lambda p: p)
sys.modules["bpy"] = bpy
sys.modules["bpy.props"] = bpy.props
gpu = types.ModuleType("gpu")
gpu.platform = NS(backend_type_get=lambda: "VULKAN")
sys.modules["gpu"] = gpu

import warp_sph_realtime  # noqa: E402
from warp_sph_realtime import install_ui, warp_installer, diagnostics  # noqa: E402

state = NS(running=False)
install_ui.register(is_running=lambda: state.running, startup_error=lambda: "", active_warp=lambda: None)
names = {c.__name__ for c in registered}
assert {"WARPSPH_Preferences", "WM_OT_warp_sph_install", "WM_OT_warp_sph_install_cancel",
        "WM_OT_warp_sph_check_hardware"} <= names, names

class FakeLayout:
    def __init__(self, log): self.log = log
    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, args, kwargs))
            return FakeLayout(self.log) if name in ("box", "column", "row") else NS()
        return call
    def prop(self, owner, key, **kw): self.log.append(("prop", (key,), kw))

def draw(source, **extra):
    prefs.warp_source = source
    for key, value in extra.items():
        setattr(prefs, key, value)
    log = []
    p = install_ui.WARPSPH_Preferences()
    p.layout = FakeLayout(log)
    for key in ("warp_source", "warp_wheel_path", "warp_folder_path", "warp_allow_uncertified"):
        setattr(p, key, getattr(prefs, key))
    p.draw(bpy.context)
    return log

def labels(log): return [kw.get("text", "") for name, _a, kw in log if name == "label"]
def operators(log): return [(a[0], kw.get("text")) for name, a, kw in log if name == "operator"]

# --- source-specific draws
log = draw("DOWNLOAD")
assert ("wm.warp_sph_install", "Repair Warp (restart required)") in operators(log), operators(log)
assert any(t.startswith("Download and Install") for _o, t in operators(log))
assert any(text.startswith("Hardware") or text == "Hardware Check" for text in labels(log))
log = draw("WHEEL")
assert ("prop", ("warp_wheel_path",)) in [(n, a) for n, a, _k in log]
assert any(t and "Wheel" in t for _o, t in operators(log))
log = draw("FOLDER", warp_folder_path="C:/nowhere")
assert not any(o == "wm.warp_sph_install" for o, _t in operators(log)), "folder source must not offer install"
assert any("not found" in text for text in labels(log)), labels(log)
assert not any("UNSUPPORTED" in text for text in labels(log))
log = draw("FOLDER", warp_folder_path="C:/nowhere", warp_allow_uncertified=True)
assert any("UNSUPPORTED" in text for text in labels(log)), labels(log)  # persistent warning
prefs.warp_allow_uncertified = False

# --- offline download shows the reason and disables the buttons
bpy.app.online_access = False
log = draw("DOWNLOAD")
assert any("Online access required" in text for text in labels(log))
bpy.app.online_access = True

# --- hardware probe is requested on a thread and the panel redraws when done
install_ui._smi_thread.join(10)
assert install_ui._smi is not None
assert install_ui._poll_hardware() is None
assert any(label.startswith("Graphics backend: Vulkan") for label in labels(draw("DOWNLOAD")))

# --- background install: progress, cancel, no bpy off-thread
info = warp_installer.WheelInfo("w.whl", "0" * 64, 10, "https://files.pythonhosted.org/x", "windows-x64")
plan = warp_installer.InstallPlan(info, Path("never-created"))
release = threading.Event()
def blocking_runner(plan, *, progress, cancel):
    progress("Downloading Warp 1.17.0 (0.0 MB)...", 0.4)
    while not cancel():
        release.wait(0.01)
    return warp_installer.InstallResult(False, "Install cancelled; nothing was installed.", cancelled=True)
warp_installer.plan_install = lambda **kw: plan
warp_installer.run_plan = blocking_runner

op = install_ui.WM_OT_warp_sph_install()
op.repair = False
state.running = True
assert op.execute(bpy.context) == {"CANCELLED"} and not install_ui.job_active()
state.running = False
assert op.execute(bpy.context) == {"FINISHED"}, op.reports
assert install_ui.job_active() and install_ui._poll_job in timers and timers[install_ui._poll_job] is True
assert op.execute(bpy.context) == {"CANCELLED"} and "already running" in op.reports[-1][1]
while "Downloading" not in install_ui.job_status_text():
    pass
assert install_ui._poll_job() == install_ui._POLL_SECONDS  # still running: keep polling
log = draw("DOWNLOAD")
assert any(name == "progress" and abs(kw["factor"] - 0.4) < 1e-9 for name, _a, kw in log), log
assert ("wm.warp_sph_install_cancel", None) in operators(log)
assert not any(t and t.startswith("Download and Install") for _o, t in operators(log))
install_ui.WM_OT_warp_sph_install_cancel().execute(bpy.context)
install_ui._job.join(10)
assert install_ui._poll_job() is None
assert not install_ui.job_active()
assert install_ui._last_result.cancelled
assert any("cancelled" in text for text in labels(draw("DOWNLOAD")))

# --- wheel source: operator hands the chosen file to the planner
seen = {}
def fake_plan(**kw):
    seen.update(kw)
    return warp_installer.InstallResult(True, "Warp 1.17.0 is already installed", already_installed=True)
warp_installer.plan_install = fake_plan
op = install_ui.WM_OT_warp_sph_install()
op.repair = False
prefs.warp_source = "WHEEL"
prefs.warp_wheel_path = ""
assert op.execute(bpy.context) == {"CANCELLED"} and "Choose a Warp wheel" in op.reports[-1][1]
prefs.warp_wheel_path = "C:/wheels/w.whl"
assert op.execute(bpy.context) == {"FINISHED"}
assert seen["local_wheel"] == Path("C:/wheels/w.whl"), seen

# --- unregister stops everything
install_ui.request_hardware_probe(force=True)
install_ui.unregister()
assert not registered and not timers, (registered, timers)
print("INSTALL_UI_OK")
'''


class InstallUiTest(unittest.TestCase):
    def test_preferences_operators_and_timer(self):
        result = subprocess.run([sys.executable, "-c", SNIPPET], cwd=str(ROOT), capture_output=True, text=True,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("INSTALL_UI_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
