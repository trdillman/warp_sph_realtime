"""Gate-owned Windows GUI probe. Run with --python-exit-code 1 --python."""
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path

import addon_utils
import bpy
from mathutils import Quaternion, Vector

out = Path(__file__).resolve().parent
zip_path = out / 'warp_sph_realtime-0.1.0.zip'
deps = Path(r'C:\Users\Tyler\Documents\Codex\2026-11-06\rev\work\warpsph-runtime\sdk')
sys.path.insert(0, str(deps))
os.environ['PROCESSOR_ARCHITECTURE'] = 'AMD64'
importlib.reload(platform)
assert Path(bpy.utils.user_resource('CONFIG')).is_relative_to(out), 'Expected isolated profile'
if not addon_utils.check('bl_pkg')[1]:
    addon_utils.enable('bl_pkg')
repo = bpy.context.preferences.extensions.repos.new(
    name='WarpSPH demo', module='demo', custom_directory=str(out / 'installed'))
assert 'FINISHED' in bpy.ops.extensions.package_install_files(
    'EXEC_DEFAULT', repo=repo.module, directory=str(out),
    files=[{'name': zip_path.name}], enable_on_install=True)
module = importlib.import_module('bl_ext.demo.warp_sph_realtime')
addon = importlib.import_module(module.__name__ + '.addon')
native = importlib.import_module(module.__name__ + '.native_output')
assert addon_utils.check(module.__name__)[1]
with zipfile.ZipFile(zip_path) as archive:
    files = [name for name in archive.namelist() if not name.endswith('/')]
    assert all((Path(module.__file__).parent / name).read_bytes() == archive.read(name) for name in files)
prefs = bpy.context.preferences.addons[module.__name__].preferences
prefs.warp_source = 'FOLDER'
prefs.warp_folder_path = str(deps)
scene = bpy.context.scene
for obj in list(scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
scene.name = 'WarpSPH | Live Geometry Nodes pacing'
scene.frame_end = 10000
scene.render.fps = 24
p = scene.warp_sph
p.particle_count = 100000
p.use_shared_sdk = p.sdk_live_geometry = True
p.render_mode = 'SPHERES'
window = bpy.context.window_manager.windows[0]
area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
region = next(r for r in area.regions if r.type == 'WINDOW')
area.spaces.active.show_region_ui = True
view = area.spaces.active.region_3d
view.view_distance = 11
view.view_location = Vector((0, 0, 1))
view.view_rotation = Quaternion((.880, .280, .116, .365)).normalized()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'WarpSPH-Live-Output.blend'))

def op(name, **kwargs):
    group, action = name.split('.')
    with bpy.context.temp_override(window=window, area=area, region=region):
        return getattr(getattr(bpy.ops, group), action)(**kwargs)

original_due = native.LiveGate.due
arms = ['baseline', 'paced', 'paced', 'baseline']
rows = []
phase = 'start'
start = time.perf_counter()
metrics = {}
pause = None

def run():
    global phase, start, metrics, pause
    try:
        q = addon._session(bpy.context)
        assert not q.last_error, q.last_error
        if phase == 'start':
            assert 'RUNNING_MODAL' in op('wm.warp_sph_start')
            phase = 'reset'
        elif phase == 'reset':
            if len(rows) == len(arms):
                native.LiveGate.due = original_due
                assert 'FINISHED' in op('wm.warp_sph_reset')
                phase = 'step'
                return .3
            native.LiveGate.due = (original_due if arms[len(rows)] == 'paced' else
                                  lambda self, *args, **kwargs: original_due(self, *args))
            assert 'FINISHED' in op('wm.warp_sph_reset')
            assert scene.frame_current == scene.frame_start and q.step_count == 0
            phase = 'settle'
        elif phase == 'settle':
            if q._output_pending or q._native_live_pending or q.session.snapshot_status != 'COMPLETED':
                return .025
            metrics = dict(q.session.metrics)
            start = time.perf_counter()
            op('screen.animation_play')
            phase = 'play'
        elif phase == 'play':
            if time.perf_counter() - start < 5:
                return .025
            elapsed = time.perf_counter() - start
            assert window.screen.is_animation_playing and q.step_count > 0 and p.sdk_live_geometry
            rows.append(dict(arm=arms[len(rows)], wall_s=elapsed, simulation_s=q.solver.sim_time,
                             frame=scene.frame_current, steps=q.step_count,
                             metrics={k: v - metrics[k] for k, v in q.session.metrics.items()}))
            assert 'FINISHED' in op('wm.warp_sph_pause')
            pause = (scene.frame_current, q.step_count, q.solver.sim_time)
            phase = 'pause'
        elif phase == 'pause':
            assert not window.screen.is_animation_playing
            assert pause == (scene.frame_current, q.step_count, q.solver.sim_time)
            phase = 'reset'
        elif phase == 'step':
            assert q.step_count == 0
            op('screen.frame_offset', delta=1)
            phase = 'step_complete'
        elif phase == 'step_complete':
            if q._timeline_elapsed + 1e-10 < 1/24 or q._native_live_pending:
                return .025
            assert scene.frame_current == 2 and not window.screen.is_animation_playing
            assert len(bpy.data.objects[p.sdk_live_name].data.vertices) == 100000
            import blender_vulkan_interop
            import gpu
            import warp
            (out / 'runtime.json').write_text(json.dumps(dict(
                passed=True, rows=rows, step_count=q.step_count, pid=os.getpid(),
                identity=dict(sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                              files=len(files), module=module.__file__, blender=bpy.app.version_string,
                              warp=warp.__version__, sdk=blender_vulkan_interop.__version__,
                              gpu=str(warp.get_device('cuda:0').name), backend=gpu.platform.backend_type_get()),
            ), indent=2))
            bpy.ops.screen.screenshot(filepath=str(out / 'demo.png'))
            bpy.ops.wm.quit_blender()
            return None
        return .3 if phase == 'pause' else .025
    except Exception:
        (out / 'runtime.json').write_text(json.dumps(dict(passed=False, phase=phase, rows=rows,
                                                          error=traceback.format_exc()), indent=2))
        traceback.print_exc()
        bpy.ops.wm.quit_blender()
        return None

bpy.app.timers.register(run, first_interval=.5)
