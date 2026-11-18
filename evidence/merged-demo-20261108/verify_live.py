import bpy
import importlib
import json
import hashlib
import os
import traceback
import time
from pathlib import Path
from mathutils import Quaternion, Vector

OUT = Path(r'C:\Users\Tyler\.codex\worktrees\live-animation-integration\warp_sph_realtime\evidence\merged-demo-20261108')
addon = importlib.import_module('bl_ext.demo.warp_sph_realtime.addon')
window = bpy.context.window_manager.windows[0]
area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
region = next(r for r in area.regions if r.type == 'WINDOW')
scene = window.scene
records = []
phase = 0
deadline = time.monotonic() + 90

def op(name, **kwargs):
    with bpy.context.temp_override(window=window, area=area, region=region):
        group, action = name.split('.')
        result = getattr(getattr(bpy.ops, group), action)(**kwargs)
        assert result.intersection({'FINISHED', 'RUNNING_MODAL', 'PASS_THROUGH'}), (name, result)

def sample(label):
    q = addon._session(bpy.context)
    import warp as wp
    wp.synchronize_device('cuda:0')
    positions = q.solver.pos_sim.numpy()
    import numpy as np
    assert np.isfinite(positions).all()
    value = dict(label=label, frame=scene.frame_current, steps=q.step_count,
                 time=q.solver.sim_time, elapsed=q._timeline_elapsed,
                 playing=window.screen.is_animation_playing,
                 position_sha=hashlib.sha256(positions.tobytes()).hexdigest(),
                 viscosity=q.solver.viscosity, error=q.last_error,
                 live_points=len(bpy.data.objects['WarpSPH Live Geometry'].data.vertices))
    assert not q.last_error, value
    records.append(value)
    return value

def same(a, b):
    assert all(a[k] == b[k] for k in ('frame','steps','time','position_sha')), (a,b)

def run():
    global phase
    try:
        assert time.monotonic() < deadline, 'Verification timed out'
        if phase == 0:
            a = sample('initial_paused'); assert a['steps'] == 0 and not a['playing']
            op('screen.animation_play')
        elif phase == 1:
            a = sample('native_play'); assert a['playing'] and a['steps'] > 0
            assert a['position_sha'] != records[0]['position_sha']
            op('screen.animation_cancel', restore_frame=False)
        elif phase == 2:
            sample('native_pause_a')
        elif phase == 3:
            same(records[-1], sample('native_pause_b'))
            op('screen.frame_jump', end=False)
        elif phase == 4:
            a = sample('timeline_reset'); assert a['frame'] == 1 and a['steps'] == 0 and a['time'] == 0
            op('screen.frame_offset', delta=1)
        elif phase == 5:
            if addon._session(bpy.context)._timeline_elapsed + 1e-10 < 1/24:
                return 0.5
            a = sample('timeline_single_step'); assert a['frame'] == 2 and a['steps'] > 0 and not a['playing']
            assert a['elapsed'] >= 1/24
        elif phase == 6:
            same(records[-1], sample('timeline_single_step_held'))
            op('wm.warp_sph_step_frame')
        elif phase == 7:
            if addon._session(bpy.context)._timeline_elapsed + 1e-10 < 2/24:
                return 0.5
            a=sample('addon_single_step'); assert a['frame'] == 3 and a['elapsed'] >= 2/24
            op('screen.animation_play')
        elif phase == 8:
            a=sample('native_resume'); assert a['steps'] > records[-2]['steps']
            op('wm.warp_sph_pause')
        elif phase == 9:
            a=sample('addon_pause_a'); assert not a['playing']
        elif phase == 10:
            same(records[-1], sample('addon_pause_b'))
            op('wm.warp_sph_pause')
        elif phase == 11:
            a=sample('addon_resume'); assert a['playing'] and a['steps'] > records[-2]['steps']
            op('wm.warp_sph_pause')
            op('wm.warp_sph_physics_preset', preset='VISCOUS')
            op('wm.warp_sph_reset')
        elif phase == 12:
            a=sample('editable_viscous_reset'); assert a['viscosity'] == 8.0 and a['steps'] == 0 and a['frame'] == 1
            op('wm.warp_sph_physics_preset', preset='WATER')
            op('wm.warp_sph_reset')
        elif phase == 13:
            sample('water_reset')
            op('wm.warp_sph_stop')
            assert not addon._session(bpy.context).running
            bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'WarpSPH-Live-Merged-Demo.blend'))
            op('wm.warp_sph_start')
        elif phase == 14:
            a=sample('restarted_paused'); assert a['steps'] == 0
            op('screen.animation_play')
        elif phase == 15:
            a=sample('handoff_running'); assert a['playing'] and a['steps'] > 0 and a['live_points'] == scene.warp_sph.particle_count
            (OUT/'runtime.json').write_text(json.dumps(dict(passed=True,pid=os.getpid(),records=records),indent=2))
            bpy.app.driver_namespace['merged_demo_verification']={'passed':True,'records':records}
            return None
        phase += 1
        return 1.2
    except Exception:
        failure=traceback.format_exc()
        (OUT/'runtime.json').write_text(json.dumps(dict(passed=False,phase=phase,error=failure,records=records),indent=2))
        bpy.app.driver_namespace['merged_demo_verification']={'passed':False,'phase':phase,'error':failure}
        traceback.print_exc()
        return None

area.spaces.active.show_region_ui=True
area.spaces.active.region_3d.view_location=Vector((0,0,0))
area.spaces.active.region_3d.view_distance=13
area.spaces.active.region_3d.view_rotation=Quaternion((0.880,0.280,0.116,0.365)).normalized()
bpy.app.timers.register(run, first_interval=0.5)
