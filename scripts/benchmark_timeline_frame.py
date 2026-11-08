"""Run inside Blender MCP: matched live single-frame latency, not hardware certification.

Set bpy.app.driver_namespace['timeline_benchmark'] to {'output': absolute_json_path,
'package_sha256': actual_installed_archive_hash, 'arm': 'baseline' or 'candidate'}.
The installed addon must already be enabled and running in the target demo.
"""
import hashlib
import importlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, cast

import bpy

config = dict(bpy.app.driver_namespace['timeline_benchmark'])
addon = importlib.import_module('bl_ext.demo.warp_sph_realtime.addon')
window = bpy.context.window_manager.windows[0]
area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
region = next(r for r in area.regions if r.type == 'WINDOW')
scene = window.scene
rows = []
phase = 'reset'
started = 0.0
trial = -1

def op(name, **kwargs):
    # The pinned bpy stubs incorrectly declare temp_override's return as None.
    with cast(Any, bpy.context).temp_override(window=window, area=area, region=region):
        group, name = name.split('.')
        return getattr(getattr(bpy.ops, group), name)(**kwargs)

def save(error=None):
    Path(config['output']).write_text(json.dumps(dict(
        config=config, rows=rows, error=error, complete=trial == 16 and error is None,
        module=addon.__file__, blender=bpy.app.version_string,
    ), indent=2))

def run():
    global phase, started, trial
    try:
        q = addon._session(bpy.context)
        if phase == 'reset':
            op('wm.warp_sph_reset')
            phase = 'settle'
            started = time.perf_counter()
        elif phase == 'settle':
            assert time.perf_counter() - started < 20, 'reset publication timeout'
            if q._output_pending or q._native_live_pending or q.session.snapshot_status != 'COMPLETED':
                return .005
            assert q.step_count == 0 and q.solver.sim_time == 0
            started = time.perf_counter()
            op('screen.frame_offset', delta=1)
            phase = 'measure'
        elif phase == 'measure':
            assert time.perf_counter() - started < 20, 'single-frame completion timeout'
            if q._timeline_elapsed + 1e-10 < 1/24 or q._native_live_pending:
                return .005
            stamp = q.session._stamp
            gate = q._native_live_gate
            native_output = importlib.import_module('bl_ext.demo.warp_sph_realtime.native_output')
            if gate.last_stamp != native_output._stamp_key(stamp) or q.session.snapshot_status != 'COMPLETED':
                return .005
            wall = time.perf_counter() - started
            import numpy as np
            import warp as wp
            wp.synchronize_device('cuda:0')
            pos=q.solver.pos_sim.numpy()
            ids=q.solver.pid.numpy()
            assert np.isfinite(pos).all() and len(np.unique(ids)) == q.solver.num_particles
            rows.append(dict(trial=trial,warmup=trial == -1,wall_s=wall,steps=q.step_count,
                             time=q.solver.sim_time,frame=scene.frame_current,count=q.solver.num_particles,
                             positions=hashlib.sha256(pos.tobytes()).hexdigest(),
                             ids=hashlib.sha256(ids.tobytes()).hexdigest(),
                             dt=q.solver.dt,substeps=scene.warp_sph.substeps,
                             live_points=len(bpy.data.objects['WarpSPH Live Geometry'].data.vertices)))
            trial += 1
            if trial == 16:
                save()
                bpy.app.driver_namespace['timeline_benchmark_result']={'complete':True,'output':config['output']}
                return None
            phase = 'reset'
            save()
        return .005
    except Exception:
        save(traceback.format_exc())
        bpy.app.driver_namespace['timeline_benchmark_result']={'complete':False,'error':traceback.format_exc()}
        return None

bpy.app.timers.register(run,first_interval=.2)
