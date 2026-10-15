"""Blender GUI entry point for the installed-extension acceptance probes.

Requires BLENDER_USER_RESOURCES pointing to a disposable profile containing the
installed candidate. See docs/verification/SSFR_REVIEW_REPAIR.md for commands.
"""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys
import time

import addon_utils
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--case', choices=('controls', 'workflow'), required=True)
parser.add_argument('--warp-path', required=True, help='Parent directory containing the Warp package')
parser.add_argument('--evidence-dir', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
profile = os.environ.get('BLENDER_USER_RESOURCES')
if not profile:
    raise RuntimeError('Set BLENDER_USER_RESOURCES to a disposable profile before launch')
profile = Path(profile).resolve()
config = Path(bpy.utils.user_resource('CONFIG')).resolve()
if not config.is_relative_to(profile):
    raise RuntimeError(f'Config path is outside the isolated profile: {config}')
sys.path.append(str(Path(args.warp_path).resolve()))
os.environ['SSFR_MODULE'] = 'bl_ext.user_default.warp_sph_realtime'
os.environ['SSFR_EVIDENCE'] = str(Path(args.evidence_dir).resolve())
addon = addon_utils.enable(os.environ['SSFR_MODULE'], default_set=False)
if addon is None:
    raise RuntimeError('Candidate extension did not enable')
loaded = Path(importlib.import_module(os.environ['SSFR_MODULE']).__file__).resolve()
if not loaded.is_relative_to(profile):
    raise RuntimeError(f'Loaded candidate is outside the isolated profile: {loaded}')
bpy.context.preferences.view.show_splash = False
bpy.ops.wm.save_userpref()
probe = 'runtime_ssfr_probe.py' if args.case == 'controls' else 'runtime_ssfr_workflow.py'
state = runpy.run_path(str(Path(__file__).with_name(probe)))
deadline = time.monotonic() + 180


def finish():
    record = state['RESULT' if args.case == 'controls' else 'R']
    if record['status'] != 'RUNNING':
        bpy.ops.wm.quit_blender()
        return None
    if time.monotonic() > deadline:
        record['status'] = 'FAIL'
        record['errors'].append('GUI runner deadline exceeded')
        state['save']()
        bpy.ops.wm.quit_blender()
        return None
    return .5


bpy.app.timers.register(finish, first_interval=2.)
