"""M5: launch one GUI Blender lane under a KILL_ON_JOB_CLOSE Job and a total deadline.

Usage: python m5_run.py vulkan|opengl [--smoke]. --smoke (pre-freeze only, never counted) writes smoke/<lane>/,
runs the probe at reduced counts (M5_SMOKE=1) under a shared lease. The probe writes progress.json and lane JSON
results under runs/<lane>/; this supervisor only contains, times and records the process.
"""
import ctypes as c
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('m0_supervisor', HERE / 'm0_supervisor.py')
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)  # import-only reuse of the M0 Job helpers (verbatim copy)

DEADLINE = {'vulkan': 5400, 'opengl': 2400}
STALL = 600  # seconds without progress.json change, counted once the probe has started
QUEUE = 7200  # blender-gate exclusive queue allowance before the probe's first progress.json


def main(lane: str, smoke: bool = False):
    package = json.loads((OUT / 'package.json').read_text())
    run_dir = OUT / ('smoke' if smoke else 'runs') / lane
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(package['lanes'][lane]['env'])
    env.update(M5_LANE=lane, M5_OUT=str(run_dir), M5_EVIDENCE=str(OUT))
    # blender.exe is intercepted by blender-gate: exclusive timing lease on this PC, verdict in bgate-report.json.
    env.update(BGATE_CLASS='exclusive', BGATE_OFFLOAD='off', BGATE_MAX_SECONDS=str(DEADLINE[lane] + 300),
               BGATE_REPORT=str(run_dir / 'bgate-report.json'), BGATE_LABEL=f'm5-requal-{lane}', BGATE_BLENDER=package['blender'])
    if smoke:
        env.update(M5_SMOKE='1', BGATE_CLASS='shared', BGATE_LABEL=f'm5-requal-smoke-{lane}')
    c.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x2 | 0x1)  # keep displays on (attempt-1 lesson)
    exe = package['blender']
    command = [exe, '--factory-startup', '--online-mode', '--gpu-backend', lane, '--window-geometry', '0', '0',
               '1600', '900', '--python', str(HERE / 'm5_probe.py')]
    record = {'lane': lane, 'smoke': smoke, 'command': command, 'deadline_seconds': DEADLINE[lane], 'started': time.time()}
    stdout = (run_dir / 'blender-stdout.log').open('w', encoding='utf-8', errors='replace')
    proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=subprocess.STDOUT)
    h = sup.open_process(0x100000 | 0x1000 | 0x100 | 1, False, proc.pid)
    sup.ck(h)
    job = sup.create_job(None, None)
    sup.ck(job)
    limits = sup.Extended()
    limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
    sup.ck(sup.set_info(job, 9, c.byref(limits), c.sizeof(limits)))
    sup.ck(sup.assign(job, h))
    record.update(pid=proc.pid, job_member=True)
    progress = run_dir / 'progress.json'
    last_change, last_mtime = time.monotonic(), None
    start = time.monotonic()
    probe_start = None  # deadline and stall count from the probe's first progress, not from the gate queue
    try:
        while True:
            if sup.wait(h, 1000) == 0:
                record['reason'] = 'natural_exit'
                break
            mtime = progress.stat().st_mtime if progress.exists() else None
            if mtime != last_mtime:
                last_mtime, last_change = mtime, time.monotonic()
                probe_start = probe_start or last_change
            if probe_start is None:
                if time.monotonic() - start > QUEUE:
                    record['reason'] = 'queue_timeout'
                else:
                    continue
            elif time.monotonic() - probe_start > DEADLINE[lane]:
                record['reason'] = 'deadline'
            elif time.monotonic() - last_change > STALL:
                record['reason'] = 'stalled'
            else:
                continue
            sup.ck(sup.terminate(job, 197))
            sup.wait(h, 10000)
            break
        code = c.c_ulong()
        sup.exit_code(h, c.byref(code))
        record.update(exit_code=code.value, elapsed_s=round(time.monotonic() - start, 1),
                      queued_s=round((probe_start or time.monotonic()) - start, 1))
    finally:
        sup.close(h)
        sup.close(job)
        stdout.close()
    sup.save(run_dir / 'supervisor.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], '--smoke' in sys.argv[2:])
