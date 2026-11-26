"""External Windows Job containment for one explicitly owned disposable process."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time

k = c.WinDLL('kernel32', use_last_error=True)
def api(name, restype, *args):
    f = getattr(k, name); f.restype = restype; f.argtypes = args; return f
open_process = api('OpenProcess', w.HANDLE, w.DWORD, w.BOOL, w.DWORD)
close = api('CloseHandle', w.BOOL, w.HANDLE)
create_job = api('CreateJobObjectW', w.HANDLE, c.c_void_p, w.LPCWSTR)
assign = api('AssignProcessToJobObject', w.BOOL, w.HANDLE, w.HANDLE)
set_info = api('SetInformationJobObject', w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
in_job = api('IsProcessInJob', w.BOOL, w.HANDLE, w.HANDLE, c.POINTER(w.BOOL))
terminate = api('TerminateJobObject', w.BOOL, w.HANDLE, w.UINT)
wait = api('WaitForSingleObject', w.DWORD, w.HANDLE, w.DWORD)
exit_code = api('GetExitCodeProcess', w.BOOL, w.HANDLE, c.POINTER(w.DWORD))
image = api('QueryFullProcessImageNameW', w.BOOL, w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD))
times = api('GetProcessTimes', w.BOOL, w.HANDLE, *([c.POINTER(w.FILETIME)] * 4))

class Basic(c.Structure):
    _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64), ('flags', w.DWORD),
                ('min_ws', c.c_size_t), ('max_ws', c.c_size_t), ('active', w.DWORD),
                ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
class IO(c.Structure):
    _fields_ = [(n, c.c_uint64) for n in ['ro', 'wo', 'oo', 'rt', 'wt', 'ot']]
class Extended(c.Structure):
    _fields_ = [('basic', Basic), ('io', IO), ('process_memory', c.c_size_t),
                ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]

def ck(ok):
    if not ok: raise c.WinError(c.get_last_error())

def save(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(data, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)

def supervise(pid, expected, out, seconds, expected_creation=None, nonce=None):
    out.mkdir(parents=True, exist_ok=False)
    h = open_process(0x100000 | 0x1000 | 0x100 | 1, False, pid); ck(h)
    job = None
    record = {'pid': pid, 'supervisor_pid': os.getpid(), 'deadline_seconds': seconds}
    try:
        name = c.create_unicode_buffer(32768); n = w.DWORD(len(name))
        ck(image(h, 0, name, c.byref(n)))
        assert Path(name.value).resolve() == Path(expected).resolve(), 'process image mismatch'
        ft = [w.FILETIME() for _ in range(4)]; ck(times(h, *[c.byref(t) for t in ft]))
        record.update(image=name.value, creation_time=(ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime)
        if expected_creation is not None:
            assert record['creation_time'] == expected_creation, 'process creation identity mismatch'
        record['nonce'] = nonce
        job = create_job(None, None); ck(job)
        limits = Extended(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway
        ck(set_info(job, 9, c.byref(limits), c.sizeof(limits)))
        ck(assign(job, h)); member = w.BOOL(); ck(in_job(h, job, c.byref(member))); assert member.value
        record.update(job_member=True, kill_on_close=True, status='ARMED')
        save(out / 'ready.json', record)
        start = time.monotonic()
        phase_name, phase_start = 'await_probe', start
        phase_limits = {'await_probe': 120, 'identity': 30, 'correctness': 120,
                        'cycles': 120, 'stress': 360, 'cleanup': 30}
        while True:
            wait_result = wait(h, 0)
            if wait_result == 0:
                record.update(reason='natural_exit', exit_confirmed=True)
                break
            if wait_result != 258:
                record.update(reason='process_wait_failed', wait_result=wait_result,
                              wait_error=c.get_last_error())
                ck(terminate(job, 198)); record['exit_confirmed']=wait(h,10000)==0
                break
            if (out / 'phase.json').exists():
                phase = json.loads((out / 'phase.json').read_text())['phase']
                if phase != phase_name:
                    assert phase in phase_limits, 'unapproved phase'
                    phase_name, phase_start = phase, time.monotonic()
            phase_expired = time.monotonic() - phase_start >= phase_limits[phase_name]
            if (out / 'abort.json').exists() or time.monotonic() - start >= seconds or phase_expired:
                record['reason'] = 'abort_request' if (out / 'abort.json').exists() else 'deadline'
                record['phase'] = phase_name
                ck(terminate(job, 197))
                record['exit_confirmed'] = wait(h, 10000) == 0
                break
            time.sleep(0.05)
        ec = w.DWORD(); ck(exit_code(h, c.byref(ec)))
        record.update(exit_code=ec.value, elapsed=time.monotonic()-start,
                      status='EXIT_CONFIRMED' if record['exit_confirmed'] else 'TERMINATION_UNCONFIRMED')
        save(out / 'result.json', record)
        return record
    finally:
        if job: close(job)
        close(h)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pid', type=int); p.add_argument('--exe')
    p.add_argument('--out', required=True); p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--self-test', action='store_true'); p.add_argument('--expected'); a = p.parse_args()
    if a.self_test:
        root = Path(a.out); root.mkdir(parents=True, exist_ok=False)
        results = []
        for label, code, deadline in [('natural', 'import time;time.sleep(1)', 5),
                                      ('parked', 'import time;time.sleep(60)', 1)]:
            child = subprocess.Popen([sys.executable, '-c', code], creationflags=0x08000000)
            result = supervise(child.pid, sys.executable, root / label, deadline)
            child.wait(timeout=2); results.append(result)
        assert results[0]['exit_code'] == 0 and results[0]['reason'] == 'natural_exit'
        assert results[1]['exit_code'] == 197 and results[1]['exit_confirmed']
        save(root / 'controls.json', {'pass': True, 'results': results})
        print('PASS natural exit and deadline termination controls')
    else:
        if a.expected:
            identity=json.loads(Path(a.expected).read_text())
            assert identity['pid']==a.pid and identity['exe']==a.exe
            print(json.dumps(supervise(a.pid,a.exe,Path(a.out),a.seconds,identity['creation_time'],identity['nonce'])))
        else:
            # Non-Blender harmless controls may use the direct test interface only.
            assert Path(a.exe).name.lower()!='blender.exe', 'Blender requires expected launch identity'
            print(json.dumps(supervise(a.pid, a.exe, Path(a.out), a.seconds)))
