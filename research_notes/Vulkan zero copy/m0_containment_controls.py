"""Harmless child-only containment controls; no GPU imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import m0_supervisor as s

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/vulkan-zero-copy/20261017-contained-m0/extra-controls'
OUT.mkdir(parents=True, exist_ok=False)
results = {}
for label in ['wrong_image', 'assignment_rejection']:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=0x08000000)
    original = s.assign
    try:
        if label == 'assignment_rejection':
            s.assign = lambda *args: False  # branch fault injection, not OS policy claim
        try:
            s.supervise(child.pid, sys.executable if label == 'assignment_rejection' else 'C:/wrong.exe', OUT / label, 1)
            raise AssertionError('unexpected acceptance')
        except (AssertionError, OSError) as e:
            assert not (OUT / label / 'ready.json').exists()
            results[label] = {'rejected_before_ready': True, 'error': str(e),
                              'injected': label == 'assignment_rejection'}
    finally:
        s.assign = original
        if child.poll() is None: child.terminate()
        child.wait(timeout=5)

# Terminating only this owned supervisor closes its sole job handle.
child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], creationflags=0x08000000)
watch = subprocess.Popen([sys.executable, str(Path(s.__file__)), '--pid', str(child.pid),
    '--exe', sys.executable, '--out', str(OUT / 'supervisor_loss'), '--seconds', '30'], creationflags=0x08000000)
try:
    until = time.monotonic() + 10
    while not (OUT / 'supervisor_loss/ready.json').exists():
        assert watch.poll() is None and time.monotonic() < until
        time.sleep(.05)
    watch.terminate(); watch.wait(timeout=5)
    child.wait(timeout=10)
    results['supervisor_loss'] = {'child_exit_confirmed': True, 'exit_code': child.returncode}
finally:
    if watch.poll() is None: watch.terminate(); watch.wait(timeout=5)
    if child.poll() is None: child.terminate(); child.wait(timeout=5)

# A receipt failure must not unwind into caller teardown; watchdog still kills.
sentinel = OUT / 'unexpected-return.txt'
code = """import time
try:
    try:
        raise OSError('injected receipt failure')
    finally:
        while True:
            try: time.sleep(1)
            except BaseException: pass
finally:
    open(%r, 'w').write('returned')
""" % str(sentinel)
child = subprocess.Popen([sys.executable, '-c', code], creationflags=0x08000000)
record = s.supervise(child.pid, sys.executable, OUT / 'receipt_failure', 1)
child.wait(timeout=5)
assert record['exit_confirmed'] and not sentinel.exists()
results['receipt_failure'] = {'nonreturning': True, 'exit_confirmed': True}
s.save(OUT / 'results.json', {'pass': True, 'controls': results})
print(json.dumps(results, indent=2))
