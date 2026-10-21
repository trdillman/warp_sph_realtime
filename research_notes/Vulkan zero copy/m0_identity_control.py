"""Reject a wrong creation identity before assigning a harmless owned process."""
import subprocess
import sys
from pathlib import Path
import m0_supervisor as s

out=Path(__file__).resolve().parents[2]/'evidence/vulkan-zero-copy/20261017-contained-m0/creation-control'
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],creationflags=0x08000000)
try:
    try:
        s.supervise(child.pid,sys.executable,out,1,expected_creation=1,nonce='negative-control')
        raise RuntimeError('identity accepted')
    except AssertionError as e:
        assert str(e)=='process creation identity mismatch' and not (out/'ready.json').exists()
        s.save(out/'rejection.json',{'pass':True,'error':str(e),'before_job_assignment':True})
finally:
    child.terminate();child.wait(timeout=5)
print('PASS incorrect creation time rejected before assignment')
