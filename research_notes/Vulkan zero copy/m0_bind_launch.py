"""Freeze the just-launched candidate and expected process identity before arming."""
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import sys
import m0_supervisor as s

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0'
suffix='-corrected' if '--corrected' in sys.argv else ''
lane=sys.argv[sys.argv.index('--lane')+1] if '--lane' in sys.argv else f'native{suffix}'
assert lane in ['native','native-corrected','lifecycle','lifecycle-corrected','final-window','context-identity','lifecycle-owned','qualification']
launch=json.loads((OUT/f'launch-{lane}.json').read_text())['structuredContent']
pid=launch['pid']; exe=launch['command'][0]
h=s.open_process(0x1000,False,pid);s.ck(h)
try:
    name=c.create_unicode_buffer(32768); n=w.DWORD(len(name));s.ck(s.image(h,0,name,c.byref(n)))
    assert Path(name.value)==Path(exe)
    ft=[w.FILETIME() for _ in range(4)];s.ck(s.times(h,*[c.byref(v) for v in ft]))
    creation=(ft[0].dwHighDateTime<<32)|ft[0].dwLowDateTime
    unix=creation/10000000-11644473600
    assert 0 <= launch['addon_receipt']['timestamp_unix']-unix <= 120
    assert launch['listener_pids']==[pid] and launch['addon_receipt']['pid']==pid
    identity={'pid':pid,'exe':exe,'creation_time':creation,'nonce':uuid.uuid4().hex}
finally:s.close(h)
(OUT/lane).mkdir(exist_ok=False)
s.save(OUT/lane/'expected.json',identity)
snapshot=OUT/(f'source-{lane}' if '--lane' in sys.argv else f'source{suffix}');snapshot.mkdir(exist_ok=False)
files=list(Path(__file__).parent.glob('m0_*.py'))
for p in files:shutil.copy2(p,snapshot/p.name)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
s.save(OUT/f'source-manifest-{lane}.json',hashes)
header=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include\cuda.h')
if suffix:
    lines=header.read_text().splitlines()
    (OUT/'cuda-header-excerpt.txt').write_text('\n'.join(f'{i+1}: {lines[i]}' for i in range(3695,3770)))
    s.save(OUT/'cuda-header-identity.json',{'path':str(header),'sha256':hashlib.sha256(header.read_bytes()).hexdigest()})
print(json.dumps(identity))
