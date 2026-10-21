"""Bind one owned M1 resolver-only session to the proven external supervisor."""
import ctypes as c
from ctypes import wintypes as w
import hashlib,json,shutil,sys,uuid
from pathlib import Path
import m0_supervisor as supervisor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261018-m1'
lane=sys.argv[1];assert lane in ('fresh-a','fresh-b')
launch=json.loads((OUT/f'launch-{lane}.json').read_text())['structuredContent']
pid=launch['pid'];exe=launch['command'][0]
h=supervisor.open_process(0x1000,False,pid);supervisor.ck(h)
try:
    name=c.create_unicode_buffer(32768);length=w.DWORD(len(name));supervisor.ck(supervisor.image(h,0,name,c.byref(length)))
    assert Path(name.value)==Path(exe)
    ft=[w.FILETIME() for _ in range(4)];supervisor.ck(supervisor.times(h,*[c.byref(x) for x in ft]))
    creation=(ft[0].dwHighDateTime<<32)|ft[0].dwLowDateTime
    assert 0<=launch['addon_receipt']['timestamp_unix']-(creation/10000000-11644473600)<=120
    assert launch['listener_pids']==[pid] and launch['addon_receipt']['pid']==pid
finally:supervisor.close(h)
target=OUT/lane;target.mkdir(exist_ok=False)
supervisor.save(target/'expected.json',{'pid':pid,'exe':exe,'creation_time':creation,'nonce':uuid.uuid4().hex})
print(json.dumps({'pid':pid,'lane':lane,'creation_time':creation}))
