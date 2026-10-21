"""Read-only M0 binary/debug identity and pinned upstream source capture; no resolver."""
import hashlib
import json
import pathlib
import struct
import urllib.request
import uuid

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-m0'
host=json.loads((OUT/'host-preflight.json').read_text())
pdb=pathlib.Path(host['candidate']['exe']).with_suffix('.pdb')
data=pdb.read_bytes()
assert data.startswith(b'Microsoft C/C++ MSF 7.00')
block, _, _, dirbytes, _, blockmap=struct.unpack_from('<6I',data,32)
count=(dirbytes+block-1)//block
blocks=struct.unpack_from('<'+str(count)+'I',data,blockmap*block)
directory=b''.join(data[b*block:(b+1)*block] for b in blocks)[:dirbytes]
n=struct.unpack_from('<I',directory)[0]
sizes=struct.unpack_from('<'+str(n)+'I',directory,4)
cursor=4+4*n
streams=[]
for size in sizes:
    bc=0 if size==0xffffffff else (size+block-1)//block
    ids=struct.unpack_from('<'+str(bc)+'I',directory,cursor)
    streams.append((size,ids)); cursor+=bc*4
size,ids=streams[1]
info=b''.join(data[b*block:(b+1)*block] for b in ids)[:size]
version,signature,age=struct.unpack_from('<III',info)
guid=str(uuid.UUID(bytes_le=info[12:28]))
rsds=host['candidate']['rsds'][0]
result={'pdb_guid':guid,'pdb_age':age,'pdb_version':version,
        'rsds_match':guid==rsds['guid'] and age==rsds['age'],
        'native_calls':0,'source':[],'abi_profile_complete':False}
files=['source/blender/python/gpu/gpu_py_texture.hh',
       'source/blender/python/gpu/gpu_py_platform.cc',
       'source/blender/gpu/vulkan/vk_fence.cc',
       'source/blender/gpu/vulkan/vk_device.hh',
       'source/blender/gpu/vulkan/vk_device.cc',
       'source/blender/gpu/GPU_state.hh',
       'source/blender/gpu/vulkan/vk_device_submission.cc',
       'source/blender/gpu/GPU_platform.hh',
       'source/blender/gpu/intern/gpu_platform.cc']
dest=OUT/'upstream-fbe6228777e7'; dest.mkdir(exist_ok=True)
for name in files:
    url='https://raw.githubusercontent.com/blender/blender/fbe6228777e7/'+name
    try:
        content=urllib.request.urlopen(url,timeout=20).read()
        (dest/pathlib.Path(name).name).write_bytes(content)
        result['source'].append({'path':name,'url':url,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
    except Exception as exc:
        result['source'].append({'path':name,'url':url,'error':str(exc)})
(OUT/'offline-identity.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'rsds_match':result['rsds_match'],'guid':guid,'age':age,'sources':len(result['source']),'errors':[x for x in result['source'] if 'error' in x]}))
