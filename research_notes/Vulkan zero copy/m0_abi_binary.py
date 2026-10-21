"""Disassemble exact PE function spans offline; no native Blender calls or resolver."""
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-offline-followup'
sys.path.insert(0,str(OUT/'tools'))
import capstone
import pefile

symbols=json.loads((OUT/'decorated-symbols.json').read_text())
exe=pathlib.Path(symbols['exe'])
assert hashlib.sha256(exe.read_bytes()).hexdigest()==symbols['sha256']
pe=pefile.PE(str(exe),fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
base=pe.OPTIONAL_HEADER.ImageBase
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
md.detail=True
spans=[(x.struct.BeginAddress,x.struct.EndAddress) for x in pe.DIRECTORY_ENTRY_EXCEPTION]
functions=[]
selected=['GPU_platform_uuid','GPU_pixel_buffer_get_native_handle','GPU_texture_update_sub_from_pixel_buffer','GPU_texture_width','GPU_texture_height','GPU_create_secondary_context','GPU_activate_secondary_context','GPU_deactivate_secondary_context','GPU_destroy_secondary_context','?wait@VKFence','?signal@VKFence','?wait_for_timeline@','BPyGPUTexture_CreatePyObject','BPyGPUTexture__tp_dealloc','GPU_fence_create','GPU_pixel_buffer_create','?get@GPUBackend','?fence_alloc@VKBackend','wm_window_make_drawable','GPU_context_active_get','vkWaitSemaphores']
for group in symbols['patterns'].values():
    for s in group['hits']:
        if not any(k in s['name'] for k in selected): continue
        if s['name'].startswith(('__imp_', '??_C')): continue
        rva=int(s['rva'],16)
        span=next(((a,b) for a,b in spans if a==rva),None)
        if span is None and s['size']: span=(rva,rva+s['size'])
        # Leaf functions have no unwind table: bounded bytes through first ret only.
        limit=span[1]-rva if span else 192
        raw=pe.get_data(rva,limit)
        ins=[]; refs=[]
        for i in md.disasm(raw,base+rva):
            ins.append({'address':hex(i.address),'bytes':i.bytes.hex(),'mnemonic':i.mnemonic,'operands':i.op_str})
            for operand in i.operands:
                if operand.type==capstone.x86.X86_OP_MEM and operand.mem.base==capstone.x86.X86_REG_RIP:
                    target=i.address+i.size+operand.mem.disp
                    refs.append({'from':hex(i.address),'target':hex(target),'rva':hex(target-base)})
            if span is None and i.mnemonic in ['ret','retf','jmp']: break
        functions.append({**s,'range_source':'PE .pdata' if span else 'bounded leaf through ret',
            'span':span,'sha256':hashlib.sha256(raw).hexdigest(),'instructions':ins,'rip_refs':refs})
result={'exe_sha256':symbols['sha256'],'image_base':hex(base),'capstone':capstone.__version__,
        'functions':functions,'native_blender_calls':0}
(OUT/'binary-disassembly.json').write_text(json.dumps(result,indent=2))
text=[]
for f in functions:
    text.append('\n'+f['name']+' '+f['address']+' '+f['range_source'])
    text += [f"{i['address']} {i['mnemonic']} {i['operands']}" for i in f['instructions']]
(OUT/'binary-disassembly.txt').write_text('\n'.join(text))
files=['source/blender/blenlib/BLI_span.hh','source/blender/python/gpu/gpu_py_texture.hh',
       'source/blender/python/gpu/gpu_py_texture.cc','source/blender/gpu/GPU_texture.hh',
       'source/blender/gpu/GPU_context.hh','source/blender/gpu/intern/gpu_context.cc',
       'source/blender/gpu/GPU_platform.hh','source/blender/gpu/intern/gpu_platform.cc',
       'source/blender/gpu/vulkan/vk_fence.hh','source/blender/gpu/vulkan/vk_fence.cc',
       'source/blender/gpu/intern/gpu_state_private.hh','source/blender/gpu/vulkan/vk_device.hh',
       'source/blender/gpu/vulkan/vk_device_submission.cc',
       'source/blender/gpu/vulkan/vk_pixel_buffer.cc','source/blender/gpu/vulkan/vk_buffer.cc',
       'intern/ghost/intern/GHOST_ContextVK.cc',
       'source/blender/gpu/intern/gpu_framebuffer.cc',
       'source/blender/gpu/vulkan/vk_resource_pool.cc',
       'source/blender/windowmanager/intern/wm_window.cc',
       'source/blender/windowmanager/intern/wm_files.cc',
       'source/blender/windowmanager/intern/wm_event_system.cc']
folder=OUT/'upstream-fbe6228777e7'; folder.mkdir(exist_ok=True)
receipts=[]
for name in files:
    url='https://raw.githubusercontent.com/blender/blender/fbe6228777e7/'+name
    target=folder/pathlib.Path(name).name
    try:
        data=target.read_bytes() if target.exists() else urllib.request.urlopen(url,timeout=25).read()
        target.write_bytes(data)
        receipts.append({'source':name,'url':url,'sha256':hashlib.sha256(data).hexdigest()})
    except Exception as exc: receipts.append({'source':name,'error':str(exc)})
(OUT/'source-capture.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps({'functions':len(functions),'source_files':len(receipts),'errors':[r for r in receipts if 'error' in r]}))
