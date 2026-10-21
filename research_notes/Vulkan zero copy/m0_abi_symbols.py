"""Exact-candidate offline PDB discovery only. Never loads/executes blender.exe."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-offline-followup'
BASELINE=ROOT/'evidence/vulkan-zero-copy/20261017-m0'
EXE=pathlib.Path(json.loads((BASELINE/'host-preflight.json').read_text())['candidate']['exe'])
EXPECTED='0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327'
assert hashlib.sha256(EXE.read_bytes()).hexdigest()==EXPECTED
OUT.mkdir(parents=True,exist_ok=True)
dbg=ctypes.WinDLL('dbghelp',use_last_error=True)
proc=wintypes.HANDLE(0x5821)
dbg.SymSetOptions(0x4 | 0x10) # preserve decorated names
dbg.SymInitializeW.argtypes=[wintypes.HANDLE,wintypes.LPCWSTR,wintypes.BOOL]
dbg.SymInitializeW.restype=wintypes.BOOL
assert dbg.SymInitializeW(proc,str(BASELINE/'symbols'),False),ctypes.get_last_error()
dbg.SymLoadModuleExW.argtypes=[wintypes.HANDLE,wintypes.HANDLE,wintypes.LPCWSTR,wintypes.LPCWSTR,ctypes.c_uint64,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
dbg.SymLoadModuleExW.restype=ctypes.c_uint64
BASE=0x140000000
base=dbg.SymLoadModuleExW(proc,None,str(EXE),None,BASE,0,None,0)
assert base,ctypes.get_last_error()
class Symbol(ctypes.Structure):
    _fields_=[('SizeOfStruct',wintypes.ULONG),('TypeIndex',wintypes.ULONG),('Reserved',ctypes.c_uint64*2),('Index',wintypes.ULONG),('Size',wintypes.ULONG),('ModBase',ctypes.c_uint64),('Flags',wintypes.ULONG),('Value',ctypes.c_uint64),('Address',ctypes.c_uint64),('Register',wintypes.ULONG),('Scope',wintypes.ULONG),('Tag',wintypes.ULONG),('NameLen',wintypes.ULONG),('MaxNameLen',wintypes.ULONG),('Name',wintypes.WCHAR*1)]
CB=ctypes.WINFUNCTYPE(wintypes.BOOL,ctypes.POINTER(Symbol),wintypes.ULONG,ctypes.c_void_p)
dbg.SymEnumSymbolsW.argtypes=[wintypes.HANDLE,ctypes.c_uint64,wintypes.LPCWSTR,CB,ctypes.c_void_p]
dbg.SymEnumSymbolsW.restype=wintypes.BOOL
dbg.SymCleanup.argtypes=[wintypes.HANDLE]
patterns=['*GPU_platform_uuid*','*GPU_fence*','*GPU_pixel_buffer*','*GPU_texture_update_sub_from_pixel_buffer*','*GPU_texture_width*','*GPU_texture_height*','*GPU_*secondary_context*','*VKFence*','*submission_finished_timeline*','*wait_for_timeline*','*VKBackend*get*','*BPyGPUTexture*','*vkGetSemaphoreCounterValue*','*get@GPUBackend*','*?GPU_backend*','*vkWaitSemaphores*','*?fence_alloc@VKBackend*','*??_7VKBackend*','*wm_window_make_drawable*','*GPU_context_active_get*']
result={'exe':str(EXE),'sha256':EXPECTED,'base':hex(BASE),'native_blender_calls':0,'patterns':{}}
try:
    for pattern in patterns:
        hits=[]
        def callback(ptr,size,context):
            s=ptr.contents
            name=ctypes.wstring_at(ctypes.addressof(s)+Symbol.Name.offset,s.NameLen).rstrip('\x00')
            hits.append({'name':name,'rva':hex(s.Address-base),'address':hex(s.Address),'size':s.Size,'tag':s.Tag,'type_index':s.TypeIndex})
            return True
        ok=dbg.SymEnumSymbolsW(proc,base,pattern,CB(callback),None)
        result['patterns'][pattern]={'ok':bool(ok),'error':0 if ok else ctypes.get_last_error(),'hits':hits}
finally:
    dbg.SymCleanup(proc)
(OUT/'decorated-symbols.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v['hits'] for k,v in result['patterns'].items()},indent=2))
