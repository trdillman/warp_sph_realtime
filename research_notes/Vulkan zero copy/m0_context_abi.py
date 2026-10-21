"""Offline, matched-PDB evidence for restoring a borrowed owner graphics context."""
from pathlib import Path
import sys
bootstrap=Path(__file__).with_name('m0_abi_symbols.py').read_text().split('patterns=')[0]
exec(compile(bootstrap,str(Path(__file__).with_name('m0_abi_symbols.py')),'exec'))
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0'
patterns=['*wm_window_reset_drawable*','*wm_window_set_drawable*','*GPU_context_active_set*','*wm_window_make_drawable*']
symbols=[]
try:
    for pattern in patterns:
        def callback(ptr,size,context):
            s=ptr.contents;name=ctypes.wstring_at(ctypes.addressof(s)+Symbol.Name.offset,s.NameLen).rstrip('\x00')
            if name.startswith('??_C'):return True
            symbols.append({'name':name,'rva':s.Address-base,'size':s.Size});return True
        assert dbg.SymEnumSymbolsW(proc,base,pattern,CB(callback),None)
finally:dbg.SymCleanup(proc)
sys.path.insert(0,str(ROOT/'evidence/vulkan-zero-copy/20261017-offline-followup/tools'))
import pefile
import capstone
pe=pefile.PE(str(EXE));md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
for s in symbols:
    rva=s['rva'];span=next(((x.struct.BeginAddress,x.struct.EndAddress) for x in pe.DIRECTORY_ENTRY_EXCEPTION if x.struct.BeginAddress==rva),None)
    raw=pe.get_data(rva,span[1]-rva if span else s['size'] or 192)
    ins=[]
    for i in md.disasm(raw,base+rva):
        ins.append(f'{i.address:x}: {i.mnemonic} {i.op_str}')
        if span is None and i.mnemonic=='ret':break
    s.update(sha256=hashlib.sha256(raw).hexdigest(),instructions=ins)
(OUT/'context-abi.json').write_text(json.dumps({'exe_sha256':EXPECTED,'native_calls':0,'symbols':symbols},indent=2))
print(json.dumps([{'name':s['name'],'rva':hex(s['rva']),'instructions':s['instructions']} for s in symbols],indent=2))
