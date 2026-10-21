"""Preserve exact compiled disable body reached by the public DRW wrapper."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0'
sys.path.insert(0,str(ROOT/'evidence/vulkan-zero-copy/20261017-offline-followup/tools'))
import pefile,capstone
exe=Path(json.loads((ROOT/'evidence/vulkan-zero-copy/20261017-m0/host-preflight.json').read_text())['candidate']['exe'])
digest=hashlib.sha256(exe.read_bytes()).hexdigest()
assert digest=='0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327'
pe=pefile.PE(str(exe));rva=0xb45850
end=next(x.struct.EndAddress for x in pe.DIRECTORY_ENTRY_EXCEPTION if x.struct.BeginAddress==rva)
raw=pe.get_data(rva,end-rva)
instructions=[f'{i.address:x}: {i.mnemonic} {i.op_str}' for i in capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64).disasm(raw,0x140000000+rva)]
result={'exe_sha256':digest,'rva':hex(rva),'sha256':hashlib.sha256(raw).hexdigest(),'instructions':instructions,'native_calls':0}
(OUT/'drw-disable-body.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
