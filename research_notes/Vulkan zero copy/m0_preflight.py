"""Bounded M0 host evidence; reads binaries, never calls Blender native addresses."""
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/vulkan-zero-copy/20261017-m0'
EXE = pathlib.Path(r'C:\Users\Tyler\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\blender.exe')

def digest(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()

def run(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'args': args, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = EXE.read_bytes()
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    machine, sections, stamp = struct.unpack_from('<HHI', data, pe + 4)
    optsize = struct.unpack_from('<H', data, pe + 20)[0]
    opt = pe + 24
    image_size = struct.unpack_from('<I', data, opt + 56)[0]
    section_table = opt + optsize
    def rva_offset(rva):
        for i in range(sections):
            pos = section_table + i * 40
            size, addr, rawsize, raw = struct.unpack_from('<IIII', data, pos + 8)
            if addr <= rva < addr + max(size, rawsize):
                return raw + rva - addr
        raise ValueError('RVA outside sections')
    debug_rva, debug_size = struct.unpack_from('<II', data, opt + 112 + 6 * 8)
    records = []
    for off in range(rva_offset(debug_rva), rva_offset(debug_rva) + debug_size, 28):
        kind, size, _, ptr = struct.unpack_from('<IIII', data, off + 12)
        if kind == 2 and data[ptr:ptr+4] == b'RSDS':
            import uuid
            records.append({'guid': str(uuid.UUID(bytes_le=data[ptr+4:ptr+20])),
                            'age': struct.unpack_from('<I', data, ptr+20)[0],
                            'pdb_name': data[ptr+24:ptr+size].split(b'\0')[0].decode(errors='replace')})
    files = [EXE, EXE.with_suffix('.pdb'), ROOT/'docs/VULKAN_ZERO_COPY_PLAN.md',
             ROOT/'docs/VULKAN_ZERO_COPY_PLAN_REVIEW.md', pathlib.Path(__file__)]
    files += sorted((ROOT/'warp_sph_realtime').rglob('*.py'))
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'candidate': {'exe': str(EXE), 'machine': machine, 'pe_timestamp': stamp,
                            'image_size': image_size, 'rsds': records},
              'hashes': {str(p): digest(p) for p in files},
              'head': run(['git','rev-parse','HEAD']), 'dirty': run(['git','status','--short']),
              'processes': run(['pwsh','-NoProfile','-Command',
                  "Get-CimInstance Win32_Process -Filter \"Name='blender.exe'\" | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3"]),
              'gpu': run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),
              'native_calls': 0, 'external_imports': 0,
              'abi_gate': 'BLOCKED: historical probe has hardcoded RVAs/layout; no complete verified ABI/UUID/cleanup profile established'}
    target = OUT/'host-preflight.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({'receipt':str(target),'candidate':result['candidate'],'abi_gate':result['abi_gate']},indent=2))

if __name__ == '__main__':
    main()
