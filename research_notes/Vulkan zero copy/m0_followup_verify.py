"""Offline artifact audit only. Never loads Blender, Warp or CUDA."""
import ctypes as c
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/vulkan-zero-copy/20261017-offline-followup'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

class Span(c.Structure):
    _fields_ = [('data', c.c_void_p), ('count', c.c_int64)]

class WaitInfo(c.Structure):
    _fields_ = [('type', c.c_uint32), ('next', c.c_void_p), ('flags', c.c_uint32),
                ('count', c.c_uint32), ('semaphores', c.c_void_p), ('values', c.c_void_p)]

def uuid_bytes(value):
    if value.startswith('GPU-'):
        value = value[4:]
    if not re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', value):
        raise ValueError('invalid UUID')
    return bytes.fromhex(value.replace('-', ''))

host = json.loads((OUT.parent / '20261017-m0/host-preflight.json').read_text())
symbols = json.loads((OUT / 'decorated-symbols.json').read_text())
binary = json.loads((OUT / 'binary-disassembly.json').read_text())
package = {p: h for p, h in host['hashes'].items() if str(ROOT / 'warp_sph_realtime') in p}
current_python = {str(p) for p in (ROOT / 'warp_sph_realtime').rglob('*.py') if '__pycache__' not in p.parts}
checks = {
    'exact_executable': sha(pathlib.Path(symbols['exe'])) == symbols['sha256'] == binary['exe_sha256'],
    'package_source_unchanged': bool(package) and current_python == set(package) and
        all(sha(pathlib.Path(p)) == h for p, h in package.items()),
    'span_layout': c.sizeof(Span) == 16 and c.alignment(Span) == 8 and Span.count.offset == 8,
    'wait_layout': c.sizeof(WaitInfo) == 40 and c.alignment(WaitInfo) == 8 and
        [getattr(WaitInfo, n).offset for n, _ in WaitInfo._fields_] == [0, 8, 16, 20, 24, 32],
    'uuid_byte_order_fixture': uuid_bytes('GPU-5d09f77b-a936-a6d9-054d-0324aef67adb') ==
        bytes.fromhex('5d09f77ba936a6d9054d0324aef67adb'),
    'uuid_mismatch_fixture': uuid_bytes('00000000-0000-0000-0000-000000000000') !=
        uuid_bytes('00000000-0000-0000-0000-000000000001'),
    'disassembly_no_native_calls': binary['native_blender_calls'] == 0 and len(binary['functions']) == 21,
}
for value in ['', 'GPU-unknown', '00000000-0000-0000-0000-00000000000z']:
    try:
        uuid_bytes(value)
        checks['uuid_malformed_rejected'] = False
        break
    except ValueError:
        checks['uuid_malformed_rejected'] = True
result = {'checks': checks, 'all_pass': all(checks.values()), 'M0_pass': False,
          'runtime_profile_approved': False, 'scope': 'offline layouts/fixtures/artifact audit only',
          'package_files_compared': len(package)}
(OUT / 'verification.json').write_text(json.dumps(result, indent=2))
paths = [p for p in OUT.rglob('*') if p.is_file() and 'tools' not in p.parts and p.name != 'manifest.json']
paths += list((ROOT / 'research_notes/Vulkan zero copy').glob('m0_*'))
paths += [ROOT / 'docs/VULKAN_ZERO_COPY_PLAN.md']
(OUT / 'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}, indent=2))
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['all_pass'] else 1)
