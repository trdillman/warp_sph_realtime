"""Final preservation and candidate audit; never modifies M0 evidence."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/vulkan-zero-copy/20261018-m1'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
before = json.loads((OUT / 'preservation-before.json').read_text())
checks = {}
copies = before['historical_source_copies']
assert all(digest(OUT / 'm0-baseline' / p) == h for p,h in copies.items())
checks['baseline_copies_unchanged'] = len(copies)
original = {p:h for p,h in copies.items() if Path(p).parts[0] == 'evidence'}
assert all(digest(ROOT / p) == h for p,h in original.items())
checks['original_M0_evidence_unchanged'] = len(original)
assert digest(ROOT / 'evidence/vulkan-zero-copy/20261017-contained-m0/manifest-final.json') == before['m0_manifest_sha256']
checks['original_M0_manifest_unchanged'] = True
old = before['package_before']
assert all(digest(ROOT / p) == h for p,h in old.items())
checks['existing_package_files_unchanged'] = len(old)
frozen = {
 'warp_sph_realtime/render/blender_symbols.py':'bc1f40d166192843c2eb6f987543e6939c2c271a396d4e722fe9e0145e3f3c58',
 'warp_sph_realtime/render/blender_symbols_helper.py':'23fa929c43fe65bf2831a0af432cf9be40f7bacf3c349fd399bd824c10f40baa',
 'warp_sph_realtime/render/profiles/blender-5.2.0-fbe6228777e7-windows-amd64-cp313-v1.json':'ad33ae7ce58b1744a4af821d6dbe462ad9518ffce7b96cab7cb3a6d819577ad3',
 'tests/test_blender_symbols.py':'4f328b7ba15f7bd800687fed857635034306688934060883aaf575827c6f4e2f'}
assert all(digest(ROOT / p) == h for p,h in frozen.items())
checks['reviewed_source_hashes'] = frozen
for lane, route in [('fresh-a','helper'),('fresh-b','cache')]:
    result = json.loads((OUT/lane/'resolver.json').read_text())
    assert result['result']['available'] and result['result']['detail'] == route
    assert result['gpu_native_calls'] == result['texture_pointer_extractions'] == 0
    watch = json.loads((OUT/lane/'watch/result.json').read_text())
    assert watch['exit_confirmed'] and watch['exit_code'] == 0 and watch['reason'] == 'natural_exit'
    checks[lane] = {'route':route,'pid':result['pid'],'natural_exit':True}
assert '120 passed, 2 skipped, 8 subtests passed' in (OUT/'tests-full.log').read_text()
assert 'All checks passed!' in (OUT/'ruff-final.log').read_text()
assert '0 errors, 0 warnings' in (OUT/'pyright-cli-final.log').read_text()
package = json.loads((OUT/'package.json').read_text())
assert all(x['exit_code'] == 0 for x in package['checks'])
assert digest(ROOT/package['archive']) == package['sha256']
checks['package_sha256'] = package['sha256']
checks['status'] = 'PASS_M1_RESOLVER_ONLY'
for rel in [*frozen,'docs/VULKAN_ZERO_COPY_PLAN.md','docs/VULKAN_RESOLVER.md',
            'docs/PRODUCT_DELIVERY.md','docs/RELEASE_PLAN.md','docs/TRACKER_UPDATE_PENDING.md',
            'research_notes/Vulkan zero copy/m1_task.md','research_notes/Vulkan zero copy/m1_audit.py']:
    target = OUT/'source-final'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/rel,target)
(OUT/'verification-final.json').write_text(json.dumps(checks,indent=2))
manifest = {str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*')
            if p.is_file() and p.name != 'manifest-final.json'}
(OUT/'manifest-final.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(checks,indent=2))
print('Manifest entries:',len(manifest))
