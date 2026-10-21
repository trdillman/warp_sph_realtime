"""Audit retained M0 receipts and preserve final hashes; does not execute GPU work."""
import hashlib
import json
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-m0'
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
host=json.loads((OUT/'host-preflight.json').read_text())
checks={}
for backend in ['vulkan','opengl']:
    r=json.loads((OUT/f'public-{backend}.json').read_text())
    checks[backend+'_public_prerequisites']=(r['status']=='PASS_PUBLIC_PREREQUISITES_ONLY'
        and r['warp_version']=='1.17.0' and r['warp_stream_known_answer']
        and [x['n'] for x in r['rows']]==[0,1,4095,4096,4097,4194304]
        and all(x['status']=='PASS' and x.get('mismatches',0)==0 for x in r['rows'])
        and r['native_calls']==0 and r['external_imports']==0)
    checks[backend+'_isolated_user_paths']=bool(r['user_paths']) and all(
        str(OUT).lower() in v.lower() for v in r['user_paths'].values())
checks['pdb_guid_age_match']=json.loads((OUT/'offline-identity.json').read_text())['rsds_match']
checks['package_source_unchanged']=all(sha(pathlib.Path(p))==h for p,h in host['hashes'].items()
    if str(ROOT/'warp_sph_realtime') in p)
src=pathlib.Path(r'C:\Users\Tyler\AppData\Roaming\Python\Python314\site-packages\warp')
dst=OUT/'runtime/warp'
files=[p for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
checks['staged_warp_matches_source']=all((dst/p.relative_to(src)).is_file()
    and sha(p)==sha(dst/p.relative_to(src)) for p in files)
p=subprocess.run(['pwsh','-NoProfile','-Command',
    "@(Get-Process -Id 69012,76576,28384 -ErrorAction SilentlyContinue).Count"],capture_output=True,text=True)
checks['owned_processes_exited']=p.stdout.strip()=='0'
result={'checks':checks,'all_pass':all(checks.values()),'warp_files_compared':len(files),
        'scope':'receipt audit only; does not pass blocked native transport gates'}
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
paths=[p for p in OUT.rglob('*') if p.is_file() and not any(s in p.parts for s in ['runtime','sessions','symbols','warp-cache']) and p.name!='manifest.json']
paths+=list((ROOT/'research_notes/Vulkan zero copy').glob('m0_*'))
(OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},indent=2))
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['all_pass'] else 1)
