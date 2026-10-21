"""Preserve M0 identities before M1; never rewrite M0 evidence or receipts."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261018-m1'
OUT.mkdir(exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0/manifest-final.json'
expected=json.loads(manifest.read_text())
bad=[p for p,h in expected.items() if sha(ROOT/p)!=h]
assert not bad,bad
target=OUT/'m0-baseline';target.mkdir(exist_ok=False)
mutable={p:h for p,h in expected.items() if Path(p).parts[0]!='evidence'}
for p in mutable:
    dest=target/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
receipt={'m0_manifest_sha256':sha(manifest),'verified_files':len(expected),
         'preserved_evidence_hashes':{p:h for p,h in expected.items() if Path(p).parts[0]=='evidence'},
         'historical_source_copies':mutable,
         'package_before':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'warp_sph_realtime').rglob('*') if p.is_file() and '__pycache__' not in p.parts},
         'git_status':subprocess.run(['git','status','--short','--branch'],capture_output=True,text=True).stdout}
(OUT/'preservation-before.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'M0_files_verified':len(expected),'historical_sources_copied':len(mutable)}))
