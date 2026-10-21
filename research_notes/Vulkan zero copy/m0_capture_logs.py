"""Retain launcher-owned receipts/logs locally without modifying the source artifacts."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0'
records=[]
for path in OUT.glob('launch-*.json'):
    launch=json.loads(path.read_text())['structuredContent']
    dest=OUT/'retained-launch-logs'/path.stem
    dest.mkdir(parents=True,exist_ok=True)
    for field in ['receipt_path','log_path']:
        original=Path(launch[field]); target=dest/original.name
        if target.exists():continue  # Preserve first retained snapshot.
        shutil.copy2(original,target)
        records.append({'source':str(original),'retained':str(target.relative_to(ROOT)),
                        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'note':'launcher stdout log can contain appended prior sessions'})
if records:
    target=OUT/'retained-launch-logs'/('capture-'+str(len(list((OUT/'retained-launch-logs').glob('capture-*.json'))))+'.json')
    target.write_text(json.dumps(records,indent=2))
print(f'Retained {len(records)} files')
