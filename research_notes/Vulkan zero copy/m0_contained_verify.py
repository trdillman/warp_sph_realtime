"""Audit exact retained M0 evidence; no GPU or Blender execution."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0'
def read(path):return json.loads((OUT/path).read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
core=read('native-corrected/receipt.json')
life=read('lifecycle-corrected/lifecycle.json')
final=read('final-window/cancel.json')
checks={}
checks['primitive_containment']=read('supervisor-controls-reviewed/controls.json')['pass'] and read('extra-controls/results.json')['pass'] and read('creation-control/rejection.json')['pass']
checks['final_window_contained']=final['synthetic_pending'] and final['exit_pre_seen'] and final['captured_owner']==final['callback_window'] and read('final-window/watch/result.json')['exit_confirmed']
checks['exact_uuid_equal']=core['vulkan_uuid']==core['cuda_uuid']=='5d09f77ba936a6d9054d0324aef67adb' and core['uuid_negative_rejected']
checks['native_full_pattern']=core['status']=='PASS_BOUNDED_NATIVE_CORE' and [r['n'] for r in core['rows']]==[1,4095,4096,4097,4194304] and all(r['mismatches']==0 for r in core['rows'])
checks['gpu_stress']=core['stress_frames']==10000 and core['stress_error_sum']==0 and core['readbacks_stress']==1
checks['recreate_50']=len(core['cycles'])==50 and len({x['process_handles'] for x in core['cycles']})==1 and len({x['free_device_bytes'] for x in core['cycles']})==1
checks['lifecycle']=life['status']=='PASS_BOUNDED_LIFECYCLE' and [x['case'] for x in life['normal']]==['unregister','producer_stop','load_pre','last_view_loss'] and all(x['seconds']<.1 for x in life['normal'])
checks['partial_rollback']=len(life['partial'])==6 and all(x['owned_resources']==0 for x in life['partial'])
checks['negative_gpu_control']=life.get('gpu_negative_control',{}).get('observed_mismatches')==4097
checks['context_restored']=bool(life.get('contexts')) and all(x['after_reset']==x['captured'] and x['after_restore']==x['prior'] for x in life['contexts'])
checks['normal_process_exits']=all(read(p+'/watch/result.json')['exit_code']==0 and read(p+'/watch/result.json')['exit_confirmed'] for p in ['native-corrected','lifecycle-corrected'])
host=json.loads((OUT.parent/'20261017-m0/host-preflight.json').read_text())
package={p:h for p,h in host['hashes'].items() if str(ROOT/'warp_sph_realtime') in p}
current={str(p) for p in (ROOT/'warp_sph_realtime').rglob('*.py') if '__pycache__' not in p.parts}
checks['package_python_unchanged']=set(package)==current and all(sha(Path(p))==h for p,h in package.items())
owned=[json.loads(p.read_text())['structuredContent']['pid'] for p in OUT.glob('launch-*.json')]
result=subprocess.run(['pwsh','-NoProfile','-Command',f"@(Get-Process -Id {','.join(map(str,owned))} -ErrorAction SilentlyContinue).Count"],capture_output=True,text=True)
checks['owned_processes_exited']=result.stdout.strip()=='0'
snapshots=[('source','source-manifest.json'),('source-corrected','source-manifest-corrected.json'),
           ('source-lifecycle','source-manifest-lifecycle.json'),('source-final-window','source-manifest-final-window.json'),
           ('source-lifecycle-corrected','source-manifest-lifecycle-corrected.json')]
checks['frozen_harnesses_match']=all(all(sha(OUT/folder/name)==digest for name,digest in read(manifest).items()) for folder,manifest in snapshots)
verification={'checks':checks,'all_pass':all(checks.values()),'scope':'bounded Windows exact-build M0 prerequisite evidence only',
              'M1_started':False,'owned_pids':owned,'package_python_files':len(package),
              'limits':['No product integration/release acceptance','No 4M latency benchmark','No shared-GPU fault isolation','Blender retirement is deferred; numeric no-growth is bounded']}
verification['gate_status']={'M0':'BLOCKED' if not all(checks.values()) else 'PASS_BOUNDED',
    'unregister':'PASS' if checks['lifecycle'] else 'FAILED',
    'producer_stop':'PASS' if checks['lifecycle'] else 'NOT_RUN',
    'load_pre':'PASS' if checks['lifecycle'] else 'NOT_RUN',
    'last_view_loss':'PASS' if checks['lifecycle'] else 'NOT_RUN',
    'negative_gpu_control':'PASS' if checks['negative_gpu_control'] else 'UNVERIFIED_RECEIPT',
    'artifact_integrity':'PASS' if all(checks[x] for x in ['package_python_unchanged','owned_processes_exited','frozen_harnesses_match']) else 'FAIL'}
(OUT/'environment.json').write_text(json.dumps({'gpu_inventory':subprocess.run(
    ['nvidia-smi','--query-gpu=name,driver_version,uuid,memory.total,memory.used','--format=csv,noheader'],
    capture_output=True,text=True).stdout.strip()},indent=2))
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
paths=[p for p in OUT.rglob('*') if p.is_file() and 'sessions' not in p.parts and '__pycache__' not in p.parts and p.name!='manifest.json']
paths+=list((ROOT/'research_notes/Vulkan zero copy').glob('m0_*'))
paths+=[ROOT/'docs/VULKAN_ZERO_COPY_PLAN.md']
(OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},indent=2))
print(json.dumps(verification,indent=2))
raise SystemExit(0 if verification['all_pass'] else 1)
