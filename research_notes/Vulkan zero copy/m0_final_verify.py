"""Reconcile original M0 gates with preserved evidence and supplemental receipt."""
import hashlib,json,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261017-contained-m0'
BASE=OUT.parent/'20261017-m0'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
q=read(OUT/'qualification/qualification.json');core=read(OUT/'native-corrected/receipt.json')
life=read(OUT/'lifecycle-owned/lifecycle.json');host=read(BASE/'host-preflight.json')
checks={}
for backend in ['vulkan','opengl']:
    pub=read(BASE/f'public-{backend}.json')
    checks[f'{backend}_instanced_full_pattern']=pub['status']=='PASS_PUBLIC_PREREQUISITES_ONLY' and [x['n'] for x in pub['rows']]==[0,1,4095,4096,4097,4194304] and all(x['status']=='PASS' and x.get('mismatches',0)==0 for x in pub['rows']) and not pub['rows'][0]['allocated'] and not pub['rows'][0]['drawn']
    old=read(BASE/'manifest.json')
    public_source=ROOT/'research_notes/Vulkan zero copy/m0_public_probe.py'
    checks[f'{backend}_public_receipt_hash']=sha(BASE/f'public-{backend}.json')==old[str((BASE/f'public-{backend}.json').relative_to(ROOT))] and sha(public_source)==old[str(public_source.relative_to(ROOT))]
    checks[f'{backend}_isolated']=all(str(BASE).lower() in v.lower() for v in pub['user_paths'].values()) and bool(pub['user_paths'])
checks['prior_core_and_cleanup_audit']=read(OUT/'verification-owned.json')['all_pass']
checks['supplement_status']=q['status']=='PASS_BOUNDED_QUALIFICATION'
checks['sizing']=all(q['sizing'][k] for k in ['zero_no_allocation','device_limit_rejected_before_native','insufficient_budget_rejected_before_native','actual_limits_checked']) and q['sizing']['native_calls_during_validation']==q['sizing']['imports_during_validation']==0 and len(q['sizing']['invalid'])==5 and all(x['rejected'] for x in q['sizing']['invalid']) and q['limits']['max_texture_size']>=4096 and q['limits']['free_device_bytes']>=8*4194304*16+(1<<30)
checks['uuid_constructor_negative']=len(q['uuid_constructor_rejections'])==2 and all(x['rejected'] for x in q['uuid_constructor_rejections'])
checks['cadence_counts']=q['consumed']==120 and q['accepted']==q['consumed']+q['stale_ready_retired'] and all(q[k]>0 for k in ['publication_drops','stale_ready_retired','repeated_texture_draws','cadence_deferred_draws']) and q['cadence_consumed']=={'1':40,'3':40,'7':40}
samples=q['completion_samples'];frames=[x['frame'] for x in samples]
checks['generation_compare']=q['gpu_error_sum']==0 and q['readbacks']==1 and q['generation_rejections']==2 and len(q['generations'])==2 and all(x['consumed_end']-x['consumed_start']==60 for x in q['generations']) and frames==sorted(set(frames)) and all(x['frame']//1000000==x['generation'] for x in samples)
totals={};allowed={'CUDA':{0,600},'VULKAN':{0,2}}
for tick,api,result,stamp in q['query_results']:
    assert result in allowed[api]
    key=(tick,api);totals[key]=totals.get(key,0)+1
checks['query_caps_and_outcomes']=bool(totals) and max(totals.values())<=2 and q['max_cuda_queries_per_tick']<=2 and q['max_vulkan_queries_per_tick']<=2
checks['completion_clock']=len(samples)==120 and q['clock']['monotonic'] and not q['clock']['adjustable'] and all(x['published_ns']<=x['cuda_ready_observed_ns']<=x['upload_returned_ns']<=x['submitted_ns']<=x['completed_observed_ns'] and x['upper_bound_ns']==x['completed_observed_ns']-x['published_ns'] for x in samples)
watch=read(OUT/'qualification/watch/result.json');expected=read(OUT/'qualification/expected.json')
checks['new_process_exit']=watch['exit_confirmed'] and watch['exit_code']==0 and watch['reason']=='natural_exit' and all(watch[k]==expected[k] for k in ['pid','creation_time','nonce'])
package={p:h for p,h in host['hashes'].items() if str(ROOT/'warp_sph_realtime') in p}
checks['package_unchanged']=set(package)=={str(p) for p in (ROOT/'warp_sph_realtime').rglob('*.py')} and all(sha(Path(p))==h for p,h in package.items())
checks['supplement_snapshot']=all(sha(OUT/'source-qualification'/name)==digest for name,digest in read(OUT/'source-manifest-qualification.json').items())
checks['prior_snapshots']=all(all(sha(OUT/folder/name)==digest for name,digest in read(OUT/manifest).items()) for folder,manifest in [('source-corrected','source-manifest-corrected.json'),('source-lifecycle-owned','source-manifest-lifecycle-owned.json')])
owned=[read(p)['structuredContent']['pid'] for p in OUT.glob('launch-*.json')]
proc=subprocess.run(['pwsh','-NoProfile','-Command',f"@(Get-Process -Id {','.join(map(str,owned))} -ErrorAction SilentlyContinue).Count"],capture_output=True,text=True)
checks['all_owned_processes_exited']=proc.stdout.strip()=='0'
values=sorted(x['upper_bound_ns']/1e6 for x in samples)
timing={'scope':'N4097 host completion-observed upper bounds; includes cadence, polling and consumer; NOT M5 performance',
        'samples':len(values),'p50_ms':values[math.ceil(.5*len(values))-1],'p95_ms':values[math.ceil(.95*len(values))-1],'max_ms':max(values),'clock':q['clock']}
gates={'candidate_identity_and_isolation':checks['prior_core_and_cleanup_audit'] and checks['package_unchanged'],
       'texture_abi_uuid_and_rejection':checks['uuid_constructor_negative'] and core['vulkan_uuid']==core['cuda_uuid'],
       'bounded_fences_timer_reuse':checks['query_caps_and_outcomes'] and checks['cadence_counts'],
       'synchronous_lifecycle':life['status']=='PASS_BOUNDED_LIFECYCLE' and len(life['normal'])==4,
       'both_backends_counts_padding_limits':all(checks[k] for k in ['vulkan_instanced_full_pattern','opengl_instanced_full_pattern','sizing']),
       'stress_and_cadence':core['stress_frames']==10000 and core['stress_error_sum']==0 and core['readbacks_stress']==1 and checks['generation_compare'],
       'recreate_and_partial_rollback':len(core['cycles'])==50 and len(life['partial'])==6,
       'completion_observation_instrumentation':checks['completion_clock']}
result={'status':'PASS_BOUNDED_M0' if all(checks.values()) and all(gates.values()) else 'FAIL',
        'checks':checks,'M0_gates':gates,'timing_observation':timing,'M1_started':False,
        'package_python_files':len(package),'owned_pids':owned,
        'limits':['Exact Windows Blender candidate only','Scratch probes, not product integration',
                  'M5 4M performance NOT MEASURED','No real GPU fault injection or shared-GPU isolation',
                  'Final-window timeout safety is disposable process containment',
                  'Reported texture capability and injected allocation-budget guards; no actual OOM test',
                  'Export rounding and other-process VRAM races require production handling',
                  'Physical Vulkan deallocation and universal bounded native-call duration not established']}
(OUT/'verification-final.json').write_text(json.dumps(result,indent=2))
paths=[p for p in OUT.rglob('*') if p.is_file() and 'sessions' not in p.parts and '__pycache__' not in p.parts and p.name!='manifest-final.json']
paths+=list((ROOT/'research_notes/Vulkan zero copy').glob('m0_*'))
paths.append(ROOT/'research_notes/Vulkan zero copy/SESSION_SUMMARY.md')
paths += [ROOT/'docs'/name for name in ['VULKAN_ZERO_COPY_PLAN.md','RELEASE_PLAN.md','PRODUCT_DELIVERY.md','TRACKER_UPDATE_PENDING.md']]
(OUT/'manifest-final.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},indent=2))
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['status']=='PASS_BOUNDED_M0' else 1)
