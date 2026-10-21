"""Real candidate CPU-only cache repair and actual bounded helper-timeout checks."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from warp_sph_realtime.render import blender_symbols as bs
OUT=ROOT/'evidence/vulkan-zero-copy/20261018-m1'
exe=Path(json.loads((ROOT/'evidence/vulkan-zero-copy/20261017-m0/host-preflight.json').read_text())['candidate']['exe'])
profile=bs.load_profile();ident=profile['identity']
args={'module_base':0x140000000,'build_hash':ident['build_hash'],'python_abi':ident['python_abi']}
rows=[]
def record(name,result,begin,**extra):
    rows.append({'case':name,'available':result.available,'reason':result.reason,'detail':result.detail,
                 'seconds':time.perf_counter()-begin,**extra})
    (OUT/'real-rejection-probes.json').write_text(json.dumps({'rows':rows,'GPU_calls':0,'pointer_extractions':0},indent=2))

begin=time.perf_counter()
result=bs.resolve_symbols(exe,OUT/'missing.pdb',**args)
record('missing_real_PDB',result,begin)
assert not result.available and result.reason=='pdb_missing'

begin=time.perf_counter()
result=bs.resolve_symbols(exe,exe.with_suffix('.pdb'),**args,helper_timeout=.001)
record('actual_helper_timeout_1ms',result,begin)
assert not result.available and result.reason=='helper_timeout'

cache=OUT/'poisoned-cache.json'
trusted=json.loads((OUT/'resolver-cache.json').read_text())
bad=json.loads(json.dumps(trusted));key='GPU_pixel_buffer_create'
bad['rvas'][key]+=1
cache.write_text(json.dumps(bad))
calls=[]
def observed_helper(request,timeout):
    calls.append(time.perf_counter());return bs._run_helper(request,timeout)
begin=time.perf_counter()
result=bs.resolve_symbols(exe,exe.with_suffix('.pdb'),**args,cache_path=cache,helper_runner=observed_helper,helper_timeout=90)
repaired=json.loads(cache.read_text())
record('in_range_wrong_RVA_rejected_then_repaired_once',result,begin,helper_calls=len(calls),
       poisoned_rva=bad['rvas'][key],repaired_rva=repaired['rvas'][key])
assert result.available and len(calls)==1 and repaired==trusted
print(json.dumps({'status':'PASS_REAL_CPU_REJECTIONS','rows':rows},indent=2))
