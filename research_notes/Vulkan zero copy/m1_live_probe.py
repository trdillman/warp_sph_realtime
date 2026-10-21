"""Fresh Blender process resolver proof. No GPU calls, pointer extraction or imports."""
import dataclasses,hashlib,json,os,sys,time,traceback
from collections.abc import Mapping
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[2]
LANE=globals().get('LANE','fresh-a')
assert LANE in ('fresh-a','fresh-b')
OUT=ROOT/'evidence/vulkan-zero-copy/20261018-m1'
TARGET=OUT/LANE
sys.path.insert(0,str(Path(__file__).parent))
from m0_supervisor import save
ready=json.loads((TARGET/'watch/ready.json').read_text());expected=json.loads((TARGET/'expected.json').read_text())
assert ready['job_member'] and ready['kill_on_close'] and ready['pid']==os.getpid()
assert all(ready[k]==expected[k] for k in ('pid','creation_time','nonce'))
sys.path.insert(0,str(ROOT))
def plain(value):
    if dataclasses.is_dataclass(value):return {f.name:plain(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,Mapping):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value
def run():
    record={'pid':os.getpid(),'lane':LANE,'gpu_native_calls':0,'texture_pointer_extractions':0,
            'scope':'resolver-only; no transport integration','build_hash':bpy.app.build_hash.decode(),
            'user_paths':{k:v for k,v in os.environ.items() if k.startswith('BLENDER_USER_')}}
    try:
        from warp_sph_realtime.render import blender_symbols
        begin=time.perf_counter()
        result=blender_symbols.resolve_current_blender(cache_path=OUT/'resolver-cache.json',helper_timeout=90.0)
        record.update(result=plain(result),seconds=time.perf_counter()-begin,
                      module_sha256=hashlib.sha256(Path(blender_symbols.__file__).read_bytes()).hexdigest())
        assert result.available,plain(result)
        record['status']='PASS_RESOLVER_ONLY'
    except BaseException:record.update(status='FAIL',error=traceback.format_exc())
    finally:
        save(TARGET/'resolver.json',record)
        bpy.ops.wm.quit_blender()
    return None
bpy.app.timers.register(run,first_interval=.1)
