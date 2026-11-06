"""Narrow UI/SimSession routing checks reuse existing isolated lifecycle harness."""

import subprocess
import sys
from pathlib import Path

from test_transport_session import LIFECYCLE

ROOT = Path(__file__).resolve().parents[1]
EXTRA = r"""
from warp_sph_realtime import interop_bridge,sdk_dependency
shared=[];preflight=[]
sdk_dependency.validate_environment=lambda:preflight.append('checked')
class Shared(ts.TransportSession):
    is_shared_sdk=True
    def __init__(self,solver,scene,window):
        super().__init__(Transport(True),'sdk_positions',Renderer(),'VULKAN',count=solver.num_particles)
        self.solver=solver;self.scene=scene;self.binds=[];self.revisions=0;self.serviced=0
        self.snapshot_status='IDLE'
    def try_bind_solver(self,solver):self.binds.append(solver);self.solver=solver;return True
    def note_scene_applied(self):self.revisions+=1
    def service_snapshot(self):self.serviced+=1;return False

def create_shared(solver,scene,window):
    result=Shared(solver,scene,window);shared.append(result);return result
interop_bridge.open_session=create_shared
ctx=reset();ctx.scene.warp_sph.use_shared_sdk=True
op,result=start(ctx)
assert result=={'RUNNING_MODAL'},op.reports
assert len(shared)==1 and preflight==['checked'],'explicit SDK selection not routed through shared bridge'
actual=sim_session.get(ctx.scene)
assert actual.session is shared[0]
actual.paused=True
assert actual.tick(ctx,lambda *a:None)
assert shared[0].serviced==1,'paused timer failed to service captured native snapshot'
from warp_sph_realtime import sdk_ui
requests=[]
shared[0].request_snapshot=lambda request:requests.append(request)
actual.step_count=1
ctx.scene.warp_sph.sdk_snapshot_name='fixture snapshot'
ctx.scene.warp_sph.sdk_snapshot_owner='ownerA'
ctx.scene.warp_sph.sdk_native_budget_mb=2
assert sdk_ui.WM_OT_warp_sph_sdk_snapshot().execute(ctx)=={'FINISHED'}
assert requests[0]['max_bytes']==requests[0]['max_native_bytes']==2*1024*1024
ctx.scene.warp_sph.sdk_snapshot_owner='ownerB'
assert sdk_ui.WM_OT_warp_sph_sdk_snapshot().execute(ctx)=={'FINISHED'}
assert requests[0]['collection']!=requests[1]['collection'],'two scenes share the same native output collection name'
actual.step_count=0

assert actual.reset(ctx,lambda *a:None)
assert shared[0].binds and shared[0].solver is actual.solver,'reset left bridge bound to stale solver arrays'
assert shared[0].revisions>0
from warp_sph_realtime import native_output
requests=[]
native_output.begin_bake=lambda owner,context,report,**kw:requests.append((owner,kw)) or True
assert actual.begin_bake(ctx,lambda *a:None,first=1,last=4,resume=False)
assert requests==[(actual,{'first':1,'last':4,'resume':False})],'shared native animation route not used'
actual.stop(ctx);assert shared[0].close_result=='completed'
ctx=reset();ctx.scene.warp_sph.use_shared_sdk=True
before=len(shared)
def bad_sdk():raise RuntimeError('wrong exact SDK package identity')
sdk_dependency.validate_environment=bad_sdk
op,result=start(ctx)
assert result=={'CANCELLED'} and len(shared)==before
assert any('identity' in msg for kind,msg in op.reports)
assert not sim_session.get(ctx.scene).running
print('SDK_SELECTION_LIFECYCLE_OK')
"""


def test_shared_sdk_selection_reset_snapshot_and_missing_dependency():
    script = LIFECYCLE.split("reset()\next.unregister()")[0] + EXTRA
    result = subprocess.run([sys.executable, "-"], input=script, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SDK_SELECTION_LIFECYCLE_OK" in result.stdout
