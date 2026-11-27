"""Start the installed demo through a dedicated Blender Gate GUI lease."""
import json
import time
from pathlib import Path

path = Path(__file__).resolve().with_name('demo_probe.py')
# Reuse the verifiable archive installation and scene setup. Keep this
# launcher separate from the exclusive performance probe.
setup = path.read_text(encoding='utf-8').split('original_due = ', 1)[0]
setup = setup.replace("out / 'installed'", "out / 'review-installed'")
setup = setup.replace('WarpSPH-Live-Output.blend', 'WarpSPH-Live-Output-Demo.blend')
exec(compile(setup, str(path), 'exec'))

def check():
    q = addon._session(bpy.context)
    assert q.running and not q.last_error, q.last_error
    assert window.screen.is_animation_playing
    if q.step_count == 0 or scene.warp_sph.sdk_live_name not in bpy.data.objects:
        assert time.monotonic() < deadline, 'Live output did not start'
        return .5
    (out / 'open-demo.json').write_text(json.dumps(dict(
        pid=os.getpid(), scene=bpy.data.filepath, steps=q.step_count, playing=True,
        count=q.solver.num_particles, enabled=addon_utils.check(module.__name__)[1],
        archive=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    ), indent=2))
    return None

def start_demo():
    assert 'RUNNING_MODAL' in op('wm.warp_sph_start')
    op('screen.animation_play')
    bpy.app.timers.register(check, first_interval=.5)
    return None

deadline = time.monotonic() + 30
bpy.app.timers.register(start_demo, first_interval=.5)
