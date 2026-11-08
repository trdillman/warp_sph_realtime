import pytest

from warp_sph_realtime.timeline import needs_step


def test_initial_pause_and_frame_target_do_not_run_free():
    assert not needs_step(1, 1, 24, 0, 0.002)
    assert needs_step(2, 1, 24, 0, 0.002)
    assert not needs_step(2, 1, 24, 0.042, 0.002)


def test_full_solver_step_includes_substeps_and_skipped_frames():
    elapsed = 0.0
    steps = 0
    while needs_step(4, 1, 24, elapsed, 0.001 * 4):
        elapsed += 0.004
        steps += 1
    assert steps == 32
    assert 0 <= elapsed - 3 / 24 < 0.004
    assert not needs_step(2, 1, 24, elapsed, 0.004)


@pytest.mark.parametrize('fps,step', [(0, 0.1), (24, 0), (float('nan'), 0.1)])
def test_invalid_timing_rejects(fps, step):
    with pytest.raises(ValueError):
        needs_step(2, 1, fps, 0, step)


def test_step_longer_than_frame_rejects_instead_of_silent_zero_step():
    with pytest.raises(ValueError, match='exceeds a timeline frame'):
        needs_step(2, 1, 24, 0, 0.1)


def test_real_session_timeline_pause_seek_backpressure_and_reset():
    import subprocess
    import sys
    from pathlib import Path

    from test_transport_session import LIFECYCLE

    extra = r'''
ctx=reset()
ctx.screen.is_animation_playing=False
ctx.scene.frame_current=1
ctx.scene.frame_start=1
ctx.scene.frame_end=200
ctx.scene.render=NS(fps=25,fps_base=1.0)
ctx.scene.frame_set=lambda f:setattr(ctx.scene,'frame_current',f)
op,result=start(ctx)
assert result=={'RUNNING_MODAL'}
rt=sim_session.get(ctx.scene)
for _ in range(4):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==0 and rt.paused
ctx.scene.frame_set(2)
rt.session.transport.ready=False
rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==0 and rt._timeline_elapsed==0
rt.session.transport.ready=True
for _ in range(30):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==20,(rt.solver.steps,rt._timeline_elapsed)
ctx.screen.is_animation_playing=True
ctx.scene.frame_set(4)
for _ in range(60):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==60
ctx.scene.frame_set(100)
for _ in range(3):rt.tick(ctx,lambda *a:None)
before_pause=rt.solver.steps
assert before_pause==63
ctx.screen.is_animation_playing=False
ctx.scene.frame_set(101)  # final playback frame can arrive together with Pause
for _ in range(30):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==before_pause,'timeline Pause must cancel unfinished playback catchup'
assert rt._timeline_elapsed==before_pause*0.002 or abs(rt._timeline_elapsed-before_pause*0.002)<1e-10
ctx.screen.is_animation_playing=True
ctx.scene.frame_set(102)
rt.tick(ctx,lambda *a:None)
before_pause=rt.solver.steps
def cancel_playback(**kwargs):ctx.screen.is_animation_playing=False;return {'FINISHED'}
def play_timeline():ctx.screen.is_animation_playing=True;return {'FINISHED'}
bpy.ops.screen=NS(animation_cancel=cancel_playback,animation_play=play_timeline)
assert addon.WM_OT_warp_sph_pause().execute(ctx)=={'FINISHED'}
ctx.scene.frame_set(103)
for _ in range(30):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==before_pause and rt.paused,'addon Pause must cancel physics backlog'
assert addon.WM_OT_warp_sph_pause().execute(ctx)=={'FINISHED'}
assert ctx.screen.is_animation_playing
rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==before_pause+1,'addon Resume must use timeline playback'
assert addon.WM_OT_warp_sph_step_frame().execute(ctx)=={'FINISHED'}
before_step=rt.solver.steps
rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==before_step+1 and not ctx.screen.is_animation_playing
ctx.screen.is_animation_playing=False
rt.tick(ctx,lambda *a:None)
ctx.scene.frame_set(4)  # explicit paused seek may evaluate the requested frame
for _ in range(65):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==60
ctx.screen.is_animation_playing=False
for _ in range(3):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==60
ctx.scene.warp_sph.particle_count=2000
for _ in range(65):rt.tick(ctx,lambda *a:None)
assert rt.start_frame==1 and ctx.scene.frame_current==4 and rt.solver.steps==60
ctx.scene.frame_set(2)
for _ in range(30):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==20 and ctx.scene.frame_current==2
rt._timeline_rebuild=True  # legacy bake completion invalidates the solver clock
ctx.screen.is_animation_playing=True
for _ in range(30):rt.tick(ctx,lambda *a:None)
assert rt.solver.steps==20 and not rt._timeline_rebuild
assert not ctx.screen.is_animation_playing
assert rt.reset(ctx,lambda *a:None) and ctx.scene.frame_current==1
assert rt.solver.steps==0
rt.stop(ctx)
ctx.scene.frame_current=10
op,result=start(ctx)
assert ctx.scene.frame_current==1 and sim_session.get(ctx.scene).start_frame==1
sim_session.get(ctx.scene).stop(ctx)
print('TIMELINE_OK')
'''
    script = LIFECYCLE.split("reset()\next.unregister()")[0] + extra
    result = subprocess.run([sys.executable, '-'], input=script, text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TIMELINE_OK' in result.stdout
