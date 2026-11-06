"""Admission must cover later populations before native geometry replacement."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def test_later_frame_cannot_exceed_persisted_native_budget(tmp_path, monkeypatch):
    from warp_sph_realtime.cache import native_sequence

    bpy = NS(app=NS(is_job_running=lambda _: False), path=NS(abspath=lambda p: p))
    monkeypatch.setitem(sys.modules, 'bpy', bpy)
    path = Path(__file__).parents[1] / 'warp_sph_realtime/cache/native_playback.py'
    spec = importlib.util.spec_from_file_location('warp_sph_realtime.cache._budget_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'frame.npz').write_bytes(b'frame')
    population = [1]
    reads, applied = [], []

    class Reader:
        directory = tmp_path
        span = (1, 2)

        @property
        def document(self):
            return {'frames': [dict(file='frame.npz', payload_bytes=100, count=population[0])] * 2}

        def clamp(self, number):
            return number

        def read(self, number):
            reads.append(number)
            return NS(nbytes=100, count=population[0])

    monkeypatch.setattr(native_sequence, 'SequenceReader', lambda _: Reader())
    monkeypatch.setattr(module, 'apply_native_frame', lambda *args: applied.append(args[1].count))

    class Object(dict):
        type = 'EMPTY'
        name = 'player'
        hide_render = False

        def as_pointer(self):
            return 1

    obj = Object(warpsph_native_sequence=str(tmp_path), warpsph_native_sequence_max_bytes=10000)
    scene = NS(objects=[obj], frame_current=1, render=NS(use_lock_interface=True),
               unit_settings=NS(scale_length=1), warp_sph=NS(sdk_native_budget_mb=1))
    module.update_scene(scene)
    assert applied == [1] and reads == [1]
    population[0] = 10
    scene.frame_current = 2
    module.update_scene(scene)
    assert applied == [1] and reads == [1]
    assert 'budget' in obj['warpsph_native_error'] and obj['warpsph_native_frame'] == -1
    assert obj.hide_render
    with pytest.raises(ValueError, match='budget'):
        module._check_frame_budget(100, 10, 10000)


def test_manual_snapshot_and_load_use_configured_budget(monkeypatch):
    import warp_sph_realtime
    sim_session = NS(get=None, active=None)
    monkeypatch.setattr(warp_sph_realtime, 'sim_session', sim_session, raising=False)
    monkeypatch.setitem(sys.modules, 'warp_sph_realtime.sim_session', sim_session)

    bpy = NS(types=NS(Operator=object, Panel=object),
             props=NS(StringProperty=lambda **kw: None), path=NS(abspath=lambda p: p))
    monkeypatch.setitem(sys.modules, 'bpy', bpy)
    path = Path(__file__).parents[1] / 'warp_sph_realtime/sdk_ui.py'
    spec = importlib.util.spec_from_file_location('warp_sph_realtime._budget_ui', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = NS(sdk_snapshot_name='snapshot', sdk_snapshot_owner='owner', sdk_native_budget_mb=2,
                  sdk_animation_dir='cache', sdk_animation_name='animation')
    scene = NS(warp_sph=settings)
    requests, loads = [], []
    monkeypatch.setattr(sim_session, 'get', lambda _: NS(session=NS(request_snapshot=requests.append)))
    monkeypatch.setattr(sim_session, 'active', lambda: None)
    snapshot = module.WM_OT_warp_sph_sdk_snapshot()
    snapshot.poll = lambda _: True
    snapshot.report = lambda *args: None
    assert snapshot.execute(NS(scene=scene)) == {'FINISHED'}
    assert requests[0]['max_bytes'] == requests[0]['max_native_bytes'] == 2 * 1024 * 1024
    from warp_sph_realtime import cache
    native_playback = NS(attach=lambda *args, **kw: loads.append(kw))
    monkeypatch.setattr(cache, 'native_playback', native_playback, raising=False)
    monkeypatch.setitem(sys.modules, 'warp_sph_realtime.cache.native_playback', native_playback)
    load = module.WM_OT_warp_sph_sdk_load_animation()
    load.report = lambda *args: None
    assert load.execute(NS(scene=scene)) == {'FINISHED'}
    assert loads == [{'max_bytes': 2 * 1024 * 1024}]
