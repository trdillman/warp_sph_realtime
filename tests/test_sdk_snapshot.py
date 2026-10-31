"""Explicit paused-solver capture contract, separate from Blender/GPU qualification."""
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


def api():
    assert find_spec('warp_sph_realtime.sdk_snapshot'), 'real-solver snapshot adapter missing'
    from warp_sph_realtime import sdk_snapshot
    return sdk_snapshot

class Array:

    def __init__(self, value, dtype, reads):
        self.value = np.asarray(value)
        self.dtype = dtype
        self.device = 'cuda:0'
        self.ndim = 1
        self.shape = (len(value),)
        self.is_contiguous = True
        self.reads = reads

    def __getitem__(self, index):
        return Array(self.value[index], self.dtype, self.reads)

    def numpy(self):
        self.reads.append(len(self.value))
        return self.value.copy()

class Solver:
    pass

def setup():
    reads = []
    scene = NS(warp_sph=NS(particle_radius=0.02, particle_color=(0.2, 0.4, 0.6, 1.0)))
    solver = Solver()
    solver.device = 'cuda:0'
    solver.num_particles = 5
    solver.active_count = 3
    solver.sim_time = 0.25
    solver._next_id = 5
    solver.h = 0.04
    solver.pos_sim = Array(np.arange(15, dtype=np.float32).reshape(5, 3), 'vec3', reads)
    solver.vel = Array(np.arange(15, dtype=np.float32).reshape(5, 3) * 0.5, 'vec3', reads)
    solver.pid = Array(np.array([4, 1, 3, -7, -7], dtype=np.int32), 'int32', reads)
    solver.density_sorted = Array(np.array([1000, 1001, 1002, np.nan, np.nan], dtype=np.float32), 'float32', reads)
    session = NS(scene=scene, scene_uid=44, solver=solver, running=True, paused=True, bake=None, step_once=False, step_count=12, token=2, restart_required='', _output_pending=False)
    wp = NS(__version__='1.17.0', array=Array, vec3='vec3', int32='int32', float32='float32', get_device=lambda name: name, synchronize_device=lambda name: reads.append('sync'))
    return (scene, session, wp, reads)

def test_paused_capture_preserves_solver_ids_sort_order_and_units():
    m = api()
    scene, session, wp, reads = setup()
    before = session.solver.pos_sim.value.copy()
    payload = m.capture_paused(session, scene, wp_module=wp)
    values = {a.name: np.frombuffer(a.data, dtype={'float32': '<f4', 'uint64': '<u8'}[a.scalar_type]).reshape(a.shape) for a in payload.arrays}
    np.testing.assert_array_equal(values['id'], [4, 1, 3])
    assert values['id'].dtype == np.uint64
    np.testing.assert_array_equal(values['position'], before[:3])
    np.testing.assert_array_equal(session.solver.pos_sim.value, before)
    assert reads == ['sync', 3, 3, 3, 3]
    assert payload.spatial.meters_per_unit == 1 and payload.stamp.simulation_time_s == 0.25 and (payload.stamp.frame_id == 12)
    assert payload.counts['active_count'] == 3 and payload.metadata['density_sampling'] == 'last_substep_force_density'
    assert session.solver.sim_time == 0.25 and session.step_count == 12 and session.paused

@pytest.mark.parametrize('field,value', [('paused', False), ('running', False), ('bake', object()), ('step_once', True), ('restart_required', 'fatal'), ('_output_pending', True)])
def test_unstable_session_rejects_before_readback(field, value):
    m = api()
    scene, session, wp, reads = setup()
    setattr(session, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        m.capture_paused(session, scene, wp_module=wp)
    assert not reads

def test_foreign_scene_and_memory_budget_reject_before_readback():
    m = api()
    scene, session, wp, reads = setup()
    with pytest.raises(ValueError):
        m.capture_paused(session, NS(), wp_module=wp)
    with pytest.raises(ValueError, match='budget'):
        m.capture_paused(session, scene, wp_module=wp, max_readback_bytes=1)
    assert not reads

@pytest.mark.parametrize('ids', [[-1, 1, 3], [4, 4, 1], [6, 1, 3]])
def test_invalid_or_reused_ids_reject_without_payload(ids):
    m = api()
    scene, session, wp, reads = setup()
    session.solver.pid.value[:3] = ids
    with pytest.raises(ValueError, match='identit|ID'):
        m.capture_paused(session, scene, wp_module=wp)

def test_solver_generation_changes_epoch_not_survivor_identity():
    m = api()
    scene, session, wp, _ = setup()
    first = m.capture_paused(session, scene, wp_module=wp)
    session.step_count += 1
    session.solver.sim_time += 0.01
    second = m.capture_paused(session, scene, wp_module=wp)
    assert first.stamp.epoch == second.stamp.epoch
    _, other, _, _ = setup()
    session.solver = other.solver
    session.step_count = 0
    third = m.capture_paused(session, scene, wp_module=wp)
    assert third.stamp.epoch == first.stamp.epoch + 1
    assert 'density' not in {a.name for a in third.arrays}, 'initial zero buffer is not a measured density'

def test_zero_count_does_not_read_capacity_or_export_ghosts():
    m = api()
    scene, session, wp, reads = setup()
    session.solver.active_count = 0
    p = m.capture_paused(session, scene, wp_module=wp)
    assert p.counts['active_count'] == 0
    assert all(a.shape[0] == 0 and not a.data for a in p.arrays)
    assert reads == ['sync']

def test_worker_thread_rejected_before_runtime_or_readback():
    m = api()
    scene, session, wp, reads = setup()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='main thread'):
            pool.submit(m.capture_paused, session, scene, wp_module=wp).result()
    assert not reads

def test_sdk_integrity_rejects_wrong_or_extra_payload_before_import(tmp_path):
    m = api()
    root = tmp_path / 'blender_vulkan_interop'
    root.mkdir()
    (root / '__init__.py').write_bytes(b'print("never execute")')
    lock = {'schema_version': 1, 'payload_sha256': {'__init__.py': hashlib.sha256((root / '__init__.py').read_bytes()).hexdigest()}}
    m.verify_sdk_payload(root, lock)
    (root / 'foreign.py').write_bytes(b'foreign')
    with pytest.raises(RuntimeError, match='identity|unexpected'):
        m.verify_sdk_payload(root, lock)
    (root / 'foreign.py').unlink()
    (root / '__init__.py').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='identity'):
        m.verify_sdk_payload(root, lock)

def test_adapter_import_does_not_load_sdk_warp_or_blender():
    m = api()
    root = Path(m.__file__).resolve().parents[1]
    code = 'import sys;sys.path.insert(0,' + repr(str(root)) + ');import warp_sph_realtime.sdk_snapshot;assert not set(("bpy","gpu","warp","blender_vulkan_interop")).intersection(sys.modules)'
    run = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stderr

@pytest.mark.parametrize('where', ['sync', 'array'])
def test_cuda_readback_failure_stops_session_instead_of_allowing_resume(where):
    m = api()
    scene, session, wp, reads = setup()

    def fail(*_args):
        raise RuntimeError('CUDA context lost')
    if where == 'sync':
        wp.synchronize_device = fail
    else:
        session.solver.pos_sim.numpy = fail
        original = session.solver.pos_sim.__class__.__getitem__

        def failing_slice(self, index):
            result = original(self, index)
            if self is session.solver.pos_sim:
                result.numpy = fail
            return result
        session.solver.pos_sim.__class__.__getitem__ = failing_slice
    try:
        with pytest.raises(RuntimeError, match='CUDA'):
            m.capture_paused(session, scene, wp_module=wp)
        assert not session.running and session.restart_required, 'unknown CUDA state was left resumable'
    finally:
        if where == 'array':
            Array.__getitem__ = original


def test_disposed_and_recreated_session_uses_new_identity_namespace():
    m = api()
    scene, session, wp, _ = setup()
    first = m.capture_paused(session, scene, wp_module=wp)
    _, replacement, _, _ = setup()
    replacement.scene = scene
    second = m.capture_paused(replacement, scene, wp_module=wp)
    assert first.stamp.source_id != second.stamp.source_id, 'new session recycled old particle identity namespace'
    assert first.stamp.epoch == second.stamp.epoch == 0
    again = m.capture_paused(session, scene, wp_module=wp)
    assert again.stamp.source_id == first.stamp.source_id
