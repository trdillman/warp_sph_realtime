"""Independent solver replay state: CPU doubles test contracts, not GPU acceptance."""
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    assert find_spec('warp_sph_realtime.cache.solver_replay'), 'isolated solver replay missing'
    from warp_sph_realtime.cache import solver_replay
    return solver_replay

@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    module = api()
    registry = SimpleNamespace(restart_required='', quarantine=[])
    monkeypatch.setattr(module, 'process_state', lambda: registry)
    monkeypatch.setattr(module, '_QUARANTINE', registry.quarantine)

class Array:

    def __init__(self, data):
        self.data = np.array(data, copy=True)
        self.size = self.data.size
        self.capacity = self.data.nbytes
        self.dtype = self.data.dtype
        self.device = 'cuda:0'
        self.ptr = id(self.data)

    def numpy(self):
        return self.data.copy()

class Stream:

    def __init__(self, device):
        self.waited = []
        self.synced = 0
        self.fail = False

    def wait_stream(self, other):
        self.waited.append(other)

    def _complete(self):
        self.synced += 1
        if self.fail:
            raise RuntimeError('CUDA stream failed')

class Grid:

    def __init__(self, *dims, device):
        self.dims = dims
        self.device = device
        self.reserved = 0

    def reserve(self, count):
        self.reserved = count

@dataclass
class Emitter:
    emitted: int = 3
    rate: float = 12.0

class Solver:

    def __init__(self):
        self.device = 'cuda:0'
        self.num_particles = 4
        self.active_count = 3
        self._next_id = 8
        self.sim_time = 0.4
        self.dt = 0.1
        self.substeps = 2
        self.unit_scale = 1.0
        self.fps = 24.0
        self.start_frame = 1.0
        self.h = 0.04
        self.stiffness = 10.0
        self.viscosity = 0.1
        self.mass = 1.0
        self.rest_density = 1.0
        self.cfl_factor = 0.2
        self.max_dt = None
        self.domain_min = (-1.0, -1.0, -1.0)
        self.domain_max = (1.0, 1.0, 1.0)
        self.gravity = (0.0, 0.0, -9.0)
        self.enable_obstacles = 0
        self.scene_mode = True
        self.colliders = SimpleNamespace(readonly=True)
        self.collider_radius = 0.04
        self._motions = ()
        self._key_frames = {}
        self._emitters = [Emitter()]
        self.emission_exhausted = False
        self._exhaustion_unreported = False
        self.boundary_fingerprint = 'fixed'
        self.grid = Grid(128, 128, 128, device='cuda:0')
        self.cuda_graph = object()
        self.pos_sim = Array(np.arange(12, dtype=np.float32).reshape(4, 3))
        self.vel = Array(np.ones((4, 3), dtype=np.float32))
        self.pid = Array(np.array([4, 2, 7, -1], dtype=np.int32))
        self.pos_sorted = Array(np.zeros((4, 3), dtype=np.float32))
        self.vel_sorted = Array(np.zeros((4, 3), dtype=np.float32))
        self.pid_sorted = Array(np.zeros(4, dtype=np.int32))
        self.density_sorted = Array(np.ones(4, dtype=np.float32))
        self.active = Array([3])
        self.collider_xform = self.collider_lin = self.collider_ang = self._key_p = self._key_q = self._motion_collider = None
        self.nozzle_pos = self.nozzle_vel = None

    def step(self, output, use_graph=True):
        for _ in range(self.substeps):
            self.pos_sim.data += self.vel.data * self.dt
            self.sim_time += self.dt
            self._emitters[0].emitted += 1
        output.data[:] = self.pos_sim.data

class Warp:
    __version__ = '1.17.0'
    array = Array
    Stream = Stream
    HashGrid = Grid
    vec3 = 'vec3'

    def __init__(self):
        self.operations = []
        self.stream = Stream('cuda:0')

    def clone(self, value):
        self.operations.append('clone')
        return Array(value.data)

    def empty(self, n, dtype, device):
        self.operations.append('empty')
        return Array(np.empty((n, 3), dtype=np.float32))

    def synchronize_stream(self, stream):
        stream._complete()

    def get_stream(self, device):
        return self.stream

    def ScopedStream(self, stream):
        return nullcontext()

def state(solver):
    return (solver.pos_sim.data.tobytes(), solver.pid.data.tobytes(), solver.sim_time, solver.dt, solver.substeps, solver._next_id, solver._emitters[0].emitted, dict(solver._key_frames))

def test_clone_copies_mutable_state_and_retains_only_immutable_mesh_owner():
    m = api()
    wp = Warp()
    live = Solver()
    before = state(live)
    clone = m.SolverReplay.clone_current(wp, live, expected_type=Solver, max_clone_bytes=128 << 20)
    assert clone.solver is not live and clone.solver.grid is not live.grid
    assert clone.solver.colliders is live.colliders
    for name in ('pos_sim', 'vel', 'pid', 'pid_sorted', 'active'):
        assert getattr(clone.solver, name).ptr != getattr(live, name).ptr
    assert clone.solver._emitters is not live._emitters
    assert clone.solver._emitters[0] is not live._emitters[0]
    assert clone.solver._key_frames is not live._key_frames
    clone.advance_steps(3, max_steps=3)
    assert clone.solver.sim_time == pytest.approx(0.7) and state(live) == before
    assert clone.close() == clone.close() == 'completed'

def test_exact_time_sampling_shortens_only_final_clone_substep():
    m = api()
    wp = Warp()
    live = Solver()
    before = state(live)
    c = m.SolverReplay.clone_current(wp, live, expected_type=Solver, max_clone_bytes=128 << 20)
    assert not c.advance_to(0.65, max_steps=1)
    assert c.solver.sim_time == pytest.approx(0.5)
    assert c.advance_to(0.65, max_steps=3)
    assert c.solver.sim_time == pytest.approx(0.65, abs=1e-12) and c.solver.dt == live.dt
    np.testing.assert_allclose(c.solver.pos_sim.data, live.pos_sim.data + 0.25, atol=2e-06)
    assert state(live) == before
    c.close()

def test_budget_and_future_animated_motion_reject_before_allocations():
    m = api()
    wp = Warp()
    s = Solver()
    with pytest.raises(ValueError, match='budget'):
        m.SolverReplay.clone_current(wp, s, expected_type=Solver, max_clone_bytes=1)
    s._motions = (object(),)
    with pytest.raises(ValueError, match='animated'):
        m.SolverReplay.clone_current(wp, s, expected_type=Solver, max_clone_bytes=128 << 20)
    assert not wp.operations

def test_unreviewed_state_field_or_solver_type_rejects_before_clone():
    m = api()
    wp = Warp()
    s = Solver()
    s.new_unknown_rng = object()
    with pytest.raises(ValueError, match='field'):
        m.SolverReplay.clone_current(wp, s, expected_type=Solver, max_clone_bytes=128 << 20)
    del s.new_unknown_rng
    with pytest.raises(TypeError):
        m.SolverReplay.clone_current(wp, s, expected_type=object, max_clone_bytes=128 << 20)
    assert not wp.operations

@pytest.mark.parametrize('target', [float('nan'), float('inf'), -0.1, 0.3])
def test_invalid_or_backwards_sample_is_rejected(target):
    m = api()
    c = m.SolverReplay.clone_current(Warp(), Solver(), expected_type=Solver, max_clone_bytes=128 << 20)
    with pytest.raises(ValueError):
        c.advance_to(target, max_steps=2)
    c.close()

def test_zero_work_and_closed_replay_do_not_advance():
    m = api()
    c = m.SolverReplay.clone_current(Warp(), Solver(), expected_type=Solver, max_clone_bytes=128 << 20)
    before = state(c.solver)
    assert c.advance_to(c.initial_time_s, max_steps=1)
    assert state(c.solver) == before
    with pytest.raises(ValueError):
        c.advance_steps(-1, max_steps=2)
    c.close()
    with pytest.raises(RuntimeError, match='closed'):
        c.advance_steps(1, max_steps=1)

def test_failed_completion_retains_replay_not_destroying_inflight_arrays():
    m = api()
    c = m.SolverReplay.clone_current(Warp(), Solver(), expected_type=Solver, max_clone_bytes=128 << 20)
    solver = c.solver
    c._stream.fail = True
    assert c.close() == 'quarantined' and c.solver is solver and (c in m._QUARANTINE)
    assert c.close() == 'quarantined'
