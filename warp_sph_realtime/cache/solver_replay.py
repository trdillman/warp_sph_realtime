"""Isolated replay of the exact current WarpSPH state for native animation.

No Blender or Warp import at module load. Mutable GPU arrays, hash grid and host
emitter schedules are copied. Existing mesh collider topology is immutable in
this solver and is retained by reference; future animated keys must be supplied
by a separately qualified source, so this adapter rejects them rather than
holding an old pose silently. Live solver settings/counters are never advanced.
"""
from __future__ import annotations

import copy
import math
import threading

from ..render.transport import process_state

# The producer's existing fatal ownership registry survives module reload.
_QUARANTINE = process_state().quarantine
_GRID_DIMS = (128, 128, 128)
_FIELDS = frozenset({
    '_emitters', '_exhaustion_unreported', '_key_frames', '_key_p', '_key_q',
    '_motion_collider', '_motions', '_next_id', '_viscosity_cap_reported',
    'active', 'active_count', 'boundary_fingerprint', 'cfl_factor', 'collider_ang',
    'collider_lin', 'collider_radius', 'collider_xform', 'colliders', 'cuda_graph',
    'density_sorted', 'device', 'domain_max', 'domain_min', 'dt', 'emission_exhausted',
    'enable_obstacles', 'fps', 'gravity', 'grid', 'h', 'mass', 'max_dt', 'nozzle_pos',
    'nozzle_vel', 'num_particles', 'pid', 'pid_sorted', 'pos_sim', 'pos_sorted',
    'rest_density', 'scene_mode', 'sim_time', 'start_frame', 'stiffness', 'substeps',
    'unit_scale', 'vel', 'vel_sorted', 'viscosity',
})
_REQUIRED = frozenset({'device', 'num_particles', 'active_count', '_next_id', 'dt',
                       'sim_time', 'substeps', 'grid', 'pos_sim', 'vel', 'pid',
                       'active', '_emitters', '_motions', 'cuda_graph'})


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return value


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite seconds')
    return float(value)


def _admit(wp, source, expected_type, max_clone_bytes):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('solver replay creation requires the Blender main thread')
    if type(source) is not expected_type:
        raise TypeError('source is not the qualified runtime solver type')
    if getattr(wp, '__version__', '') != '1.17.0' or str(source.device) != 'cuda:0':
        raise ValueError('solver replay requires Warp1.17.0 on cuda:0')
    state = vars(source)
    unknown, missing = set(state) - _FIELDS, _REQUIRED - set(state)
    if unknown or missing:
        raise ValueError(f'unreviewed solver state field set: extra={sorted(unknown)}, missing={sorted(missing)}')
    if source._motions:
        raise ValueError('animated collider replay needs independently recorded future motion; not frozen poses')
    n = _integer(source.num_particles, 'particle capacity', 1)
    if not 0 <= _integer(source.active_count, 'active count') <= n:
        raise ValueError('active count exceeds the solver capacity')
    if not 0 <= _integer(source._next_id, 'next particle ID') <= (1 << 31):
        raise ValueError('solver identity range is unsupported')
    if _seconds(source.dt, 'solver time step') <= 0 or _seconds(source.sim_time, 'solver time') < 0:
        raise ValueError('solver time step/time is invalid')
    _integer(max_clone_bytes, 'clone memory budget', 1)
    array_bytes = sum(int(a.capacity) for a in state.values() if isinstance(a, wp.array))
    # Qualified Warp hashgrid.cpp: two dense int32 cell tables; double-buffered
    # point-cell and point-ID arrays allocated at 1.5x reserved point capacity.
    # Explicit conservative radix workspace allowance is charged too. This is
    # an admission estimate, not a claim about driver/module/global allocations.
    grid_bytes = 2 * math.prod(_GRID_DIMS) * 4 + 4 * (n * 3 // 2) * 4
    scratch_reservation = max(n * 64, 1 << 20)
    estimate = array_bytes + n * 12 + grid_bytes + scratch_reservation
    if estimate > max_clone_bytes:
        raise ValueError(f'replay clone memory budget exceeded: estimated {estimate} > {max_clone_bytes}')
    return estimate


class SolverReplay:
    """Replay owner. Clone step kernels are eager; live CUDA graph is untouched."""

    @classmethod
    def estimate_clone_bytes(cls, wp, source, *, expected_type):
        return _admit(wp, source, expected_type, (1 << 63) - 1)

    @classmethod
    def clone_current(cls, wp, source, *, expected_type, max_clone_bytes):
        estimated = _admit(wp, source, expected_type, max_clone_bytes)
        if process_state().restart_required:
            raise RuntimeError('CUDA state requires restart; no replay can be created')
        obj = cls.__new__(cls)
        obj.wp = wp
        obj.solver = None
        obj.output = None
        obj._stream = None
        obj._closed = ''
        obj.estimated_bytes = estimated
        obj.initial_time_s = float(source.sim_time)
        obj.base_dt = float(source.dt)
        obj.steps_advanced = 0
        try:
            obj._stream = wp.Stream(source.device)
            obj._stream.wait_stream(wp.get_stream(source.device))
            with wp.ScopedStream(obj._stream):
                clone = object.__new__(expected_type)
                obj.solver = clone
                for name, value in vars(source).items():
                    if name in ('grid', 'cuda_graph'):
                        continue
                    if isinstance(value, wp.array):
                        value = wp.clone(value)
                        if value is getattr(source, name) or (value.ptr and value.ptr == getattr(source, name).ptr):
                            raise RuntimeError(f'replay array aliases live source: {name}')
                    elif name != 'colliders':
                        value = copy.deepcopy(value)
                    setattr(clone, name, value)
                clone.cuda_graph = None
                clone.grid = wp.HashGrid(*_GRID_DIMS, device=source.device)
                clone.grid.reserve(source.num_particles)
                obj.output = wp.empty(source.num_particles, dtype=wp.vec3, device=source.device)
            # Snapshot copying must finish before the live owner may resume.
            wp.synchronize_stream(obj._stream)
            return obj
        except Exception:
            # Any partially submitted copy must finish before dropping references.
            obj.close()
            raise

    def _check(self):
        if self._closed:
            raise RuntimeError('solver replay is closed')
        if process_state().restart_required:
            raise RuntimeError(process_state().restart_required)
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('solver replay advances on Blender main thread only')

    def _poison(self, error):
        message = f'isolated solver replay CUDA ownership uncertain; restart Blender: {error}'
        state = process_state()
        state.restart_required = state.restart_required or message
        if self not in state.quarantine:
            state.quarantine.append(self)
        return RuntimeError(message)

    def _step(self, steps, dt):
        self._check()
        solver = self.solver
        previous_steps, previous_dt = solver.substeps, solver.dt
        try:
            solver.substeps, solver.dt = steps, dt
            with self.wp.ScopedStream(self._stream):
                # Same solver equations and sorting, new independent storage.
                # Eager launch avoids accidentally replaying the live graph.
                solver.step(self.output, use_graph=False)
            self.steps_advanced += steps
        except Exception as exc:
            raise self._poison(exc) from exc
        finally:
            solver.substeps, solver.dt = previous_steps, previous_dt

    def advance_steps(self, steps, *, max_steps):
        self._check()
        _integer(steps, 'steps')
        _integer(max_steps, 'maximum steps per slice', 1)
        taken = min(steps, max_steps)
        if taken:
            self._step(taken, self.base_dt)
        return taken

    def advance_to(self, time_s, *, max_steps):
        """Advance toward an exact requested timestamp in a bounded step count.

        Full original substeps are retained; only the isolated clone's final
        remainder uses a shorter eager substep to land exactly on the sample.
        Returns True when reached. No live timestep or scene clock is changed.
        """
        self._check()
        target = _seconds(time_s, 'sample time')
        _integer(max_steps, 'maximum steps per slice', 1)
        current = float(self.solver.sim_time)
        tolerance = max(math.ulp(target) * 4, 1e-12)
        if target < current - tolerance:
            raise ValueError('replay sample time cannot move backwards')
        for _ in range(max_steps):
            remaining = target - float(self.solver.sim_time)
            if remaining <= tolerance:
                return True
            self._step(1, min(self.base_dt, remaining))
        return abs(float(self.solver.sim_time) - target) <= tolerance

    def synchronize(self):
        self._check()
        try:
            self.wp.synchronize_stream(self._stream)
        except Exception as exc:
            raise self._poison(exc) from exc

    def close(self):
        if self._closed:
            return self._closed
        if process_state().restart_required:
            self._closed = 'quarantined'
            if self not in process_state().quarantine:
                process_state().quarantine.append(self)
            return self._closed
        try:
            if self._stream is not None:
                self.wp.synchronize_stream(self._stream)
        except Exception as exc:
            self._poison(exc)
            self._closed = 'quarantined'
            return self._closed
        self.output = self.solver = self._stream = None
        self._closed = 'completed'
        return self._closed
