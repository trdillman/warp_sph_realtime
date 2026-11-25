"""Explicit real-solver snapshots into the shared SDK (bundled in private builds, else installed).

No imports of bpy, GPU, Warp or the SDK at registration. No live-step hook or
solver mutation. Native conversion is an explicit user action on a paused run.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
import weakref
from math import isfinite
from pathlib import Path

from . import sdk_bundle

DEFAULT_CAPTURE_BYTES = 256 * 1024 * 1024
_DEFAULT_LOCK = Path(__file__).with_name('sdk_snapshot.lock.json')


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('WarpSPH snapshots require the Blender main thread')


def verify_sdk_payload(root, lock):
    """Verify exact separately installed bytes; never import a wrong SDK candidate."""
    root = Path(root).resolve()
    if lock.get('schema_version') != 1 or not isinstance(lock.get('payload_sha256'), dict):
        raise RuntimeError('invalid SDK snapshot identity lock')
    expected = lock['payload_sha256']
    if not expected or '__init__.py' not in expected:
        raise RuntimeError('SDK snapshot identity lock has no package')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts}
    if actual != set(expected):
        raise RuntimeError('SDK package identity mismatch: missing or unexpected payload files')
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'SDK package identity mismatch: {name}')


def require_sdk():
    """Load only the byte-bound candidate (bundled, else installed); no automatic install or fallback."""
    lock = json.loads(_DEFAULT_LOCK.read_text(encoding='utf-8'))
    root = sdk_bundle.locate()
    if root is None:
        raise RuntimeError('Native snapshots need the qualified Blender Vulkan Interop SDK, bundled in a private add-on build or installed separately')
    verify_sdk_payload(root, lock)
    return sdk_bundle.load(root)


def eligibility(session, scene):
    """Read-only UI admission; imports nothing and creates no session/runtime."""
    if session is None or getattr(session, 'solver', None) is None:
        return 'Start a WarpSPH simulation first'
    if session.scene is not scene:
        return 'The simulation belongs to another scene'
    if getattr(session, 'restart_required', ''):
        return 'The GPU session needs a Blender restart'
    if not session.running:
        return 'Start a WarpSPH simulation first'
    if not session.paused:
        return 'Pause the simulation before taking a native snapshot'
    if session.bake is not None:
        return 'Finish or cancel the bake before taking a native snapshot'
    if session.step_once or getattr(session, '_output_pending', False):
        return 'Wait for the requested simulation frame to finish'
    return ''


def _source_id(session):
    source = getattr(session, '_sdk_snapshot_source', None)
    if source is None:
        # Scene UID alone survives addon disable/enable. A newly constructed
        # session must not reuse that old session's (epoch, particle ID) tuples.
        source = f'warpsph:{session.scene_uid}:{uuid.uuid4().hex}'
        session._sdk_snapshot_source = source
    return source


def _gpu_read(session, operation):
    try:
        return operation()
    except Exception as exc:
        # A synchronization/readback exception may mean lost CUDA ownership.
        # Reuse the consumer's fatal-state route; never silently resume it.
        reason = f'Native snapshot CUDA readback failed; restart Blender: {exc}'
        mark = getattr(session, 'mark_fatal', None)
        if callable(mark):
            mark(reason)
        else:
            session.restart_required = session.restart_required or reason
            session.running = False
        raise RuntimeError(reason) from exc


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def capture_paused(session, scene, *, wp_module, max_readback_bytes=DEFAULT_CAPTURE_BYTES,
                   max_capture_bytes=DEFAULT_CAPTURE_BYTES):
    """Capture only the active prefix, retaining actual solver identity and units.

    The explicit snapshot may synchronize/read back. It never advances physics,
    changes pause/timeline/settings, replaces solver arrays or rewrites their IDs.
    Density is the solver's final-substep force density, not recomputed at the
    post-integration position. Before the first step it is omitted, not invented.
    """
    _main_thread()
    reason = eligibility(session, scene)
    if reason:
        raise ValueError(reason)
    solver = session.solver
    count = _count(solver.active_count, 'active count')
    capacity = _count(solver.num_particles, 'capacity')
    steps = _count(session.step_count, 'step count')
    next_id = _count(solver._next_id, 'next particle ID')
    if count > capacity or next_id > (1 << 31):
        raise ValueError('invalid solver population/ID capacity')
    time_s = float(solver.sim_time)
    if not isfinite(time_s) or time_s < 0:
        raise ValueError('invalid solver simulation time')
    for value in (max_readback_bytes, max_capture_bytes):
        _count(value, 'snapshot byte budget')
    # Logical output includes float3 position/velocity, uint64 ID, radius,
    # float4 colour and optional scalar density. Working budget includes host
    # copies, immutable payload bytes, ID promotion and uniqueness diagnostics.
    logical_bytes = count * (56 if steps else 52)
    if logical_bytes > max_readback_bytes or count * 256 > max_capture_bytes:
        raise ValueError('snapshot byte budget exceeded before GPU readback')
    radius = float(scene.warp_sph.particle_radius)
    color = tuple(float(x) for x in scene.warp_sph.particle_color)
    if not isfinite(radius) or radius < 0 or len(color) != 4 or not all(isfinite(x) for x in color):
        raise ValueError('snapshot radius/colour must be finite')
    wp = wp_module
    if wp is None or wp.__version__ != '1.17.0' or str(solver.device) != 'cuda:0':
        raise RuntimeError('Snapshot capture requires the qualified Warp 1.17.0 solver on cuda:0')
    pairs = [('position', solver.pos_sim, wp.vec3, 3),
             ('velocity', solver.vel, wp.vec3, 3),
             ('id', solver.pid, wp.int32, 1)]
    if steps:
        pairs.append(('density', solver.density_sorted, wp.float32, 1))
    for name, array, dtype, _ in pairs:
        if (not isinstance(array, wp.array) or array.dtype != dtype or array.ndim != 1
                or not array.is_contiguous or array.shape[0] < count
                or array.device != wp.get_device('cuda:0')):
            raise ValueError(f'invalid authoritative solver array: {name}')
    require_sdk()
    import numpy as np
    # These optional imports run only after require_sdk() verifies and loads the private SDK.
    from blender_vulkan_interop.frames import FrameStamp, SpatialMetadata  # pyright: ignore[reportMissingImports]
    from blender_vulkan_interop.output.contracts import HostArray, SnapshotPayload  # pyright: ignore[reportMissingImports]
    token = session.token
    _gpu_read(session, lambda: wp.synchronize_device('cuda:0'))
    values = {}
    for name, array, _dtype, components in pairs:
        shape = (count,) if components == 1 else (count, components)
        dtype = np.int32 if name == 'id' else np.float32
        value = _gpu_read(session, lambda array=array: array[:count].numpy()) if count else np.empty(shape, dtype=dtype)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f'authoritative {name} readback shape/dtype mismatch')
        if name == 'id':
            if count and (np.any(value < 0) or np.any(value >= next_id) or len(np.unique(value)) != count):
                raise ValueError('invalid or duplicate solver particle IDs; array indices are not identities')
            value = value.astype('<u8')
        elif not np.isfinite(value).all():
            raise ValueError(f'nonfinite solver {name}')
        values[name] = value
    # No state mutation is allowed to slip between admission and publication.
    if (eligibility(session, scene) or session.solver is not solver or session.token != token
            or session.step_count != steps or solver.active_count != count or solver.sim_time != time_s):
        raise RuntimeError('simulation changed during native snapshot capture')
    previous = getattr(session, '_sdk_snapshot_generation', None)
    same = (previous is not None and previous[0]() is solver and previous[1] == token
            and time_s >= previous[3] and steps >= previous[4])
    if same:
        assert previous is not None
        epoch = previous[2]
    else:
        epoch = 0 if previous is None else previous[2] + 1
    stamp = FrameStamp(_source_id(session), epoch, steps, time_s, 0)
    values['radius'] = np.full(count, radius, dtype='<f4')
    values['color'] = np.tile(np.asarray(color, dtype='<f4'), (count, 1))
    arrays = tuple(HostArray(name, 'uint64' if name == 'id' else 'float32', value.shape,
                             np.ascontiguousarray(value, dtype='<u8' if name == 'id' else '<f4').tobytes())
                   for name, value in values.items())
    metadata = {'producer': 'WarpSPH Realtime', 'producer_base_commit': '52a383aa45ba14acfba0e5c2c7c86a91f1498181',
                'semantic.position': 'metres', 'semantic.velocity': 'metres_per_second',
                'semantic.id': 'solver_pid_promoted_losslessly_to_uint64',
                'semantic.radius': 'artist_display_radius_metres', 'semantic.color': 'artist_scene_linear_rgba',
                'scene_revision_tracking': 'not_qualified', 'solver_step_count': steps,
                'density_sampling': 'last_substep_force_density' if steps else 'omitted_before_first_step',
                'readback_bytes': count * (32 if steps else 28), 'capture_working_bound_bytes': count * 256,
                'live_zero_copy': False}
    payload = SnapshotPayload('particles', stamp, SpatialMetadata((0., 0., 0.), 1.), arrays,
                              {'active_count': count}, metadata)
    session._sdk_snapshot_generation = (weakref.ref(solver), token, epoch, time_s, steps)
    return payload


def validate_target_owner(existing, source_id):
    if (existing.type != 'MESH' or not existing.data.get('interop_snapshot')
            or existing.data.get('interop_source') != source_id):
        raise ValueError('Only a native snapshot from this WarpSPH session may be replaced; choose a new name')


class _NodeOutput:
    """Keep native replacement reversible until optional GN creation succeeds."""
    def __init__(self, staged, bpy_module, create_nodes):
        self._staged, self._bpy, self._create_nodes = staged, bpy_module, create_nodes
        self._object = self._group = None
        self._rolled = self._final = False
        self.refs = ()

    def commit(self):
        self.refs = self._staged.commit()
        name = next(ref.name for ref in self.refs if ref.kind == 'OBJECT')
        self._object = self._bpy.data.objects[name]
        already = any(m.type == 'NODES' and m.node_group is not None
                      and m.node_group.get('interop_generated') for m in self._object.modifiers)
        if not already:
            self._group = self._create_nodes(self._object)
        return self.refs

    def finalize(self):
        self._staged.finalize()
        self._final = True

    def rollback(self):
        if self._rolled or self._final:
            return
        if self._group is not None:
            materials = [node.inputs['Material'].default_value for node in self._group.nodes
                         if node.bl_idname == 'GeometryNodeSetMaterial'
                         and node.inputs['Material'].default_value is not None]
            for modifier in tuple(self._object.modifiers):
                if modifier.type == 'NODES' and modifier.node_group is self._group:
                    self._object.modifiers.remove(modifier)
            if self._group.users == 0:
                self._bpy.data.node_groups.remove(self._group)
            for material in materials:
                if material.users == 0:
                    self._bpy.data.materials.remove(material)
        self._staged.rollback()
        self._rolled = True


class _NodeConverter:
    domain = 'particles'

    def __init__(self, base, bpy_module, create_nodes):
        self.base, self.bpy, self.create_nodes = base, bpy_module, create_nodes

    def estimate(self, payload, target):
        from blender_vulkan_interop.output.contracts import OutputEstimate  # pyright: ignore[reportMissingImports]
        base = self.base.estimate(payload, target)
        # Unit icosphere: 12 vertices, 20 triangles, 60 corners. Charge generous
        # native geometry/attribute overhead plus graph/material storage. This
        # includes realized GN output, not only one-frame host readback.
        per_particle = 12 * 256 + 20 * 64 + 60 * 64
        return OutputEstimate(base.native_bytes + 131072 + payload.counts['active_count'] * per_particle,
                              payload.counts['active_count'] * 12)

    def prepare(self, payload, target):
        return _NodeOutput(self.base.prepare(payload, target), self.bpy, self.create_nodes)


def snapshot_to_native(session, scene, *, wp_module, object_name='WarpSPH Native Snapshot',
                       collection_name='WarpSPH Native Snapshots', replace_existing=False,
                       geometry_nodes=True, max_readback_bytes=DEFAULT_CAPTURE_BYTES,
                       max_native_bytes=1024 * 1024 * 1024):
    """Explicit, transactional paused snapshot; live display/solver are untouched."""
    _main_thread()
    import bpy
    if bpy.context.scene is not scene:
        raise ValueError('Native snapshot target is not the active Blender scene')
    reason = eligibility(session, scene)
    if reason:
        raise ValueError(reason)
    if type(replace_existing) is not bool or type(geometry_nodes) is not bool:
        raise ValueError('Snapshot replacement and Geometry Nodes choices must be explicit booleans')
    existing = bpy.data.objects.get(object_name)
    if existing is not None:
        if not replace_existing:
            raise ValueError('Snapshot object already exists; choose a new name or explicitly update it')
        validate_target_owner(existing, _source_id(session))
    payload = capture_paused(session, scene, wp_module=wp_module, max_readback_bytes=max_readback_bytes)
    from blender_vulkan_interop.output.contracts import SnapshotTarget  # pyright: ignore[reportMissingImports]
    from blender_vulkan_interop.output.geometry_nodes import create_particle_instancer  # pyright: ignore[reportMissingImports]
    from blender_vulkan_interop.output.particles import ParticleConverter  # pyright: ignore[reportMissingImports]
    from blender_vulkan_interop.output.registry import ConverterRegistry  # pyright: ignore[reportMissingImports]
    from blender_vulkan_interop.output.snapshots import PayloadSource, SnapshotService  # pyright: ignore[reportMissingImports]
    registry = ConverterRegistry()
    base = ParticleConverter()
    registry.register(_NodeConverter(base, bpy, create_particle_instancer) if geometry_nodes else base)
    target = SnapshotTarget(object_name, collection_name, replace_existing=replace_existing,
                            max_readback_bytes=max_readback_bytes, max_native_bytes=max_native_bytes)
    transaction = SnapshotService(registry).begin(target)
    try:
        transaction.stage(PayloadSource(payload), stamp=payload.stamp)
        result = transaction.commit()
    except Exception:
        transaction.rollback()
        raise
    return result
