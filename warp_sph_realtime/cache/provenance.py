"""Versioned effective scene inputs. No bpy, Warp, particle readback or per-tick hashing."""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import numpy as np

from ..scene.description import rigid_key

PROVENANCE_VERSION = 1
_INTERNAL_SCENES: ContextVar[tuple] = ContextVar("cache_internal_scenes", default=())


def internal_frame_change(scene) -> bool:
    """Whether cache-owned synchronous evaluation is in progress for this scene."""
    return any(item is scene for item in _INTERNAL_SCENES.get())


@contextmanager
def internal_frames(scene):
    token = _INTERNAL_SCENES.set((*_INTERNAL_SCENES.get(), scene))
    try:
        yield
    finally:
        _INTERNAL_SCENES.reset(token)


def _set_frame(scene, frame, *, session=None, **kwargs):
    # Resolve on each call: reallocate can replace PR26's tracker inside at_frame.
    tracker = getattr(session, "tracker", None)
    with tracker.muted() if tracker is not None else nullcontext():
        scene.frame_set(frame, **kwargs)


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array(values, dtype) -> dict:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    if array.dtype.kind == "f":
        if not np.isfinite(array).all():
            raise ValueError("scene provenance requires finite numeric inputs")
        array[array == 0] = 0  # signed zero has identical solver meaning
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _mesh(mesh) -> dict:
    return {"name": mesh.name, "vertices": _array(mesh.vertices, "<f4"),
            "triangles": _array(mesh.triangles, "<i4")}


def scene_provenance(desc, *, spacing: float, motion_keys=None, initializer=None) -> dict:
    """Hash the ordered inputs consumed by from_scene, plus frozen rigid keys if supplied.

    Arrays have explicit precision/endianness and shape. No pointer, repr, mtime or blend path.
    Names/order are deliberately conservative: changing collider/emitter ordering can change physics.
    Bump the schema when the input or canonicalization contract changes.
    """
    payload = {
        "version": PROVENANCE_VERSION, "initializer": initializer,
        "domain": [desc.domain_min, desc.domain_max], "gravity": desc.gravity,
        "unit_scale": desc.unit_scale, "start_frame": desc.start_frame, "fps": desc.fps,
        "particle_budget": desc.particle_budget, "spacing": float(spacing),
        "fluid": [_mesh(m) for m in desc.fluid_volumes],
        "colliders": [_mesh(m) for m in desc.colliders],
        "emitters": [
            {"mesh": _mesh(e.mesh), "velocity_mode": e.velocity_mode, "speed": e.speed,
             "vector": e.vector, "rate": e.rate, "frame_start": e.frame_start, "frame_end": e.frame_end}
            for e in desc.emitters],
        "collider_motion": [
            {"collider": m.collider, "name": m.name, "pivot": m.pivot,
             "start_linear": _array(m.start_linear, "<f8")} for m in desc.collider_motion],
    }
    if motion_keys is not None:
        payload["motion_keys"] = [
            [int(frame), [_array(matrix, "<f8") for matrix in matrices]]
            for frame, matrices in sorted(motion_keys.items())]
    return {"version": PROVENANCE_VERSION, "algorithm": "sha256", "digest": _digest(payload)}


@contextmanager
def at_frame(scene, frame: int, *, session=None):
    """Always restore the user's frame/subframe, including when evaluated geometry fails."""
    previous = scene.frame_current
    subframe = float(getattr(scene, "frame_subframe", 0.0))
    with internal_frames(scene):
        try:
            _set_frame(scene, frame, session=session)
            yield
        finally:
            _set_frame(scene, previous, subframe=subframe, session=session)


def capture_scene(scene, depsgraph, *, first: int, last: int, steps: int, dt: float, spacing: float, session=None):
    """Read initial evaluated geometry once, and freeze all integer keys needed by this bake.

    These are the same evaluated world matrices sync_collider_keyframes feeds to the solver.
    Sampling, including constraints/parents/drivers, avoids trying to serialize arbitrary RNA.
    The simulation consumes the frozen copies, so later interactive edits cannot contaminate a bake.
    """
    from ..scene.blender_scene import description_from_blender

    with at_frame(scene, first, session=session):
        desc = description_from_blender(scene, depsgraph)
        initial = scene_provenance(desc, spacing=spacing)
        keys = {}
        if desc.collider_motion:
            # Integration can end at a noninteger scene time. The solver needs both bounding keys,
            # including the key immediately after the final time.
            end = math.floor(first + (last - first) * steps * dt * desc.fps) + 2
            for frame in range(first, end + 1):
                _set_frame(scene, frame, session=session)
                matrices = []
                for motion in desc.collider_motion:
                    matrix = np.array(scene.objects[motion.name].matrix_world, dtype=np.float64, copy=True)
                    rigid_key(motion, matrix, desc.unit_scale)  # fail before writing on nonrigid motion
                    matrix.setflags(write=False)
                    matrices.append(matrix)
                keys[frame] = tuple(matrices)
        provenance = scene_provenance(desc, spacing=spacing, motion_keys=keys or None)
    return initial, provenance, keys


def feed_motion(scene, solver, keys) -> None:
    """Publish only captured matrices; never inspect mutable Blender geometry during bake."""
    for frame in solver.collider_frames_needed():
        if frame not in keys:
            raise ValueError("collider animation exceeds captured bake horizon; start a fresh bake")
        solver.set_collider_keyframe(frame, keys[frame])
    if keys and scene.frame_current != solver.current_frame:
        scene.frame_set(solver.current_frame)
