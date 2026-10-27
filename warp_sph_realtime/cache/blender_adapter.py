"""Blender side of the cache (P1.6): a point cloud object that follows the timeline. The only ``bpy`` module in
``cache/``; everything else in the package is importable without Blender.

Playback is a ``PointCloud`` datablock (Cycles renders its points as spheres of the ``radius`` attribute; verified
on Blender 5.2) with ``velocity`` (FLOAT_VECTOR), ``id`` (INT) and ``age`` (FLOAT, seconds since the particle was
first cached, see ``age.py``) point attributes. A ``frame_change_post`` handler loads the cached frame for the
scene's current frame (the USD / Alembic exporters step frames through the same handlers; verified on Blender
5.2). Frames outside the cached range show the nearest cached frame. The handler reads only ``.npz`` files:
it needs neither Warp nor a running simulation.

NumPy and ``.disk`` are imported inside the functions that need them, so registering the add-on (which
installs the handler) never imports a third-party package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    from .disk import Frame

_LOG = logging.getLogger(__name__)

PLAYBACK_PROP = "warp_sph_cache_playback"  # custom property that marks the playback object
OBJECT_NAME = "WarpSPH Cache"

_loaded: dict[int, tuple] = {}  # point cloud pointer -> (dir, frame, file mtime, count, radius) last applied
_emission: dict[str, object] = {}  # cache directory -> age.EmissionIndex
_reported: set[str] = set()     # distinct handler failures already logged
_failures: dict[str, dict[int, str]] = {}  # cache directory -> scene frame -> why it was not shown


def cache_dir(scene) -> Path:
    """The scene's cache directory: its ``cache_dir`` setting (``//`` paths resolved) or the default."""
    from .disk import resolve_cache_dir
    configured = bpy.path.abspath(scene.warp_sph.cache_dir) if scene.warp_sph.cache_dir else ""
    return resolve_cache_dir(configured, bpy.data.filepath, scene.name)


def find_playback_object(scene):
    for obj in scene.objects:
        if obj.get(PLAYBACK_PROP) and obj.type == 'POINTCLOUD':
            return obj
    return None


def ensure_playback_object(scene):
    """The scene's playback point cloud object, created (in the scene's master collection) if missing."""
    obj = find_playback_object(scene)
    if obj is not None:
        return obj
    cloud = bpy.data.pointclouds.new(OBJECT_NAME)
    obj = bpy.data.objects.new(OBJECT_NAME, cloud)
    obj[PLAYBACK_PROP] = True
    scene.collection.objects.link(obj)
    return obj


def remove_playback_object(scene) -> bool:
    obj = find_playback_object(scene)
    if obj is None:
        return False
    cloud = obj.data
    _loaded.pop(cloud.as_pointer(), None)
    _emission.clear()
    bpy.data.objects.remove(obj)
    if cloud.users == 0:
        bpy.data.pointclouds.remove(cloud)
    return True


def _attribute(cloud, name: str, data_type: str):
    attr = cloud.attributes.get(name)
    if attr is None or attr.data_type != data_type:
        if attr is not None:
            cloud.attributes.remove(attr)
        attr = cloud.attributes.new(name, data_type, 'POINT')
    return attr


def apply_frame(cloud, frame: Frame, radius: float, age=None) -> None:
    """Write one cached frame into the point cloud (positions bit-identical to the file). ``age`` (seconds per
    particle, optional) fills the ``age`` attribute."""
    import numpy as np
    count = frame.count
    if len(cloud.points) != count:
        cloud.resize(count)
    cloud.attributes["position"].data.foreach_set("vector", frame.position.ravel())
    _attribute(cloud, "velocity", 'FLOAT_VECTOR').data.foreach_set("vector", frame.velocity.ravel())
    _attribute(cloud, "id", 'INT').data.foreach_set("value", frame.id)
    if age is not None:
        _attribute(cloud, "age", 'FLOAT').data.foreach_set("value", age)
    key = cloud.as_pointer()
    if _loaded.get(key, (None,) * 5)[3:] != (count, radius):
        _attribute(cloud, "radius", 'FLOAT').data.foreach_set("value", np.full(count, radius, dtype=np.float32))
    cloud.update_tag()


def particle_ages(scene, disk, span: tuple[int, int], number: int, ids):
    """Age (seconds) of the particles ``ids`` of cached frame ``number``; zeros if the id scan fails (logged once)."""
    import numpy as np

    from .age import EmissionIndex, seconds_per_frame
    from .disk import CacheError
    try:
        index = _emission.get(str(disk.directory))
        if index is None:
            index = _emission[str(disk.directory)] = EmissionIndex(disk)
        index.sync(span[0], number)
        fps = scene.render.fps / scene.render.fps_base if hasattr(scene, "render") else 24.0
        return index.ages(ids, number, seconds_per_frame(disk.manifest(), fps))
    except CacheError as exc:
        message = f"particle age unavailable: {exc}"
        if message not in _reported:
            _reported.add(message)
            _LOG.warning("WarpSPH cache playback: %s (logged once)", message)
        return np.zeros(len(ids), dtype=np.float32)


def show_frame(scene, frame_number: int) -> bool:
    """Load the cached frame for ``frame_number`` into the scene's playback object. Returns True if it changed.
    Raises CacheError if nothing usable is cached."""
    from .disk import CacheError, DiskCache
    obj = find_playback_object(scene)
    if obj is None:
        return False
    disk = DiskCache(cache_dir(scene))
    span = disk.cached_range()
    if span is None:
        raise CacheError(f"no cached frames in {disk.directory}")
    number = min(max(frame_number, span[0]), span[1])
    try:
        mtime = disk.path(number).stat().st_mtime_ns
    except OSError as exc:
        raise CacheError(f"frame {number} is not cached") from exc
    cloud = obj.data
    radius = float(scene.warp_sph.particle_radius)
    key = cloud.as_pointer()
    state = (str(disk.directory), number, mtime, len(cloud.points), radius)
    if _loaded.get(key) == state:
        return False
    frame = disk.read_frame(number)
    apply_frame(cloud, frame, radius, particle_ages(scene, disk, span, number, frame.id))
    _loaded[key] = (str(disk.directory), number, mtime, frame.count, radius)
    return True


def playback_failures(directory) -> dict[int, str]:
    """Scene frames whose cache the handler could not show (frame -> reason): the point cloud held an older frame.

    The panel shows them and export reports them; a frame shown successfully later is removed."""
    return dict(_failures.get(str(directory), {}))


def clear_playback_failures(directory) -> None:
    _failures.pop(str(directory), None)


def on_frame_change_post(scene, depsgraph=None):
    """frame_change_post: keep the playback point cloud on the current frame. Never raises into Blender.

    A frame that cannot be shown leaves the previous one in place; it is logged once and kept in
    ``playback_failures`` so the UI and export can tell the user."""
    directory = None
    try:
        if find_playback_object(scene) is None:
            return
        directory = str(cache_dir(scene))
        show_frame(scene, scene.frame_current)
        failed = _failures.get(directory)
        if failed:
            failed.pop(int(scene.frame_current), None)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if directory is not None:
            try:
                _failures.setdefault(directory, {})[int(scene.frame_current)] = str(exc)
            except Exception:
                pass  # no usable frame number: logged below
        if message not in _reported:
            _reported.add(message)
            _LOG.warning("WarpSPH cache playback: %s (logged once)", message)


def _frame_change_handlers():
    return getattr(getattr(getattr(bpy, "app", None), "handlers", None), "frame_change_post", None)


def register() -> None:
    handlers = _frame_change_handlers()
    if handlers is None:
        return
    persistent = getattr(bpy.app.handlers, "persistent", None)
    if persistent is not None and not getattr(on_frame_change_post, "_bpy_persistent", False):
        persistent(on_frame_change_post)  # keep the handler across file loads
    if on_frame_change_post not in handlers:
        handlers.append(on_frame_change_post)


def unregister() -> None:
    handlers = _frame_change_handlers()
    while handlers is not None and on_frame_change_post in handlers:
        handlers.remove(on_frame_change_post)
    _loaded.clear()
    _emission.clear()
    _reported.clear()
    _failures.clear()
