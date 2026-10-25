"""Interactive mode (P2.9): what changed in the scene since the run was built, and what happens to each change.

Every change has one outcome, read from ``EDITS``, the single table the panel and the tests share:

    LIVE    reaches the fluid on the next step (the session applies it, then ``accept``s it)
    RESET   applies on Reset, which rebuilds the run from the scene at its start frame

During a bake no change is applied; the panel shows ``BAKING_NOTE`` instead. Physics, sub-step and display
settings are live through their own paths in ``sim_session`` and are not tracked here.

Duck-typed against the Blender API (no ``bpy`` import), so it is unit-testable with fake objects. Objects are
identified by ``session_uid``, which survives renames, not by name.
"""
from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass, replace

LIVE = "LIVE"
RESET = "RESET"

# kind -> (outcome, panel text). Object kinds are the object's role.
EDITS: dict[str, tuple[str, str]] = {
    "gravity": (LIVE, "gravity"),
    "spacing": (RESET, "particle spacing changed"),
    "max_particles": (RESET, "Max Particles changed"),
    "unit_scale": (RESET, "unit scale changed"),
    "fps": (RESET, "frame rate changed"),
    "scene_source": (RESET, "scene source changed"),
    "objects": (RESET, "objects added, removed or given another role"),
    "DOMAIN": (RESET, "domain changed"),
    "FLUID": (RESET, "fluid changed"),
    "COLLIDER": (RESET, "collider changed"),
    "EMITTER": (RESET, "emitter changed"),
    "unreadable": (RESET, "the scene could not be read"),
}

BAKING_NOTE = "Live edits are ignored while baking; they apply when the bake ends"
ANIMATED_NOTE = "Follows its animation while running"

_EMIT_FIELDS = ("velocity_mode", "speed", "vector", "rate", "frame_start", "frame_end")


def outcome(kind: str) -> str:
    """LIVE or RESET. A kind missing from EDITS raises KeyError: every detected change must be classified."""
    return EDITS[kind][0]


def uid(obj):
    """Identity that survives renames and undo: ``session_uid``, or the name for objects without one."""
    value = getattr(obj, "session_uid", None)
    return value if value is not None else ("name", getattr(obj, "name", None))


def objects_by_uid(scene) -> dict:
    return {uid(obj): obj for obj in scene.objects}


@dataclass(frozen=True)
class ObjectState:
    name: str
    role: str
    animated: bool
    matrix: tuple
    shape: tuple
    emit: tuple


@dataclass(frozen=True)
class Snapshot:
    custom: bool
    settings: dict
    objects: dict  # uid -> ObjectState


def _scene_api():
    return importlib.import_module(".scene.blender_scene", __package__)


def _matrix(obj) -> tuple:
    return tuple(tuple(float(v) for v in row) for row in obj.matrix_world)


def _shape(obj) -> tuple:
    """Cheap geometry token: mesh identity, element counts and evaluated bounds. Edits it cannot see are
    caught by the depsgraph geometry flags (ChangeTracker.note_updates)."""
    data = getattr(obj, "data", None)
    counts = tuple(len(getattr(data, name, ())) for name in ("vertices", "polygons")) if data is not None else ()
    bounds = tuple(tuple(float(v) for v in corner) for corner in getattr(obj, "bound_box", ()))
    return uid(data) if data is not None else None, counts, bounds


def _emit(obj) -> tuple:
    settings = getattr(obj, "warp_sph", None)
    values = []
    for name in _EMIT_FIELDS:
        value = getattr(settings, f"emit_{name}", None)
        values.append(tuple(value) if hasattr(value, "__len__") and not isinstance(value, str) else value)
    return tuple(values)


def _settings(scene, custom: bool) -> dict:
    settings = scene.warp_sph
    values = {"scene_source": settings.scene_source}
    if custom:
        api = _scene_api()
        render = getattr(scene, "render", None)
        values.update(
            gravity=tuple(float(g) for g in scene.gravity) if getattr(scene, "use_gravity", True) else (0.0, 0.0, 0.0),
            spacing=float(settings.particle_spacing),
            max_particles=int(getattr(settings, "max_particles", 0)),
            unit_scale=api.unit_scale(scene),
            fps=float(getattr(render, "fps", 24)) / float(getattr(render, "fps_base", 1.0) or 1.0),
        )
    return values


def _state(api, obj, role: str) -> ObjectState:
    return ObjectState(obj.name, role, api._is_animated(obj), _matrix(obj), _shape(obj),
                       _emit(obj) if role == "EMITTER" else ())


def snapshot(scene, custom: bool) -> Snapshot:
    """The scene inputs a run is built from: settings always, role objects for a Custom scene."""
    objects = {}
    if custom:
        api = _scene_api()
        for role, objs in api.objects_by_role(scene).items():
            for obj in objs:
                objects[uid(obj)] = _state(api, obj, role)
    return Snapshot(custom, _settings(scene, custom), objects)


def _object_change(old: ObjectState, new: ObjectState, reshaped: bool) -> str:
    if old.animated != new.animated:
        return "animation added or removed"
    # An animated collider follows its animation: its pose at any frame is part of the run, not an edit.
    moving_collider = new.role == "COLLIDER" and new.animated
    if old.matrix != new.matrix and not moving_collider:
        return "moved"
    if reshaped or old.shape != new.shape:
        return "reshaped"
    if old.emit != new.emit:
        return "settings changed"
    return ""


def diff(base: Snapshot, now: Snapshot, reshaped=frozenset()) -> dict[str, str]:
    """kind -> panel text for every input that differs from ``base``; kinds are EDITS keys."""
    changes: dict[str, str] = {}

    def add(kind: str, text: str = ""):
        outcome(kind)  # unclassified kinds fail here, never silently
        changes.setdefault(kind, text or EDITS[kind][1])

    if base.settings["scene_source"] != now.settings["scene_source"]:
        add("scene_source")
    if not (base.custom and now.custom):
        return changes
    for key in ("gravity", "spacing", "unit_scale", "fps"):
        if base.settings[key] != now.settings[key]:
            add(key)
    states = (*base.objects.values(), *now.objects.values())
    if any(s.role == "EMITTER" for s in states) and base.settings["max_particles"] != now.settings["max_particles"]:
        add("max_particles")
    for key, old in base.objects.items():
        new = now.objects.get(key)
        if new is None:
            add("objects", f"'{old.name}' left the simulation")
        elif new.role != old.role:
            add("objects", f"'{new.name}' is now {new.role.title()}")
        else:
            what = _object_change(old, new, key in reshaped)
            if what:
                add(old.role, f"{old.role.lower()} '{new.name}' {what}")
    for key, new in now.objects.items():
        if key not in base.objects:
            add("objects", f"'{new.name}' joined as {new.role.title()}")
    return changes


class ChangeTracker:
    """Pending scene changes of one run. ``note_updates`` (the depsgraph handler) only records what was updated;
    ``refresh`` (the session tick) re-reads it when dirty, so an idle scene costs nothing. An update that names
    only objects and the scene re-reads just those objects and the settings (a drag costs one object per tick);
    anything else (collections, actions, an empty update list, undo) re-reads the whole scene."""

    def __init__(self, scene, custom: bool):
        self.custom = custom
        self._muted = False
        self.rebase(scene)

    def rebase(self, scene):
        """The run was (re)built from the scene as it is now: nothing is pending."""
        self.reshaped: set = set()
        self.dirty = False
        self.pending: dict[str, str] = {}
        self._scene_uid = uid(scene)
        try:
            self.base = self.current = snapshot(scene, self.custom)
            self._remember(scene)
        except Exception as exc:  # never fails a Start: the panel reports it and Reset re-reads
            self.base = self.current = Snapshot(False, {"scene_source": None}, {})
            self.pending = {"unreadable": f"the scene could not be read: {exc}"}
            self._full = True

    def _remember(self, scene):
        self._objects = objects_by_uid(scene) if self.custom else {}
        self._count = len(self._objects)
        self._touched: set = set()
        self._full = False

    @contextmanager
    def muted(self):
        """Updates the session causes itself (its own frame_set) are not edits."""
        previous, self._muted = self._muted, True
        try:
            yield
        finally:
            self._muted = previous

    def after_undo(self):
        """Undo / redo swapped in another copy of the file's data: re-read everything.

        Keep geometry uncertainty until Reset: counts and bounds cannot distinguish interior vertex edits.
        Blender can flag unchanged geometry during undo, so a Reset hint may be conservative, but discarding
        those flags would silently hide actual geometry edits and pending edits an unrelated undo preserves.
        """
        self.dirty = self._full = True

    def note_updates(self, updates=()):
        if self._muted:
            return
        self.dirty = True
        if not updates:
            self._full = True  # nothing says what changed
        for update in updates:
            ident = getattr(update, "id", None)
            ident = getattr(ident, "original", ident)
            if hasattr(ident, "matrix_world"):  # an object
                key = uid(ident)
                self._touched.add(key)
                if getattr(update, "is_updated_geometry", False):
                    self.reshaped.add(key)
            elif hasattr(ident, "vertices") or uid(ident) == self._scene_uid:
                pass  # a mesh flags its objects too; scene settings are re-read on every refresh
            else:
                self._full = True  # collections, actions, ...: membership or animation may have changed

    def _read(self, scene) -> Snapshot:
        if not self.custom:
            return Snapshot(False, _settings(scene, False), {})
        if not self._full and len(scene.objects) == self._count and self._touched <= self._objects.keys():
            try:
                api = _scene_api()
                objects = dict(self.current.objects)
                for key in self._touched:
                    obj = self._objects[key]
                    role = api.object_role(obj)
                    if role is None:
                        objects.pop(key, None)
                    else:
                        objects[key] = _state(api, obj, role)
                return Snapshot(True, _settings(scene, True), objects)
            except ReferenceError:  # an object was removed under its update: read everything
                pass
        now = snapshot(scene, True)
        self._remember(scene)
        return now

    def refresh(self, scene) -> dict[str, str]:
        if self.dirty:
            self.dirty = False
            try:
                self.current = self._read(scene)
                self.pending = diff(self.base, self.current, self.reshaped)
            except Exception as exc:  # an authoring problem never stops the run
                self.pending = {"unreadable": f"the scene could not be read: {exc}"}
                self._full = True
            self._touched = set()
        return self.pending

    def accept(self, kind: str):
        """A LIVE change was applied to the solver: it is now part of the run."""
        if outcome(kind) != LIVE:
            raise ValueError(f"{kind!r} applies on Reset, not live")
        self.base = replace(self.base, settings={**self.base.settings, kind: self.current.settings[kind]})
        self.pending.pop(kind, None)

    def reset_reasons(self) -> list[str]:
        return [text for kind, text in self.pending.items() if outcome(kind) == RESET]
