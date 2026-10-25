"""Builds a SceneDescription from Blender objects.

Duck-typed against the Blender API (no `bpy` import), so it is unit-testable with a fake object model
and can never pull bpy into sim/. Objects opt in through the add-on's object settings:

    obj.warp_sph.role in {"NONE", "DOMAIN", "FLUID", "COLLIDER", "EMITTER"}

Objects whose role is NONE fall back to the early custom property `obj["warp_sph_role"]`, so scenes
authored before the UI existed keep working.

Roles:
    DOMAIN    the simulation box: the object's world-space axis-aligned bounding box (a cube is expected;
              a rotated domain uses its enclosing axis-aligned box). Exactly one is required.
    FLUID     a closed mesh volume, filled with particles when the simulation starts.
    COLLIDER  a closed mesh solid. If its object is animated (keyframes, drivers, constraints or an
              animated parent), it moves rigidly with its object: location and rotation. The mesh shape
              is read once at the start frame, so deformation (armatures, shape keys) is not followed,
              and an animated scale stops the run with an error.
    EMITTER   a mesh surface (need not be closed) that injects particles over a frame range, with the
              settings obj.warp_sph.emit_* (velocity mode, speed or vector, rate, frame start and end).

Units: the scene's unit scale (metres per Blender unit) converts geometry, gravity and motion to
metres for the solver; particle positions are converted back for display.
"""

import numpy as np

from .description import (
    DEFAULT_PARTICLE_BUDGET,
    ColliderMotion,
    EmitterData,
    MeshData,
    SceneDescription,
    SceneValidationError,
)

ROLE_PROP = "warp_sph_role"  # legacy custom-property fallback
ROLES = ("DOMAIN", "FLUID", "COLLIDER", "EMITTER")
NO_ROLE = "NONE"


def object_role(obj) -> str | None:
    """The object's simulation role, or None. The settings PropertyGroup wins over the legacy property."""
    settings = getattr(obj, "warp_sph", None)
    role = getattr(settings, "role", NO_ROLE) if settings is not None else NO_ROLE
    if role == NO_ROLE:
        getter = getattr(obj, "get", None)
        role = getter(ROLE_PROP) if callable(getter) else None
    return None if role in (None, NO_ROLE) else str(role)


def _world_mesh(obj, depsgraph, role: str, unit: float = 1.0) -> MeshData:
    """Evaluated (modifiers applied), triangulated, world-space copy of a mesh object, in metres."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        raise SceneValidationError(f"object '{obj.name}' produced no mesh (role {role})")
    try:
        calc = getattr(mesh, "calc_loop_triangles", None)  # automatic in newer Blender
        if calc is not None:
            calc()
        local = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", local)
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
    finally:
        evaluated.to_mesh_clear()

    matrix = np.array(evaluated.matrix_world, dtype=np.float64).reshape(4, 4)
    vertices = (local.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]) * unit
    triangles = tris.reshape(-1, 3)
    if np.linalg.det(matrix[:3, :3]) < 0.0:  # mirrored transform reverses face orientation
        triangles = triangles[:, ::-1]
    return MeshData(vertices.astype(np.float32), triangles.copy(), obj.name)


def objects_by_role(scene) -> dict[str, list]:
    """Scene objects grouped by role, sorted by name; objects without a role are left out."""
    by_role: dict[str, list] = {}
    for obj in sorted(scene.objects, key=lambda o: o.name):
        role = object_role(obj)
        if role is not None:
            by_role.setdefault(role, []).append(obj)
    return by_role


def world_bounds(obj) -> tuple[np.ndarray, np.ndarray] | None:
    """Cheap world-space axis-aligned bounds from Blender's local `bound_box` (no mesh evaluation)."""
    corners = getattr(obj, "bound_box", None)
    if corners is None:
        return None
    local = np.array([tuple(c) for c in corners], dtype=np.float64).reshape(-1, 3)
    matrix = np.array(obj.matrix_world, dtype=np.float64).reshape(4, 4)
    world = local @ matrix[:3, :3].T + matrix[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def scene_warnings(scene) -> list[str]:
    """Authoring problems the panel can show without evaluating meshes. Start runs the full validation."""
    by_role = objects_by_role(scene)
    warnings = []
    for role, objs in by_role.items():
        if role not in ROLES:
            warnings.extend(f"'{o.name}' has unknown role {role!r}" for o in objs)
            continue
        warnings.extend(f"'{o.name}' is a {o.type}, not a mesh" for o in objs if o.type != "MESH")
    domains = by_role.get("DOMAIN", [])
    if not domains:
        warnings.append("No domain: mark a box as Domain")
    elif len(domains) > 1:
        warnings.append(f"{len(domains)} domains; exactly one is allowed")
    if not by_role.get("FLUID") and not by_role.get("EMITTER"):
        warnings.append("No fluid: mark a closed mesh as Fluid")
    if len(domains) == 1:
        box = world_bounds(domains[0])
        if box is not None:
            lo, hi = box
            tol = 1e-5 * float(np.linalg.norm(hi - lo))
            for obj in by_role.get("FLUID", []):
                b = world_bounds(obj)
                if b is not None and ((b[0] < lo - tol).any() or (b[1] > hi + tol).any()):
                    warnings.append(f"Fluid '{obj.name}' extends outside the domain")
    return warnings


def unit_scale(scene) -> float:
    """Metres per Blender unit (Scene Properties > Units > Unit Scale); 1.0 when unset."""
    settings = getattr(scene, "unit_settings", None)
    scale = float(getattr(settings, "scale_length", 1.0) or 1.0)
    return scale if scale > 0.0 else 1.0


def _is_animated(obj) -> bool:
    """True if the object's world transform can change over time: keyframes or drivers, constraints, or
    an animated parent. Shape changes (armatures, shape keys) are not looked at; they are not followed."""
    while obj is not None:
        anim = getattr(obj, "animation_data", None)
        if anim is not None and (getattr(anim, "action", None) is not None or len(getattr(anim, "drivers", ()))):
            return True
        if len(getattr(obj, "constraints", ())):
            return True
        obj = getattr(obj, "parent", None)
    return False


def _emitter(obj, depsgraph, unit: float) -> EmitterData:
    settings = getattr(obj, "warp_sph", None)

    def get(name, default):
        return getattr(settings, f"emit_{name}", default) if settings is not None else default

    return EmitterData(
        mesh=_world_mesh(obj, depsgraph, "EMITTER", unit),
        velocity_mode=str(get("velocity_mode", "NORMAL")),
        speed=float(get("speed", 2.0)) * unit,
        vector=tuple(float(c) * unit for c in get("vector", (0.0, 0.0, -2.0))),
        rate=float(get("rate", 0.0)),
        frame_start=float(get("frame_start", 1.0)),
        frame_end=float(get("frame_end", 250.0)),
    )


def description_from_blender(scene, depsgraph) -> SceneDescription:
    """Reads every scene object with a role into a validated SceneDescription, in metres.

    Exactly one DOMAIN object is required (its world-space bounding box is the simulation domain).
    FLUID and COLLIDER objects must be closed meshes; unsupported geometry raises SceneValidationError.
    Geometry is read at the current frame, which is also the simulation's start frame; animated
    colliders get a ColliderMotion whose later poses come from sync_collider_keyframes.
    """
    by_role: dict[str, list] = {role: [] for role in ROLES}
    for role, objs in objects_by_role(scene).items():
        for obj in objs:
            if role not in by_role:
                raise SceneValidationError(
                    f"object '{obj.name}' has role {role!r}; expected one of {', '.join(ROLES)}")
            if obj.type != "MESH":
                raise SceneValidationError(f"object '{obj.name}' has role {role} but is a {obj.type}, not a MESH")
            by_role[role].append(obj)

    domains = by_role["DOMAIN"]
    if len(domains) != 1:
        found = ", ".join(o.name for o in domains) or "none"
        raise SceneValidationError(f"exactly one DOMAIN object is required; found {len(domains)} ({found})")
    unit = unit_scale(scene)
    domain_mesh = _world_mesh(domains[0], depsgraph, "DOMAIN", unit)
    if len(domain_mesh.vertices) == 0:
        raise SceneValidationError(f"domain object '{domains[0].name}' has no vertices")

    # Blender's gravity is in m/s^2 whatever the unit scale.
    gravity = tuple(float(g) for g in scene.gravity) if getattr(scene, "use_gravity", True) else (0.0, 0.0, 0.0)
    colliders = by_role["COLLIDER"]
    motion = []
    for index, obj in enumerate(colliders):
        if _is_animated(obj):
            matrix = np.array(obj.matrix_world, dtype=np.float64).reshape(4, 4)
            motion.append(ColliderMotion(index, obj.name, matrix[:3, :3], matrix[:3, 3] * unit))
    render = getattr(scene, "render", None)
    fps = float(getattr(render, "fps", 24)) / float(getattr(render, "fps_base", 1.0) or 1.0)
    settings = getattr(scene, "warp_sph", None)
    return SceneDescription(
        domain_min=tuple(domain_mesh.vertices.min(axis=0)),
        domain_max=tuple(domain_mesh.vertices.max(axis=0)),
        gravity=gravity,
        fluid_volumes=tuple(_world_mesh(o, depsgraph, "FLUID", unit) for o in by_role["FLUID"]),
        colliders=tuple(_world_mesh(o, depsgraph, "COLLIDER", unit) for o in colliders),
        emitters=tuple(_emitter(o, depsgraph, unit) for o in by_role["EMITTER"]),
        collider_motion=tuple(motion),
        unit_scale=unit,
        start_frame=float(getattr(scene, "frame_current", 1)),
        fps=fps,
        particle_budget=int(getattr(settings, "max_particles", DEFAULT_PARTICLE_BUDGET)),
    )


def _matrix_or_none(obj):
    if obj is None:
        return None
    try:
        return np.array(obj.matrix_world)
    except ReferenceError:  # deleted while the simulation runs
        return None


def sync_collider_keyframes(scene, solver, objects=None, held=None) -> None:
    """Feeds the solver the animated colliders' world matrices for the frames its next step needs.

    Main thread only. Each Blender frame is evaluated once (scene.frame_set) and its matrices handed to
    the solver; afterwards the scene is left at the simulation's current frame, so the viewport shows
    the colliders where the fluid sees them. Does nothing when no collider is animated or the needed
    frames are already known.

    ``objects`` are the animated colliders in solver order, resolved by the caller (by default looked up by
    name). A collider that is None or was deleted holds its last pose, kept in ``held`` (index -> matrix).
    """
    frames = solver.collider_frames_needed()
    if not frames:
        return
    if objects is None:
        objects = [scene.objects.get(n) for n in solver.animated_collider_names]
    held = {} if held is None else held
    for frame in frames:
        scene.frame_set(frame)
        matrices = []
        for index, obj in enumerate(objects):
            matrix = _matrix_or_none(obj)
            if matrix is None:
                matrix = held.get(index)
                if matrix is None:
                    raise SceneValidationError(f"animated collider {index} is missing and has no pose yet")
            held[index] = matrix
            matrices.append(matrix)
        solver.set_collider_keyframe(frame, matrices)
    current = solver.current_frame
    if scene.frame_current != current:
        scene.frame_set(current)
