"""Builds a SceneDescription from Blender objects.

Duck-typed against the Blender API (no `bpy` import), so it is unit-testable with a fake object model
and can never pull bpy into sim/. Objects opt in through the add-on's object settings:

    obj.warp_sph.role in {"NONE", "DOMAIN", "FLUID", "COLLIDER"}

Objects whose role is NONE fall back to the early custom property `obj["warp_sph_role"]`, so scenes
authored before the UI existed keep working.

Roles:
    DOMAIN    the simulation box: the object's world-space axis-aligned bounding box (a cube is expected;
              a rotated domain uses its enclosing axis-aligned box). Exactly one is required.
    FLUID     a closed mesh volume, filled with particles when the simulation starts.
    COLLIDER  a closed, static mesh solid.

Assumes 1 Blender unit = 1 metre.
"""

import numpy as np

from .description import MeshData, SceneDescription, SceneValidationError

ROLE_PROP = "warp_sph_role"  # legacy custom-property fallback
ROLES = ("DOMAIN", "FLUID", "COLLIDER")
NO_ROLE = "NONE"


def object_role(obj) -> str | None:
    """The object's simulation role, or None. The settings PropertyGroup wins over the legacy property."""
    settings = getattr(obj, "warp_sph", None)
    role = getattr(settings, "role", NO_ROLE) if settings is not None else NO_ROLE
    if role == NO_ROLE:
        getter = getattr(obj, "get", None)
        role = getter(ROLE_PROP) if callable(getter) else None
    return None if role in (None, NO_ROLE) else str(role)


def _world_mesh(obj, depsgraph, role: str) -> MeshData:
    """Evaluated (modifiers applied), triangulated, world-space copy of a mesh object."""
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
    vertices = local.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
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
    if not by_role.get("FLUID"):
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


def description_from_blender(scene, depsgraph) -> SceneDescription:
    """Reads every scene object with a role into a validated SceneDescription.

    Exactly one DOMAIN object is required (its world-space bounding box is the simulation domain).
    FLUID and COLLIDER objects must be closed meshes; unsupported geometry raises SceneValidationError.
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
    domain_mesh = _world_mesh(domains[0], depsgraph, "DOMAIN")
    if len(domain_mesh.vertices) == 0:
        raise SceneValidationError(f"domain object '{domains[0].name}' has no vertices")

    gravity = tuple(scene.gravity) if getattr(scene, "use_gravity", True) else (0.0, 0.0, 0.0)
    return SceneDescription(
        domain_min=tuple(domain_mesh.vertices.min(axis=0)),
        domain_max=tuple(domain_mesh.vertices.max(axis=0)),
        gravity=gravity,
        fluid_volumes=tuple(_world_mesh(o, depsgraph, "FLUID") for o in by_role["FLUID"]),
        colliders=tuple(_world_mesh(o, depsgraph, "COLLIDER") for o in by_role["COLLIDER"]),
    )
