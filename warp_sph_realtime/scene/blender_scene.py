"""Builds a SceneDescription from Blender objects.

Duck-typed against the Blender API (no `bpy` import), so it is unit-testable with a fake object model
and can never pull bpy into sim/. Objects opt in with a custom property:

    obj["warp_sph_role"] in {"DOMAIN", "FLUID", "COLLIDER"}

The role property is a stand-in until the UI exposes a PropertyGroup. Assumes 1 Blender unit = 1 metre.
"""

import numpy as np

from .description import MeshData, SceneDescription, SceneValidationError

ROLE_PROP = "warp_sph_role"
ROLES = ("DOMAIN", "FLUID", "COLLIDER")


def _world_mesh(obj, depsgraph) -> MeshData:
    """Evaluated (modifiers applied), triangulated, world-space copy of a mesh object."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        raise SceneValidationError(f"object '{obj.name}' produced no mesh (role {obj[ROLE_PROP]})")
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


def description_from_blender(scene, depsgraph) -> SceneDescription:
    """Reads every scene object carrying a `warp_sph_role` property into a validated SceneDescription.

    Exactly one DOMAIN object is required (its world-space bounding box is the simulation domain).
    FLUID and COLLIDER objects must be closed meshes; unsupported geometry raises SceneValidationError.
    """
    by_role: dict[str, list] = {role: [] for role in ROLES}
    for obj in sorted(scene.objects, key=lambda o: o.name):
        role = obj.get(ROLE_PROP)
        if role is None:
            continue
        if role not in by_role:
            raise SceneValidationError(
                f"object '{obj.name}' has {ROLE_PROP}={role!r}; expected one of {', '.join(ROLES)}")
        if obj.type != "MESH":
            raise SceneValidationError(f"object '{obj.name}' has role {role} but is a {obj.type}, not a MESH")
        by_role[role].append(obj)

    domains = by_role["DOMAIN"]
    if len(domains) != 1:
        found = ", ".join(o.name for o in domains) or "none"
        raise SceneValidationError(f"exactly one {ROLE_PROP}='DOMAIN' object is required; found {len(domains)} ({found})")
    domain_mesh = _world_mesh(domains[0], depsgraph)
    if len(domain_mesh.vertices) == 0:
        raise SceneValidationError(f"domain object '{domains[0].name}' has no vertices")

    gravity = tuple(scene.gravity) if getattr(scene, "use_gravity", True) else (0.0, 0.0, 0.0)
    return SceneDescription(
        domain_min=tuple(domain_mesh.vertices.min(axis=0)),
        domain_max=tuple(domain_mesh.vertices.max(axis=0)),
        gravity=gravity,
        fluid_volumes=tuple(_world_mesh(o, depsgraph) for o in by_role["FLUID"]),
        colliders=tuple(_world_mesh(o, depsgraph) for o in by_role["COLLIDER"]),
    )
