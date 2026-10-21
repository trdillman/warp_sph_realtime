"""Triangle-mesh colliders for the WCSPH solver.

Each mesh becomes one `wp.Mesh` (BVH built once, outside any CUDA graph capture). Kernels receive
the mesh ids as a plain uint64 array, which is capture-safe. A moving collider's mesh is stored
relative to its pivot and placed by a rigid transform at query time, so it is never rebuilt.

Signed distance takes the sign from the closest face's pseudo-normal
(`mesh_query_point_sign_normal`), which is exact for the closed, consistently oriented meshes that
scene validation admits, and confirms deep "inside" answers with the winding number
(`mesh_query_point_sign_winding_number`), which classifies against the whole mesh. A wrong deep
"inside" would teleport a particle through the solid; the confirmation rules it out, also where
separate closed parts of one object overlap. The winding number on every query near a surface
doubled the Demo 1 step time at one million particles (evidence/p2.1-scene-seam).
"""

from collections.abc import Sequence

import numpy as np
import warp as wp

from ..scene.description import MeshData

# "Inside" answers deeper than this fraction of the query radius are confirmed by the winding number.
# Shallower ones are the many particles pressed against a wall each substep; a wrong sign there moves a
# particle by less than this depth, so the costly check is skipped.
CONFIRM_DEPTH = wp.constant(0.25)


@wp.func
def mesh_signed_distance(mesh_id: wp.uint64, x: wp.vec3, radius: float) -> tuple[float, wp.vec3, wp.vec3]:
    """(signed distance, outward unit normal, closest surface point); d is `radius` when no face is closer.

    Negative inside the solid. The normal always points out of the solid, also for interior points.
    """
    d = float(radius)
    n = wp.vec3(0.0, 0.0, 0.0)
    cp = wp.vec3(x[0], x[1], x[2])
    q = wp.mesh_query_point_sign_normal(mesh_id, x, radius)
    if q.result:
        cp = wp.vec3(wp.mesh_eval_position(mesh_id, q.face, q.u, q.v))
        diff = x - cp
        dist = wp.length(diff)
        sign = q.sign
        if sign < 0.0 and dist > CONFIRM_DEPTH * radius:
            # Deep "inside": confirm against the whole mesh before a projection could move the point far.
            wq = wp.mesh_query_point_sign_winding_number(mesh_id, x, radius, 2.0, 0.5)
            if wq.result:
                sign = wq.sign
        d = sign * dist
        if dist > 1e-7:
            n = wp.vec3(sign * diff / dist)
        else:
            n = wp.vec3(wp.normalize(wp.mesh_eval_face_normal(mesh_id, q.face)))
    return d, n, cp


class MeshSet:
    """Owns the `wp.Mesh` objects (they must outlive any kernel using their ids) and their id array."""

    def __init__(self, meshes: Sequence[MeshData], device, origins: Sequence | None = None):
        """`origins` (one point per mesh, optional) are subtracted from the vertices: a moving collider's
        mesh is stored relative to its pivot and placed by a transform at query time."""
        self.count = len(meshes)
        shifts = np.zeros((self.count, 3)) if origins is None else np.asarray(origins, dtype=np.float64)
        self._meshes = [
            wp.Mesh(
                points=wp.array((m.vertices - shift).astype(np.float32), dtype=wp.vec3, device=device),
                indices=wp.array(m.triangles.reshape(-1), dtype=wp.int32, device=device),
                support_winding_number=True,
            )
            for m, shift in zip(meshes, shifts, strict=True)
        ]
        ids = [int(m.id) for m in self._meshes] or [0]  # never hand kernels a zero-length array
        self.ids = wp.array(np.array(ids, dtype=np.uint64), dtype=wp.uint64, device=device)
