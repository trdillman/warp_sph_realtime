"""Static triangle-mesh colliders for the WCSPH solver.

Each mesh becomes one `wp.Mesh` (BVH built once, outside any CUDA graph capture). Kernels receive
the mesh ids as a plain uint64 array, which is capture-safe.

Signed distance uses `wp.mesh_query_point_sign_winding_number`. The cheaper normal-based sign
(`mesh_query_point_sign_normal`) trusts the pseudo-normal of the closest face, which flips for
open, non-manifold or self-overlapping geometry that Blender users routinely produce. The winding
number classifies against the whole mesh, so thin shells and overlapping parts keep a correct sign.
It is only evaluated for particles within the query radius of a surface.
"""

from collections.abc import Sequence

import numpy as np
import warp as wp

from ..scene.description import MeshData


@wp.func
def mesh_signed_distance(mesh_id: wp.uint64, x: wp.vec3, radius: float) -> tuple[float, wp.vec3, wp.vec3]:
    """(signed distance, outward unit normal, closest surface point); d is `radius` when no face is closer.

    Negative inside the solid. The normal always points out of the solid, also for interior points.
    """
    d = float(radius)
    n = wp.vec3(0.0, 0.0, 0.0)
    cp = wp.vec3(x[0], x[1], x[2])
    q = wp.mesh_query_point_sign_winding_number(mesh_id, x, radius, 2.0, 0.5)
    if q.result:
        cp = wp.vec3(wp.mesh_eval_position(mesh_id, q.face, q.u, q.v))
        diff = x - cp
        dist = wp.length(diff)
        d = q.sign * dist
        if dist > 1e-7:
            n = wp.vec3(q.sign * diff / dist)
        else:
            n = wp.vec3(wp.normalize(wp.mesh_eval_face_normal(mesh_id, q.face)))
    return d, n, cp


class MeshSet:
    """Owns the `wp.Mesh` objects (they must outlive any kernel using their ids) and their id array."""

    def __init__(self, meshes: Sequence[MeshData], device):
        self.count = len(meshes)
        self._meshes = [
            wp.Mesh(
                points=wp.array(m.vertices, dtype=wp.vec3, device=device),
                indices=wp.array(m.triangles.reshape(-1), dtype=wp.int32, device=device),
                support_winding_number=True,
            )
            for m in meshes
        ]
        ids = [int(m.id) for m in self._meshes] or [0]  # never hand kernels a zero-length array
        self.ids = wp.array(np.array(ids, dtype=np.uint64), dtype=wp.uint64, device=device)
