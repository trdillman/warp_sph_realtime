"""Small closed-mesh builders shared by the authored-scene tests."""

import numpy as np

from warp_sph_realtime.scene.description import MeshData

# Outward-facing (counter-clockwise seen from outside) triangles of a unit cube's corner indices
# (bit 0 = x, bit 1 = y, bit 2 = z).
_BOX_TRIANGLES = np.array([
    [0, 2, 3], [0, 3, 1],  # -z
    [4, 5, 7], [4, 7, 6],  # +z
    [0, 1, 5], [0, 5, 4],  # -y
    [2, 6, 7], [2, 7, 3],  # +y
    [0, 4, 6], [0, 6, 2],  # -x
    [1, 3, 7], [1, 7, 5],  # +x
], dtype=np.int32)


def box_mesh(lo, hi, name="box") -> MeshData:
    lo, hi = np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)
    corners = np.array([[(hi if (i >> k) & 1 else lo)[k] for k in range(3)] for i in range(8)], dtype=np.float32)
    return MeshData(corners, _BOX_TRIANGLES.copy(), name)


def box_signed_distance(points, lo, hi) -> np.ndarray:
    """Exact signed distance to an axis-aligned box (negative inside)."""
    centre, half = 0.5 * (np.asarray(lo) + np.asarray(hi)), 0.5 * (np.asarray(hi) - np.asarray(lo))
    q = np.abs(points - centre) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
