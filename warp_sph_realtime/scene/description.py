"""Renderer- and Blender-independent description of an authored fluid scene.

No bpy and no Warp here: the description is plain NumPy data so it can be built by the Blender
adapter, by tests, or by a future file loader, and validated without a GPU. Unsupported geometry is
rejected with a specific message rather than approximated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Fluid volumes may overshoot the domain by float32 export noise, scaled by the domain diagonal.
DOMAIN_CONTAINMENT_TOLERANCE = 1e-5


class SceneValidationError(ValueError):
    """The scene cannot be simulated as authored."""


@dataclass(frozen=True, eq=False)
class MeshData:
    """Triangle mesh in world space. Closed meshes must have outward-facing (counter-clockwise) faces."""

    vertices: np.ndarray  # (N, 3) float32
    triangles: np.ndarray  # (M, 3) int32
    name: str

    def __post_init__(self):
        vertices = np.asarray(self.vertices)
        triangles = np.asarray(self.triangles)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise SceneValidationError(f"mesh '{self.name}': vertices must have shape (N, 3), got {vertices.shape}")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise SceneValidationError(f"mesh '{self.name}': triangles must have shape (M, 3), got {triangles.shape}")
        if triangles.size and not np.issubdtype(triangles.dtype, np.integer):
            raise SceneValidationError(f"mesh '{self.name}': triangle indices must be integers")
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(triangles, dtype=np.int32)
        if not np.isfinite(vertices).all():
            raise SceneValidationError(f"mesh '{self.name}': vertices contain NaN or infinite coordinates")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise SceneValidationError(
                f"mesh '{self.name}': triangle indices must be in [0, {len(vertices)}), "
                f"got [{triangles.min()}, {triangles.max()}]")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the vertices that faces reference (loose vertices are ignored)."""
        used = self.vertices[np.unique(self.triangles)]
        return used.min(axis=0), used.max(axis=0)


def validate_closed_mesh(mesh: MeshData, role: str) -> None:
    """Requires a non-empty, closed, consistently outward-oriented, positive-volume triangle mesh."""
    label = f"{role} '{mesh.name}'"
    tri = mesh.triangles
    if len(tri) == 0:
        raise SceneValidationError(f"{label} has no triangles")
    if (tri[:, 0] == tri[:, 1]).any() or (tri[:, 1] == tri[:, 2]).any() or (tri[:, 0] == tri[:, 2]).any():
        raise SceneValidationError(f"{label} has triangles that repeat a vertex index (degenerate faces)")

    n = np.int64(len(mesh.vertices))
    a, b, c = (tri[:, k].astype(np.int64) for k in range(3))
    directed = np.concatenate([a * n + b, b * n + c, c * n + a])
    forward = np.sort(directed)
    if (forward[1:] == forward[:-1]).any():
        raise SceneValidationError(
            f"{label} is not a consistently oriented manifold: an edge is used twice in the same direction "
            "(non-manifold edge or flipped face normals); fix normals or remove duplicate faces")
    reverse = np.sort(np.concatenate([b * n + a, c * n + b, a * n + c]))
    if not np.array_equal(forward, reverse):
        unmatched = len(np.setxor1d(forward, reverse)) // 2
        raise SceneValidationError(
            f"{label} is not closed (watertight): {unmatched} boundary edge(s); "
            "close holes before using it as a solid")

    v = mesh.vertices.astype(np.float64)
    volume = float(np.einsum("ij,ij->", v[tri[:, 0]], np.cross(v[tri[:, 1]], v[tri[:, 2]])) / 6.0)
    extent = float(np.ptp(v[np.unique(tri)], axis=0).max())
    if abs(volume) <= 1e-9 * max(extent, 1e-12) ** 3:
        raise SceneValidationError(f"{label} encloses no volume")
    if volume < 0.0:
        raise SceneValidationError(f"{label} has inward-facing normals (negative signed volume); flip its normals")


def _vec3(value, what: str) -> tuple[float, float, float]:
    try:
        x, y, z = (float(c) for c in value)
    except (TypeError, ValueError):
        raise SceneValidationError(f"{what} must be three numbers, got {value!r}") from None
    if not all(math.isfinite(c) for c in (x, y, z)):
        raise SceneValidationError(f"{what} must be finite, got {value!r}")
    return x, y, z


@dataclass(frozen=True, eq=False)
class SceneDescription:
    """Domain box, gravity, fluid-filled volumes and static solid colliders, all in world space (metres)."""

    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    gravity: tuple[float, float, float]
    fluid_volumes: tuple[MeshData, ...]
    colliders: tuple[MeshData, ...]

    def __post_init__(self):
        object.__setattr__(self, "domain_min", _vec3(self.domain_min, "domain_min"))
        object.__setattr__(self, "domain_max", _vec3(self.domain_max, "domain_max"))
        object.__setattr__(self, "gravity", _vec3(self.gravity, "gravity"))
        object.__setattr__(self, "fluid_volumes", tuple(self.fluid_volumes))
        object.__setattr__(self, "colliders", tuple(self.colliders))
        self.validate()

    def validate(self) -> None:
        lo, hi = np.array(self.domain_min), np.array(self.domain_max)
        for axis, name in enumerate("xyz"):
            if not lo[axis] < hi[axis]:
                raise SceneValidationError(
                    f"domain must have min < max on every axis; {name}: {lo[axis]} .. {hi[axis]}")
        if not self.fluid_volumes:
            raise SceneValidationError("scene has no fluid volumes; add at least one closed fluid mesh")
        tol = DOMAIN_CONTAINMENT_TOLERANCE * float(np.linalg.norm(hi - lo))
        for mesh in self.fluid_volumes:
            validate_closed_mesh(mesh, "fluid volume")
            vmin, vmax = mesh.bounds()
            if (vmin < lo - tol).any() or (vmax > hi + tol).any():
                raise SceneValidationError(
                    f"fluid volume '{mesh.name}' extends outside the domain: bounds {vmin.tolist()} .. "
                    f"{vmax.tolist()} vs domain {lo.tolist()} .. {hi.tolist()}")
        for mesh in self.colliders:
            validate_closed_mesh(mesh, "collider")
