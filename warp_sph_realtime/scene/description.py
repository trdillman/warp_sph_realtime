"""Renderer- and Blender-independent description of an authored fluid scene.

No bpy and no Warp here: the description is plain NumPy data so it can be built by the Blender
adapter, by tests, or by a future file loader, and validated without a GPU. Unsupported geometry is
rejected with a specific message rather than approximated.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# Fluid volumes may overshoot the domain by float32 export noise, scaled by the domain diagonal.
DOMAIN_CONTAINMENT_TOLERANCE = 1e-5


class SceneValidationError(ValueError):
    """The scene cannot be simulated as authored."""


class SceneFillError(ValueError):
    """The fluid volumes cannot be turned into a valid particle set (raised by scene.fill).

    Defined here, not in fill.py, so callers can catch it without importing Warp.
    """


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


EMITTER_VELOCITY_MODES = ("NORMAL", "VECTOR")

# Particle budget when the scene sets none (matches the scene setting's default).
DEFAULT_PARTICLE_BUDGET = 1_000_000


@dataclass(frozen=True, eq=False)
class EmitterData:
    """A surface that injects fluid particles while the simulation clock is in [frame_start, frame_end].

    The mesh is any triangle surface in world space (it need not be closed). NORMAL mode emits at `speed`
    (m/s) along each face's normal; VECTOR mode emits with the world-space `vector` (m/s). `rate` is in
    particles per second; 0 means full inflow (each nozzle point emits again as soon as its previous
    particle has moved one particle spacing away), which is also the upper limit.
    """

    mesh: MeshData
    velocity_mode: str = "NORMAL"
    speed: float = 2.0
    vector: tuple[float, float, float] = (0.0, 0.0, -2.0)
    rate: float = 0.0
    frame_start: float = 1.0
    frame_end: float = 250.0

    def __post_init__(self):
        label = f"emitter '{self.mesh.name}'"
        if self.velocity_mode not in EMITTER_VELOCITY_MODES:
            raise SceneValidationError(
                f"{label}: velocity mode must be one of {', '.join(EMITTER_VELOCITY_MODES)}, got {self.velocity_mode!r}")
        object.__setattr__(self, "vector", _vec3(self.vector, f"{label} velocity vector"))
        for name in ("speed", "rate", "frame_start", "frame_end"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise SceneValidationError(f"{label}: {name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.velocity_mode == "NORMAL" and self.speed <= 0.0:
            raise SceneValidationError(f"{label}: speed must be above 0 m/s (flip the normals to reverse it)")
        if self.velocity_mode == "VECTOR" and not any(self.vector):
            raise SceneValidationError(f"{label}: the velocity vector is zero")
        if self.rate < 0.0:
            raise SceneValidationError(f"{label}: rate must be 0 (full inflow) or above, got {self.rate:g}")
        if not self.frame_end > self.frame_start:
            raise SceneValidationError(
                f"{label}: frame end ({self.frame_end:g}) must be after frame start ({self.frame_start:g})")
        if len(self.mesh.triangles) == 0:
            raise SceneValidationError(f"{label} has no faces")


@dataclass(frozen=True, eq=False)
class ColliderMotion:
    """Rigid animation of collider `collider` (an index into SceneDescription.colliders).

    The collider mesh in the description is its world-space shape at the start frame. `start_linear` is the
    3x3 linear part of the object's world matrix at that frame and `pivot` its origin in metres. Each later
    world matrix becomes a key (origin, rotation) through `rigid_key`, so the mesh is never rebuilt.
    """

    collider: int
    name: str
    start_linear: np.ndarray
    pivot: tuple[float, float, float]

    def __post_init__(self):
        linear = np.asarray(self.start_linear, dtype=np.float64)
        if linear.shape != (3, 3) or not np.isfinite(linear).all() or abs(np.linalg.det(linear)) < 1e-12:
            raise SceneValidationError(f"collider '{self.name}': start transform must be an invertible 3x3 matrix")
        object.__setattr__(self, "start_linear", linear)
        object.__setattr__(self, "pivot", _vec3(self.pivot, f"collider '{self.name}' pivot"))


# Rotation tolerance for rigid keys: |R^T R - I| per entry. Blender matrices are float32.
RIGID_TOLERANCE = 1e-3


def quat_from_rotation(r) -> np.ndarray:
    """Unit quaternion (x, y, z, w), Warp's order, of a proper rotation matrix (Shepperd's method)."""
    m = np.asarray(r, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rigid_key(motion: ColliderMotion, matrix, unit_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(origin in metres, rotation quaternion xyzw) of an animated collider, from its 4x4 world matrix.

    The rotation is relative to the start frame, so the start matrix gives the identity. Raises
    SceneValidationError when the motion is not rigid (animated scale or shear): the solver moves the
    start-frame mesh and never rebuilds it.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(m).all():
        raise SceneValidationError(f"collider '{motion.name}': its world matrix is not finite")
    rotation = m[:3, :3] @ np.linalg.inv(motion.start_linear)
    if np.abs(rotation.T @ rotation - np.eye(3)).max() > RIGID_TOLERANCE or np.linalg.det(rotation) <= 0.0:
        raise SceneValidationError(
            f"collider '{motion.name}' changes scale or shear while animated; only rigid motion "
            "(location and rotation) is supported")
    u, _, vt = np.linalg.svd(rotation)  # nearest exact rotation: removes float32 noise
    return m[:3, 3] * unit_scale, quat_from_rotation(u @ vt)


@dataclass(frozen=True, eq=False)
class SceneDescription:
    """Domain box, gravity, fluid-filled volumes, emitters and solid colliders, all in world space (metres).

    `unit_scale` is metres per scene unit (Blender's unit scale): the solver works in metres and converts
    positions back for display. `start_frame` and `fps` map the simulation clock to scene frames, for
    emitter frame ranges and animated colliders. `particle_budget` caps the particle count; with emitters
    it is the preallocated pool size.
    """

    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    gravity: tuple[float, float, float]
    fluid_volumes: tuple[MeshData, ...]
    colliders: tuple[MeshData, ...]
    emitters: tuple[EmitterData, ...] = ()
    collider_motion: tuple[ColliderMotion, ...] = ()
    unit_scale: float = 1.0
    start_frame: float = 1.0
    fps: float = 24.0
    particle_budget: int = DEFAULT_PARTICLE_BUDGET

    def __post_init__(self):
        object.__setattr__(self, "domain_min", _vec3(self.domain_min, "domain_min"))
        object.__setattr__(self, "domain_max", _vec3(self.domain_max, "domain_max"))
        object.__setattr__(self, "gravity", _vec3(self.gravity, "gravity"))
        object.__setattr__(self, "fluid_volumes", tuple(self.fluid_volumes))
        object.__setattr__(self, "colliders", tuple(self.colliders))
        object.__setattr__(self, "emitters", tuple(self.emitters))
        object.__setattr__(self, "collider_motion", tuple(self.collider_motion))
        for name in ("unit_scale", "start_frame", "fps"):
            object.__setattr__(self, name, float(getattr(self, name)))
        self.validate()

    def boundary_fingerprint(self) -> str:
        """Hex digest of what shapes the motion after the initial state: domain box, colliders (start-frame
        shape and which ones move) and emitters. A resumed bake must not mix caches across a change to these.

        Fluid volumes are left out (a resume restores the cached state, not the fill). Animated colliders
        contribute name and topology only: their poses come from the Blender animation each frame and are not
        part of the description."""
        digest = hashlib.sha256()

        def put(*values):
            for value in values:
                digest.update(np.ascontiguousarray(value).tobytes() if isinstance(value, np.ndarray)
                              else repr(value).encode())

        put("domain", self.domain_min, self.domain_max)
        moving = {motion.collider: motion.name for motion in self.collider_motion}
        for index, mesh in enumerate(self.colliders):
            if index in moving:  # its start shape depends on the frame the run started at: topology and name only
                put("moving collider", moving[index], mesh.vertices.shape, mesh.triangles)
            else:
                put("collider", mesh.vertices.shape, mesh.vertices, mesh.triangles)
        for e in self.emitters:
            put("emitter", e.mesh.vertices.shape, e.mesh.vertices, e.mesh.triangles, e.velocity_mode, e.speed,
                e.vector, e.rate, e.frame_start, e.frame_end)
        return digest.hexdigest()

    def validate(self) -> None:
        lo, hi = np.array(self.domain_min), np.array(self.domain_max)
        for axis, name in enumerate("xyz"):
            if not lo[axis] < hi[axis]:
                raise SceneValidationError(
                    f"domain must have min < max on every axis; {name}: {lo[axis]} .. {hi[axis]}")
        if not self.fluid_volumes and not self.emitters:
            raise SceneValidationError(
                "scene has no fluid volumes and no emitters; add a closed fluid mesh or an emitter")
        if not (math.isfinite(self.unit_scale) and self.unit_scale > 0.0):
            raise SceneValidationError(f"unit scale must be a positive number, got {self.unit_scale}")
        if not (math.isfinite(self.fps) and self.fps > 0.0) or not math.isfinite(self.start_frame):
            raise SceneValidationError(
                f"frame rate must be positive and the start frame finite, got {self.fps} fps from {self.start_frame}")
        if isinstance(self.particle_budget, bool) or int(self.particle_budget) != self.particle_budget \
                or self.particle_budget < 1:
            raise SceneValidationError(f"particle budget must be a positive whole number, got {self.particle_budget}")
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
        seen = set()
        for motion in self.collider_motion:
            if not 0 <= motion.collider < len(self.colliders) or motion.collider in seen:
                raise SceneValidationError(
                    f"collider motion '{motion.name}' must name a distinct collider index below {len(self.colliders)}")
            seen.add(motion.collider)
