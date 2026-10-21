"""Nozzle points for emitter surfaces: where, and with which velocity, emitted particles appear.

Pure NumPy (no bpy, no Warp). Each emitter face is covered with a square lattice of nozzle points at
the particle spacing. Coplanar faces share one lattice (its axes come from the face normal and it is
anchored at the world origin), so a subdivided plane gives the same points as a single quad. Points
closer than 0.9 spacing to an earlier point (on shared edges, or where faces meet at an angle) are
dropped, so emitted particles never start on top of each other.
"""

from __future__ import annotations

import math

import numpy as np

from .description import EmitterData, SceneFillError

# Nozzle points over all emitters. Every nozzle emits about once per spacing of travel, so this many
# nozzles fill the largest particle budget in a few spacings: a larger emitter is a modelling mistake.
MAX_NOZZLES = 250_000

# Nozzle points closer than this (in spacings) are merged.
MIN_NOZZLE_GAP = 0.9


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes (u, w) for a unit normal. Rounded first so coplanar faces get identical axes."""
    n = np.round(normal, 5)
    n = n / np.linalg.norm(n)
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _face_points(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, normal: np.ndarray, spacing: float) -> np.ndarray:
    """Lattice points inside one triangle (exactly on its plane), or its centroid if none fall inside."""
    u, w = _plane_basis(normal)
    a = np.array([[v0 @ u, v0 @ w], [v1 @ u, v1 @ w], [v2 @ u, v2 @ w]])
    lo = np.ceil(a.min(axis=0) / spacing - 0.5)
    hi = np.floor(a.max(axis=0) / spacing - 0.5)
    if (hi >= lo).all():
        gi, gj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        p = (np.stack([gi.ravel(), gj.ravel()], axis=1) + 0.5) * spacing
        # Barycentric coordinates in the 2D plane frame.
        e1, e2 = a[1] - a[0], a[2] - a[0]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        rel = p - a[0]
        b1 = (rel[:, 0] * e2[1] - rel[:, 1] * e2[0]) / det
        b2 = (e1[0] * rel[:, 1] - e1[1] * rel[:, 0]) / det
        eps = 1e-9
        inside = (b1 >= -eps) & (b2 >= -eps) & (b1 + b2 <= 1.0 + eps)
        if inside.any():
            b1, b2 = b1[inside, None], b2[inside, None]
            return v0 + b1 * (v1 - v0) + b2 * (v2 - v0)
    return ((v0 + v1 + v2) / 3.0)[None, :]


def _thin(points: np.ndarray, gap: float) -> np.ndarray:
    """Indices of a greedy subset (in order) with no two points closer than `gap`."""
    keep = []
    cells: dict[tuple[int, int, int], list[int]] = {}
    keys = np.floor(points / gap).astype(np.int64)
    gap2 = gap * gap
    for i, p in enumerate(points):
        kx, ky, kz = (int(k) for k in keys[i])
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cells.get((kx + dx, ky + dy, kz + dz), ()):
                        d = points[j] - p
                        if d @ d < gap2:
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            keep.append(i)
            cells.setdefault((kx, ky, kz), []).append(i)
    return np.array(keep, dtype=np.int64)


def emitter_nozzles(emitter: EmitterData, spacing: float,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions (K, 3), velocities (K, 3), flux (K,)) for one emitter, float64, before any domain filter.

    NORMAL mode: speed along each face's normal (Blender's face normal: counter-clockwise winding).
    VECTOR mode: the emitter's vector everywhere. `flux` is |v . n| per nozzle: each nozzle covers about
    spacing^2 of surface, so at full inflow it emits flux / spacing particles per second. Points are
    thinned to MIN_NOZZLE_GAP * spacing and shuffled with a fixed seed, so round-robin emission spreads
    over the surface and is reproducible.
    """
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise SceneFillError(f"particle spacing must be a positive finite number, got {spacing}")
    verts = emitter.mesh.vertices.astype(np.float64)
    tri = emitter.mesh.triangles
    v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(cross, axis=1)
    extent = float(np.ptp(verts[np.unique(tri)], axis=0).max()) if len(tri) else 0.0
    usable = length > 1e-12 * max(extent, 1e-6) ** 2
    if not usable.any():
        raise SceneFillError(f"emitter '{emitter.mesh.name}' has no faces with area")
    area = 0.5 * float(length[usable].sum())
    if area / spacing ** 2 > 4 * MAX_NOZZLES:
        raise SceneFillError(
            f"emitter '{emitter.mesh.name}' is too large for spacing {spacing} (area {area:.3g} m^2); "
            "increase the particle spacing or use a smaller emitter")

    points, normals = [], []
    for f in np.flatnonzero(usable):
        n = cross[f] / length[f]
        p = _face_points(v0[f], v1[f], v2[f], n, spacing)
        points.append(p)
        normals.append(np.broadcast_to(n, p.shape))
    pos = np.concatenate(points)
    nrm = np.concatenate(normals)
    keep = _thin(pos, MIN_NOZZLE_GAP * spacing)
    pos, nrm = pos[keep], nrm[keep]

    if emitter.velocity_mode == "NORMAL":
        vel = nrm * emitter.speed
    else:
        vel = np.broadcast_to(np.array(emitter.vector, dtype=np.float64), pos.shape).copy()
    flux = np.abs(np.einsum("ij,ij->i", vel, nrm))
    order = np.random.default_rng(seed).permutation(len(pos))
    return pos[order], vel[order], flux[order]
