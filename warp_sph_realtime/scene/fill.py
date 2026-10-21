"""Fills an authored scene's fluid volumes with particles on the solver's cubic lattice."""

import numpy as np
import warp as wp

from ..sim.colliders import MeshSet
from .description import SceneDescription, SceneFillError

__all__ = ["MAX_LATTICE_CELLS", "SceneFillError", "emitter_nozzle_arrays", "fill_particles"]

# Lattice cells tested per fill. Bounds the temporary flag array (1 byte per cell) when a sparse volume
# or a tiny spacing would otherwise allocate gigabytes.
MAX_LATTICE_CELLS = 1 << 27


@wp.func
def _inside_any(ids: wp.array[wp.uint64], count: int, x: wp.vec3, max_dist: float) -> bool:
    inside = int(0)
    for k in range(count):
        q = wp.mesh_query_point_sign_winding_number(ids[k], x, max_dist, 2.0, 0.5)
        if q.result and q.sign < 0.0:
            inside = 1
    return inside == 1


@wp.kernel
def k_fill_lattice(
    fluid_ids: wp.array[wp.uint64],
    num_fluid: int,
    collider_ids: wp.array[wp.uint64],
    num_colliders: int,
    origin: wp.vec3,
    spacing: float,
    nx: int,
    ny: int,
    max_dist: float,
    keep: wp.array[wp.uint8],
):
    tid = int(wp.tid())
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    x = origin + wp.vec3(float(ix), float(iy), float(iz)) * spacing
    if _inside_any(fluid_ids, num_fluid, x, max_dist) and not _inside_any(collider_ids, num_colliders, x, max_dist):
        keep[tid] = wp.uint8(1)


@wp.kernel
def k_filter_nozzles(
    points: wp.array[wp.vec3],
    collider_ids: wp.array[wp.uint64],
    num_colliders: int,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    clearance: float,
    max_dist: float,
    keep: wp.array[wp.uint8],
):
    """Keeps a point at least `clearance` inside every domain wall and outside every collider surface.

    Particles closer than that would start inside the boundaries' soft layer and be thrown out of it.
    The solver's soft layer is one particle spacing wide (`wall_accel`, r0 = h / 2), so one spacing is
    where the boundary force is exactly zero.
    """
    tid = int(wp.tid())
    x = points[tid]
    for axis in range(3):
        if x[axis] < domain_min[axis] + clearance or x[axis] > domain_max[axis] - clearance:
            return
    for k in range(num_colliders):
        q = wp.mesh_query_point_sign_winding_number(collider_ids[k], x, max_dist, 2.0, 0.5)
        if q.result:
            if q.sign < 0.0 or wp.length(x - wp.mesh_eval_position(collider_ids[k], q.face, q.u, q.v)) < clearance:
                return
    keep[tid] = wp.uint8(1)


def clear_of_boundaries(points: np.ndarray, desc: SceneDescription, colliders: MeshSet, clearance: float,
                        device) -> np.ndarray:
    """Boolean mask of the `points` (K, 3) at least `clearance` inside every domain wall and outside every
    collider surface (at the start pose), and outside every collider."""
    d_lo = np.array(desc.domain_min, dtype=np.float64)
    d_hi = np.array(desc.domain_max, dtype=np.float64)
    keep = wp.zeros(len(points), dtype=wp.uint8, device=device)
    wp.launch(
        k_filter_nozzles, dim=len(points),
        inputs=[wp.array(points.astype(np.float32), dtype=wp.vec3, device=device), colliders.ids,
                colliders.count, wp.vec3(*d_lo), wp.vec3(*d_hi), float(clearance),
                float(np.linalg.norm(d_hi - d_lo)) * 2.0, keep],
        device=device,
    )
    return keep.numpy().astype(bool)


def emitter_nozzle_arrays(desc: SceneDescription, spacing: float, device=None,
                          clearance: float | None = None) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """All emitters' nozzle points at least `clearance` (default: `spacing`) from every domain wall and
    collider surface (at the start pose), and outside every collider.

    Returns (positions (K, 3) float32, velocities (K, 3) float32, per-emitter (offset, count, natural
    rate in particles/s)). Raises SceneFillError when an emitter keeps no nozzle, emits nothing (its
    velocity is parallel to its surface), or the emitters together exceed MAX_NOZZLES.
    """
    from .emitters import MAX_NOZZLES, emitter_nozzles

    device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    raw = [emitter_nozzles(e, spacing) for e in desc.emitters]
    total = sum(len(p) for p, _, _ in raw)
    if total > MAX_NOZZLES:
        raise SceneFillError(
            f"emitters have {total} nozzle points at spacing {spacing} (limit {MAX_NOZZLES}); "
            "increase the particle spacing or use smaller emitters")
    clearance = spacing if clearance is None else clearance
    colliders = MeshSet(desc.colliders, device)
    positions, velocities, ranges = [], [], []
    offset = 0
    for emitter, (pos, vel, flux) in zip(desc.emitters, raw, strict=True):
        name = emitter.mesh.name
        mask = clear_of_boundaries(pos, desc, colliders, clearance, device)
        if not mask.any():
            raise SceneFillError(
                f"emitter '{name}' is outside the domain, inside a collider, or closer than {clearance:.4g} m "
                "to a domain wall or collider; move it into open space")
        rate = float(flux[mask].sum()) / spacing
        if rate <= 0.0:
            raise SceneFillError(f"emitter '{name}' velocity is parallel to its surface, so it emits nothing")
        positions.append(pos[mask])
        velocities.append(vel[mask])
        ranges.append((offset, int(mask.sum()), rate))
        offset += int(mask.sum())
    return (np.concatenate(positions).astype(np.float32), np.concatenate(velocities).astype(np.float32), ranges)


def fill_particles(desc: SceneDescription, spacing: float, device=None, clearance: float | None = None) -> np.ndarray:
    """Particle positions (N, 3) float32 for the scene's fluid volumes.

    Cell-centred cubic lattice at `spacing`, anchored at the domain minimum so overlapping volumes
    share one lattice and never double-count. A lattice point is kept when it is inside any fluid
    volume, at least `clearance` (default: `spacing`) inside the domain walls, and outside every collider
    by the same margin. The solver's soft wall layer is one spacing wide and pushes a particle inside
    it away, so a lattice started closer (the cell-centred half spacing) is expelled violently; at one
    spacing the boundary force is exactly zero. Winding-number inside tests, so non-manifold or
    overlapping parts classify correctly. Raises SceneFillError instead of returning
    zero particles, except for a scene without fluid volumes (emitters only), which returns (0, 3).
    """
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise SceneFillError(f"particle spacing must be a positive finite number, got {spacing}")
    device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    if not desc.fluid_volumes:  # an emitter-only scene starts empty
        return np.zeros((0, 3), dtype=np.float32)

    d_lo = np.array(desc.domain_min, dtype=np.float64)
    d_hi = np.array(desc.domain_max, dtype=np.float64)
    bounds = [m.bounds() for m in desc.fluid_volumes]
    lo = np.maximum(np.min([b[0] for b in bounds], axis=0).astype(np.float64), d_lo)
    hi = np.minimum(np.max([b[1] for b in bounds], axis=0).astype(np.float64), d_hi)

    # Lattice points at d_lo + (i + 0.5) * spacing that fall inside [lo, hi].
    first = np.ceil((lo - d_lo) / spacing - 0.5).astype(np.int64)
    last = np.floor((hi - d_lo) / spacing - 0.5).astype(np.int64)
    counts = np.maximum(last - first + 1, 0)
    cells = int(np.prod(counts))
    if cells == 0:
        raise SceneFillError(f"no lattice point fits in the fluid volumes at spacing {spacing}; decrease the spacing")
    if cells > MAX_LATTICE_CELLS:
        raise SceneFillError(
            f"fluid bounds need {cells} lattice cells at spacing {spacing} (limit {MAX_LATTICE_CELLS}); "
            "increase the particle spacing")

    fluid = MeshSet(desc.fluid_volumes, device)
    colliders = MeshSet(desc.colliders, device)
    origin = d_lo + (first + 0.5) * spacing
    keep = wp.zeros(cells, dtype=wp.uint8, device=device)
    wp.launch(
        k_fill_lattice,
        dim=cells,
        inputs=[fluid.ids, fluid.count, colliders.ids, colliders.count, wp.vec3(*origin), spacing,
                int(counts[0]), int(counts[1]), float(np.linalg.norm(d_hi - d_lo)) * 2.0, keep],
        device=device,
    )
    idx = np.flatnonzero(keep.numpy())
    ix = idx % counts[0]
    iy = (idx // counts[0]) % counts[1]
    iz = idx // (counts[0] * counts[1])
    points = (origin + np.stack([ix, iy, iz], axis=1) * spacing).astype(np.float32)
    if len(points):
        points = points[clear_of_boundaries(points, desc, colliders, spacing if clearance is None else clearance,
                                            device)]
    if len(points) == 0:
        raise SceneFillError(
            f"fluid volumes contain no particles at spacing {spacing} (fully inside colliders, thinner than "
            "the spacing, or within one spacing of the domain walls and colliders); decrease the spacing or "
            "check the volumes")
    return points
