"""Fills an authored scene's fluid volumes with particles on the solver's cubic lattice."""

import numpy as np
import warp as wp

from ..sim.colliders import MeshSet
from .description import SceneDescription

# Lattice cells tested per fill. Bounds the temporary flag array (1 byte per cell) when a sparse volume
# or a tiny spacing would otherwise allocate gigabytes.
MAX_LATTICE_CELLS = 1 << 27


class SceneFillError(ValueError):
    """The fluid volumes cannot be turned into a valid particle set."""


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


def fill_particles(desc: SceneDescription, spacing: float, device=None) -> np.ndarray:
    """Particle positions (N, 3) float32 for the scene's fluid volumes.

    Cell-centred cubic lattice at `spacing`, anchored at the domain minimum so overlapping volumes
    share one lattice and never double-count. A lattice point is kept when it is inside any fluid
    volume, inside the domain, and outside every collider. Winding-number inside tests, so
    non-manifold or overlapping parts classify correctly. Raises SceneFillError instead of returning
    zero particles.
    """
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise SceneFillError(f"particle spacing must be a positive finite number, got {spacing}")
    device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")

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
    if len(idx) == 0:
        raise SceneFillError(
            f"fluid volumes contain no particles at spacing {spacing} (fully inside colliders or thinner than "
            "the spacing); decrease the spacing or check the volumes")
    ix = idx % counts[0]
    iy = (idx // counts[0]) % counts[1]
    iz = idx // (counts[0] * counts[1])
    points = origin + np.stack([ix, iy, iz], axis=1) * spacing
    return points.astype(np.float32)
