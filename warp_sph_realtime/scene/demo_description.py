"""The built-in demos as SceneDescriptions: the second adapter of the scene seam (the first is Blender).

No bpy and no Warp. The demo obstacles (central sphere and four pillars of Demo 1) are closed triangle
meshes that go through the same mesh-collider path as authored colliders. The fluid volumes are boxes
around the demos' particle blocks; the demo init kernels (scene.demo_scenes) still place the particles,
so the starting particle sets are unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from .description import MeshData, SceneDescription

DOMAIN_MIN = (-2.5, -2.5, 0.0)
DOMAIN_MAX = (2.5, 2.5, 5.0)
GRAVITY = (0.0, 0.0, -9.81)

SPHERE_CENTER = (0.0, 0.0, 0.55)
SPHERE_RADIUS = 0.55
PILLAR_CENTERS = ((-0.85, -0.55), (-0.85, 0.55), (0.85, -0.55), (0.85, 0.55))
PILLAR_RADIUS = 0.18
PILLAR_TOP = 2.2
PILLAR_BOTTOM = -0.1  # below the floor, so the domain floor and the pillar never leave a gap

# Inner x-extent of each dam slab; pillars occupy |x| in [0.67, 1.03].
DAM_SLAB_INNER_X = 1.15

SEGMENTS = 64


def demo_spacing(num_particles: int) -> float:
    """Particle spacing of a demo with `num_particles` (about 0.024 m at one million)."""
    return max(0.016, 0.024 * ((1_000_000 / max(num_particles, 1)) ** (1.0 / 3.0)))


def wall_inset(base: float, spacing: float) -> float:
    """Distance of a demo's outermost particle centres from a wall: `base`, but at least 1.25 spacings.

    The solver's soft wall layer is one spacing wide and expels particles inside it, so a demo block
    must start outside it (the same clearance `fill_particles` keeps). The default one-million particle
    demos are unchanged because `base` is already larger than that at their 0.024 m spacing.
    """
    return max(base, 1.25 * spacing)


def dam_layout(spacing: float) -> tuple[float, float, int, int]:
    """Demo 1's slabs: (edge, z0, nx, ny). Each slab spans x in [INNER, edge] by |y| <= edge from height z0."""
    edge = DOMAIN_MAX[0] - wall_inset(0.15, spacing)
    z0 = DOMAIN_MIN[2] + wall_inset(0.08, spacing)
    return edge, z0, max(1, int((edge - DAM_SLAB_INNER_X) / spacing)), max(1, int(2.0 * edge / spacing))


def flop_layout(spacing: float) -> tuple[float, float, int]:
    """Demo 2's puddle: (edge, z0, nx): a square of side 2 * edge on the floor, nx particles each way."""
    edge = DOMAIN_MAX[0] - wall_inset(0.1, spacing)
    return edge, DOMAIN_MIN[2] + wall_inset(0.05, spacing), max(1, int(2.0 * edge / spacing))


def uv_sphere(center, radius: float, segments: int = SEGMENTS, rings: int = SEGMENTS // 2) -> tuple:
    """Closed, outward-facing UV sphere (vertices, triangles). Vertices sit slightly outside `radius` so
    the flat faces approximate the true sphere from both sides instead of cutting inside it."""
    r = radius / math.cos(math.pi / segments)
    verts = [(0.0, 0.0, -r)]
    for i in range(1, rings):
        theta = math.pi * i / rings - math.pi / 2.0
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((r * math.cos(theta) * math.cos(phi), r * math.cos(theta) * math.sin(phi),
                          r * math.sin(theta)))
    verts.append((0.0, 0.0, r))
    top = len(verts) - 1
    tris = []
    for j in range(segments):
        tris.append((0, 1 + (j + 1) % segments, 1 + j))
    for i in range(rings - 2):
        a0, b0 = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            j1 = (j + 1) % segments
            tris.append((a0 + j, a0 + j1, b0 + j1))
            tris.append((a0 + j, b0 + j1, b0 + j))
    last = 1 + (rings - 2) * segments
    for j in range(segments):
        tris.append((last + j, last + (j + 1) % segments, top))
    return np.array(verts) + np.array(center), np.array(tris)


def cylinder(cx: float, cy: float, radius: float, z0: float, z1: float, segments: int = SEGMENTS) -> tuple:
    """Closed, outward-facing vertical cylinder (vertices, triangles), radius corrected like uv_sphere."""
    r = radius / math.cos(math.pi / segments)
    ring = [(cx + r * math.cos(2.0 * math.pi * j / segments), cy + r * math.sin(2.0 * math.pi * j / segments))
            for j in range(segments)]
    verts = [(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring] + [(cx, cy, z0), (cx, cy, z1)]
    bottom, top = 2 * segments, 2 * segments + 1
    tris = []
    for j in range(segments):
        j1 = (j + 1) % segments
        tris.append((j, j1, segments + j1))
        tris.append((j, segments + j1, segments + j))
        tris.append((bottom, j1, j))
        tris.append((top, segments + j, segments + j1))
    return np.array(verts), np.array(tris)


def merge_meshes(parts, name: str) -> MeshData:
    """One MeshData from disjoint closed parts: one collider, so one mesh query per particle."""
    verts, tris, base = [], [], 0
    for v, t in parts:
        verts.append(v)
        tris.append(t + base)
        base += len(v)
    return MeshData(np.concatenate(verts), np.concatenate(tris), name)


def demo_obstacles() -> MeshData:
    """Demo 1's central sphere and four pillars as one closed collider mesh."""
    parts = [uv_sphere(SPHERE_CENTER, SPHERE_RADIUS)]
    parts += [cylinder(cx, cy, PILLAR_RADIUS, PILLAR_BOTTOM, PILLAR_TOP) for cx, cy in PILLAR_CENTERS]
    return merge_meshes(parts, "Demo obstacles")


def _box(lo, hi, name: str) -> MeshData:
    lo = np.maximum(lo, DOMAIN_MIN)
    hi = np.minimum(hi, DOMAIN_MAX)
    corners = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return MeshData(corners, np.array(tris), name)


def _block(x0, y0, z0, nx, ny, count, spacing, name):
    """Box around `count` lattice points laid out x-fastest from (x0, y0, z0) like the init kernels."""
    nz = math.ceil(count / (nx * ny))
    half = 0.5 * spacing
    return _box((x0 - half, y0 - half, z0 - half),
                (x0 + (nx - 1) * spacing + half, y0 + (ny - 1) * spacing + half, z0 + (nz - 1) * spacing + half),
                name)


def demo_description(scene_type: str, num_particles: int) -> SceneDescription:
    """SceneDescription of a demo: its domain, gravity, obstacle collider and fluid blocks."""
    s = demo_spacing(num_particles)
    if scene_type == "DOUBLE_DAM_PILLARS":
        edge, z0, nx, ny = dam_layout(s)
        half_n = num_particles // 2  # the init kernel puts the remaining particles in the mirrored slab
        fluids = (_block(-edge, -edge, z0, nx, ny, half_n, s, "Dam left"),
                  _block(edge - (nx - 1) * s, -edge, z0, nx, ny, num_particles - half_n, s, "Dam right"))
        colliders = (demo_obstacles(),)
    else:
        puddle = int(num_particles * 0.60)
        edge, z0, nx = flop_layout(s)
        side = max(1, int(round((num_particles - puddle) ** (1.0 / 3.0))))
        half_w = 0.5 * side * s
        fluids = (_block(-edge, -edge, z0, nx, nx, puddle, s, "Puddle"),
                  _block(-half_w, -half_w, 2.2, side, side, num_particles - puddle, s, "Block"))
        colliders = ()
    return SceneDescription(DOMAIN_MIN, DOMAIN_MAX, GRAVITY, fluids, colliders)
