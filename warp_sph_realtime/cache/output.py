"""Output helpers (P2.3): meshing defaults and export planning. Pure Python: no ``bpy``, no Warp, no NumPy.

The Blender side (node group, modifier, exporters) is ``cache/output_blender.py``; this module only decides the
numbers, so they can be unit-tested without Blender.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Meshing is tuned in units of the particle spacing so one preset fits any resolution.
RADIUS_PER_SPACING = 1.5       # splat radius: neighbours overlap deeply, so the surface is smooth, not a bead chain
VOXEL_PER_SPACING = 0.5        # voxel size: 2 voxels per particle spacing, ~3 per splat radius
THRESHOLD = 0.1                # iso level of the density field (the node's own default)
SMOOTH_ITERATIONS = 2          # Blur Attribute passes over the vertex positions
SMOOTH_WEIGHT = 0.5


def positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


EXPORT_FORMATS = {"USD": ".usdc", "ALEMBIC": ".abc"}


@dataclass(frozen=True)
class MeshingDefaults:
    radius: float
    voxel_size: float
    threshold: float
    smooth_iterations: int
    smooth_weight: float


def particle_spacing(manifest: dict | None, setting_spacing: float, setting_radius: float = 0.0) -> float:
    """Particle spacing of a cache in scene units: half the smoothing length ``h`` stored in the bake signature
    (the solver's ``h`` is twice the spacing, in metres; ``unit_scale``, metres per scene unit, is stored when it is
    not 1), else the scene's ``particle_spacing`` setting, else the particle radius."""
    signature = (manifest or {}).get("signature") or {}
    h, unit = signature.get("h"), signature.get("unit_scale", 1.0)
    if positive(h) and positive(unit):
        return float(h) / 2.0 / float(unit)
    if positive(setting_spacing):
        return float(setting_spacing)
    return float(setting_radius) if positive(setting_radius) else 0.02


def meshing_defaults(spacing: float) -> MeshingDefaults:
    """Splat radius, voxel size, threshold and smoothing for particles ``spacing`` apart."""
    if not positive(spacing):
        raise ValueError("particle spacing must be positive")
    return MeshingDefaults(radius=RADIUS_PER_SPACING * spacing, voxel_size=VOXEL_PER_SPACING * spacing,
                           threshold=THRESHOLD, smooth_iterations=SMOOTH_ITERATIONS, smooth_weight=SMOOTH_WEIGHT)


def export_path(filepath: str, fmt: str) -> Path:
    """``filepath`` with the extension of ``fmt`` ('USD' or 'ALEMBIC') unless it already has a matching one."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format {fmt!r}")
    path = Path(filepath)
    allowed = {".usd", ".usdc", ".usda"} if fmt == "USD" else {".abc"}
    return path if path.suffix.lower() in allowed else path.with_suffix(EXPORT_FORMATS[fmt])


def export_frames(cached: tuple[int, int] | None, requested: tuple[int, int] | None = None) -> tuple[int, int]:
    """Frame range to export: the baked range, optionally narrowed to ``requested``. Only baked frames are ever
    exported; ValueError if nothing is baked or the request does not overlap the cache."""
    if cached is None:
        raise ValueError("nothing is baked: bake first")
    first, last = cached
    if requested is not None:
        first, last = max(first, requested[0]), min(last, requested[1])
    if last < first:
        raise ValueError(f"frames {requested[0]}-{requested[1]} are not baked (cached {cached[0]}-{cached[1]})"
                         if requested else "empty frame range")
    return first, last
