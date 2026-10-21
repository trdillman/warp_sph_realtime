"""Scene settings (P1.3): one PropertyGroup at ``scene.warp_sph`` and the GPU memory estimate.

Replaces the flat ``scene.warp_sph_*`` properties of earlier versions. Their values in older .blend
files are copied in once by ``migrate_scenes`` (run from a ``load_post`` handler) and then unset.
No ``from __future__ import annotations``: Blender reads the property definitions from the class
annotations, so they must stay real objects.
"""
import logging
import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty

_LOG = logging.getLogger(__name__)

SCENE_SOURCE_ITEMS = [
    ("DEMO", "Demo", "Run one of the built-in demo scenes"),
    ("CUSTOM", "Custom", "Build the simulation from objects marked Domain, Fluid and Collider"),
]
DEMO_ITEMS = [
    ("DOUBLE_DAM_PILLARS", "Demo 1: Double Dam + Pillars",
     "Two opposing fluid columns colliding with a central sphere and 4 pillar obstacles"),
    ("BLOCK_FLOP_SPLASH", "Demo 2: Block Flop Crown Splash",
     "Heavy elevated fluid cube dropping into a shallow resting puddle basin"),
]
RENDER_MODE_ITEMS = [
    ("SSFR_LIQUID", "SSFR Liquid (Narrow-Range + Beer-Lambert)",
     "5-pass Screen-Space Fluid Renderer with Truong & Yuksel Narrow-Range Filter and Fresnel shading"),
    ("SPHERES", "Shaded Sphere Impostors",
     "Single-pass eye-space shaded 3D spheres with per-fragment depth correction"),
    ("POINTS", "Raw Points (Max FPS)", "Fastest raw GL_POINTS mode for telemetry benchmarking"),
]

STIFFNESS_UNIT = 1000.0  # the stiffness setting is in thousands, so one UI click (step 100 = 1.0) is 1000


class WARPSPH_SceneSettings(bpy.types.PropertyGroup):
    scene_source: EnumProperty(name="Scene Source", items=SCENE_SOURCE_ITEMS, default="DEMO")
    particle_spacing: FloatProperty(
        name="Particle Spacing",
        description="Distance between fluid particles in a Custom scene, in scene length units (it follows "
                    "Scene > Units > Unit Scale, like the objects). Smaller = more particles (about fluid "
                    "volume / spacing^3) and finer detail; the smoothing length is twice this",
        subtype='DISTANCE', unit='LENGTH',
        default=0.02, min=0.0001, max=1000.0, soft_min=0.005, soft_max=0.5,
        step=0.1,        # 0.001 scene units per click
        precision=4,
    )
    max_particles: IntProperty(
        name="Max Particles",
        description="Particle budget of a Custom scene with emitters: the GPU pool is allocated once at this "
                    "size and emitters stop when it is full. Scenes without emitters use exactly their fill",
        default=1_000_000, min=1000, max=4_000_000, step=100_000,
    )
    demo_scene: EnumProperty(name="Demo Scene", items=DEMO_ITEMS, default="DOUBLE_DAM_PILLARS")
    particle_count: IntProperty(
        name="Particles",
        description="Total fluid particles (dynamically reallocates GPU SoA buffers on change)",
        default=1_000_000, min=100_000, max=4_000_000, step=100_000,
    )
    substeps: IntProperty(
        name="Sub-steps / Frame", description="Physics sub-steps executed per viewport redraw",
        default=2, min=1, max=10,
    )
    time_step_scale: FloatProperty(
        name="Time Step (CFL)",
        description="Physics time step as a fraction of the stability limit. Higher = faster simulation, "
                    "but above 0.4 hard impacts can blow up at high particle counts",
        default=0.4, min=0.1, max=1.0,
        step=1,          # Blender steps are 1/100 units: 0.01 per click
        precision=3,
    )
    stiffness: FloatProperty(
        name="Fluid Stiffness (x1000)",
        description="Pressure stiffness in thousands (90 = 90,000; sets sound speed). Lower = larger time step "
                    "and faster simulation, but more compressible and less stable on impacts",
        default=90.0, min=10.0, max=400.0,
        step=100,        # 1.0 (= 1,000) per click; Blender allows steps of 1..100 hundredths
        precision=1,
    )
    viscosity: FloatProperty(
        name="Viscosity",
        description="How strongly each particle's velocity is pulled toward its neighbours' average, in kg/s "
                    "per particle (0.15 = water-like, 1 to 3 = thick like honey). The effect scales with "
                    "1 / particle mass, so the same value is thicker at finer spacing. Capped automatically "
                    "where the explicit step would become unstable",
        default=0.15, min=0.0, max=20.0, soft_max=5.0,
        step=1,          # 0.01 per click
        precision=2,
    )
    show_particles: BoolProperty(name="Show Particles", default=True)
    render_mode: EnumProperty(name="Render Mode", items=RENDER_MODE_ITEMS, default="SSFR_LIQUID")
    point_size: FloatProperty(
        name="Point Size", description="Particle point size in screen pixels (Raw Points mode)",
        default=2.5, min=1.0, max=20.0,
        step=10,         # 0.1 px per click
        precision=1,
    )
    particle_radius: FloatProperty(
        name="Particle Radius (m)", description="World-space sphere radius for SSFR and Sphere Impostor modes",
        default=0.022, min=0.002, max=0.10,
        soft_min=0.004,  # narrow drag range so mouse travel maps to useful radii
        soft_max=0.04,
        step=0.05,       # 0.0005 m (0.5 mm) per click
        precision=4,
    )
    filter_iterations: IntProperty(
        name="Narrow-Range Filter Passes",
        description="Number of 1D separable Horizontal + Vertical Narrow-Range smoothing iterations",
        default=2, min=1, max=5,
    )
    filter_sigma: FloatProperty(
        name="Filter Sigma (px)",
        description="Screen-space Gaussian standard deviation inside the Narrow-Range depth window",
        default=4.0, min=1.0, max=12.0,
        step=10,         # 0.1 px per click
        precision=1,
    )
    absorption: FloatVectorProperty(
        name="Beer-Lambert Absorption (RGB)",
        description="Wavelength-dependent light extinction coefficients exp(-a * thickness)",
        size=3, default=(0.85, 0.22, 0.05), min=0.0, max=5.0, soft_max=2.0,
        step=1,          # 0.01 per click
        precision=3,
    )
    particle_color: FloatVectorProperty(
        name="Particle Color", subtype='COLOR', size=4, default=(0.12, 0.68, 1.0, 1.0), min=0.0, max=1.0,
    )
    depth_test: BoolProperty(
        name="Hide Behind Objects",
        description="Particles are hidden behind scene geometry (off = always draw on top)",
        default=True,
    )
    use_cuda_graph: BoolProperty(
        name="Warp 1.17 CUDA Graph Capture",
        description="Execute Coalesced Density + WCSPH Force + Copy-back via captured CUDA Graph",
        default=True,
    )
    cache_dir: StringProperty(
        name="Cache Folder", subtype='DIR_PATH', default="",
        description="Where this scene's baked frames are stored. Empty: next to the saved .blend file "
                    "(a temporary folder while the file is unsaved)",
    )
    bake_start: IntProperty(name="Start", description="First frame of the bake (the initial state)", default=1)
    bake_end: IntProperty(name="End", description="Last frame of the bake", default=120)
    bake_steps: IntProperty(
        name="Steps / Frame",
        description="Solver steps simulated per cached frame. 0 = real time: the steps that make one frame "
                    "last 1 / (scene frame rate) seconds of simulated time",
        default=0, min=0, max=2000,
    )


def stiffness_pa(settings) -> float:
    """Solver stiffness from the setting (stored in thousands)."""
    return settings.stiffness * STIFFNESS_UNIT


def spacing_metres(settings, unit_scale: float) -> float:
    """Particle spacing in metres: the setting is in scene length units, `unit_scale` is metres per unit."""
    return settings.particle_spacing * unit_scale


# -- migration of the flat properties of earlier versions ------------------------------------------

# Blender 5 keeps bpy.props values in a store that ``scene.get()`` cannot see, so the old definitions are
# registered again (without limits, so saved values read back unchanged) only while the values are copied.
LEGACY_PREFIX = "warp_sph_"
_ENUMS = {"scene_source": SCENE_SOURCE_ITEMS, "demo_scene": DEMO_ITEMS, "render_mode": RENDER_MODE_ITEMS}
_INTS = ("particle_count", "substeps", "filter_iterations")
_BOOLS = ("show_particles", "depth_test", "use_cuda_graph")
_VECTORS = {"absorption": 3, "particle_color": 4}
_FLOATS = ("particle_spacing", "time_step_scale", "stiffness", "point_size", "particle_radius", "filter_sigma")
LEGACY_NAMES = (*_ENUMS, *_INTS, *_BOOLS, *_VECTORS, *_FLOATS)


def _legacy_definition(name: str):
    if name in _ENUMS:
        return EnumProperty(items=_ENUMS[name])
    if name in _VECTORS:
        return FloatVectorProperty(size=_VECTORS[name])
    if name in _BOOLS:
        return BoolProperty()
    if name in _INTS:
        return IntProperty()
    return FloatProperty()


def _converted(name: str, value):
    """A legacy value as the new setting's value; ValueError when it cannot be used."""
    if name in _ENUMS:
        if value not in [item[0] for item in _ENUMS[name]]:
            raise ValueError(f"unknown item {value!r}")
        return value
    if name in _VECTORS:
        return tuple(float(v) for v in value)
    if name in _BOOLS:
        return bool(value)
    if name in _INTS:
        return int(value)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"not a finite number: {value!r}")
    return value / STIFFNESS_UNIT if name == "stiffness" else value


def migrate_scene(scene) -> list[str]:
    """Copy one scene's flat ``warp_sph_*`` values (older files) into ``scene.warp_sph``, once.

    Needs the legacy definitions registered (``migrate_scenes`` does that). Only values saved in the file
    are copied, and each is unset after it is read, so a later load never overwrites newer settings. A
    legacy value is dropped, not copied, when the new setting is already set (it is stale). An unusable
    value keeps the default and is logged. Returns the names copied.
    """
    settings = getattr(scene, "warp_sph", None)
    if settings is None:
        return []
    copied = []
    for name in LEGACY_NAMES:
        key = LEGACY_PREFIX + name
        if not scene.is_property_set(key):
            continue
        if settings.is_property_set(name):
            _LOG.info("WarpSPH: dropped the old %s of scene %r; it is already set in scene.warp_sph",
                      name, scene.name)
            scene.property_unset(key)
            continue
        value = getattr(scene, key)
        try:
            setattr(settings, name, _converted(name, value))
            copied.append(name)
        except (TypeError, ValueError, AttributeError) as exc:
            _LOG.warning("WarpSPH: kept the default %s; the saved value %r cannot be used: %s", name, value, exc)
        scene.property_unset(key)
    return copied


def migrate_scenes(scenes) -> dict[str, list[str]]:
    """Run ``migrate_scene`` on every scene with the legacy definitions registered; returns the copies by
    scene name. Never raises: a file must open even if its old settings cannot be read."""
    scene_type = bpy.types.Scene
    added = []
    migrated = {}
    try:
        for name in LEGACY_NAMES:
            key = LEGACY_PREFIX + name
            if not hasattr(scene_type, key):
                setattr(scene_type, key, _legacy_definition(name))
                added.append(key)
        for scene in scenes:
            copied = migrate_scene(scene)
            if copied:
                migrated[scene.name] = copied
                _LOG.info("WarpSPH: moved the settings %s of scene %r to scene.warp_sph", copied, scene.name)
    except Exception:
        _LOG.exception("WarpSPH: could not migrate the settings of an older file")
    finally:
        for key in added:
            try:
                delattr(scene_type, key)
            except Exception:
                _LOG.exception("WarpSPH: could not remove the legacy property %s", key)
    return migrated


# -- GPU memory estimate ---------------------------------------------------------------------------

# Measured on Warp 1.17 (evidence/p1.3-settings/vram_per_particle.json): solver SoA buffers, hash grid and
# the position buffer take ~83 B/particle plus ~27 MB. The transport adds up to 64 B/particle (Vulkan: two
# RGBA32F pixel-buffer + texture slots; OpenGL: staging + texture), so 150 B/particle and 32 MB bound it.
VRAM_BYTES_PER_PARTICLE = 150
VRAM_FIXED_MB = 32.0
VRAM_WARN_FRACTION = 0.5     # Blender, the desktop and other apps share the GPU
VRAM_WARN_MB = 4096.0        # used until this GPU's memory is known (it is read on the first Start)


def estimate_vram_mb(particles: int) -> float:
    return VRAM_FIXED_MB + particles * VRAM_BYTES_PER_PARTICLE / 2**20


def custom_particle_estimate(fluids, spacing: float) -> int:
    """Upper bound for a Custom scene's fill: the fluid objects' bounding-box volumes / spacing^3."""
    if spacing <= 0.0:
        return 0
    volume = 0.0
    for obj in fluids:
        dims = getattr(obj, "dimensions", None)
        if dims is not None:
            volume += abs(float(dims[0]) * float(dims[1]) * float(dims[2]))
    return int(volume / spacing ** 3)


def vram_warning(particles: int, total_mb: float | None) -> str:
    """A short warning when the estimate is over half of the GPU's memory (or over 4 GB if unknown)."""
    need = estimate_vram_mb(particles)
    if total_mb:
        if need > VRAM_WARN_FRACTION * total_mb:
            return f"Needs about {need / 1024:.1f} GB of this GPU's {total_mb / 1024:.0f} GB: may not fit"
        return ""
    if need > VRAM_WARN_MB:
        return f"Needs about {need / 1024:.1f} GB of GPU memory: may not fit"
    return ""
