import bpy
import warp as wp

# Physical regime presets for daily VFX tasks (from pouring milk to underwater shockwaves)
REGIME_PRESETS = {
    "WATER": {
        "name": "Standard Water (WCSPH / IPBF)",
        "rest_density": 1000.0,
        "stiffness": 35000.0,
        "viscosity": 0.15,
        "dt": 0.0015,
    },
    "MILK": {
        "name": "High-Viscosity Pouring Milk",
        "rest_density": 1030.0,
        "stiffness": 40000.0,
        "viscosity": 0.65,
        "dt": 0.0012,
    },
    "SHOCKWAVE": {
        "name": "Underwater Explosion / Compressible",
        "rest_density": 1000.0,
        "stiffness": 85000.0,
        "viscosity": 0.05,
        "dt": 0.0008,
    },
}

def get_domain_bounds(scene) -> tuple[wp.vec3, wp.vec3]:
    """Returns simulation domain min/max from scene settings or active Domain object."""
    d_min = wp.vec3(-2.5, -2.5, 0.0)
    d_max = wp.vec3(2.5, 2.5, 5.0)
    domain_obj = getattr(scene, "warp_sph_domain_obj", None)
    if domain_obj and domain_obj.type == 'MESH':
        bbox = [domain_obj.matrix_world @ v.co for v in domain_obj.data.vertices]
        if bbox:
            xs = [p.x for p in bbox]
            ys = [p.y for p in bbox]
            zs = [p.z for p in bbox]
            d_min = wp.vec3(min(xs), min(ys), min(zs))
            d_max = wp.vec3(max(xs), max(ys), max(zs))
    return d_min, d_max

def apply_regime_preset(solver, preset_key: str):
    """Applies physical regime parameters to the active GPU solver."""
    cfg = REGIME_PRESETS.get(preset_key, REGIME_PRESETS["WATER"])
    solver.rest_density = cfg["rest_density"]
    solver.stiffness = cfg["stiffness"]
    solver.viscosity = cfg["viscosity"]
    solver.dt = cfg["dt"]
    if hasattr(solver, "h") and hasattr(solver, "mass"):
        solver.mass = 0.8 * (solver.h * 0.5) ** 3 * solver.rest_density
    if hasattr(solver, "rebuild_cuda_graph"):
        solver.rebuild_cuda_graph()
