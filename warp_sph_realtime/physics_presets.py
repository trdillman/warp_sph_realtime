"""Fluid physics presets: named groups of the settings that change how the fluid behaves (P2.5).

Pure data and two helpers, no ``bpy`` and no Warp, so the values can be tested headless. A preset sets
only the physics: stiffness (in thousands, like the setting), the time-step factor, viscosity and the
sub-steps per redraw. Particle spacing, render and cache settings are the scene's and are never touched.

Every preset is verified stable at the default resolutions on the solver itself (tests/test_physics_presets.py).
"""

from types import SimpleNamespace

PRESET_ITEMS = [
    ("WATER", "Water", "The tuned default: free-flowing water at the stiffness the demos were calibrated for"),
    ("VISCOUS", "Viscous", "Thick, slow-flowing fluid like honey or syrup: strong velocity smoothing"),
    ("STYLISED", "Stylised", "Snappy, bouncy and a little blobby: softer fluid and a larger time step, so it "
                             "simulates faster at the cost of accuracy on hard impacts"),
]

# Setting names are the scene.warp_sph properties.
PRESETS = {
    "WATER": {"stiffness": 90.0, "time_step_scale": 0.4, "viscosity": 0.15, "substeps": 2},
    "VISCOUS": {"stiffness": 90.0, "time_step_scale": 0.4, "viscosity": 8.0, "substeps": 2},
    "STYLISED": {"stiffness": 60.0, "time_step_scale": 0.4, "viscosity": 1.5, "substeps": 3},
}

_TOLERANCE = 1e-4  # settings are stored as float32


def apply_preset(settings, key: str) -> None:
    """Set the preset's values on `settings` (a ``scene.warp_sph`` group or any object with those names)."""
    for name, value in PRESETS[key].items():
        setattr(settings, name, value)


def active_preset(settings) -> str | None:
    """The preset whose values `settings` currently holds, or None when any of them was changed."""
    for key, values in PRESETS.items():
        if all(abs(getattr(settings, name) - value) <= _TOLERANCE * max(1.0, abs(value))
               for name, value in values.items()):
            return key
    return None


def preset_settings(key: str) -> SimpleNamespace:
    """The preset as an object with the setting names (for building a solver without Blender)."""
    return SimpleNamespace(**PRESETS[key])
