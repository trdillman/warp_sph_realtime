"""Warp SPH Realtime: GPU fluid simulation in the Blender viewport.

Package layout:
    addon         Blender operators, panel, scene properties (imports ``bpy``)
    sim/          WCSPH solver kernels and CUDA-graph stepping
    scene/        Regime presets and demo scene initialisation
    render/       OpenGL-CUDA interop, SSFR shaders, viewport drawing
    experimental/ Phase 2-4 modules not yet wired into the runtime

``addon`` is imported lazily so the simulation modules stay importable
without Blender (tests, tooling, headless use).
"""


def register():
    from . import addon

    addon.register()


def unregister():
    from . import addon

    addon.unregister()
