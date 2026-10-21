"""Viscous drop: a block of thick fluid falls into a shallow pool.

Uses the Viscous preset (honey or syrup). Change PRESET to "WATER" or "STYLISED" to compare. Run it with
``blender --python examples/viscous_drop.py`` or paste it into Blender's Text editor and press Run Script, then
press "Start Realtime Sim" in the WarpSPH tab (N panel). Set WARPSPH_EXAMPLE_START=1 to start from the command line.
Viscosity is per particle mass, so after changing SPACING you may want to adjust the Viscosity value in the Fluid box.
"""
import os

import bpy

PRESET = "VISCOUS"  # WATER, VISCOUS or STYLISED
SPACING = 0.03

BOX_QUADS = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))


def add_box(name, size, location, role):
    hx, hy, hz = (0.5 * v for v in size)
    corners = [(hx if i & 1 else -hx, hy if i & 2 else -hy, hz if i & 4 else -hz) for i in range(8)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], BOX_QUADS)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    obj.warp_sph.role = role
    if role in ("DOMAIN", "FLUID"):
        obj.display_type = 'WIRE'
    return obj


def build():
    scene = bpy.context.scene
    for obj in [o for o in bpy.data.objects if o.name.startswith("Example ")]:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Domain x, y in [-0.6, 0.6], z in [0, 1.2]: a pool 0.2 m deep and a block of fluid above it.
    add_box("Example Domain", (1.2, 1.2, 1.2), (0.0, 0.0, 0.6), "DOMAIN")
    add_box("Example Pool", (1.1, 1.1, 0.2), (0.0, 0.0, 0.15), "FLUID")
    add_box("Example Drop", (0.3, 0.3, 0.3), (0.0, 0.0, 0.8), "FLUID")

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    bpy.ops.wm.warp_sph_physics_preset(preset=PRESET)
    scene.frame_current = 1


def start():
    bpy.ops.wm.warp_sph_start()
    return None


build()
if os.environ.get("WARPSPH_EXAMPLE_START") == "1":
    bpy.app.timers.register(start, first_interval=1.0)
