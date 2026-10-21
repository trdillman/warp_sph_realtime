"""Dam break: a column of water collapses across a tank and hits an obstacle.

Builds a Custom WarpSPH scene (domain, one fluid column, one collider block). Run it with
``blender --python examples/dam_break.py`` or paste it into Blender's Text editor and press Run Script, then press
"Start Realtime Sim" in the WarpSPH tab (N panel). Set WARPSPH_EXAMPLE_START=1 to start from the command line.
The particle count is about 0.6 x 0.6 x 1.0 / SPACING^3 (about 30,000 at the default).
"""
import os

import bpy

SPACING = 0.04  # particle spacing in scene units (metres at Unit Scale 1); halve it for 8x the particles

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

    # Tank 3 m long, 1 m wide, 1.5 m high: x in [-1.5, 1.5], y in [-0.5, 0.5], z in [0, 1.5].
    add_box("Example Domain", (3.0, 1.0, 1.5), (0.0, 0.0, 0.75), "DOMAIN")
    # The water column leans on the left wall. The fill keeps one spacing of clearance from the walls.
    add_box("Example Water Column", (0.6, 0.9, 1.0), (-1.1, 0.0, 0.55), "FLUID")
    # A low block in the way of the flood.
    add_box("Example Obstacle", (0.2, 0.5, 0.3), (0.5, 0.0, 0.15), "COLLIDER")

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")
    scene.frame_current = 1


def start():
    bpy.ops.wm.warp_sph_start()
    return None


build()
if os.environ.get("WARPSPH_EXAMPLE_START") == "1":
    bpy.app.timers.register(start, first_interval=1.0)
