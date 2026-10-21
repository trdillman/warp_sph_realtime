"""Moving paddle: a keyframed collider sweeps through a tank of water.

The paddle is a collider whose location is animated. WarpSPH follows animated location and rotation, so it pushes
the fluid; scale must not be animated. Run it with ``blender --python examples/moving_paddle.py`` or paste it into
Blender's Text editor and press Run Script, then press "Start Realtime Sim" in the WarpSPH tab (N panel). While it
runs the timeline follows the simulation's frame. Set WARPSPH_EXAMPLE_START=1 to start from the command line.
"""
import os

import bpy

SPACING = 0.04
LAST_FRAME = 120

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

    # Tank x in [-1.5, 1.5], y in [-0.5, 0.5], z in [0, 1.2]; a shallow pool 0.3 m deep.
    add_box("Example Domain", (3.0, 1.0, 1.2), (0.0, 0.0, 0.6), "DOMAIN")
    add_box("Example Pool", (2.8, 0.9, 0.3), (0.0, 0.0, 0.2), "FLUID")

    # The paddle starts at the left end, partly under the surface, and sweeps to the right.
    paddle = add_box("Example Paddle", (0.1, 0.9, 0.6), (-1.2, 0.0, 0.4), "COLLIDER")
    paddle.keyframe_insert("location", index=0, frame=1)
    paddle.location.x = 1.0
    paddle.keyframe_insert("location", index=0, frame=LAST_FRAME)

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")
    scene.frame_start, scene.frame_end = 1, LAST_FRAME
    scene.frame_current = 1


def start():
    bpy.ops.wm.warp_sph_start()
    return None


build()
if os.environ.get("WARPSPH_EXAMPLE_START") == "1":
    bpy.app.timers.register(start, first_interval=1.0)
