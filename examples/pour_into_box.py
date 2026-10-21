"""Pour from an emitter into an open box.

Builds a Custom WarpSPH scene: a domain, four collider walls that form an open box, and an emitter plane above it
that pours a stream downward. Run it with ``blender --python examples/pour_into_box.py`` or paste it into
Blender's Text editor and press Run Script, then press "Start Realtime Sim" in the WarpSPH tab (N panel).
Set the environment variable WARPSPH_EXAMPLE_START=1 to start the simulation automatically from the command line.
"""
import os

import bpy

SPACING = 0.04          # particle spacing in scene units (metres at Unit Scale 1)
POUR_FRAMES = (1, 120)  # emitter Start / End frame
MAX_PARTICLES = 300_000

# Corner i of a box has x/y/z from bit 0/1/2; quads are counter-clockwise seen from outside.
BOX_QUADS = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))


def new_object(name, vertices, faces, location=(0.0, 0.0, 0.0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_box(name, size, location, role):
    hx, hy, hz = (0.5 * v for v in size)
    corners = [(hx if i & 1 else -hx, hy if i & 2 else -hy, hz if i & 4 else -hz) for i in range(8)]
    obj = new_object(name, corners, BOX_QUADS, location)
    obj.warp_sph.role = role
    if role == "DOMAIN":
        obj.display_type = 'WIRE'
    return obj


def clear_previous():
    for obj in [o for o in bpy.data.objects if o.name.startswith("Example ")]:
        bpy.data.objects.remove(obj, do_unlink=True)


def build():
    scene = bpy.context.scene
    clear_previous()

    # Domain: x, y in [-1, 1], z in [0, 2].
    add_box("Example Domain", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0), "DOMAIN")

    # Open box on the floor: inner floor area 1 m x 1 m, walls 0.5 m high and 0.05 m thick (no overlaps).
    add_box("Example Wall -Y", (1.1, 0.05, 0.5), (0.0, -0.525, 0.25), "COLLIDER")
    add_box("Example Wall +Y", (1.1, 0.05, 0.5), (0.0, 0.525, 0.25), "COLLIDER")
    add_box("Example Wall -X", (0.05, 1.0, 0.5), (-0.525, 0.0, 0.25), "COLLIDER")
    add_box("Example Wall +X", (0.05, 1.0, 0.5), (0.525, 0.0, 0.25), "COLLIDER")

    # Emitter: a 0.3 m square plane with its normal pointing down, 1.6 m above the floor.
    half = 0.15
    plane = new_object("Example Emitter", [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0),
                                           (-half, half, 0.0)], [(0, 3, 2, 1)], (0.0, 0.0, 1.6))
    emitter = plane.warp_sph
    emitter.role = "EMITTER"
    emitter.emit_velocity_mode = "NORMAL"
    emitter.emit_speed = 2.0
    emitter.emit_rate = 0.0  # 0 = a full stream
    emitter.emit_frame_start, emitter.emit_frame_end = POUR_FRAMES

    settings = scene.warp_sph
    settings.scene_source = "CUSTOM"
    settings.particle_spacing = SPACING
    settings.max_particles = MAX_PARTICLES
    bpy.ops.wm.warp_sph_physics_preset(preset="WATER")
    scene.frame_start, scene.frame_end = 1, POUR_FRAMES[1]
    scene.frame_current = 1


def start():
    bpy.ops.wm.warp_sph_start()
    return None


build()
if os.environ.get("WARPSPH_EXAMPLE_START") == "1":
    bpy.app.timers.register(start, first_interval=1.0)
