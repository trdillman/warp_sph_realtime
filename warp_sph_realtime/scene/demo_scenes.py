import bpy
import warp as wp

# Inner x-extent of each dam slab; pillars occupy |x| in [0.67, 1.03].
DAM_SLAB_INNER_X = 1.15

@wp.kernel
def k_init_double_dambreak_pillars(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    num_particles: int,
    spacing: float,
    slab_nx: int,
    slab_ny: int
):
    """Scene 1: Opposing Double Dam-Break crashing into a central obstacle/pillar array.

    Each slab spans the full Y width and stays at |x| >= DAM_SLAB_INNER_X so no particle
    starts inside the pillars (|x| <= 1.03) or the central sphere.
    """
    i = int(wp.tid())
    half_n = num_particles // 2

    idx = i
    side = 1.0
    if i >= half_n:
        idx = i - half_n
        side = -1.0

    ix = idx % slab_nx
    iy = (idx // slab_nx) % slab_ny
    iz = idx // (slab_nx * slab_ny)
    pos[i] = wp.vec3(
        side * (-2.35 + float(ix) * spacing),
        -2.35 + float(iy) * spacing,
        0.08 + float(iz) * spacing
    )
    vel[i] = wp.vec3(side * 1.2, 0.0, 0.0)


@wp.kernel
def k_init_block_flop_puddle(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    num_particles: int,
    spacing: float,
    puddle_count: int,
    puddle_nx: int,
    puddle_ny: int,
    drop_side: int
):
    """Scene 2: High-density 'Block Flop' / Crown Splash dropped into a resting puddle basin."""
    i = int(wp.tid())
    if i < puddle_count:
        # Resting shallow fluid basin covering the floor
        ix = i % puddle_nx
        iy = (i // puddle_nx) % puddle_ny
        iz = i // (puddle_nx * puddle_ny)
        pos[i] = wp.vec3(
            -2.4 + float(ix) * spacing,
            -2.4 + float(iy) * spacing,
            0.05 + float(iz) * spacing
        )
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    else:
        # Elevated fluid block suspended above the center of the puddle
        idx = i - puddle_count
        ix = idx % drop_side
        iy = (idx // drop_side) % drop_side
        iz = idx // (drop_side * drop_side)
        half_w = 0.5 * float(drop_side) * spacing
        pos[i] = wp.vec3(
            -half_w + float(ix) * spacing,
            -half_w + float(iy) * spacing,
            2.2 + float(iz) * spacing
        )
        vel[i] = wp.vec3(0.0, 0.0, -3.5)


DEMO_HELPERS_COLLECTION = "WarpSPH_Demo_Helpers"


def remove_blender_demo_helpers():
    """Deletes the demo wireframe/collider helper objects (an authored scene does not use them)."""
    col = bpy.data.collections.get(DEMO_HELPERS_COLLECTION)
    if col is not None:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)


def build_blender_demo_helpers(scene_type: str):
    """Creates non-destructive wireframe domain & obstacle visualizers in the Blender 5.2+ viewport."""
    scene = bpy.context.scene
    if scene is None:
        return

    col = bpy.data.collections.get(DEMO_HELPERS_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(DEMO_HELPERS_COLLECTION)
        scene.collection.children.link(col)

    remove_blender_demo_helpers()  # clear previous demo helper meshes

    # 1. Create Domain Wireframe Box (-2.5..2.5 X/Y, 0..5.0 Z)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 2.5))
    domain_obj = bpy.context.active_object
    if domain_obj is not None:
        domain_obj.name = "WarpSPH_Domain_Bounds"
        domain_obj.scale = (5.0, 5.0, 5.0)
        domain_obj.display_type = 'WIRE'
        for c in list(domain_obj.users_collection):
            c.objects.unlink(domain_obj)
        col.objects.link(domain_obj)

    # 2. Scene 1: Create Central Obstacle Sphere + 4 Vertical Pillars
    if scene_type == "DOUBLE_DAM_PILLARS":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55, location=(0.0, 0.0, 0.55))
        sph = bpy.context.active_object
        if sph is not None:
            sph.name = "WarpSPH_Center_Collider"
            sph.display_type = 'SOLID'
            for c in list(sph.users_collection):
                c.objects.unlink(sph)
            col.objects.link(sph)

        pillar_offsets = [(-0.85, -0.55), (-0.85, 0.55), (0.85, -0.55), (0.85, 0.55)]
        for idx, (px, py) in enumerate(pillar_offsets):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=2.2, location=(px, py, 1.1))
            pil = bpy.context.active_object
            if pil is not None:
                pil.name = f"WarpSPH_Pillar_{idx+1}"
                pil.display_type = 'SOLID'
                for c in list(pil.users_collection):
                    c.objects.unlink(pil)
                col.objects.link(pil)


def launch_demo_scene_init(solver, scene_type: str):
    """Initializes GPU SoA particle positions, velocities, and SDF obstacles for the chosen demo."""
    num_p = solver.num_particles
    spacing = max(0.016, 0.024 * ((1_000_000 / max(num_p, 1)) ** (1.0 / 3.0)))
    # Smoothing length and particle mass must follow the spawn spacing, otherwise
    # dense presets start far above rest density and the Tait EOS explodes.
    solver.set_particle_spacing(spacing)

    if scene_type == "DOUBLE_DAM_PILLARS":
        solver.enable_obstacles = 1
        slab_width = 2.35 - DAM_SLAB_INNER_X
        slab_nx = max(1, int(slab_width / spacing))
        slab_ny = max(1, int(4.7 / spacing))
        wp.launch(
            kernel=k_init_double_dambreak_pillars,
            dim=num_p,
            inputs=[solver.pos_sim, solver.vel, num_p, spacing, slab_nx, slab_ny],
            device=solver.pos_sim.device,
        )
    else:
        # Scene 2: Block Flop into Puddle (High-Speed Crown Splash)
        solver.enable_obstacles = 0
        puddle_count = int(num_p * 0.60)
        puddle_nx = max(1, int(4.8 / spacing))
        puddle_ny = max(1, int(4.8 / spacing))
        drop_count = num_p - puddle_count
        drop_side = max(1, int(round(drop_count ** (1.0 / 3.0))))
        wp.launch(
            kernel=k_init_block_flop_puddle,
            dim=num_p,
            inputs=[
                solver.pos_sim,
                solver.vel,
                num_p,
                spacing,
                puddle_count,
                puddle_nx,
                puddle_ny,
                drop_side,
            ],
            device=solver.pos_sim.device,
        )
