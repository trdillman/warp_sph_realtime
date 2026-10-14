import bpy
import warp as wp

@wp.kernel
def k_init_double_dambreak_pillars(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    num_particles: int,
    spacing: float,
    side_half: int
):
    """Scene 1: Opposing Double Dam-Break crashing into a central obstacle/pillar array."""
    i = wp.tid()
    half_n = num_particles // 2

    if i < half_n:
        # Left fluid column (-X corner)
        idx = i
        ix = idx % side_half
        iy = (idx // side_half) % side_half
        iz = idx // (side_half * side_half)
        pos[i] = wp.vec3(
            -2.35 + float(ix) * spacing,
            -1.2 + float(iy) * spacing,
            0.08 + float(iz) * spacing
        )
        vel[i] = wp.vec3(1.2, 0.0, 0.0)
    else:
        # Right opposing fluid column (+X corner)
        idx = i - half_n
        ix = idx % side_half
        iy = (idx // side_half) % side_half
        iz = idx // (side_half * side_half)
        pos[i] = wp.vec3(
            2.35 - float(ix) * spacing,
            -1.2 + float(iy) * spacing,
            0.08 + float(iz) * spacing
        )
        vel[i] = wp.vec3(-1.2, 0.0, 0.0)


@wp.kernel
def k_init_block_flop_puddle(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    num_particles: int,
    spacing: float,
    puddle_count: int,
    puddle_nx: int,
    puddle_ny: int,
    drop_side: int
):
    """Scene 2: High-density 'Block Flop' / Crown Splash dropped into a resting puddle basin."""
    i = wp.tid()
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


def build_blender_demo_helpers(scene_type: str):
    """Creates non-destructive wireframe domain & obstacle visualizers in the Blender 5.2+ viewport."""
    col_name = "WarpSPH_Demo_Helpers"
    col = bpy.data.collections.get(col_name)
    if col is None:
        col = bpy.data.collections.new(col_name)
        bpy.context.scene.collection.children.link(col)

    # Clear previous demo helper meshes
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # 1. Create Domain Wireframe Box (-2.5..2.5 X/Y, 0..5.0 Z)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 2.5))
    domain_obj = bpy.context.active_object
    domain_obj.name = "WarpSPH_Domain_Bounds"
    domain_obj.scale = (5.0, 5.0, 5.0)
    domain_obj.display_type = 'WIRE'
    for c in domain_obj.users_collection:
        c.objects.unlink(domain_obj)
    col.objects.link(domain_obj)

    # 2. Scene 1: Create Central Obstacle Sphere + 4 Vertical Pillars
    if scene_type == "DOUBLE_DAM_PILLARS":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55, location=(0.0, 0.0, 0.55))
        sph = bpy.context.active_object
        sph.name = "WarpSPH_Center_Collider"
        sph.display_type = 'SOLID'
        for c in sph.users_collection:
            c.objects.unlink(sph)
        col.objects.link(sph)

        pillar_offsets = [(-0.85, -0.55), (-0.85, 0.55), (0.85, -0.55), (0.85, 0.55)]
        for idx, (px, py) in enumerate(pillar_offsets):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=2.2, location=(px, py, 1.1))
            pil = bpy.context.active_object
            pil.name = f"WarpSPH_Pillar_{idx+1}"
            pil.display_type = 'SOLID'
            for c in pil.users_collection:
                c.objects.unlink(pil)
            col.objects.link(pil)


def launch_demo_scene_init(solver, scene_type: str):
    """Initializes GPU SoA particle positions, velocities, and SDF obstacles for the chosen demo."""
    num_p = solver.num_particles
    spacing = max(0.016, 0.024 * ((1_000_000 / max(num_p, 1)) ** (1.0 / 3.0)))

    if scene_type == "DOUBLE_DAM_PILLARS":
        solver.enable_obstacles = 1
        side_half = max(1, int(round((num_p // 2) ** (1.0 / 3.0))))
        wp.launch(
            kernel=k_init_double_dambreak_pillars,
            dim=num_p,
            inputs=[solver.pos_sim, solver.vel, num_p, spacing, side_half],
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
