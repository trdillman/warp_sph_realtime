import bpy
import warp as wp

from .demo_description import DAM_SLAB_INNER_X, dam_layout, demo_description, demo_spacing, flop_layout

__all__ = ["DAM_SLAB_INNER_X", "build_blender_demo_helpers", "launch_demo_scene_init", "remove_blender_demo_helpers"]

@wp.kernel
def k_init_double_dambreak_pillars(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    num_particles: int,
    spacing: float,
    slab_nx: int,
    slab_ny: int,
    edge: float,
    z0: float
):
    """Scene 1: Opposing Double Dam-Break crashing into a central obstacle/pillar array.

    Each slab spans the full Y width and stays at |x| >= DAM_SLAB_INNER_X so no particle
    starts inside the pillars (|x| <= 1.03) or the central sphere. It starts `edge` from the centre
    and `z0` above the floor, outside the solver's one-spacing wall layer (see wall_inset).
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
        side * (-edge + float(ix) * spacing),
        -edge + float(iy) * spacing,
        z0 + float(iz) * spacing
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
    drop_side: int,
    edge: float,
    z0: float
):
    """Scene 2: High-density 'Block Flop' / Crown Splash dropped into a resting puddle basin."""
    i = int(wp.tid())
    if i < puddle_count:
        # Resting shallow fluid basin covering the floor
        ix = i % puddle_nx
        iy = (i // puddle_nx) % puddle_ny
        iz = i // (puddle_nx * puddle_ny)
        pos[i] = wp.vec3(
            -edge + float(ix) * spacing,
            -edge + float(iy) * spacing,
            z0 + float(iz) * spacing
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
    """Deletes the demo wireframe/collider helper objects and their meshes (an authored scene does not use them)."""
    col = bpy.data.collections.get(DEMO_HELPERS_COLLECTION)
    if col is not None:
        for obj in list(col.objects):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def _helper(col, name, build, location, display_type, scale=(1.0, 1.0, 1.0)):
    """One helper object built with bmesh and bpy.data: no operator, so the user's selection, active object and
    edit state are never touched."""
    import bmesh  # Blender-only module

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        build(bmesh.ops, bm)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location, obj.scale, obj.display_type = location, scale, display_type
    col.objects.link(obj)
    return obj


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

    # 1. Domain wireframe box (-2.5..2.5 X/Y, 0..5.0 Z)
    _helper(col, "WarpSPH_Domain_Bounds", lambda ops, bm: ops.create_cube(bm, size=1.0), (0.0, 0.0, 2.5), 'WIRE',
            scale=(5.0, 5.0, 5.0))

    # 2. Scene 1: central obstacle sphere + 4 vertical pillars (the primitive operators' default resolution)
    if scene_type == "DOUBLE_DAM_PILLARS":
        _helper(col, "WarpSPH_Center_Collider",
                lambda ops, bm: ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.55),
                (0.0, 0.0, 0.55), 'SOLID')
        pillar_offsets = [(-0.85, -0.55), (-0.85, 0.55), (0.85, -0.55), (0.85, 0.55)]
        for idx, (px, py) in enumerate(pillar_offsets):
            _helper(col, f"WarpSPH_Pillar_{idx+1}",
                    lambda ops, bm: ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.18, radius2=0.18,
                                                    depth=2.2),
                    (px, py, 1.1), 'SOLID')


def launch_demo_scene_init(solver, scene_type: str):
    """Initializes GPU SoA particle positions and velocities, and adopts the demo's SceneDescription.

    The demo obstacles go through the mesh-collider path (solver.set_scene), like authored colliders.
    Call solver.rebuild_cuda_graph() afterwards.
    """
    num_p = solver.num_particles
    spacing = demo_spacing(num_p)
    # Smoothing length and particle mass must follow the spawn spacing, otherwise
    # dense presets start far above rest density and the Tait EOS explodes.
    solver.set_particle_spacing(spacing)
    solver.set_scene(demo_description(scene_type, num_p))

    if scene_type == "DOUBLE_DAM_PILLARS":
        edge, z0, slab_nx, slab_ny = dam_layout(spacing)
        wp.launch(
            kernel=k_init_double_dambreak_pillars,
            dim=num_p,
            inputs=[solver.pos_sim, solver.vel, num_p, spacing, slab_nx, slab_ny, edge, z0],
            device=solver.pos_sim.device,
        )
    else:
        # Scene 2: Block Flop into Puddle (High-Speed Crown Splash)
        puddle_count = int(num_p * 0.60)
        edge, z0, puddle_nx = flop_layout(spacing)
        puddle_ny = puddle_nx
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
                edge,
                z0,
            ],
            device=solver.pos_sim.device,
        )
