import time
import bpy
import gpu
import warp as wp
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
try:
    from .interop_bridge import ZeroCopyGLBuffer52
    from .sim_core import RealtimeWCSPHSolver
    from .demo_scenes import build_blender_demo_helpers, launch_demo_scene_init
except ImportError:
    from interop_bridge import ZeroCopyGLBuffer52
    from sim_core import RealtimeWCSPHSolver
    from demo_scenes import build_blender_demo_helpers, launch_demo_scene_init

class _SimRuntime:
    gl_buf: ZeroCopyGLBuffer52 = None
    solver: RealtimeWCSPHSolver = None
    draw_handle = None
    running: bool = False
    active_demo: str = "DOUBLE_DAM_PILLARS"
    fps_avg: float = 0.0
    frame_time_ms: float = 0.0
    vram_mb: float = 0.0
    step_count: int = 0

RT = _SimRuntime()

def draw_particles_blender52():
    if not RT.running or RT.gl_buf is None:
        return

    scene = bpy.context.scene
    pt_size = getattr(scene, "warp_sph_point_size", 2.5)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
    shader.uniform_float("ModelViewProjectionMatrix", mvp)

    # Distinct fluid tint per demo scene
    if RT.active_demo == "DOUBLE_DAM_PILLARS":
        shader.uniform_float("color", (0.12, 0.68, 1.0, 1.0))
    else:
        shader.uniform_float("color", (0.05, 0.85, 0.72, 1.0))

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.point_size_set(pt_size)
    RT.gl_buf.draw_points()
    gpu.state.depth_test_set('NONE')


def _reallocate_and_init_scene(scene, num_p: int, demo_type: str):
    if RT.gl_buf:
        RT.gl_buf.destroy()

    RT.active_demo = demo_type
    enable_obs = 1 if demo_type == "DOUBLE_DAM_PILLARS" else 0

    RT.gl_buf = ZeroCopyGLBuffer52(num_p)
    RT.solver = RealtimeWCSPHSolver(
        num_particles=num_p,
        h=0.04,
        substeps=scene.warp_sph_substeps,
        enable_obstacles=enable_obs,
    )
    RT.vram_mb = RT.solver.get_vram_usage_mb()
    RT.step_count = 0

    # Spawn matching Blender 3D viewport wireframe domain & obstacle collider meshes
    build_blender_demo_helpers(demo_type)

    # Initialize GPU SoA particle distribution for the chosen demo
    launch_demo_scene_init(RT.solver, demo_type)
    RT.solver.rebuild_cuda_graph()

    pos_vbo_wp = RT.gl_buf.map_to_warp()
    wp.copy(pos_vbo_wp, RT.solver.pos_sim)
    RT.gl_buf.unmap()


class WM_OT_warp_sph_start(bpy.types.Operator):
    bl_idname = "wm.warp_sph_start"
    bl_label = "Start Warp 1.16+ Realtime SPH"
    _timer = None

    def modal(self, context, event):
        if not RT.running or event.type == 'ESC':
            self.cleanup(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            scene = context.scene
            target_count = scene.warp_sph_particle_count
            target_demo = scene.warp_sph_demo_scene

            # Dynamic Buffer & Demo Manager: reallocate/reset on-the-fly if slider or demo changed
            if RT.solver.num_particles != target_count or RT.active_demo != target_demo:
                _reallocate_and_init_scene(scene, target_count, target_demo)

            t0 = time.perf_counter()

            RT.solver.substeps = scene.warp_sph_substeps
            try:
                pos_vbo_wp = RT.gl_buf.map_to_warp()
                RT.solver.step(pos_vbo_wp, use_graph=scene.warp_sph_use_cuda_graph)
                RT.gl_buf.unmap()
                wp.synchronize_device("cuda:0")
            except Exception as exc:
                self.report({'ERROR'}, f"WarpSPH stopped: {exc}")
                self.cleanup(context)
                return {'CANCELLED'}
            dt_ms = (time.perf_counter() - t0) * 1000.0

            RT.frame_time_ms = 0.9 * RT.frame_time_ms + 0.1 * dt_ms if RT.step_count > 0 else dt_ms
            RT.fps_avg = 1000.0 / max(RT.frame_time_ms, 0.001)
            RT.step_count += 1

            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'PASS_THROUGH'}

    def execute(self, context):
        if RT.running:
            return {'FINISHED'}

        scene = context.scene
        _reallocate_and_init_scene(scene, scene.warp_sph_particle_count, scene.warp_sph_demo_scene)

        RT.running = True
        RT.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_particles_blender52, (), 'WINDOW', 'POST_VIEW'
        )
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.001, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        RT.running = False
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if RT.draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(RT.draw_handle, 'WINDOW')
            RT.draw_handle = None
        if RT.gl_buf:
            RT.gl_buf.destroy()
            RT.gl_buf = None
        RT.solver = None  # release solver SoA buffers and CUDA graph
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class WM_OT_warp_sph_load_demo(bpy.types.Operator):
    bl_idname = "wm.warp_sph_load_demo"
    bl_label = "Quick-Launch Demo Scene"

    demo_id: bpy.props.StringProperty(default="DOUBLE_DAM_PILLARS")

    def execute(self, context):
        scene = context.scene
        scene.warp_sph_demo_scene = self.demo_id
        if RT.running:
            _reallocate_and_init_scene(scene, scene.warp_sph_particle_count, self.demo_id)
        else:
            bpy.ops.wm.warp_sph_start()
        return {'FINISHED'}


class WM_OT_warp_sph_stop(bpy.types.Operator):
    bl_idname = "wm.warp_sph_stop"
    bl_label = "Stop Simulation"

    def execute(self, context):
        RT.running = False
        return {'FINISHED'}


class VIEW3D_PT_warp_sph(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WarpSPH"
    bl_label = "Warp 1.16+ Realtime Fluid"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Quick-Run Demo Scene Buttons
        box_demo = layout.box()
        box_demo.label(text="Quick-Run Demo Scenes", icon='SCENE_DATA')
        col_demo = box_demo.column(align=True)
        op1 = col_demo.operator(
            "wm.warp_sph_load_demo",
            icon='MOD_FLUIDSIM',
            text="Demo 1: Double Dam + Pillars"
        )
        op1.demo_id = "DOUBLE_DAM_PILLARS"

        op2 = col_demo.operator(
            "wm.warp_sph_load_demo",
            icon='FORCE_TURBULENCE',
            text="Demo 2: Block Flop Crown Splash"
        )
        op2.demo_id = "BLOCK_FLOP_SPLASH"

        layout.separator()
        col = layout.column(align=True)
        col.prop(scene, "warp_sph_demo_scene", text="Scene")
        col.prop(scene, "warp_sph_particle_count", slider=True)
        col.prop(scene, "warp_sph_substeps")
        col.prop(scene, "warp_sph_point_size")
        col.prop(scene, "warp_sph_use_cuda_graph")

        layout.separator()
        row = layout.row(align=True)
        if not RT.running:
            row.operator("wm.warp_sph_start", icon='PLAY', text="Start Realtime Sim")
        else:
            row.operator("wm.warp_sph_stop", icon='PAUSE', text="Stop Sim (ESC)")

        # Live GPU Telemetry Readout
        box = layout.box()
        box.label(text="GPU Performance Telemetry", icon='PREFERENCES')
        col_stat = box.column(align=True)
        col_stat.label(text=f"Status: {'RUNNING' if RT.running else 'IDLE'}")
        col_stat.label(text=f"Active Demo: {RT.active_demo}")
        col_stat.label(text=f"FPS Avg: {RT.fps_avg:.1f} FPS")
        col_stat.label(text=f"Frame Time: {RT.frame_time_ms:.2f} ms")
        col_stat.label(text=f"SoA VRAM Footprint: {RT.vram_mb:.1f} MB")
        if RT.running and RT.solver:
            throughput = (RT.solver.num_particles / max(RT.frame_time_ms, 0.001))
            col_stat.label(text=f"Throughput: {throughput:,.0f} particles/ms")


def register():
    bpy.types.Scene.warp_sph_demo_scene = EnumProperty(
        name="Demo Scene",
        items=[
            ("DOUBLE_DAM_PILLARS", "Demo 1: Double Dam + Pillars", "Two opposing fluid columns colliding with a central sphere and 4 pillar obstacles"),
            ("BLOCK_FLOP_SPLASH", "Demo 2: Block Flop Crown Splash", "Heavy elevated fluid cube dropping into a shallow resting puddle basin"),
        ],
        default="DOUBLE_DAM_PILLARS",
    )
    bpy.types.Scene.warp_sph_particle_count = IntProperty(
        name="Particles",
        description="Total fluid particles (dynamically reallocates GPU SoA buffers on change)",
        default=1_000_000,
        min=100_000,
        max=4_000_000,
        step=100_000,
    )
    bpy.types.Scene.warp_sph_substeps = IntProperty(
        name="Sub-steps / Frame",
        description="Physics sub-steps executed per viewport redraw",
        default=2,
        min=1,
        max=10,
    )
    bpy.types.Scene.warp_sph_point_size = FloatProperty(
        name="Viewport Point Size",
        default=2.5,
        min=1.0,
        max=10.0,
    )
    bpy.types.Scene.warp_sph_use_cuda_graph = BoolProperty(
        name="Warp 1.16+ CUDA Graph Capture",
        description="Execute Coalesced Density + WCSPH Force + Copy-back via captured CUDA Graph",
        default=True,
    )
    bpy.utils.register_class(WM_OT_warp_sph_start)
    bpy.utils.register_class(WM_OT_warp_sph_load_demo)
    bpy.utils.register_class(WM_OT_warp_sph_stop)
    bpy.utils.register_class(VIEW3D_PT_warp_sph)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_warp_sph)
    bpy.utils.unregister_class(WM_OT_warp_sph_stop)
    bpy.utils.unregister_class(WM_OT_warp_sph_load_demo)
    bpy.utils.unregister_class(WM_OT_warp_sph_start)
    del bpy.types.Scene.warp_sph_use_cuda_graph
    del bpy.types.Scene.warp_sph_point_size
    del bpy.types.Scene.warp_sph_substeps
    del bpy.types.Scene.warp_sph_particle_count
    del bpy.types.Scene.warp_sph_demo_scene
