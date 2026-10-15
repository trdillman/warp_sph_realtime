"""Blender operators, panel, scene properties, and the modal simulation runtime.

NVIDIA Warp and every CUDA-dependent module are imported only in
``_activate_runtime()``, after the owned/version/device checks, so the
extension registers (and can install Warp) on machines without it.
"""

from __future__ import annotations

import importlib
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty

from . import warp_installer, warp_runtime

wp = None
ZeroCopyGLBuffer52 = None
RealtimeWCSPHSolver = None
build_blender_demo_helpers = None
launch_demo_scene_init = None
cleanup_viewport_renderer = None
draw_particles_viewport = None
_startup_error = ""


def _activate_runtime():
    """Import CUDA-dependent modules only after owned/version/device checks."""
    global wp, ZeroCopyGLBuffer52, RealtimeWCSPHSolver
    global build_blender_demo_helpers, launch_demo_scene_init
    global cleanup_viewport_renderer, draw_particles_viewport
    if wp is not None:
        return  # already active; keep existing bindings
    loaded = warp_runtime.require_cuda_runtime()
    bridge = importlib.import_module(".render.interop_bridge", __package__)
    viewport = importlib.import_module(".render.viewport_draw", __package__)
    solver = importlib.import_module(".sim.sim_core", __package__)
    demos = importlib.import_module(".scene.demo_scenes", __package__)
    wp = loaded
    ZeroCopyGLBuffer52 = bridge.ZeroCopyGLBuffer52
    RealtimeWCSPHSolver = solver.RealtimeWCSPHSolver
    build_blender_demo_helpers = demos.build_blender_demo_helpers
    launch_demo_scene_init = demos.launch_demo_scene_init
    cleanup_viewport_renderer = viewport.cleanup_viewport_renderer
    draw_particles_viewport = viewport.draw_particles_viewport

class _SimRuntime:
    gl_buf: ZeroCopyGLBuffer52 | None = None
    solver: RealtimeWCSPHSolver | None = None
    draw_handle = None
    running: bool = False
    paused: bool = False      # physics frozen; GPU state and drawing kept alive
    step_once: bool = False   # advance exactly one frame while paused
    active_demo: str = "DOUBLE_DAM_PILLARS"
    fps_avg: float = 0.0
    frame_time_ms: float = 0.0
    vram_mb: float = 0.0
    step_count: int = 0


RT = _SimRuntime()


def draw_particles_blender52():
    draw_particles_viewport(RT.gl_buf, RT.running)


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
    RT.solver.stiffness = scene.warp_sph_stiffness
    RT.solver.cfl_factor = scene.warp_sph_time_step_scale
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
    bl_label = "Start WarpSPH Realtime"
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
            elif (RT.solver.stiffness != scene.warp_sph_stiffness
                  or RT.solver.cfl_factor != scene.warp_sph_time_step_scale):
                RT.solver.stiffness = scene.warp_sph_stiffness
                RT.solver.cfl_factor = scene.warp_sph_time_step_scale
                RT.solver.update_time_step()
                RT.solver.rebuild_cuda_graph()

            if RT.paused and not RT.step_once:
                # Keep redrawing so display settings stay live on the frozen frame
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                return {'PASS_THROUGH'}
            RT.step_once = False

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

        try:
            _activate_runtime()
            scene = context.scene
            _reallocate_and_init_scene(scene, scene.warp_sph_particle_count, scene.warp_sph_demo_scene)
        except Exception as exc:
            RT.running = False
            self.report({'ERROR'}, f"WarpSPH cannot start: {exc}")
            return {'CANCELLED'}

        RT.running = True
        RT.paused = False
        RT.step_once = False
        RT.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_particles_blender52, (), 'WINDOW', 'POST_VIEW'
        )
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.001, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        RT.running = False
        RT.paused = False
        RT.step_once = False
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if RT.draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(RT.draw_handle, 'WINDOW')
            RT.draw_handle = None
        if cleanup_viewport_renderer is not None:
            cleanup_viewport_renderer()
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


class WM_OT_warp_sph_pause(bpy.types.Operator):
    bl_idname = "wm.warp_sph_pause"
    bl_label = "Pause / Resume Simulation"
    bl_description = "Freeze or continue the physics without releasing GPU buffers"

    @classmethod
    def poll(cls, context):
        return RT.running

    def execute(self, context):
        RT.paused = not RT.paused
        return {'FINISHED'}


class WM_OT_warp_sph_step_frame(bpy.types.Operator):
    bl_idname = "wm.warp_sph_step_frame"
    bl_label = "Step One Frame"
    bl_description = "Pause (if running) and advance the simulation by exactly one frame"

    @classmethod
    def poll(cls, context):
        return RT.running

    def execute(self, context):
        RT.paused = True
        RT.step_once = True
        return {'FINISHED'}
class WM_OT_warp_sph_install(bpy.types.Operator):
    bl_idname = "wm.warp_sph_install"
    bl_label = "Install NVIDIA Warp"
    bl_description = "Download and checksum-verify Warp 1.17.0 CUDA 12 into this extension's isolated user directory"

    repair: BoolProperty(name="Repair Warp", default=False)

    def execute(self, context):
        if RT.running:
            self.report({'ERROR'}, "Stop the simulation before changing the Warp runtime")
            return {'CANCELLED'}
        result = warp_installer.install_warp(force_reinstall=self.repair)
        self.report({'INFO'} if result.success else {'ERROR'}, result.message)
        return {'FINISHED'} if result.success else {'CANCELLED'}


class WARPSPH_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or "warp_sph_realtime"

    def draw(self, context):
        box = self.layout.box()
        box.label(text="NVIDIA Warp Runtime (CUDA 12.9)", icon='PREFERENCES')
        state = warp_runtime.status()
        box.label(text=state.message, icon='CHECKMARK' if state.available else 'ERROR')
        if _startup_error:
            box.label(text=_startup_error, icon='ERROR')
        if not state.available:
            box.operator("wm.warp_sph_install", text="Install NVIDIA Warp 1.17.0", icon='IMPORT')
        row = box.row()
        op = row.operator("wm.warp_sph_install", text="Repair Warp (restart required)", icon='FILE_REFRESH')
        op.repair = True
        box.label(text="WarpSPH never modifies Blender's Python or other extensions.")


class VIEW3D_PT_warp_sph(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WarpSPH"
    bl_label = "WarpSPH Realtime Fluid"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        state = warp_runtime.status()
        status_box = layout.box()
        status_box.label(text="Runtime: " + ("Installed (GPU check on Start)" if state.available else "Not ready"), icon='CHECKMARK' if state.available else 'ERROR')
        status_box.label(text=state.message)
        if _startup_error:
            status_box.label(text=_startup_error, icon='ERROR')
        if not state.available:
            status_box.operator("wm.warp_sph_install", text="Install NVIDIA Warp", icon='IMPORT')
        op = status_box.operator("wm.warp_sph_install", text="Repair Warp", icon='FILE_REFRESH')
        op.repair = True

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
        col.prop(scene, "warp_sph_use_cuda_graph")

        box_speed = layout.box()
        box_speed.label(text="Simulation Speed vs. Stability", icon='TIME')
        col_speed = box_speed.column(align=True)
        col_speed.prop(scene, "warp_sph_substeps")
        col_speed.prop(scene, "warp_sph_time_step_scale", slider=True)
        col_speed.prop(scene, "warp_sph_stiffness")
        if RT.solver:
            col_speed.label(
                text=f"dt = {RT.solver.dt * 1000:.3f} ms  |  sim speed ≈ "
                     f"{RT.solver.dt * RT.solver.substeps * RT.fps_avg:.2f}x real time"
            )

        box_vis = layout.box()
        box_vis.label(text="Viewport Display & SSFR", icon='SHADING_RENDERED')
        col_vis = box_vis.column(align=True)
        col_vis.prop(scene, "warp_sph_show_particles")
        col_vis.prop(scene, "warp_sph_render_mode", text="Mode")
        col_vis.prop(scene, "warp_sph_particle_color", text="")
        col_vis.prop(scene, "warp_sph_depth_test")

        if scene.warp_sph_render_mode == "POINTS":
            col_vis.prop(scene, "warp_sph_point_size")
        elif scene.warp_sph_render_mode == "SPHERES":
            col_vis.prop(scene, "warp_sph_particle_radius")
        else:
            col_vis.prop(scene, "warp_sph_particle_radius")
            col_vis.prop(scene, "warp_sph_filter_iterations")
            col_vis.prop(scene, "warp_sph_filter_sigma")
            col_vis.prop(scene, "warp_sph_absorption")

        layout.separator()
        row = layout.row(align=True)
        if not RT.running:
            row.enabled = state.available and not bool(_startup_error)
            row.operator("wm.warp_sph_start", icon='PLAY', text="Start Realtime Sim")
        else:
            if RT.paused:
                row.operator("wm.warp_sph_pause", icon='PLAY', text="Resume")
            else:
                row.operator("wm.warp_sph_pause", icon='PAUSE', text="Pause")
            row.operator("wm.warp_sph_step_frame", icon='FRAME_NEXT', text="Step")
            row.operator("wm.warp_sph_stop", icon='SNAP_FACE', text="Stop (ESC)")

        # Live GPU Telemetry Readout
        box = layout.box()
        box.label(text="GPU Performance Telemetry", icon='PREFERENCES')
        col_stat = box.column(align=True)
        status = 'IDLE' if not RT.running else ('PAUSED' if RT.paused else 'RUNNING')
        col_stat.label(text=f"Status: {status}")
        col_stat.label(text=f"Active Demo: {RT.active_demo}")
        col_stat.label(text=f"FPS Avg: {RT.fps_avg:.1f} FPS")
        col_stat.label(text=f"Frame Time: {RT.frame_time_ms:.2f} ms")
        col_stat.label(text=f"SoA VRAM Footprint: {RT.vram_mb:.1f} MB")
        if RT.running and RT.solver:
            throughput = (RT.solver.num_particles / max(RT.frame_time_ms, 0.001))
            col_stat.label(text=f"Throughput: {throughput:,.0f} particles/ms")


def register():
    global _startup_error
    _startup_error = ""
    try:
        warp_installer.apply_pending_repair_on_startup()
    except Exception as exc:
        _startup_error = str(exc)
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
    bpy.types.Scene.warp_sph_time_step_scale = FloatProperty(
        name="Time Step (CFL)",
        description="Physics time step as a fraction of the stability limit. Higher = faster simulation, "
                    "but above ~0.5 high-speed impacts can blow up",
        default=0.4,
        min=0.1,
        max=1.0,
        step=1,          # Blender steps are 1/100 units: 0.01 per click
        precision=3,
    )
    bpy.types.Scene.warp_sph_stiffness = FloatProperty(
        name="Fluid Stiffness",
        description="Pressure stiffness (sets sound speed). Lower = larger time step and faster simulation, "
                    "but more compressible and less stable on impacts",
        default=90000.0,
        min=10000.0,
        max=400000.0,
        step=500000,
    )
    bpy.types.Scene.warp_sph_show_particles = BoolProperty(
        name="Show Particles",
        default=True,
    )
    bpy.types.Scene.warp_sph_render_mode = EnumProperty(
        name="Render Mode",
        items=[
            ("SSFR_LIQUID", "SSFR Liquid (Narrow-Range + Beer-Lambert)", "5-pass Screen-Space Fluid Renderer with Truong & Yuksel Narrow-Range Filter and Fresnel shading"),
            ("SPHERES", "Shaded Sphere Impostors", "Single-pass eye-space shaded 3D spheres with per-fragment depth correction"),
            ("POINTS", "Raw Points (Max FPS)", "Fastest raw GL_POINTS mode for telemetry benchmarking"),
        ],
        default="SSFR_LIQUID",
    )
    bpy.types.Scene.warp_sph_point_size = FloatProperty(
        name="Point Size",
        description="Particle point size in screen pixels (Raw Points mode)",
        default=2.5,
        min=1.0,
        max=20.0,
        step=10,         # 0.1 px per click
        precision=1,
    )
    bpy.types.Scene.warp_sph_particle_radius = FloatProperty(
        name="Particle Radius (m)",
        description="World-space sphere radius for SSFR and Sphere Impostor modes",
        default=0.022,
        min=0.002,
        max=0.10,
        soft_min=0.004,  # narrow drag range so mouse travel maps to useful radii
        soft_max=0.04,
        step=0.05,       # 0.0005 m (0.5 mm) per click
        precision=4,
    )
    bpy.types.Scene.warp_sph_filter_iterations = IntProperty(
        name="Narrow-Range Filter Passes",
        description="Number of 1D separable Horizontal + Vertical Narrow-Range smoothing iterations",
        default=2,
        min=1,
        max=5,
    )
    bpy.types.Scene.warp_sph_filter_sigma = FloatProperty(
        name="Filter Sigma (px)",
        description="Screen-space Gaussian standard deviation inside the Narrow-Range depth window",
        default=4.0,
        min=1.0,
        max=12.0,
        step=10,         # 0.1 px per click
        precision=1,
    )
    bpy.types.Scene.warp_sph_absorption = FloatVectorProperty(
        name="Beer-Lambert Absorption (RGB)",
        description="Wavelength-dependent light extinction coefficients exp(-a * thickness)",
        size=3,
        default=(0.85, 0.22, 0.05),
        min=0.0,
        max=5.0,
        soft_max=2.0,
        step=1,          # 0.01 per click
        precision=3,
    )
    bpy.types.Scene.warp_sph_particle_color = FloatVectorProperty(
        name="Particle Color",
        subtype='COLOR',
        size=4,
        default=(0.12, 0.68, 1.0, 1.0),
        min=0.0,
        max=1.0,
    )
    bpy.types.Scene.warp_sph_depth_test = BoolProperty(
        name="Hide Behind Objects",
        description="Particles are hidden behind scene geometry (off = always draw on top)",
        default=True,
    )
    bpy.types.Scene.warp_sph_use_cuda_graph = BoolProperty(
        name="Warp 1.17 CUDA Graph Capture",
        description="Execute Coalesced Density + WCSPH Force + Copy-back via captured CUDA Graph",
        default=True,
    )
    bpy.utils.register_class(WM_OT_warp_sph_start)
    bpy.utils.register_class(WM_OT_warp_sph_load_demo)
    bpy.utils.register_class(WM_OT_warp_sph_stop)
    bpy.utils.register_class(WM_OT_warp_sph_pause)
    bpy.utils.register_class(WM_OT_warp_sph_step_frame)
    bpy.utils.register_class(WM_OT_warp_sph_install)
    bpy.utils.register_class(WARPSPH_Preferences)
    bpy.utils.register_class(VIEW3D_PT_warp_sph)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_warp_sph)
    bpy.utils.unregister_class(WM_OT_warp_sph_step_frame)
    bpy.utils.unregister_class(WM_OT_warp_sph_pause)
    bpy.utils.unregister_class(WARPSPH_Preferences)
    bpy.utils.unregister_class(WM_OT_warp_sph_install)
    bpy.utils.unregister_class(WM_OT_warp_sph_stop)
    bpy.utils.unregister_class(WM_OT_warp_sph_load_demo)
    bpy.utils.unregister_class(WM_OT_warp_sph_start)
    del bpy.types.Scene.warp_sph_use_cuda_graph
    del bpy.types.Scene.warp_sph_depth_test
    del bpy.types.Scene.warp_sph_particle_color
    del bpy.types.Scene.warp_sph_absorption
    del bpy.types.Scene.warp_sph_filter_sigma
    del bpy.types.Scene.warp_sph_filter_iterations
    del bpy.types.Scene.warp_sph_particle_radius
    del bpy.types.Scene.warp_sph_point_size
    del bpy.types.Scene.warp_sph_render_mode
    del bpy.types.Scene.warp_sph_show_particles
    del bpy.types.Scene.warp_sph_stiffness
    del bpy.types.Scene.warp_sph_time_step_scale
    del bpy.types.Scene.warp_sph_substeps
    del bpy.types.Scene.warp_sph_particle_count
    del bpy.types.Scene.warp_sph_demo_scene
