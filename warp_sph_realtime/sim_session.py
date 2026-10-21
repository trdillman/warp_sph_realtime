"""Per-scene simulation session (P1.1): every piece of runtime state behind start/stop/tick/dispose.

``addon.py`` is the Blender UI around it: operators, panels and the modal timer call these methods.
One ``SimSession`` exists per scene (``get(scene)``); at most one runs per process, because the solver,
its 1 ms timer and its viewport draw handler are process resources (``active()``).

CUDA-dependent bindings live in ``runtime`` and are filled only by ``addon._activate_runtime()``, the
single Warp import point. The sticky restart-required / interop-unavailable flags and the quarantine
are process-wide (``render.transport.process_state``) and outlive add-on reloads.
"""
from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable

import bpy

from .props import spacing_metres, stiffness_pa
from .render import session as transport_session
from .render.gpu_points import settings_from_scene
from .render.transport import COMPLETED, FAILED, QUARANTINED, FatalTransportError, process_state

_LOG = logging.getLogger(__name__)

# OpenGL display modes drawn from the GLInterop texture (session) path. SSFR and sphere impostors
# stay on the legacy raw-GL VBO path.
GL_TEXTURE_MODES = ("POINTS",)
_BACKEND_NAMES = {"VULKAN": "Vulkan", "OPENGL": "OpenGL"}

# SimSession.active_demo while an authored (tagged-object) scene runs; never a scene.warp_sph.demo_scene value.
CUSTOM_SCENE = "CUSTOM"

TIMER_INTERVAL_S = 0.001  # P0.5: 1 ms timer chosen by measured latency; do not change
BAKE_CHUNK_S = 0.004      # P2.2: most time one timer tick spends queueing bake frames (never a blocking loop)
BAKE_RING_SLOTS = 3       # frames in flight between the solver, the host copy and the disk writer

_QUARANTINE: list = process_state().quarantine  # objects never freed until exit, across reloads

Report = Callable[[set, str], object]  # Operator.report signature


class SceneSetupError(Exception):
    """The authored scene cannot be simulated as it stands (bad roles, geometry or spacing)."""


class Runtime:
    """CUDA-dependent bindings. Set only by ``addon._activate_runtime()``; ``wp`` is None until then."""
    wp = None
    ZeroCopyGLBuffer52 = None
    RealtimeWCSPHSolver = None
    build_blender_demo_helpers = None
    remove_blender_demo_helpers = None
    launch_demo_scene_init = None
    cleanup_viewport_renderer = None
    draw_particles_viewport = None


runtime = Runtime()


def gpu_backend() -> str:
    import gpu
    return gpu.platform.backend_type_get()


_device_total_mb: list[float] = []


def device_total_mb() -> float | None:
    """Total memory of the simulation GPU in MB, once Warp is loaded (read once: each query is a CUDA call).
    None before the first Start, after a fatal GPU error or if the query fails."""
    if _device_total_mb:
        return _device_total_mb[0]
    if runtime.wp is None or process_state().restart_required:
        return None
    try:
        _device_total_mb.append(runtime.wp.get_device("cuda:0").total_memory / 2**20)
    except Exception:
        return None
    return _device_total_mb[0]


def _scene_api():
    """NumPy-backed scene helpers (scene.blender_scene), imported on first use."""
    return importlib.import_module(".scene.blender_scene", __package__)


def _redraw_views(screen):
    for area in getattr(screen, "areas", ()):
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def physics_changed(solver, settings) -> bool:
    """True when the solver's stiffness, CFL factor or viscosity differs from the scene settings."""
    return (solver.stiffness != stiffness_pa(settings) or solver.cfl_factor != settings.time_step_scale
            or solver.viscosity != settings.viscosity)


def apply_physics(solver, settings):
    """Copy the physics settings into the solver and re-derive dt. The caller recaptures the CUDA graph."""
    solver.stiffness = stiffness_pa(settings)
    solver.cfl_factor = settings.time_step_scale
    solver.viscosity = settings.viscosity
    solver.update_time_step()


class SimSession:
    """Runtime state of one scene's simulation. ``session`` is the render transport session
    (render/session.py; Vulkan/other backends and OpenGL Points), ``gl_buf`` the legacy OpenGL VBO
    (SSFR, spheres); at most one of the two is live."""

    def __init__(self, scene=None):
        self.scene = scene
        self.gl_buf = None
        self.session: transport_session.TransportSession | None = None
        self.solver = None
        self.draw_handle = None
        self.timer = None
        self.token = 0             # bumped on every start/stop; orphaned modals exit
        self.running = False
        self.paused = False        # physics frozen; GPU state and drawing kept alive
        self.step_once = False     # advance exactly one frame while paused
        self.active_demo = "DOUBLE_DAM_PILLARS"
        self.fps_avg = 0.0
        self.frame_time_ms = 0.0
        self.vram_mb = 0.0
        self.step_count = 0
        self.last_tick = 0.0
        self.last_close = ""
        self.last_error = ""
        self.draw_errors: set = set()  # distinct draw failures already logged this run
        self.bake = None               # cache.bake.BakeJob while a bake runs (P2.2); driven by tick()
        self.bake_result = ""          # outcome of the last bake, for the panel

    # Process-lifetime flags: they outlive add-on disable/enable (module reload).
    @property
    def restart_required(self) -> str:  # fatal GPU/CUDA state: no further CUDA work this process
        return process_state().restart_required

    @restart_required.setter
    def restart_required(self, value: str):
        process_state().restart_required = value

    @property
    def interop_unavailable(self) -> str:  # recoverable interop failure: Readback until restart
        return process_state().interop_unavailable

    @interop_unavailable.setter
    def interop_unavailable(self, value: str):
        process_state().interop_unavailable = value

    # -- state queries -------------------------------------------------------------------------------

    def readback_active(self) -> bool:
        """True while Readback runs on a backend where OpenGL would help (from what runs, not prefs)."""
        session = self.session
        return session is not None and session.backend != 'OPENGL' and not session.transport.capabilities().zero_copy

    def gl_uses_texture(self, scene) -> bool:
        """OpenGL route, fixed for a run: chosen from the display mode at Start, kept through resizes."""
        if self.session is not None or self.gl_buf is not None:
            return self.session is not None
        if self.interop_unavailable:
            # The texture interop failed earlier this process: the legacy VBO interop (no CPU copy)
            # beats Readback, and was the only OpenGL route before the texture adapter.
            return False
        return scene.warp_sph.render_mode in GL_TEXTURE_MODES

    # -- viewport draw -------------------------------------------------------------------------------

    def draw(self):
        """SpaceView3D POST_VIEW callback; never raises into Blender."""
        session = self.session
        try:
            if session is None:
                # Legacy OpenGL path: the SSFR/sphere passes restore GL state in a finally block
                # (_preserve_gl_state) and draw_points restores its own, so a failure leaves none behind.
                runtime.draw_particles_viewport(self.gl_buf, self.running)
                return
            scene = bpy.context.scene
            if not self.running or not scene.warp_sph.show_particles:
                return
            session.draw(settings_from_scene(scene, session.backend))
        except FatalTransportError as exc:
            self.mark_fatal(str(exc))
        except Exception as exc:
            self._report_draw_failure(exc)

    def _report_draw_failure(self, exc: Exception):
        """Stop the run and log each distinct draw failure once, never once per redraw."""
        self.last_error = f"Viewport draw failed: {exc}"
        self.running = False  # the modal timer stops the session; never raise every redraw
        key = f"{type(exc).__name__}: {exc}"
        if key not in self.draw_errors:
            self.draw_errors.add(key)
            _LOG.error("WarpSPH viewport draw failed (logged once per distinct error): %s", key, exc_info=exc)

    # -- failure handling ----------------------------------------------------------------------------

    def mark_fatal(self, reason: str):
        self.restart_required = self.restart_required or reason
        self.running = False

    def _retire(self, session):
        result = session.close()
        self.last_close = result
        if result != COMPLETED:
            _QUARANTINE.append(session)
        if result == QUARANTINED:  # Blender objects retained: do not create more interop objects
            self.interop_unavailable = self.interop_unavailable or \
                f"{_BACKEND_NAMES.get(session.backend, session.backend)} interop cleanup was quarantined (no GPU-context route)"
        if result == FAILED:
            self.mark_fatal(session.fatal_reason or "transport cleanup did not complete; ownership retained")

    # -- solver and output allocation ----------------------------------------------------------------

    def _custom_solver(self, scene, depsgraph):
        """Solver for the tagged objects; the particle count is the fill at the scene's spacing, or the
        Max Particles budget when the scene has emitters."""
        from .scene.description import SceneFillError, SceneValidationError
        try:
            desc = _scene_api().description_from_blender(scene, depsgraph)
            settings = scene.warp_sph
            return runtime.RealtimeWCSPHSolver.from_scene(
                desc, spacing_metres(settings, _scene_api().unit_scale(scene)), substeps=settings.substeps,
                stiffness=stiffness_pa(settings), cfl_factor=settings.time_step_scale,
                viscosity=settings.viscosity)
        except (SceneValidationError, SceneFillError) as exc:
            raise SceneSetupError(str(exc)) from exc

    def _new_solver(self, scene, num_p: int, demo_type: str, depsgraph=None):
        if demo_type == CUSTOM_SCENE:
            return self._custom_solver(
                scene, depsgraph if depsgraph is not None else bpy.context.evaluated_depsgraph_get())
        settings = scene.warp_sph
        solver = runtime.RealtimeWCSPHSolver(
            num_particles=num_p,
            h=0.04,
            substeps=settings.substeps,
            enable_obstacles=0,  # the demos set a scene: their obstacles are mesh colliders
        )
        solver.stiffness = stiffness_pa(settings)
        solver.cfl_factor = settings.time_step_scale
        solver.viscosity = settings.viscosity
        # Initialize GPU SoA particle distribution for the chosen demo
        runtime.launch_demo_scene_init(solver, demo_type)
        solver.rebuild_cuda_graph()
        return solver

    def reallocate(self, scene, num_p: int, demo_type: str, window=None, depsgraph=None):
        """Transactional: the new output and solver are built while the old ones stay live;
        on failure the new ones are released and the running session is untouched.

        demo_type CUSTOM_SCENE builds the authored scene from tagged objects; num_p is then ignored and
        every output is sized from the solver's fill count."""
        backend = gpu_backend()
        if backend == 'OPENGL' and not self.gl_uses_texture(scene):
            # An authored scene's particle count is known only after the fill, so its solver comes first.
            solver = self._new_solver(scene, num_p, demo_type, depsgraph) if demo_type == CUSTOM_SCENE else None
            gl_buf = runtime.ZeroCopyGLBuffer52(solver.num_particles if solver is not None else num_p)
            try:
                if solver is None:
                    solver = self._new_solver(scene, num_p, demo_type)
                with gl_buf.mapped() as pos_vbo_wp:
                    solver.copy_positions(pos_vbo_wp)
            except BaseException:
                gl_buf.destroy()
                raise
            # Commit point: output and solver switch together, before the old output is released.
            old_buf, self.gl_buf = self.gl_buf, gl_buf
            self.solver, self.active_demo = solver, demo_type
            if old_buf:
                old_buf.destroy()
        else:
            solver = self._new_solver(scene, num_p, demo_type, depsgraph)
            window = window or bpy.context.window
            try:
                session = transport_session.open_session(
                    backend, solver.num_particles, window.as_pointer(), interop_unavailable=self.interop_unavailable)
            except FatalTransportError:
                _QUARANTINE.append(solver)
                raise
            try:
                session.tick(solver.copy_positions)  # publish before first draw
            except FatalTransportError:
                _QUARANTINE.extend((solver, session))  # no CUDA cleanup after a fatal error
                raise
            except BaseException:
                self._retire(session)
                if self.restart_required:  # its cleanup failed: CUDA state unknown, keep the solver too
                    _QUARANTINE.append(solver)
                raise
            zero_copy = session.transport.capabilities().zero_copy
            old_session, old_solver = self.session, self.solver
            if backend in _BACKEND_NAMES and not zero_copy:
                if old_session is not None and old_session.transport.capabilities().zero_copy:
                    # A resize must not silently downgrade a working interop session.
                    self._retire(session)
                    if self.restart_required:
                        _QUARANTINE.append(solver)
                        raise FatalTransportError(self.restart_required)
                    raise RuntimeError(
                        f"new size cannot use {_BACKEND_NAMES[backend]} interop ({session.reason}); kept the current size")
                self.interop_unavailable = session.reason  # fall back once; no interop retries
            # Commit point: output and solver switch together, before the old session is retired.
            self.session = session
            self.solver, self.active_demo = solver, demo_type
            if old_session is not None:
                self._retire(old_session)
                if self.restart_required:  # old cleanup failed: stop; stop() quarantines the new pair
                    _QUARANTINE.append(old_solver)
                    raise FatalTransportError(self.restart_required)

        self.vram_mb = solver.get_vram_usage_mb()
        self.step_count = 0
        self.last_tick = 0.0
        if demo_type == CUSTOM_SCENE:
            if runtime.remove_blender_demo_helpers is not None:
                runtime.remove_blender_demo_helpers()  # demo wireframes would only confuse an authored scene
        else:
            # Spawn matching Blender 3D viewport wireframe domain & obstacle collider meshes
            runtime.build_blender_demo_helpers(demo_type)

    def apply_scene(self, context, count: int, demo: str, report: Report) -> bool:
        """Resize or switch demo while running; on failure the running session is kept and the
        scene settings are put back so the same change is not retried every tick."""
        try:
            self.reallocate(self.scene, count, demo, context.window)
            return True
        except FatalTransportError as exc:
            self.mark_fatal(str(exc))
            report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
        except Exception as exc:
            report({'ERROR'}, f"WarpSPH could not rebuild the scene: {exc}")
            if self.solver is not None:  # keep the running session; do not retry the same change
                settings = self.scene.warp_sph
                if self.active_demo == CUSTOM_SCENE:
                    settings.scene_source = 'CUSTOM'
                else:
                    settings.particle_count = self.solver.num_particles
                    settings.demo_scene = self.active_demo
        return False

    # -- lifecycle -----------------------------------------------------------------------------------

    def start(self, context, report: Report, activate: Callable[[], None]) -> int | None:
        """Build the solver and output for ``context.scene``, then add the draw handler and 1 ms timer.

        Returns the run token for the modal operator, or None after reporting why it could not start
        (nothing is left behind). ``activate`` is ``addon._activate_runtime`` (the Warp import point)."""
        if self.restart_required:
            report({'ERROR'}, f"WarpSPH needs a Blender restart: {self.restart_required}")
            return None
        self.scene = context.scene
        self.last_error = ""
        self.draw_errors.clear()
        try:
            activate()
            scene = self.scene
            settings = scene.warp_sph
            if settings.scene_source == 'CUSTOM':
                self.reallocate(scene, 0, CUSTOM_SCENE, context.window, depsgraph=context.evaluated_depsgraph_get())
            else:
                self.reallocate(scene, settings.particle_count, settings.demo_scene, context.window)
        except SceneSetupError as exc:
            self.stop(context)
            self.last_error = f"Scene not valid: {exc}"
            report({'ERROR'}, f"WarpSPH cannot start this scene: {exc}")
            return None
        except FatalTransportError as exc:
            self.mark_fatal(str(exc))
            self.stop(context)
            report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
            return None
        except Exception as exc:
            self.stop(context)
            report({'ERROR'}, f"WarpSPH cannot start: {exc}")
            return None

        self.running = True
        self.paused = False
        self.step_once = False
        self.token += 1
        try:
            self.draw_handle = bpy.types.SpaceView3D.draw_handler_add(self.draw, (), 'WINDOW', 'POST_VIEW')
            self.timer = context.window_manager.event_timer_add(TIMER_INTERVAL_S, window=context.window)
        except Exception as exc:
            self.stop(context)
            report({'ERROR'}, f"WarpSPH cannot start: {exc}")
            return None
        return self.token

    def tick(self, context, report: Report) -> bool:
        """One timer event. Returns False when the run ended here (the session is already stopped)."""
        screen = context.screen
        if screen is None or not any(area.type == 'VIEW_3D' for area in screen.areas):
            report({'WARNING'}, "WarpSPH stopped: its window has no 3D viewport")
            self.stop(context)
            return False
        try:
            settings = self.scene.warp_sph
            target_count = settings.particle_count
            target_demo = settings.demo_scene
        except ReferenceError:  # the scene was deleted under the run
            report({'WARNING'}, "WarpSPH stopped: its scene was removed")
            self.stop(context)
            return False

        if self.bake is not None:  # a bake owns the solver: settings changes apply to the next bake
            return self._tick_bake(context, report, screen)

        # Dynamic Buffer & Demo Manager: reallocate/reset on-the-fly if slider or demo changed.
        # An authored scene is fixed for the run: role or geometry changes need Stop + Start.
        if self.active_demo != CUSTOM_SCENE and (
                self.solver.num_particles != target_count or self.active_demo != target_demo):
            if not self.apply_scene(context, target_count, target_demo, report):
                if not self.running:
                    self.stop(context)
                    return False
                _redraw_views(screen)  # kept the running session; no step this tick
                return True
        elif physics_changed(self.solver, settings):
            try:
                # Stiffness, time step and viscosity are captured by value: recapture the graph (no
                # allocation; the same path the stiffness slider has always used).
                apply_physics(self.solver, settings)
                self.solver.rebuild_cuda_graph()
            except Exception as exc:
                if self.session is not None:  # as a failed step (session.tick): CUDA state unknown
                    self.mark_fatal(f"CUDA graph rebuild failed; CUDA state unknown: {exc}")
                    report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
                else:
                    report({'ERROR'}, f"WarpSPH stopped: {exc}")
                self.stop(context)
                return False

        if self.paused and not self.step_once:
            _redraw_views(screen)  # keep redrawing so display settings stay live on the frozen frame
            return True

        t0 = time.perf_counter()

        self.solver.substeps = settings.substeps
        solver, use_graph = self.solver, settings.use_cuda_graph
        try:
            if self.active_demo == CUSTOM_SCENE:  # animated colliders: one transform read per frame
                _scene_api().sync_collider_keyframes(self.scene, solver)
            if self.session is not None:
                # No host sync: the adapter orders publication on the solver stream and
                # skips the step (backpressure) while a step is in flight or both slots are busy.
                stepped = self.session.tick(lambda out: solver.step(out, use_graph=use_graph))
            else:
                with self.gl_buf.mapped() as pos_vbo_wp:
                    solver.step(pos_vbo_wp, use_graph=use_graph)
                runtime.wp.synchronize_device("cuda:0")
                stepped = True
        except FatalTransportError as exc:
            self.mark_fatal(str(exc))
            report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
            self.stop(context)
            return False
        except Exception as exc:
            report({'ERROR'}, f"WarpSPH stopped: {exc}")
            self.stop(context)
            return False
        now = time.perf_counter()
        if self.session is None:
            dt_ms = (now - t0) * 1000.0
        elif stepped and self.last_tick:
            dt_ms = (now - self.last_tick) * 1000.0  # asynchronous: wall-clock step interval
        else:
            dt_ms = None
        if stepped and self.session is not None:
            self.last_tick = now
        if dt_ms is not None:
            self.frame_time_ms = 0.9 * self.frame_time_ms + 0.1 * dt_ms if self.step_count > 0 else dt_ms
            self.fps_avg = 1000.0 / max(self.frame_time_ms, 0.001)
        if stepped:
            self.step_count += 1
            self.step_once = False  # a single-step request survives a backpressure skip
        if getattr(solver, "emission_exhausted", False) and solver.take_exhaustion_notice():
            report({'WARNING'}, "WarpSPH: particle budget reached; emitters stopped (raise Max Particles)")

        # GL draws synchronously after every step; a session redraws only for a new frame
        if self.session is None or self.session.take_new_frame():
            _redraw_views(screen)
        return True

    # -- bake (P2.2) ---------------------------------------------------------------------------------

    def begin_bake(self, context, report: Report, *, first: int, last: int, resume: bool) -> bool:
        """Bake frames ``first`` ... ``last`` of the running scene into its cache folder, driven by ``tick``.

        Fresh bakes start from the scene's initial state; resume continues after the last intact cached frame.
        Returns False after reporting why nothing started (or that everything is already cached)."""
        from .cache import bake as cache_bake
        from .cache import blender_adapter
        from .cache import disk as cache_disk
        from .cache.frame_store import FrameStore
        from .cache.gpu import SolverSource, VramRing

        if not self.running or self.solver is None or runtime.wp is None:
            report({'ERROR'}, "Start the simulation before baking")
            return False
        if self.bake is not None:
            report({'ERROR'}, "A bake is already running")
            return False
        scene = self.scene
        settings = scene.warp_sph
        fps = scene.render.fps / scene.render.fps_base
        try:
            if not resume and self.step_count > 0:  # a fresh bake starts from the initial state
                self.paused = False
                if settings.scene_source == 'CUSTOM':
                    self.reallocate(scene, 0, CUSTOM_SCENE, context.window,
                                    depsgraph=context.evaluated_depsgraph_get())
                else:
                    self.reallocate(scene, settings.particle_count, settings.demo_scene, context.window)
            solver = self.solver
            apply_physics(solver, settings)
            solver.rebuild_cuda_graph()
            steps = cache_bake.steps_per_frame(settings.bake_steps, solver.dt, fps)
            signature = {
                "particles": solver.num_particles, "steps_per_frame": steps, "fps": fps, "dt": solver.dt,
                "h": solver.h, "stiffness": solver.stiffness, "viscosity": solver.viscosity,
                "scene": scene_name(scene), "demo": self.active_demo,
            }
            unit = float(getattr(solver, "unit_scale", 1.0))
            if unit != 1.0:  # h is in metres; meshing needs scene units. Absent at 1 so older caches still resume
                signature["unit_scale"] = unit
            disk = cache_disk.DiskCache(blender_adapter.cache_dir(scene))
            remaining = max(last - first + 1 - (disk.cached_count() if resume else 0), 0)
            free = cache_disk.free_bytes(disk.directory)
            need = cache_disk.estimate_bytes(solver.num_particles, remaining)
            if free is not None and need > free:
                report({'ERROR'}, f"Not enough disk space in {disk.directory}: the bake needs about "
                                  f"{need / 2**30:.1f} GB, {free / 2**30:.1f} GB free")
                return False
            # With emitters a frame holds the active particles (up to the pool): restore_state checks the count.
            particles = None if getattr(solver, "has_emitters", False) else solver.num_particles
            plan = cache_bake.plan_bake(disk, signature, first, last, resume=resume, particles=particles)
            if plan.complete:
                report({'INFO'}, f"Frames {first}-{last} are already cached")
                return False
            ring = VramRing(runtime.wp, solver.num_particles, BAKE_RING_SLOTS)  # the pool: emitters never overflow it
            store = FrameStore(disk, ring)
            before_step = None
            if self.active_demo == CUSTOM_SCENE:  # animated colliders: one transform read per frame, as in tick
                def before_step(scene=scene, solver=solver):
                    _scene_api().sync_collider_keyframes(scene, solver)
            try:
                self.bake = cache_bake.BakeJob(
                    store, SolverSource(runtime.wp, solver, steps, settings.use_cuda_graph,
                                        before_step=before_step, first_frame=first), plan)
            except BaseException:
                store.close()
                raise
        except FatalTransportError as exc:
            self.mark_fatal(str(exc))
            report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
            return False
        except Exception as exc:  # CacheError, CacheMismatchError, ValueError (resume mismatch), solver errors
            report({'ERROR'}, f"WarpSPH cannot bake: {exc}")
            return False
        self.bake_result = ""
        self.paused = False
        self.step_once = False
        return True

    def cancel_bake(self):
        """Stop capturing; frames already queued are still written, so the cache stays a whole prefix."""
        if self.bake is not None:
            self.bake.cancel()

    def _publish_solver(self):
        """Show the solver's latest state in the viewport (the bake bypasses the normal step path).

        ``copy_positions`` writes scene units, as ``step()`` does (the solver works in metres)."""
        wp, solver = runtime.wp, self.solver
        if self.session is not None:
            self.session.tick(solver.copy_positions)
        elif wp.get_stream("cuda:0").is_complete:  # the legacy VBO route waits for the GPU: only when idle
            with self.gl_buf.mapped() as pos_vbo_wp:
                solver.copy_positions(pos_vbo_wp)
            wp.synchronize_device("cuda:0")

    def _tick_bake(self, context, report: Report, screen) -> bool:
        bake = self.bake
        done_before = bake.frames_done
        try:
            if bake.run_chunk(BAKE_CHUNK_S):
                self._publish_solver()
        except FatalTransportError as exc:
            self.mark_fatal(str(exc))
            report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
            self.stop(context)
            return False
        except Exception as exc:  # the solver step or a Warp copy failed
            if self.session is not None:  # as a failed step (session.tick): CUDA state unknown
                self.mark_fatal(f"bake step failed; CUDA state unknown: {exc}")
                report({'ERROR'}, f"WarpSPH GPU transport failed; restart Blender: {exc}")
            else:
                report({'ERROR'}, f"WarpSPH bake stopped: {exc}")
            self.stop(context)
            return False
        finished = not bake.active
        if finished:
            self._finish_bake(report)
        if finished or bake.frames_done != done_before:
            _redraw_views(screen)
        elif self.session is not None and self.session.take_new_frame():
            _redraw_views(screen)
        return True

    def _release_bake(self, bake):
        """Drain the job's writer and free its ring (kept alive, never freed, after a fatal CUDA error)."""
        bake.store.close()
        if self.restart_required:
            _QUARANTINE.append(bake)

    def _finish_bake(self, report: Report):
        from .cache import bake as cache_bake
        bake, self.bake = self.bake, None
        self._release_bake(bake)
        span = f"frames {bake.plan.first}-{bake.plan.first + bake.frames_done - 1}"
        if bake.state == cache_bake.DONE:
            self.bake_result = f"Baked {span} ({bake.frames_done} frames)"
            report({'INFO'}, "WarpSPH " + self.bake_result)
        elif bake.state == cache_bake.CANCELLED:
            self.bake_result = f"Bake cancelled: {bake.frames_done} frames cached ({span}); Resume continues"
            report({'WARNING'}, "WarpSPH " + self.bake_result)
        else:
            self.bake_result = f"Bake failed: {bake.error} ({bake.frames_done} frames cached)"
            report({'ERROR'}, "WarpSPH " + self.bake_result)
        if self.solver is not None:
            self.solver.substeps = self.scene.warp_sph.substeps
        self.paused = True  # keep the final state on screen; Resume continues the live view

    def _discard_bake(self):
        """Teardown path (stop, file load, unregister): leave the cache as a whole prefix and free the ring."""
        bake, self.bake = self.bake, None
        if bake is not None:
            bake.cancel()
            self._release_bake(bake)
            self.bake_result = f"Bake stopped: {bake.frames_done} frames cached"

    def stop(self, context=None):
        """Idempotent teardown for stop, cancel, unregister, file load and viewport loss."""
        context = context or getattr(bpy, "context", None)
        self.running = False
        self.paused = False
        self.step_once = False
        self.token += 1
        self._discard_bake()
        if self.timer is not None:
            wm = getattr(context, "window_manager", None)
            if wm is not None:
                try:
                    wm.event_timer_remove(self.timer)
                except (ReferenceError, RuntimeError, ValueError):
                    pass  # already freed with its window
            self.timer = None
        if self.draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handle, 'WINDOW')
            self.draw_handle = None
        session, solver = self.session, self.solver
        self.session = None
        self.solver = None
        if session is not None:
            if self.restart_required:
                _QUARANTINE.append(session)  # no speculative CUDA cleanup after a fatal error
            else:
                self._retire(session)
        if self.restart_required and solver is not None:
            _QUARANTINE.append(solver)  # dropping it would free CUDA allocations
        if self.gl_buf:
            if runtime.cleanup_viewport_renderer is not None:
                runtime.cleanup_viewport_renderer()
            self.gl_buf.destroy()
            self.gl_buf = None
        _redraw_views(getattr(context, "screen", None))

    def dispose(self, context=None):
        """Stop and forget this session; get() then hands its scene a fresh one."""
        self.stop(context)
        for key, session in list(_SESSIONS.items()):
            if session is self:
                del _SESSIONS[key]
        self.scene = None


# -- per-scene registry ------------------------------------------------------------------------------

_SESSIONS: dict[int, SimSession] = {}


def _scene_key(scene) -> int:
    as_pointer = getattr(scene, "as_pointer", None)
    return as_pointer() if as_pointer is not None else id(scene)


def _alive(scene) -> bool:
    try:
        getattr(scene, "name", None)  # ReferenceError once Blender freed it (file load, scene delete)
    except ReferenceError:
        return False
    return scene is not None


def _is_scene(stored, scene) -> bool:
    """The stored scene is still alive and is ``scene`` (a freed scene's address can be reused)."""
    return _alive(stored) and stored == scene


def scene_name(scene) -> str:
    try:
        return str(getattr(scene, "name", "") or "")
    except ReferenceError:
        return ""


def get(scene) -> SimSession:
    """The session of ``scene``, created on first use."""
    key = _scene_key(scene)
    session = _SESSIONS.get(key)
    if session is None or (not session.running and session.solver is None and not _is_scene(session.scene, scene)):
        for stale_key, stale in list(_SESSIONS.items()):  # idle sessions of freed scenes (e.g. after a file load)
            if not stale.running and stale.solver is None and not _alive(stale.scene):
                del _SESSIONS[stale_key]
        session = _SESSIONS[key] = SimSession(scene)
    return session


def active() -> SimSession | None:
    """The running session, if any (at most one per process)."""
    return next((s for s in _SESSIONS.values() if s.running), None)


def sessions() -> list[SimSession]:
    return list(_SESSIONS.values())


def stop_all(context=None):
    for session in sessions():
        session.stop(context)


def dispose_all(context=None):
    for session in sessions():
        session.dispose(context)
