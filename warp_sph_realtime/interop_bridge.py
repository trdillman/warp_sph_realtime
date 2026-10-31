"""Opt-in shared SDK rendering of the real WarpSPH solver and native snapshots.

Imports no bpy/Warp/SDK at module load. Existing simulation and cache ownership
remain with SimSession. The private candidate is exact-byte pinned and never an
implicit fallback. Native snapshot readback is explicit and outside normal ticks.
"""

from __future__ import annotations

import time
import uuid

from .render.transport import COMPLETED, FAILED, Capabilities, FatalTransportError, Frame, process_state


class InteropBridge:
    """Render-session adapter retaining the actual solver's sorted ID lineage."""

    is_shared_sdk = True

    def __init__(self, solver, scene, window, *, _bindings=None):
        self.solver, self.scene, self.window = solver, scene, window
        self.count = solver.num_particles
        self.backend = "VULKAN"
        self.reason = "Pinned experimental shared SDK; no hidden fallback"
        self.transport = self  # Existing render-session status/capabilities interface.
        self.positions = None
        self.waits = self.drops = 0
        self.close_result = ""
        self._bindings = _bindings
        self._owner = self._native = None
        self._arrays = {}
        self._receipt = self._stamp = None
        self._source_id = f"warpsph:{getattr(scene, 'session_uid', id(scene))}:{uuid.uuid4().hex}"
        self._epoch = self._frame = self._scene_revision = 0
        self._shown = -1
        self._started = False
        self._snapshot_request = self._snapshot_payload = self._snapshot_stamp = None
        self.snapshot_status = "IDLE"
        self.snapshot_message = ""
        self.snapshot_result = None
        self.metrics = {
            "published": 0,
            "solver_submit_s": 0.0,
            "pack_publish_s": 0.0,
            "draws": 0,
            "snapshot_readback_s": 0.0,
            "snapshot_commit_s": 0.0,
        }

    @property
    def label(self):
        return "Shared SDK (experimental GPU copy; native snapshot on request)"

    @property
    def fatal_reason(self):
        return self._bindings.fatal_reason() if self._bindings is not None else ""

    def capabilities(self):
        return Capabilities(
            "SHARED_SDK_V2",
            True,
            4194304,
            self._started and not self.close_result and not self.fatal_reason,
            self.fatal_reason,
        )

    def start(self):
        if self._started:
            raise RuntimeError("shared SDK bridge already started")
        if self.close_result:
            raise RuntimeError("closed bridge cannot be restarted")
        if self._bindings is None:
            self._bindings = _SDKBindings(self.scene, self.window)
        self._owner, self._native, self._arrays = self._bindings.create(self.count)
        self.positions = self._arrays["position"]
        self._started = True

    def _fatal(self, exc):
        reason = f"shared SDK output failed; GPU ownership uncertain: {exc}"
        self._bindings.poison(reason, self)
        return FatalTransportError(reason)

    def can_publish(self):
        if not self._started or self.close_result:
            return False
        if self.fatal_reason:
            raise FatalTransportError(self.fatal_reason)
        if self.snapshot_status in ("QUEUED", "CAPTURED"):
            return False
        return self._native.can_publish()

    def tick(self, produce):
        """The existing solver callback executes only after source/slot admission."""
        if not self.can_publish():
            self.waits += 1
            return False
        try:
            # Existing solver IDs are nonnegative int32, capped by its capacity.
            # Do not silently re-number particles or wrap a future overflow.
            if not 0 <= self.solver.active_count <= self.count or self.solver._next_id > 1 << 31:
                raise ValueError("solver population/ID range cannot be represented safely")
            begin = time.perf_counter()
            produce(self.positions)
            produced = time.perf_counter()
            channels = self._bindings.pack(self.solver, self._arrays, self.scene)
            frame = self._bindings.frame(
                self._source_id, self._epoch, self._frame + 1, self._scene_revision, self.solver, channels
            )
            receipt = self._native.publish(frame)
            if receipt.status != "ACCEPTED":
                raise RuntimeError(f"publication rejected after the real solver step: {receipt.status}")
            self._receipt, self._stamp = receipt, frame.stamp
            self._frame += 1
            self.metrics["published"] += 1
            self.metrics["solver_submit_s"] += produced - begin
            self.metrics["pack_publish_s"] += time.perf_counter() - produced
            return True
        except Exception as exc:
            raise self._fatal(exc) from exc

    def try_bind_solver(self, solver):
        """Reuse compatible output only after the previous producer is complete."""
        if self.close_result or solver.num_particles != self.count or self.snapshot_status in ("QUEUED", "CAPTURED"):
            return False
        if self._receipt is not None and not self._receipt.source_released():
            return False
        self.solver = solver
        self._epoch += 1
        self._frame = 0
        self._scene_revision += 1
        self._stamp = None
        self._shown = -1
        return True

    def note_scene_applied(self):
        """Called only after SimSession's existing input boundary accepted edits."""
        self._scene_revision += 1

    def apply_scene(self, delta):
        return self._bindings.apply_receipt(
            delta.snapshot.revision,
            "REBUILD_REQUIRED",
            "Use WarpSPH's existing authored-input and Reset workflow; arbitrary SDK scene deltas are not applied",
        )

    def take_new_frame(self):
        return self._stamp is not None and self._shown < self._stamp.frame_id

    def draw(self, settings):
        if not self._started or self.close_result or self._stamp is None:
            return False
        try:
            result = self._bindings.draw(self, settings)
            if result:
                self.metrics["draws"] += 1
            return result
        except Exception as exc:
            raise self._fatal(exc) from exc

    def request_snapshot(self, request):
        if not self._started or self.close_result or self._stamp is None:
            raise RuntimeError("no published solver frame is available")
        if self.snapshot_status in ("QUEUED", "CAPTURED"):
            raise RuntimeError("a snapshot request is already pending")
        self._snapshot_request, self._snapshot_stamp = request, self._stamp
        self._snapshot_payload = None
        self.snapshot_status, self.snapshot_message = "QUEUED", "Waiting for the exact displayed solver frame"

    def service_snapshot(self):
        """Timer-only native mutation after draw-context readback has completed."""
        if self.snapshot_status != "CAPTURED":
            return False
        start = time.perf_counter()
        try:
            self.snapshot_result = self._bindings.commit_snapshot(
                self._snapshot_payload, self._snapshot_request, self.scene
            )
            self.snapshot_status = "COMPLETED"
            self.snapshot_message = "Native snapshot and Geometry Nodes updated"
            return True
        except Exception as exc:
            self.snapshot_status, self.snapshot_message = "FAILED", str(exc)
            return False
        finally:
            self.metrics["snapshot_commit_s"] += time.perf_counter() - start
            self._snapshot_request = self._snapshot_payload = self._snapshot_stamp = None

    def close(self):
        if self.close_result:
            return self.close_result
        if self.snapshot_status in ("QUEUED", "CAPTURED"):
            self.snapshot_status, self.snapshot_message = "CANCELLED", "Stopped before snapshot commit"
        self._snapshot_request = self._snapshot_payload = self._snapshot_stamp = None
        if self._owner is None:
            self.close_result = COMPLETED
            return self.close_result
        try:
            status = self._owner.close()
        except Exception as exc:
            self._bindings.poison(f"shared SDK close raised: {exc}", self)
            status = FAILED
        self.close_result = status
        if status == COMPLETED:
            self._bindings.free_renderer()
            self.positions = None
            self._arrays.clear()
            self.solver = None
        else:
            self._bindings.poison(f"shared SDK close retained resources ({status}); restart Blender", self)
        return status


class _SDKBindings:
    def __init__(self, scene, window):
        from .sdk_dependency import validate_environment

        validate_environment()
        import bpy
        import warp as wp

        if wp.__version__ != "1.17.0" or bpy.context.window.as_pointer() != window:
            raise RuntimeError("shared SDK requires Warp1.17 and the current owner window")
        self.wp = wp
        self.scene, self.window = scene, window
        self.consumer = None
        self.region = None
        from .render.gpu_ssfr import ViewportRenderer

        self.renderer = ViewportRenderer()

    def create(self, capacity):
        from blender_vulkan_interop._impl.particle_frames import _create_candidate
        from blender_vulkan_interop.frames import ChannelSpec, FrameSchema, TransportConfig
        from blender_vulkan_interop.lifecycle import LifecycleOwner

        wp = self.wp
        owner = LifecycleOwner.for_current_window()
        try:
            schema = FrameSchema(
                2,
                (
                    ChannelSpec("position", "float32", 3, "position"),
                    ChannelSpec("velocity", "float32", 3, "velocity_m_s"),
                    ChannelSpec("density", "float32", 1, "density_kg_m3_last_solver_evaluation"),
                    ChannelSpec("radius", "float32", 1, "artist_radius"),
                    ChannelSpec("color", "float32", 4, "scene_linear_rgba"),
                    ChannelSpec("id", "uint64", 1, "particle_id"),
                ),
            )
            native = _create_candidate(TransportConfig(capacity, schema, slots=2), owner=owner.owner_token())
            arrays = {
                name: wp.empty(capacity, dtype=dtype, device="cuda:0")
                for name, dtype in (
                    ("position", wp.vec3),
                    ("id", wp.uint64),
                    ("radius", wp.float32),
                    ("color", wp.vec4),
                )
            }
            self.owner = owner
            return owner, native, arrays
        except Exception:
            if owner.close() != COMPLETED:
                self.poison("shared SDK activation did not close safely", owner)
            raise

    def pack(self, solver, arrays, scene):
        from .sdk_channels import pack_fields

        settings = scene.warp_sph
        pack_fields(
            solver.pid,
            arrays["id"],
            arrays["radius"],
            arrays["color"],
            solver.active_count,
            float(settings.particle_radius),
            tuple(settings.particle_color),
        )
        return {**arrays, "velocity": solver.vel, "density": solver.density_sorted}

    def frame(self, source, epoch, number, revision, solver, channels):
        from blender_vulkan_interop.frames import FrameInput, FrameStamp, SpatialMetadata

        stamp = FrameStamp(source, epoch, number, float(solver.sim_time), revision)
        return FrameInput(
            stamp, SpatialMetadata((0.0, 0.0, 0.0), float(solver.unit_scale)), solver.active_count, channels
        )

    @staticmethod
    def apply_receipt(revision, status, reason):
        from blender_vulkan_interop.adapters.scene_updates import SceneApplyReceipt

        return SceneApplyReceipt(revision, status, reason)

    @staticmethod
    def fatal_reason():
        from blender_vulkan_interop._impl.transport import process_state as sdk_state

        return process_state().restart_required or sdk_state().restart_required

    @staticmethod
    def poison(reason, owner):
        from blender_vulkan_interop._impl.transport import process_state as sdk_state

        for state in (process_state(), sdk_state()):
            state.restart_required = state.restart_required or reason
            if owner not in state.quarantine:
                state.quarantine.append(owner)

    def free_renderer(self):
        self.renderer.free()

    def draw(self, bridge, settings):
        import bpy

        if (
            bpy.context.window is None
            or bpy.context.window.as_pointer() != self.window
            or bpy.context.scene is not bridge.scene
        ):
            return False
        region = bpy.context.region.as_pointer()
        if self.consumer is None:
            self.consumer = self.owner.register_consumer_for_current_context()
            self.region = region
        if region != self.region:
            return False
        stamp = bridge._snapshot_stamp if bridge.snapshot_status == "QUEUED" else None
        try:
            lease = bridge._native.acquire_frame(self.consumer, stamp=stamp)
        except ValueError as exc:
            if str(exc) == "requested stamp is unavailable":
                return False
            raise
        if lease is None:
            return False
        with lease as view:
            binding = view.channels.get("position")
            frame = (
                None
                if binding is None
                else Frame(
                    binding.texture,
                    view.active_count,
                    binding.width,
                    binding.height,
                    view.stamp.epoch,
                    view.stamp.frame_id,
                )
            )
            drawn = self.renderer.draw(frame, settings) if bridge.scene.warp_sph.show_particles else False
            if view.stamp.epoch == bridge._epoch:
                bridge._shown = view.stamp.frame_id
        if bridge.snapshot_status == "QUEUED" and view.stamp == bridge._snapshot_stamp:
            from blender_vulkan_interop.output.snapshots import ParticleSnapshotSource

            snapshot_lease = ParticleSnapshotSource(bridge._native, self.consumer).pin(bridge._snapshot_stamp)
            start = time.perf_counter()
            try:
                bridge._snapshot_payload = snapshot_lease.readback(max_bytes=268435456)
                bridge.snapshot_status = "CAPTURED"
                bridge.snapshot_message = "GPU snapshot captured; awaiting native commit"
            except ValueError as exc:
                bridge.snapshot_status, bridge.snapshot_message = "FAILED", str(exc)
                bridge._snapshot_request = bridge._snapshot_stamp = None
            finally:
                snapshot_lease.close()
                bridge.metrics["snapshot_readback_s"] += time.perf_counter() - start
        return drawn

    @staticmethod
    def commit_snapshot(payload, request, scene):
        import bpy
        from blender_vulkan_interop.output.contracts import SnapshotResult, SnapshotTarget
        from blender_vulkan_interop.output.geometry_nodes import create_particle_instancer
        from blender_vulkan_interop.output.particles import ParticleConverter

        if bpy.context.scene is not scene:
            raise ValueError("snapshot scene changed before native commit")
        target_name = request["name"]
        collection_name = request["collection"]
        owner_key = scene.warp_sph.sdk_snapshot_owner
        if not owner_key:
            raise ValueError("snapshot ownership was not initialized by the explicit request")
        existing = bpy.data.objects.get(target_name)
        if existing is not None and existing.get("warpsph_sdk_scene") != owner_key:
            raise ValueError("snapshot target is not owned by this WarpSPH scene")
        target = SnapshotTarget(target_name, collection_name, replace_existing=existing is not None)
        converter = ParticleConverter()
        if converter.estimate(payload, target).native_bytes > target.max_native_bytes:
            raise ValueError("native snapshot exceeds memory budget")
        staged = converter.prepare(payload, target)
        group = obj = None
        try:
            refs = staged.commit()
            obj = bpy.data.objects[next(ref.name for ref in refs if ref.kind == "OBJECT")]
            if obj.modifiers.get("Interop Particle Instances") is None:
                group = create_particle_instancer(obj)
            obj["warpsph_sdk_scene"] = owner_key
            staged.finalize()
            return SnapshotResult(payload.stamp, refs, sum(len(a.data) for a in payload.arrays))
        except Exception:
            if group is not None and obj is not None:
                material_nodes = [node for node in group.nodes if node.bl_idname == "GeometryNodeSetMaterial"]
                materials = [node.inputs["Material"].default_value for node in material_nodes]
                for modifier in tuple(obj.modifiers):
                    if getattr(modifier, "node_group", None) is group:
                        obj.modifiers.remove(modifier)
                if group.users == 0:
                    bpy.data.node_groups.remove(group)
                for material in materials:
                    if material is not None and material.users == 0:
                        bpy.data.materials.remove(material)
            staged.rollback()
            raise


def open_session(solver, scene, window):
    bridge = InteropBridge(solver, scene, window)
    bridge.start()
    return bridge
