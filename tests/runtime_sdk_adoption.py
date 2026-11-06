"""Actual installed WarpSPH modal solver -> shared SDK -> native GN qualification.

Only correctness snapshots use CPU readback. Timing samples prior to pause do not.
The existing Start/Pause/Step/Reset/Snapshot operators and SimSession are used.
"""

import hashlib
import importlib
import json
import os
import time
import traceback
from pathlib import Path

import bpy
import gpu
import numpy as np
import warp as wp

MODULE = "bl_ext.user_default.warp_sph_realtime"
_probe = None


def read_attribute(mesh, name, width, dtype=np.float32):
    attr = mesh.attributes.get(name)
    assert attr is not None and attr.domain == "POINT", name
    count = len(mesh.vertices)
    values = np.empty(count * width, dtype=dtype)
    field = "value" if width == 1 else ("color" if width == 4 else "vector")
    attr.data.foreach_get(field, values)
    return values if width == 1 else values.reshape(count, width)


def native_ids(mesh):
    lo = read_attribute(mesh, "particle_id_low", 1, np.int32).view(np.uint32).astype(np.uint64)
    hi = read_attribute(mesh, "particle_id_high", 1, np.int32).view(np.uint32).astype(np.uint64)
    return lo | (hi << np.uint64(32))


class AdoptionProbe:
    def __init__(self, output, count=100000, steps=24):
        self.output = Path(output)
        assert self.output.is_absolute() and not self.output.exists()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        assert gpu.platform.backend_type_get() == "VULKAN" and bpy.app.build_hash.decode() == "fbe6228777e7"
        assert wp.__version__ == "1.17.0"
        self.smod = importlib.import_module(MODULE + ".sim_session")
        self.steps = steps
        self.count = count
        self.started = time.perf_counter()
        self.phase = "advance"
        self.samples = []
        self.expected = None
        self.old_epoch = None
        self.old_source = None
        self.reset_source = None
        self.report = {
            "status": "running",
            "pid": os.getpid(),
            "backend": "VULKAN",
            "build": bpy.app.build_hash.decode(),
            "warp": wp.__version__,
            "consumer_module": MODULE,
            "count": count,
            "target_steps": steps,
            "scope": "actual WarpSPH solver/modal output and explicit native snapshots; not full ADOPT-01/performance certification",
            "checks": [],
        }

    def save(self):
        temp = self.output.with_suffix(".tmp")
        temp.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        temp.replace(self.output)

    def check(self, name, value):
        self.report["checks"].append({"name": name, "passed": bool(value)})
        assert value, name

    def begin(self):
        assert self.smod.active() is None
        self.original = bpy.context.window.scene
        self.scene = bpy.data.scenes.new("WarpSPH SDK Real Solver")
        bpy.context.window.scene = self.scene
        settings = self.scene.warp_sph
        settings.use_shared_sdk = True
        settings.particle_count = self.count
        settings.scene_source = "DEMO"
        settings.render_mode = "SPHERES"
        settings.sdk_snapshot_name = "WarpSPH SDK Snapshot " + str(self.scene.session_uid)
        settings.show_particles = True
        settings.use_cuda_graph = True
        result = bpy.ops.wm.warp_sph_start()
        self.session = self.smod.get(self.scene)
        self.check("actual_start_operator", result == {"RUNNING_MODAL"} and self.session.running)
        self.check("shared_sdk_is_actual_render_session", getattr(self.session.session, "is_shared_sdk", False))
        self.report["consumer_origin"] = importlib.import_module(MODULE).__file__
        self.report["solver_class"] = type(self.session.solver).__module__ + "." + type(self.session.solver).__name__
        self.report["solver_capacity"] = self.session.solver.num_particles
        self.save()
        bpy.app.timers.register(self.tick, first_interval=0.02)

    def solver_data(self):
        solver = self.session.solver
        wp.synchronize_device("cuda:0")
        count = solver.active_count
        return {
            "position": solver.pos_sim.numpy()[:count] / solver.unit_scale,
            "velocity": solver.vel.numpy()[:count],
            "density": solver.density_sorted.numpy()[:count],
            "id": solver.pid.numpy()[:count].astype(np.uint64),
        }

    def compare_snapshot(self):
        obj = bpy.data.objects.get(self.scene.warp_sph.sdk_snapshot_name)
        assert obj is not None
        mesh = obj.data
        expected = self.expected
        count = len(expected["id"])
        self.check("snapshot_logical_count", len(mesh.vertices) == count)
        coords = np.empty(count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        np.testing.assert_allclose(coords.reshape(count, 3), expected["position"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(native_ids(mesh), expected["id"])
        np.testing.assert_array_equal(
            read_attribute(mesh, "velocity", 3), expected["velocity"] / self.session.solver.unit_scale
        )
        np.testing.assert_array_equal(read_attribute(mesh, "density", 1), expected["density"])
        np.testing.assert_allclose(read_attribute(mesh, "radius", 1), self.scene.warp_sph.particle_radius)
        np.testing.assert_allclose(
            read_attribute(mesh, "color", 4), np.tile(self.scene.warp_sph.particle_color, (count, 1)), rtol=1e-6
        )
        self.check("all_native_attributes_match_actual_solver", True)
        after = self.solver_data()
        for key in expected:
            np.testing.assert_array_equal(after[key], expected[key])
        self.check("snapshot_did_not_modify_solver", True)
        from blender_vulkan_interop.output.geometry_nodes import set_particle_id_filter

        for index in sorted(set([0, count // 2, count - 1])):
            identity = int(expected["id"][index])
            set_particle_id_filter(obj, identity)
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            evaluated = obj.evaluated_get(dg)
            realized = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
            try:
                self.check(
                    "GN_exact_solver_ID_" + str(identity),
                    len(realized.vertices) == 12 and np.all(native_ids(realized) == identity),
                )
                points = np.empty(36, dtype=np.float32)
                realized.vertices.foreach_get("co", points)
                points = points.reshape(12, 3)
                np.testing.assert_allclose(points.mean(axis=0), expected["position"][index], rtol=0, atol=2e-5)
                np.testing.assert_allclose(
                    np.linalg.norm(points - expected["position"][index], axis=1),
                    self.scene.warp_sph.particle_radius,
                    rtol=2e-4,
                    atol=2e-5,
                )
            finally:
                evaluated.to_mesh_clear()
        set_particle_id_filter(obj, None)
        bpy.context.view_layer.update()
        self.report["native_object"] = obj.name
        self.report["node_group"] = obj.modifiers["Interop Particle Instances"].node_group.name

    def tick(self):
        try:
            if time.perf_counter() - self.started > 300:
                raise TimeoutError("actual solver scenario timeout")
            if self.session.restart_required:
                raise RuntimeError(self.session.restart_required)
            if self.phase != "stopping" and not self.session.running:
                raise RuntimeError("solver stopped: " + self.session.last_error)
            self.samples.append((time.perf_counter(), self.session.step_count))
            bridge = self.session.session
            if self.phase == "advance" and self.session.step_count >= self.steps:
                self.report["sampled_step_progress"] = self.samples[-100:]
                self.report["solver_steps_before_pause"] = self.session.step_count
                self.report["sim_time_before_pause"] = self.session.solver.sim_time
                self.report["transport_metrics_before_readback"] = dict(bridge.metrics)
                self.check("actual_pause_operator", bpy.ops.wm.warp_sph_pause() == {"FINISHED"} and self.session.paused)
                self.phase = "wait_source"
            if self.phase == "wait_source" and bridge._receipt.source_released():
                self.expected = self.solver_data()
                self.check("finite_real_solver_output", all(np.isfinite(v).all() for v in self.expected.values()))
                self.check("solver_IDs_unique", len(np.unique(self.expected["id"])) == self.count)
                self.report["solver_array_sha256"] = {
                    key: hashlib.sha256(values.tobytes()).hexdigest() for key, values in self.expected.items()
                }
                self.check("snapshot_operator", bpy.ops.wm.warp_sph_sdk_snapshot() == {"FINISHED"})
                self.phase = "snapshot"
            if self.phase == "snapshot":
                if bridge.snapshot_status == "FAILED":
                    raise RuntimeError(bridge.snapshot_message)
                if bridge.snapshot_status == "COMPLETED":
                    self.compare_snapshot()
                    self.old_epoch = bridge._epoch
                    self.old_source = bridge._source_id
                    self.previous_bridge = bridge
                    self.check("actual_reset_operator", bpy.ops.wm.warp_sph_reset() == {"FINISHED"})
                    self.phase = "reset_prime"
            if self.phase == "reset_prime" and not self.session._output_pending:
                bridge = self.session.session
                self.check(
                    "reset_rebound_real_solver", bridge.solver is self.session.solver and bridge is self.previous_bridge
                )
                self.check(
                    "reset_new_identity_epoch",
                    bridge._epoch == self.old_epoch + 1 and bridge._source_id == self.old_source,
                )
                self.check("reset_remains_paused", self.session.paused)
                self.reset_source = self.session.solver
                self.check("actual_single_step_operator", bpy.ops.wm.warp_sph_step_frame() == {"FINISHED"})
                self.phase = "single_step"
            if (
                self.phase == "single_step"
                and self.session.step_count >= 1
                and not self.session.step_once
                and bridge._receipt.source_released()
            ):
                self.check("exactly_one_step_after_reset", self.session.step_count == 1 and self.session.paused)
                self.report["after_reset_epoch"] = bridge._epoch
                self.report["after_reset_sim_time"] = self.session.solver.sim_time
                self.report["final_transport_metrics"] = dict(bridge.metrics)
                self.final_bridge = bridge
                self.check("actual_stop_operator", bpy.ops.wm.warp_sph_stop() == {"FINISHED"})
                self.phase = "stopping"
            if self.phase == "stopping" and self.session.timer is None:
                bridge = self.final_bridge
                self.check(
                    "native_transport_closed",
                    bridge.close_result == "completed" and self.session.last_close == "completed",
                )
                self.check(
                    "runtime_callbacks_removed",
                    self.session.timer is None and self.session.draw_handle is None and self.session.solver is None,
                )
                self.report["status"] = "passed"
                self.report["elapsed_seconds"] = time.perf_counter() - self.started
                self.report["saved_snapshot_is_pre_reset_frame"] = True
                path = self.output.parent / "WarpSPH-Real-Solver-SDK-Snapshot.blend"
                assert bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True) == {"FINISHED"}
                self.report["saved_blend"] = str(path)
                self.save()
                return None
            self.save()
            return 0.02
        except Exception:
            self.report["status"] = "failed"
            self.report["error"] = traceback.format_exc()
            if getattr(self, "session", None) is not None:
                self.session.stop(bpy.context)
                self.report["failure_close"] = self.session.last_close
            self.save()
            return None


def start(output, count=100000, steps=24):
    global _probe
    if _probe is not None:
        raise RuntimeError("probe already started")
    _probe = AdoptionProbe(output, count, steps)
    _probe.begin()
    return _probe
