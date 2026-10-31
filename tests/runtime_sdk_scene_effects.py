"""Real authored WarpSPH collider/emitter A/B through existing Reset/Step operators.

Small diagnostic scenes deliberately read positions/IDs after completion to prove
physics and identity effects. This script is not a steady-state performance test.
"""

import importlib
import json
import os
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
import warp as wp

MODULE = "bl_ext.user_default.warp_sph_realtime"
_probe = None
PLATE_LO = np.array([-0.5, -0.5, 0.3])
PLATE_HI = np.array([0.5, 0.5, 0.5])


def distance(points):
    q = np.abs(points - (PLATE_LO + PLATE_HI) * 0.5) - (PLATE_HI - PLATE_LO) * 0.5
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def box(scene, name, lo, hi, role):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.warp_sph.role = role
    obj.display_type = "WIRE"
    return obj


class Effects:
    def __init__(self, output):
        self.output = Path(output)
        assert self.output.is_absolute() and not self.output.exists()
        self.smod = importlib.import_module(MODULE + ".sim_session")
        assert self.smod.active() is None
        self.started = time.monotonic()
        self.phase = "collider_present"
        self.requested = 0
        self.previous = None
        self.report = {
            "status": "running",
            "pid": os.getpid(),
            "scope": "actual authored collider and emitter influence using unchanged WarpSPH physics; diagnostic readbacks only",
            "cases": {},
            "checks": [],
        }
        self.case_stats = {}
        self.scene = None
        self.session = None

    def check(self, name, condition):
        self.report["checks"].append({"name": name, "passed": bool(condition)})
        assert condition, name

    def save(self):
        self.output.write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def scene_base(self, name):
        scene = bpy.data.scenes.new(name)
        bpy.context.window.scene = scene
        scene.unit_settings.scale_length = 1.0
        scene.gravity = (0.0, 0.0, -9.81)
        scene.use_gravity = True
        s = scene.warp_sph
        s.use_shared_sdk = True
        s.scene_source = "CUSTOM"
        s.particle_spacing = 0.04
        s.max_particles = 1200
        s.use_cuda_graph = True
        s.render_mode = "POINTS"
        s.substeps = 2
        s.show_particles = True
        s.sdk_snapshot_name = name + " Snapshot"
        return scene

    def start_paused(self):
        self.check(self.phase + "_start", bpy.ops.wm.warp_sph_start() == {"RUNNING_MODAL"})
        self.session = self.smod.get(self.scene)
        self.check(self.phase + "_pause", bpy.ops.wm.warp_sph_pause() == {"FINISHED"})
        self.requested = 0
        self.previous = None
        self.case_stats = {
            "initial_count": self.session.solver.active_count,
            "capacity": self.session.solver.num_particles,
            "min_plate_distance": None,
            "max_survivor_step_distance": 0.0,
            "count_history": [],
            "spatial_reordering_observed": False,
        }

    def begin(self):
        self.scene = self.scene_base("SDK Collider A-B")
        box(self.scene, "SDK Domain", (-1.0, -1.0, 0.0), (1.0, 1.0, 2.0), "DOMAIN")
        box(self.scene, "SDK Fluid", (-0.12, -0.12, 0.92), (0.12, 0.12, 1.24), "FLUID")
        self.plate = box(self.scene, "SDK Plate", PLATE_LO, PLATE_HI, "COLLIDER")
        bpy.context.view_layer.update()
        self.start_paused()
        self.save()
        bpy.app.timers.register(self.tick, first_interval=0.002)

    def capture(self):
        solver = self.session.solver
        wp.synchronize_device("cuda:0")
        n = solver.active_count
        ids = solver.pid.numpy()[:n]
        positions = solver.pos_sim.numpy()[:n]
        assert np.isfinite(positions).all() and len(np.unique(ids)) == n
        self.case_stats["count_history"].append(n)
        self.case_stats["spatial_reordering_observed"] |= not np.array_equal(ids, np.arange(n))
        if self.phase.startswith("collider"):
            d = float(distance(positions).min())
            old = self.case_stats["min_plate_distance"]
            self.case_stats["min_plate_distance"] = d if old is None else min(old, d)
        else:
            assert sorted(ids.tolist()) == list(range(n)), "solver emitter ID lineage is not exact"
            if self.previous is not None:
                old = self.previous
                matched = [(int(i), p) for i, p in zip(ids, positions, strict=True) if int(i) in old]
                movement = max((float(np.linalg.norm(p - old[i])) for i, p in matched), default=0.0)
                self.case_stats["max_survivor_step_distance"] = max(
                    self.case_stats["max_survivor_step_distance"], movement
                )
                new = positions[[int(i) not in old for i in ids]]
                if len(new):
                    assert (new[:, 2] > 0.8).all(), "new IDs not at authored emitter"
            self.previous = {int(i): p.copy() for i, p in zip(ids, positions, strict=True)}
        self.final_ids = ids.astype(np.uint64)
        self.final_position = positions.copy()

    def finish_case(self):
        self.case_stats.update(
            steps=self.session.step_count,
            sim_time=self.session.solver.sim_time,
            final_count=self.session.solver.active_count,
            scene_revision=self.session.session._scene_revision,
            epoch=self.session.session._epoch,
        )
        self.report["cases"][self.phase] = self.case_stats

    def reset_case(self, new_phase):
        old_epoch = self.session.session._epoch
        self.phase = new_phase
        self.check(new_phase + "_reset", bpy.ops.wm.warp_sph_reset() == {"FINISHED"})
        self.check(new_phase + "_new_epoch", self.session.session._epoch == old_epoch + 1)
        self.requested = 0
        self.previous = None
        self.case_stats = {
            "initial_count": self.session.solver.active_count,
            "capacity": self.session.solver.num_particles,
            "min_plate_distance": None,
            "max_survivor_step_distance": 0.0,
            "count_history": [],
            "spatial_reordering_observed": False,
        }

    def tick(self):
        try:
            if time.monotonic() - self.started > 300:
                raise TimeoutError("authored A/B timeout")
            if self.session.restart_required:
                raise RuntimeError(self.session.restart_required)
            if self.phase == "stopping_collider":
                if self.session.timer is not None:
                    return 0.002
                self.check("collider_sdk_close", self.session.last_close == "completed")
                self.phase = "emitter_low"
                self.scene = self.scene_base("SDK Emitter A-B")
                box(self.scene, "Emitter Domain", (-0.5, -0.5, 0.0), (0.5, 0.5, 1.2), "DOMAIN")
                box(self.scene, "Resting Fluid", (-0.2, -0.2, 0.0), (0.2, 0.2, 0.12), "FLUID")
                mesh = bpy.data.meshes.new("SDK Nozzle")
                mesh.from_pydata(
                    [(-0.08, -0.08, 1.0), (0.08, -0.08, 1.0), (0.08, 0.08, 1.0), (-0.08, 0.08, 1.0)], [], [(0, 3, 2, 1)]
                )
                mesh.update()
                self.tap = bpy.data.objects.new("SDK Emitter", mesh)
                self.scene.collection.objects.link(self.tap)
                self.tap.warp_sph.role = "EMITTER"
                self.tap.warp_sph.emit_speed = 1.5
                self.tap.warp_sph.emit_rate = 80.0
                bpy.context.view_layer.update()
                self.start_paused()
                return 0.002
            if self.phase == "stopping_final":
                if self.session.timer is not None:
                    return 0.002
                self.check("emitter_sdk_close", self.session.last_close == "completed")
                self.report["status"] = "passed"
                self.report["elapsed_seconds"] = time.monotonic() - self.started
                self.save()
                return None
            if not self.session.running:
                raise RuntimeError("actual session stopped: " + self.session.last_error)
            bridge = self.session.session
            if self.phase == "emitter_snapshot":
                if bridge.snapshot_status == "FAILED":
                    raise RuntimeError(bridge.snapshot_message)
                if bridge.snapshot_status == "COMPLETED":
                    from blender_vulkan_interop.output.particles import ids_from_words

                    obj = bpy.data.objects[self.scene.warp_sph.sdk_snapshot_name]
                    mesh = obj.data
                    count = len(mesh.vertices)
                    lo = np.empty(count, dtype=np.int32)
                    hi = np.empty(count, dtype=np.int32)
                    mesh.attributes["particle_id_low"].data.foreach_get("value", lo)
                    mesh.attributes["particle_id_high"].data.foreach_get("value", hi)
                    np.testing.assert_array_equal(ids_from_words(lo, hi), self.final_ids)
                    self.check(
                        "emitted_native_snapshot_uses_logical_count",
                        count == len(self.final_ids) < self.session.solver.num_particles,
                    )
                    self.check("emitted_stable_IDs_match_real_solver", True)
                    self.report["native_emitter_object"] = obj.name
                    self.check("emitter_stop_operator", bpy.ops.wm.warp_sph_stop() == {"FINISHED"})
                    self.phase = "stopping_final"
                self.save()
                return 0.002
            if (
                self.session._output_pending
                or self.session.step_once
                or self.session.step_count != self.requested
                or not bridge._receipt.source_released()
            ):
                return 0.002
            self.capture()
            target = 300 if self.phase.startswith("collider") else 40
            if self.requested < target:
                self.check(
                    self.phase + "_single_step_" + str(self.requested + 1),
                    bpy.ops.wm.warp_sph_step_frame() == {"FINISHED"},
                )
                self.requested += 1
                return 0.002
            self.finish_case()
            if self.phase == "collider_present":
                self.check("actual_contact_reached", self.case_stats["min_plate_distance"] < 0.04)
                self.check("collider_prevents_penetration", self.case_stats["min_plate_distance"] >= -1e-4)
                self.plate.warp_sph.role = "NONE"
                self.plate.update_tag()
                bpy.context.view_layer.update()
                self.reset_case("collider_absent")
            elif self.phase == "collider_absent":
                self.check("removed_collider_changes_physical_path", self.case_stats["min_plate_distance"] < -0.02)
                a = self.report["cases"]["collider_present"]
                b = self.report["cases"]["collider_absent"]
                self.check(
                    "A_B_same_time_and_population",
                    abs(a["sim_time"] - b["sim_time"]) < 1e-12 and a["final_count"] == b["final_count"],
                )
                self.check("collider_stop_operator", bpy.ops.wm.warp_sph_stop() == {"FINISHED"})
                self.phase = "stopping_collider"
            elif self.phase == "emitter_low":
                self.check(
                    "authored_emitter_adds_particles", self.case_stats["final_count"] > self.case_stats["initial_count"]
                )
                self.tap.warp_sph.emit_rate = 160.0
                self.tap.update_tag()
                bpy.context.view_layer.update()
                self.reset_case("emitter_high")
            elif self.phase == "emitter_high":
                a = self.report["cases"]["emitter_low"]
                b = self.report["cases"]["emitter_high"]
                self.check(
                    "higher_authored_rate_changes_particle_count",
                    b["final_count"] > a["final_count"] > a["initial_count"],
                )
                self.check("emitter_A_B_same_elapsed_time", abs(a["sim_time"] - b["sim_time"]) < 1e-12)
                self.check(
                    "stable_IDs_survive_real_spatial_sort",
                    a["spatial_reordering_observed"] and b["spatial_reordering_observed"],
                )
                self.check(
                    "survivor_ID_motion_is_local",
                    max(a["max_survivor_step_distance"], b["max_survivor_step_distance"]) < 0.04,
                )
                self.check("actual_emitter_snapshot_operator", bpy.ops.wm.warp_sph_sdk_snapshot() == {"FINISHED"})
                self.phase = "emitter_snapshot"
            self.save()
            return 0.002
        except Exception:
            self.report.update(status="failed", phase=self.phase, error=traceback.format_exc())
            if self.session:
                self.session.stop(bpy.context)
                self.report["failure_close"] = self.session.last_close
            self.save()
            return None


def start(output):
    global _probe
    if _probe is not None:
        raise RuntimeError("probe already used")
    _probe = Effects(output)
    _probe.begin()
    return _probe
