"""Installed actual-solver live GN, isolated animation and native render proof."""

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

_probe = None


def read_attr(mesh, name, components=1, integer=False):
    attr = mesh.attributes.get(name)
    assert attr is not None, (name, "missing")
    values = np.empty(len(mesh.vertices) * components, dtype=np.int32 if integer else np.float32)
    attr.data.foreach_get("value" if components == 1 else "color" if components == 4 else "vector", values)
    return values.reshape(-1, components) if components > 1 else values


def native_arrays(obj):
    mesh = obj.data
    pos = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", pos)
    lo = read_attr(mesh, "particle_id_low", integer=True).view(np.uint32).astype(np.uint64)
    hi = read_attr(mesh, "particle_id_high", integer=True).view(np.uint32).astype(np.uint64)
    return {
        "position": pos.reshape(-1, 3),
        "velocity": read_attr(mesh, "velocity", 3),
        "density": read_attr(mesh, "density"),
        "radius": read_attr(mesh, "radius"),
        "color": read_attr(mesh, "color", 4),
        "id": lo | (hi << np.uint64(32)),
    }


def hashes(arrays):
    return {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in arrays.items()}


def solver_fingerprint(session):
    import warp as wp

    wp.synchronize_device("cuda:0")
    solver = session.solver
    arrays = {
        name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        for name, value in vars(solver).items()
        if isinstance(value, wp.array)
    }
    return {
        "arrays": arrays,
        "time": float(solver.sim_time),
        "next_id": int(solver._next_id),
        "active": int(solver.active_count),
        "step_count": session.step_count,
        "paused": session.paused,
        "token": session.token,
        "frame": session.scene.frame_current,
        "range": [session.scene.frame_start, session.scene.frame_end],
        "solver_identity": id(solver),
        "bridge_identity": id(session.session),
        "emitter_state": repr(solver._emitters),
        "substeps": solver.substeps,
        "dt": solver.dt,
    }


def validate_mesh(obj, frame, scene):
    actual = native_arrays(obj)
    scale = scene.unit_settings.scale_length
    for name in actual:
        expected = frame.fields[name]
        if name in ("position", "velocity", "radius"):
            expected = (expected.astype(np.float64) / scale).astype(np.float32)
        np.testing.assert_array_equal(actual[name], expected)
    return hashes(actual)


def evaluate_ids(obj, expected_count):
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(dg)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
    try:
        assert len(mesh.vertices) == expected_count * 12, (len(mesh.vertices), expected_count)
        lo = read_attr(mesh, "particle_id_low", integer=True).view(np.uint32).astype(np.uint64)
        hi = read_attr(mesh, "particle_id_high", integer=True).view(np.uint32).astype(np.uint64)
        ids = lo | (hi << np.uint64(32))
        return {
            "vertices": len(mesh.vertices),
            "ids_sha": hashlib.sha256(ids.tobytes()).hexdigest(),
            "unique_ids": len(np.unique(ids)),
        }
    finally:
        evaluated.to_mesh_clear()


class Probe:
    def __init__(self, output, consumer):
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "live-animation.json"
        assert not self.path.exists()
        self.consumer = consumer
        self.sim = importlib.import_module(consumer + ".sim_session")
        self.sequence = importlib.import_module(consumer + ".cache.native_sequence")
        self.playback = importlib.import_module(consumer + ".cache.native_playback")
        self.stage = "LIVE"
        self.started = time.monotonic()
        self.seen = []
        self.report = {
            "status": "running",
            "pid": os.getpid(),
            "backend": gpu.platform.backend_type_get(),
            "checks": [],
            "live_samples": self.seen,
            "scope": "actual solver live native GN plus private-copy animation; correctness not realtime FPS",
        }

    def save(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def check(self, name, value=True, **details):
        assert value, name
        self.report["checks"].append({"name": name, "passed": True, **details})

    def begin(self):
        assert (
            gpu.platform.backend_type_get() == "VULKAN"
            and bpy.app.build_hash.decode() == "fbe6228777e7"
            and not bpy.app.background
        )
        assert self.sim.active() is None
        scene = bpy.data.scenes.new("Live GN and Animation Qualification")
        bpy.context.window.scene = scene
        self.scene = scene
        s = scene.warp_sph
        s.use_shared_sdk = True
        s.scene_source = "DEMO"
        s.demo_scene = "DOUBLE_DAM_PILLARS"
        s.particle_count = 100_000
        s.substeps = 2
        s.render_mode = "POINTS"
        s.particle_radius = 0.025
        s.particle_color = (0.05, 0.4, 0.95, 1.0)
        s.sdk_live_geometry = True
        s.sdk_live_rate = 20
        s.sdk_live_name = "Live GN Fluid"
        s.sdk_native_budget_mb = 1024
        s.sdk_animation_name = "Native Fluid Animation"
        s.bake_start = 1
        s.bake_end = 8
        s.bake_steps = 16
        s.sdk_animation_dir = str(self.out / "fluid-cache")
        scene.render.fps = 24
        assert bpy.ops.wm.warp_sph_start() == {"RUNNING_MODAL"}
        self.session = self.sim.get(scene)
        self.particle_count = int(self.session.solver.active_count)
        assert self.particle_count == s.particle_count
        self.report["particle_count"] = self.particle_count
        self.save()
        bpy.app.timers.register(self.tick, first_interval=0.02)

    def ready_live(self):
        obj = bpy.data.objects.get(self.scene.warp_sph.sdk_live_name)
        if obj is None:
            return None
        bridge = self.session.session
        return (
            obj
            if bridge._stamp is not None
            and int(obj.data.get("interop_frame", -1)) == bridge._stamp.frame_id
            and bridge.snapshot_status == "COMPLETED"
            else None
        )

    def tick(self):
        try:
            if time.monotonic() - self.started > 600:
                raise TimeoutError("live-animation native scenario exceeded10minutes")
            if self.session.restart_required:
                raise RuntimeError(self.session.restart_required)
            s = self.scene.warp_sph
            if self.stage == "LIVE":
                obj = bpy.data.objects.get(s.sdk_live_name)
                if obj:
                    stamp = int(obj.data.get("interop_frame", -1))
                    if stamp not in self.seen:
                        self.seen.append(stamp)
                if len(self.seen) >= 5:
                    assert bpy.ops.wm.warp_sph_pause() == {"FINISHED"}
                    self.stage = "PAUSED_LIVE"
            elif self.stage == "PAUSED_LIVE":
                obj = self.ready_live()
                if obj is not None:
                    import warp as wp

                    wp.synchronize_device("cuda:0")
                    n = self.session.solver.active_count
                    values = native_arrays(obj)
                    np.testing.assert_array_equal(values["id"], self.session.solver.pid[:n].numpy().astype(np.uint64))
                    np.testing.assert_array_equal(values["position"], self.session.solver.pos_sim[:n].numpy())
                    np.testing.assert_array_equal(values["velocity"], self.session.solver.vel[:n].numpy())
                    self.check(
                        "live_GN_updates_without_snapshot_clicks", len(self.seen) >= 5, frames=self.seen, count=n
                    )
                    self.live_group = obj.modifiers["Interop Particle Instances"].node_group
                    self.live_hash = hashes(values)
                    self.live_obj = obj
                    s.sdk_live_geometry = False
                    self.target_step = self.session.step_count + 6
                    assert bpy.ops.wm.warp_sph_pause() == {"FINISHED"}
                    self.stage = "LIVE_OFF"
            elif self.stage == "LIVE_OFF" and self.session.step_count >= self.target_step:
                assert bpy.ops.wm.warp_sph_pause() == {"FINISHED"}
                self.check("disabled_live_keeps_last_geometry", hashes(native_arrays(self.live_obj)) == self.live_hash)
                self.live_before = solver_fingerprint(self.session)
                assert bpy.ops.wm.warp_sph_sdk_bake_animation() == {"FINISHED"}
                self.check(
                    "bake_has_independent_solver", self.session.native_bake.replay.solver is not self.session.solver
                )
                self.stage = "BAKING"
            elif self.stage == "BAKING" and self.session.native_bake is None:
                self.check(
                    "bake_preserves_all_live_arrays_state_and_timeline",
                    solver_fingerprint(self.session) == self.live_before,
                )
                reader = self.sequence.SequenceReader(self.scene.warp_sph.sdk_animation_dir)
                assert reader.status == "COMPLETE" and reader.span == (1, 8)
                first = reader.read(1)
                last = reader.read(8)
                self.check("bake_all_requested_frames", len(reader.document["frames"]) == 8)
                self.check(
                    "actual_private_solver_moves", not np.array_equal(first.fields["position"], last.fields["position"])
                )
                expected_delta = 7 * s.bake_steps * self.session.solver.dt
                self.check(
                    "declared_sampling_matches_solver_time",
                    abs(last.stamp["simulation_time_s"] - first.stamp["simulation_time_s"] - expected_delta) < 1e-8,
                )
                self.report["animation_signature"] = reader.document["signature"]
                self.saved_reader = reader
                s.sdk_animation_dir = str(self.out / "cancel-cache")
                s.bake_end = 50
                self.cancel_before = solver_fingerprint(self.session)
                assert bpy.ops.wm.warp_sph_sdk_bake_animation() == {"FINISHED"}
                self.stage = "CANCEL"
            elif (
                self.stage == "CANCEL"
                and self.session.native_bake is not None
                and self.session.native_bake.captured >= 2
            ):
                assert bpy.ops.wm.warp_sph_sdk_cancel_animation() == {"FINISHED"}
                self.stage = "CANCEL_DONE"
            elif self.stage == "CANCEL_DONE" and self.session.native_bake is None:
                self.check(
                    "cancel_preserves_live_arrays_state_and_timeline",
                    solver_fingerprint(self.session) == self.cancel_before,
                )
                reader = self.sequence.SequenceReader(s.sdk_animation_dir)
                self.check(
                    "cancel_retains_whole_prefix",
                    reader.status == "CANCELLED" and reader.span[1] < 50,
                    span=reader.span,
                )
                s.sdk_animation_dir = str(self.saved_reader.directory)
                s.bake_end = 8
                assert bpy.ops.wm.warp_sph_stop() == {"FINISHED"}
                self.stage = "STOPPED"
            elif self.stage == "STOPPED" and self.session.solver is None:
                self.check("GPU_teardown_completed", self.session.last_close == "completed")
                self.verify_animation()
                self.report.update(status="passed", elapsed_seconds=time.monotonic() - self.started)
                self.save()
                return None
            self.save()
            return 0.02
        except Exception:
            self.report.update(status="failed", error=traceback.format_exc(), stage=self.stage)
            try:
                self.session.stop(bpy.context)
            except Exception:
                self.report["cleanup_error"] = traceback.format_exc()
            self.save()
            return None

    def verify_animation(self):
        settings = self.scene.warp_sph
        obj = bpy.data.objects[settings.sdk_animation_name]
        graph = obj.modifiers["Interop Particle Instances"].node_group
        checked = {}
        for frame in [8, 1, 5, 2, 7]:
            self.scene.frame_set(frame)
            bpy.context.view_layer.update()
            checked[str(frame)] = validate_mesh(obj, self.saved_reader.read(frame), self.scene)
            assert obj.modifiers["Interop Particle Instances"].node_group is graph
        self.check("timeline_scrub_exact_native_attributes", True, frames=list(checked))
        self.report["frame_hashes"] = checked
        # Corruption must never silently show an old frame. Restore the test bytes.
        bad = self.saved_reader.directory / "frame_000003.npz"
        data = bad.read_bytes()
        bad.write_bytes(b"controlled corruption")
        try:
            self.scene.frame_set(3)
            bpy.context.view_layer.update()
            self.check(
                "corrupt_cache_hidden_and_reported",
                bool(obj.get("warpsph_native_error")) and obj.hide_render and obj["warpsph_native_frame"] == -1,
            )
        finally:
            bad.write_bytes(data)
        self.scene.frame_set(3)
        bpy.context.view_layer.update()
        validate_mesh(obj, self.saved_reader.read(3), self.scene)
        self.check("restored_cache_recovers", not obj.get("warpsph_native_error") and not obj.hide_render)
        # A too-small standalone budget must reject before any scene mutation.
        before = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups))
        try:
            self.playback.make_standalone(self.scene, obj, max_bytes=1)
        except ValueError:
            self.check("standalone_budget_rejects_without_mutation", before ==
                       (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups)))
        else:
            raise AssertionError("standalone embedding ignored its memory budget")
        assert bpy.ops.wm.warp_sph_sdk_embed_animation() == {"FINISHED"}
        standalone = bpy.data.objects["WarpSPH Standalone Animation"]
        standalone_values = {}
        for frame in [1, 4, 8]:
            self.scene.frame_set(frame)
            bpy.context.view_layer.update()
            standalone_values[str(frame)] = evaluate_ids(standalone, self.particle_count)
        self.report["standalone_evaluation"] = standalone_values
        # Hide earlier static/live geometry so the exported standalone is unambiguous.
        for item in self.scene.objects:
            if item.type == "MESH" and item is not standalone:
                item.hide_render = True
        self.live_obj.hide_set(True)
        self.scene.frame_start = 1
        self.scene.frame_end = 8
        self.setup_render(standalone)
        self.scene.frame_set(1)
        path = self.out / "Live-GN-Fluid-Animation.blend"
        assert not path.exists()
        assert bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True) == {"FINISHED"}
        self.report["standalone_blend"] = str(path)
        expected = {
            "creator_pid": os.getpid(),
            "object": standalone.name,
            "count": self.particle_count,
            "frame_hashes": checked,
            "source_cache": str(self.saved_reader.directory),
            "frame_range": [1, 8],
        }
        (self.out / "standalone-expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")

    def setup_render(self, obj):
        from mathutils import Vector

        points = self.saved_reader.read(1).fields["position"]
        low = points.min(axis=0)
        high = points.max(axis=0)
        center = Vector(((low + high) * 0.5).tolist())
        extent = max(float(np.max(high - low)), 1.0)
        camera_data = bpy.data.cameras.new("Animation Camera")
        camera = bpy.data.objects.new("Animation Camera", camera_data)
        self.scene.collection.objects.link(camera)
        camera.location = center + Vector((extent * 1.6, -extent * 2.2, extent * 1.5))
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = extent * 1.8
        self.scene.camera = camera
        light_data = bpy.data.lights.new("Animation Key", "AREA")
        light_data.energy = 1800
        light_data.shape = "DISK"
        light_data.size = extent * 2
        light = bpy.data.objects.new("Animation Key", light_data)
        self.scene.collection.objects.link(light)
        light.location = center + Vector((extent, -extent, extent * 2))
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        self.scene.render.engine = "BLENDER_EEVEE"
        self.scene.render.resolution_x = 192
        self.scene.render.resolution_y = 192
        self.scene.render.resolution_percentage = 100
        self.scene.render.film_transparent = True
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.image_settings.color_mode = "RGBA"


def start(output, consumer="bl_ext.user_default.warp_sph_realtime"):
    global _probe
    assert _probe is None
    _probe = Probe(output, consumer)
    try:
        _probe.begin()
    except Exception:
        _probe.report.update(status="failed", startup_error=traceback.format_exc())
        _probe.save()
        raise
    return _probe


def verify_standalone(output):
    import importlib.util

    out = Path(output)
    expected = json.loads((out / "standalone-expected.json").read_text())
    assert os.getpid() != expected["creator_pid"]
    assert importlib.util.find_spec("warp") is None and importlib.util.find_spec("blender_vulkan_interop") is None
    assert not any("warp_sph_realtime" in name for name in bpy.context.preferences.addons)
    obj = bpy.data.objects[expected["object"]]
    samples = {}
    for frame in [1, 4, 8]:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        result = evaluate_ids(obj, expected["count"])
        dg = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(dg)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
        try:
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            result["position_sha"] = hashlib.sha256(coords.tobytes()).hexdigest()
        finally:
            evaluated.to_mesh_clear()
        samples[str(frame)] = result
    assert samples["1"]["position_sha"] != samples["8"]["position_sha"]
    report = {
        "status": "passed",
        "pid": os.getpid(),
        "SDK_Warp_consumer_absent": True,
        "native_GN_time_samples": samples,
        "source_files_not_read_for_playback": True,
    }
    (out / "standalone-reopen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def render_frame(output, frame):
    out = Path(output)
    path = out / f"animation_{frame:04d}.png"
    assert not path.exists()
    scene = bpy.context.scene
    scene.frame_set(frame)
    scene.render.filepath = str(path)
    assert bpy.ops.render.render(write_still=True) == {"FINISHED"}
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgba = pixels.reshape(-1, 4)
        count = int(np.count_nonzero(rgba[:, 3] > 0.1))
        assert count > 25, "render is blank"
        result = {
            "status": "passed",
            "frame": frame,
            "path": str(path),
            "visible_pixels": count,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        }
    finally:
        bpy.data.images.remove(image)
    (out / f"render_{frame:04d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
