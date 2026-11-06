"""Independent embedded-animation oracle. Blender playback uses no SDK or cache.

Reference snapshots are read only by the oracle; their original playback directory
must be absent. This separates data verification from actual Scene-Time GN playback.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import bpy
import gpu
import numpy as np


def _field(mesh, name, components, integer=False):
    attr = mesh.attributes.get(name)
    assert attr is not None and attr.domain == "POINT", (name, "missing point field")
    values = np.empty(len(mesh.vertices) * components, dtype=np.int32 if integer else np.float32)
    attr.data.foreach_get("value" if components == 1 else "color" if components == 4 else "vector", values)
    return values.reshape(-1, components) if components > 1 else values


def check_frame(obj, reference, number, scale):
    bpy.context.scene.frame_set(number)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        with np.load(reference, allow_pickle=False) as arrays:
            ids = arrays["id"].copy()
            position = arrays["position"].copy()
            radius = arrays["radius"].copy()
            velocity = arrays["velocity"].copy()
            color = arrays["color"].copy()
            density = arrays["density"].copy()
        count = len(ids)
        assert len(mesh.vertices) == 12 * count and len(mesh.polygons) == 20 * count
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        if not count:
            return dict(frame=number, particles=0, vertices=0)
        low = _field(mesh, "particle_id_low", 1, True).view(np.uint32).astype(np.uint64)
        high = _field(mesh, "particle_id_high", 1, True).view(np.uint32).astype(np.uint64)
        realized = low | (high << np.uint64(32))
        keys, inverse, occurrences = np.unique(realized, return_inverse=True, return_counts=True)
        order = np.argsort(ids)
        np.testing.assert_array_equal(keys, ids[order])
        np.testing.assert_array_equal(occurrences, np.full(count, 12))
        centers = np.stack(
            [np.bincount(inverse, weights=coords[:, axis], minlength=count) / 12.0 for axis in range(3)], axis=1
        )
        expected = position[order].astype(np.float64) / scale
        np.testing.assert_allclose(centers, expected, atol=2e-5, rtol=0)
        measured = np.linalg.norm(coords - expected[inverse], axis=1)
        np.testing.assert_allclose(measured, radius[order][inverse] / scale, atol=2e-5, rtol=5e-5)
        for name, width, source in [("velocity", 3, velocity / scale), ("color", 4, color), ("density", 1, density)]:
            np.testing.assert_allclose(_field(mesh, name, width), source[order][inverse], atol=1e-6, rtol=1e-6)
        return dict(
            frame=number,
            particles=count,
            vertices=len(mesh.vertices),
            ids_exact=True,
            all_native_fields_match=True,
            positions_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
        )
    finally:
        evaluated.to_mesh_clear()


def run(output_dir, oracle_manifest):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(oracle_manifest).read_text())
    report = {"pid": os.getpid(), "status": "running", "frames": [], "SDK_Warp_consumer_absent": False}
    assert not (out / "independent-standalone.json").exists()
    try:
        assert os.getpid() != manifest["creator_pid"]
        assert gpu.platform.backend_type_get() == "VULKAN" and bpy.app.build_hash.decode() == "fbe6228777e7"
        assert importlib.util.find_spec("blender_vulkan_interop") is None and importlib.util.find_spec("warp") is None
        assert not any("warp_sph_realtime" in addon.module for addon in bpy.context.preferences.addons)
        assert not Path(manifest["original_cache_path"]).exists(), (
            "original cache must be physically hidden for standalone proof"
        )
        report["SDK_Warp_consumer_absent"] = True
        report["original_cache_absent"] = True
        obj = bpy.data.objects[manifest["object"]]
        scale = float(bpy.context.scene.unit_settings.scale_length)
        frames = manifest["frames"]
        numbers = sorted(int(n) for n in frames)
        # Nonmonotonic seeking probes stale-geometry behavior instead of sequential-only playback.
        order = numbers[::2][::-1] + numbers[1::2]
        for number in order:
            record = frames[str(number)]
            path = Path(record["oracle_file"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            report["frames"].append(check_frame(obj, path, number, scale))
        assert len({frame["positions_sha256"] for frame in report["frames"]}) > 1, "all animation frames are frozen"
        report["status"] = "passed"
        return report
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (out / "independent-standalone.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def render_animation(output_dir):
    out = Path(output_dir)
    directory = out / "animation-render"
    assert not directory.exists()
    directory.mkdir()
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    scene.render.filepath = str(directory / "frame_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 192
    scene.render.resolution_y = 192
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    assert bpy.ops.render.render(animation=True) == {"FINISHED"}
    results = []
    for number in range(first, last + 1):
        path = directory / f"frame_{number:04d}.png"
        assert path.is_file()
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            visible = int(np.count_nonzero(pixels.reshape(-1, 4)[:, 3] > 0.1))
            assert visible > 25
            results.append(
                dict(
                    frame=number,
                    visible_pixels=visible,
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    path=str(path),
                )
            )
        finally:
            bpy.data.images.remove(image)
    assert len({x["pixel_sha256"] for x in results}) > 1, "render animation produced identical frames"
    receipt = dict(
        status="passed", pid=os.getpid(), render_operator="animation=True", frames=results, engine=scene.render.engine
    )
    (out / "render-animation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
