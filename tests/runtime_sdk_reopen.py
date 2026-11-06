"""Fresh Blender proof of a real WarpSPH native snapshot, without runtime add-ons."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import bpy
import gpu
import numpy as np


def _absent(name):
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


def run(expected_path, output_path, *, save_demo=None):
    expected = json.loads(Path(expected_path).read_text())
    output = Path(output_path)
    assert output.is_absolute() and not output.exists()
    assert os.getpid() != expected["creator_pid"]
    assert bpy.app.build_hash.decode() == "fbe6228777e7" and gpu.platform.backend_type_get() == "VULKAN"
    assert ".verification-temp" in Path(bpy.utils.user_resource("CONFIG")).parts
    missing = {
        name: _absent(name) for name in ("warp", "blender_vulkan_interop", "bl_ext.user_default.warp_sph_realtime")
    }
    assert all(missing.values()), missing
    obj = bpy.data.objects[expected["object"]]
    mesh = obj.data
    count = len(mesh.vertices)
    assert count == expected["count"] == 100000
    values = {}
    position = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", position)
    values["position"] = position.reshape(count, 3)
    fields = [
        ("velocity", 3, "vector", np.float32),
        ("density", 1, "value", np.float32),
        ("radius", 1, "value", np.float32),
        ("color", 4, "color", np.float32),
        ("particle_id_low", 1, "value", np.int32),
        ("particle_id_high", 1, "value", np.int32),
    ]
    for name, width, field, dtype in fields:
        attr = mesh.attributes[name]
        assert attr.domain == "POINT"
        data = np.empty(count * width, dtype=dtype)
        attr.data.foreach_get(field, data)
        values[name] = data.reshape(count, width) if width > 1 else data
    hashes = {name: hashlib.sha256(data.tobytes()).hexdigest() for name, data in values.items()}
    assert hashes == expected["array_sha256"]
    modifier = obj.modifiers["Interop Particle Instances"]
    group = modifier.node_group
    inputs = {
        item.name: item.identifier
        for item in group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    }
    verified = []
    try:
        for sample in expected["samples"]:
            identity = sample["id"]
            low, high = identity & 0xFFFFFFFF, identity >> 32
            if low >= 1 << 31:
                low -= 1 << 32
            if high >= 1 << 31:
                high -= 1 << 32
            getattr(modifier.properties.inputs, inputs["Filter ID"]).value = True
            getattr(modifier.properties.inputs, inputs["ID Low"]).value = low
            getattr(modifier.properties.inputs, inputs["ID High"]).value = high
            obj.update_tag()
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = obj.evaluated_get(depsgraph)
            realized = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                assert len(realized.vertices) == 12
                lo = np.empty(12, dtype=np.int32)
                hi = np.empty(12, dtype=np.int32)
                realized.attributes["particle_id_low"].data.foreach_get("value", lo)
                realized.attributes["particle_id_high"].data.foreach_get("value", hi)
                ids = lo.view(np.uint32).astype(np.uint64) | (hi.view(np.uint32).astype(np.uint64) << np.uint64(32))
                assert np.all(ids == identity)
                points = np.empty(36, dtype=np.float32)
                realized.vertices.foreach_get("co", points)
                points = points.reshape(12, 3)
                np.testing.assert_allclose(points.mean(axis=0), sample["position"], rtol=0, atol=2e-5)
                np.testing.assert_allclose(
                    np.linalg.norm(points - np.asarray(sample["position"]), axis=1),
                    sample["radius"],
                    rtol=2e-4,
                    atol=2e-5,
                )
                verified.append(identity)
            finally:
                evaluated.to_mesh_clear()
    finally:
        getattr(modifier.properties.inputs, inputs["Filter ID"]).value = False
        obj.update_tag()
        bpy.context.view_layer.update()
    report = {
        "status": "passed",
        "pid": os.getpid(),
        "creator_pid": expected["creator_pid"],
        "particle_count": count,
        "runtime_modules_absent": missing,
        "array_sha256": hashes,
        "exact_GN_selected_solver_ids": verified,
        "saved_file": bpy.data.filepath,
        "scope": "saved actual-solver native attributes and GN, no live simulation or bake cache",
    }
    if save_demo is not None:
        destination = Path(save_demo)
        assert destination.is_absolute() and not destination.exists()
        for item in bpy.context.view_layer.objects:
            item.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        center = (values["position"].min(axis=0) + values["position"].max(axis=0)) * 0.5
        extent = float(np.ptp(values["position"], axis=0).max())
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "MATERIAL"
                area.spaces.active.region_3d.view_location = center
                area.spaces.active.region_3d.view_distance = max(1.0, extent * 1.7)
        readme = bpy.data.texts.get("READ ME - Actual WarpSPH Snapshot") or bpy.data.texts.new(
            "READ ME - Actual WarpSPH Snapshot"
        )
        readme.clear()
        readme.write(
            "ACTUAL WARPSPH SOLVER -> SHARED SDK -> NATIVE GEOMETRY NODES\n\n"
            "This static snapshot contains 100,000 particles from the real WarpSPH CUDA solver, not a synthetic particle fixture.\n"
            "Position, velocity, density and stable ID words matched the solver at capture. Radius and colour come from artist display settings.\n\n"
            "The saved object and editable Geometry Nodes graph were verified in a fresh Blender5.2 process without WarpSPH, Warp or the SDK installed.\n"
            "Use the Spreadsheet to inspect the mesh point attributes. Filter ID selects by both integer words, not by current array index.\n\n"
            "This is a native snapshot, not a simulation cache or a claim of live zero-copy Geometry Nodes.\n"
            "The private development extension adds an opt-in Shared SDK Output mode: Start, Pause, Snapshot to Geometry Nodes, Stop.\n"
        )
        assert bpy.ops.wm.save_as_mainfile(filepath=str(destination), copy=True) == {"FINISHED"}
        report["artist_demo"] = str(destination)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
