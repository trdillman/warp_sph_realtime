"""Native animation replay and handler-free standalone GN output.

Only Blender/NumPy is used during replay; neither Warp nor the SDK is needed.
Creating the initial particle graph uses the already-admitted shared SDK bridge.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import bpy

_MARKER = "warpsph_native_sequence"
_ERRORS = {}
_LOADED = {}
_LOG = logging.getLogger(__name__)


def _scale(scene):
    value = float(scene.unit_settings.scale_length)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("positive scene unit scale required")
    return value


def mesh_from_frame(frame, scene, name):
    import numpy as np

    from .native_sequence import NativeFrame

    if not isinstance(frame, NativeFrame):
        raise ValueError("validated NativeFrame required")
    scale = _scale(scene)
    mesh = bpy.data.meshes.new(name)
    try:
        fields = frame.fields
        mesh.vertices.add(frame.count)
        positions = (fields["position"].astype(np.float64) / scale).astype(np.float32)
        if not np.isfinite(positions).all():
            raise ValueError("scene-unit conversion overflow")
        mesh.vertices.foreach_set("co", positions.ravel())
        for name, kind, prop, values in (
            ("radius", "FLOAT", "value", fields["radius"] / scale),
            ("color", "FLOAT_COLOR", "color", fields["color"]),
            ("velocity", "FLOAT_VECTOR", "vector", fields["velocity"] / scale),
            ("density", "FLOAT", "value", fields["density"]),
            ("particle_id_low", "INT", "value", fields["id"].astype(np.uint32).view(np.int32)),
            ("particle_id_high", "INT", "value", (fields["id"] >> np.uint64(32)).astype(np.uint32).view(np.int32)),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"scene-unit conversion overflow for {name}")
            attr = mesh.attributes.new(name=name, type=kind, domain="POINT")
            attr.data.foreach_set(prop, values.ravel())
        mesh["interop_source"] = frame.stamp["source_id"]
        mesh["interop_epoch"] = frame.stamp["epoch"]
        mesh["interop_frame"] = frame.stamp["frame_id"]
        mesh["interop_snapshot"] = True
        mesh.update()
        return mesh
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise


def _object_identity(obj):
    from mathutils import Matrix

    if obj.type != "MESH" or obj.mode != "OBJECT" or obj.library or obj.data.library:
        raise ValueError("animation output must be an editable mesh in Object Mode")
    if obj.matrix_world != Matrix.Identity(4):
        raise ValueError("apply/reset output transform before native playback; positions are world-space")


def apply_native_frame(obj, frame, scene):
    _object_identity(obj)
    replacement = mesh_from_frame(frame, scene, obj.name + " Frame")
    previous = obj.data
    try:
        obj.data = replacement
        obj.update_tag()
    except Exception:
        if replacement.users == 0:
            bpy.data.meshes.remove(replacement)
        raise
    if previous.users == 0:
        bpy.data.meshes.remove(previous)


def attach(scene, directory, name, create_graph, *, max_bytes=1024 * 1024 * 1024, configure_range=True):
    from .native_sequence import SequenceReader

    if not isinstance(name, str) or not name.strip():
        raise ValueError("give the native animation object a name")
    reader = SequenceReader(directory)
    for item in reader.document["frames"]:
        _check_frame_budget(item["payload_bytes"], item["count"], max_bytes)
    first, last = reader.validate_all()
    owner = scene.warp_sph.sdk_snapshot_owner
    obj = bpy.data.objects.get(name)
    created = obj is None
    if obj is not None:
        if obj.get("warpsph_sdk_scene") != owner or obj.name not in scene.objects:
            raise ValueError("animation target is not owned by the active WarpSPH scene")
        _object_identity(obj)
    frame = reader.read(reader.clamp(scene.frame_current))
    _check_frame_budget(frame.nbytes, frame.count, max_bytes)
    mesh = mesh_from_frame(frame, scene, name + " Frame")
    previous = None if created else obj.data
    group = None
    try:
        if created:
            obj = bpy.data.objects.new(name, mesh)
            scene.collection.objects.link(obj)
        else:
            obj.data = mesh
        if obj.modifiers.get("Interop Particle Instances") is None:
            group = create_graph(obj)
        obj["warpsph_sdk_scene"] = owner
        obj[_MARKER] = str(Path(directory).resolve())
        obj[_MARKER + "_enabled"] = True
        obj[_MARKER + "_max_bytes"] = max_bytes
        obj.hide_render = False
        obj.hide_set(False)
        obj["warpsph_native_frame"] = reader.clamp(scene.frame_current)
        obj["warpsph_native_error"] = ""
        scene.render.use_lock_interface = True
        if configure_range:
            scene.frame_start, scene.frame_end = first, last
        _LOADED.pop(obj.as_pointer(), None)
        obj.update_tag()
        if previous is not None and previous.users == 0:
            bpy.data.meshes.remove(previous)
        return obj
    except Exception:
        if group is not None and obj is not None:
            materials = [
                node.inputs["Material"].default_value
                for node in group.nodes
                if node.bl_idname == "GeometryNodeSetMaterial" and node.inputs["Material"].default_value is not None
            ]
            for modifier in tuple(obj.modifiers):
                if modifier.type == "NODES" and modifier.node_group is group:
                    obj.modifiers.remove(modifier)
            if group.users == 0:
                bpy.data.node_groups.remove(group)
            for material in materials:
                if material.users == 0:
                    bpy.data.materials.remove(material)
        if created and obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        elif obj is not None:
            obj.data = previous
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise


def _check_frame_budget(payload_bytes, count, max_bytes):
    if max_bytes <= 0 or payload_bytes * 3 + count * 4096 > max_bytes:
        raise ValueError("animation player/GN estimate exceeds native memory budget")


def update_scene(scene, depsgraph=None):
    from .native_sequence import SequenceReader

    for obj in tuple(scene.objects):
        directory = obj.get(_MARKER)
        if not directory or not obj.get(_MARKER + "_enabled", True):
            continue
        try:
            if bpy.app.is_job_running("RENDER") and not scene.render.use_lock_interface:
                raise ValueError("Enable Render Lock Interface for disk-backed native animation")
            reader = SequenceReader(bpy.path.abspath(directory))
            number = reader.clamp(scene.frame_current)
            span = reader.span
            assert span is not None
            item = reader.document["frames"][number - span[0]]
            max_bytes = min(int(obj.get(_MARKER + "_max_bytes", 1024 * 1024 * 1024)),
                            int(scene.warp_sph.sdk_native_budget_mb) * 1024 * 1024)
            _check_frame_budget(item["payload_bytes"], item["count"], max_bytes)
            path = reader.directory / item["file"]
            # Include mtime and size so edits/corruption invalidate already shown frames.
            state = (str(reader.directory), number, path.stat().st_mtime_ns, path.stat().st_size,
                     _scale(scene), max_bytes)
            if _LOADED.get(obj.as_pointer()) == state:
                continue
            frame = reader.read(number)
            _check_frame_budget(frame.nbytes, frame.count, max_bytes)
            apply_native_frame(obj, frame, scene)
            obj["warpsph_native_frame"] = number
            obj["warpsph_native_error"] = ""
            if "warpsph_restore_render" in obj:
                obj.hide_render = bool(obj["warpsph_restore_render"])
                del obj["warpsph_restore_render"]
            _LOADED[obj.as_pointer()] = state
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            obj["warpsph_native_error"] = message
            obj["warpsph_native_frame"] = -1
            _LOADED.pop(obj.as_pointer(), None)
            # Never show an old frame as if it were current after a bad cache read.
            if obj.type == "MESH" and obj.mode == "OBJECT" and not bpy.app.is_job_running("RENDER"):
                empty = bpy.data.meshes.new(obj.name + " Missing Frame")
                previous, obj.data = obj.data, empty
                if previous.users == 0:
                    bpy.data.meshes.remove(previous)
            if "warpsph_restore_render" not in obj:
                obj["warpsph_restore_render"] = obj.hide_render
            obj.hide_render = True
            if _ERRORS.get(obj.as_pointer()) != message:
                _LOG.error("Native animation stopped for %s: %s", obj.name, message)
                _ERRORS[obj.as_pointer()] = message


def make_standalone(scene, player, *, name="WarpSPH Standalone Animation", max_bytes=1024 * 1024 * 1024):
    """Embed an exact discrete frame sequence in ordinary GN, no Python driver.

    Uses one Object Info per stored point mesh and a SceneTime Index Switch.
    Frames hold discretely; no interpolation or motion-blur claim is implied.
    """
    from .native_sequence import SequenceReader

    _object_identity(player)
    if player.name not in scene.objects or not player.get(_MARKER):
        raise ValueError("select an active native cache player")
    if bpy.data.objects.get(name) is not None:
        raise ValueError("standalone output name already exists; no overwrite")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("positive standalone memory budget required")
    reader = SequenceReader(bpy.path.abspath(player[_MARKER]))
    first, last = reader.validate_all()
    frames = reader.document["frames"]
    # Retained point meshes + largest realized icosphere frame + graph overhead.
    estimate = (
        sum(item["count"] * 192 + 2048 for item in frames)
        + max(item["count"] for item in frames) * 4096
        + len(frames) * 8192
    )
    if estimate > max_bytes:
        raise ValueError(
            f"standalone animation estimate {estimate} bytes exceeds {max_bytes}; use streaming playback or reduce range/count"
        )
    template = player.modifiers.get("Interop Particle Instances")
    if template is None or template.type != "NODES" or template.node_group is None:
        raise ValueError("native particle Geometry Nodes template missing")
    collection = bpy.data.collections.new(name + " Frames")
    scene.collection.children.link(collection)
    objects, meshes = [], []
    group = output = None
    try:
        group = bpy.data.node_groups.new(name, "GeometryNodeTree")
        group.is_modifier = True
        group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        nodes, links = group.nodes, group.links
        clock = nodes.new("GeometryNodeInputSceneTime")
        clock.location = (-800, 300)
        offset = nodes.new("ShaderNodeMath")
        offset.operation = "SUBTRACT"
        offset.inputs[1].default_value = first
        offset.location = (-600, 300)
        links.new(clock.outputs["Frame"], offset.inputs[0])
        lower = nodes.new("ShaderNodeMath")
        lower.operation = "MAXIMUM"
        lower.inputs[1].default_value = 0
        lower.location = (-400, 300)
        links.new(offset.outputs[0], lower.inputs[0])
        upper = nodes.new("ShaderNodeMath")
        upper.operation = "MINIMUM"
        upper.inputs[1].default_value = last - first
        upper.location = (-200, 300)
        links.new(lower.outputs[0], upper.inputs[0])
        choose = nodes.new("GeometryNodeIndexSwitch")
        choose.data_type = "GEOMETRY"
        choose.location = (50, 100)
        while len(choose.index_switch_items) > 0:
            choose.index_switch_items.remove(choose.index_switch_items[-1])
        for _ in frames:
            choose.index_switch_items.new()
        links.new(upper.outputs[0], choose.inputs["Index"])
        for index, item in enumerate(frames):
            frame = reader.read(item["frame"])
            mesh = mesh_from_frame(frame, scene, f"{name} Frame {item['frame']:06d}")
            meshes.append(mesh)
            obj = bpy.data.objects.new(mesh.name, mesh)
            collection.objects.link(obj)
            objects.append(obj)
            obj.hide_render = True
            obj.hide_set(True)
            obj["warpsph_embedded_frame"] = item["frame"]
            info = nodes.new("GeometryNodeObjectInfo")
            info.transform_space = "ORIGINAL"
            info.inputs["Object"].default_value = obj
            info.inputs["As Instance"].default_value = False
            info.location = (-200, -index * 180)
            links.new(info.outputs["Geometry"], choose.inputs[index + 1])
        particles = nodes.new("GeometryNodeGroup")
        particles.node_tree = template.node_group
        particles.location = (300, 100)
        links.new(choose.outputs[0], particles.inputs["Geometry"])
        result = nodes.new("NodeGroupOutput")
        result.location = (550, 100)
        links.new(particles.outputs["Geometry"], result.inputs["Geometry"])
        mesh = bpy.data.meshes.new(name)
        meshes.append(mesh)
        output = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(output)
        mod = output.modifiers.new("Native Animation", "NODES")
        mod.node_group = group
        output["warpsph_standalone_animation"] = True
        output["frame_start"] = first
        output["frame_end"] = last
        output["source_sequence"] = str(reader.directory)
        output["native_memory_estimate"] = estimate
        output["frame_semantics"] = "discrete cached samples; held outside range"
        player.hide_render = True
        player.hide_set(True)
        # The baked object no longer depends on handlers, source object or cache path.
        player[_MARKER + "_enabled"] = False
        output.update_tag()
        return output
    except Exception:
        if output is not None:
            bpy.data.objects.remove(output, do_unlink=True)
        if group is not None and group.users == 0:
            bpy.data.node_groups.remove(group)
        for obj in objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        bpy.data.collections.remove(collection)
        raise


def _frame_handler(scene, depsgraph=None):
    # A standalone baked output is independent of this handler and remains valid
    # when the extension, SDK and Warp are all absent.
    if any(obj.get(_MARKER) and obj.get(_MARKER + "_enabled", True) for obj in getattr(scene, "objects", ())):
        update_scene(scene, depsgraph)


def register():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None:
        return
    persistent = getattr(handlers, "persistent", None)
    for callback, name in [(_frame_handler, "frame_change_post"), (_loaded_file, "load_post")]:
        items = getattr(handlers, name, None)
        if persistent is not None:
            persistent(callback)
        if items is not None and callback not in items:
            items.append(callback)


def _loaded_file(_unused):
    _LOADED.clear()
    _ERRORS.clear()
    for scene in bpy.data.scenes:
        _frame_handler(scene)


def unregister():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is not None:
        for callback, name in [(_frame_handler, "frame_change_post"), (_loaded_file, "load_post")]:
            items = getattr(handlers, name, ())
            while callback in items:
                items.remove(callback)
    _LOADED.clear()
    _ERRORS.clear()
