"""Blender side of the output tools (P2.3): the Geometry Nodes meshing preset and the Alembic / USD export of the
baked sequence. With ``blender_adapter`` the only ``bpy`` modules in ``cache/``; the numbers come from ``output.py``.

Meshing preset: a mesh object ("WarpSPH Mesh") whose Geometry Nodes modifier reads the playback point cloud through
an Object Info node and builds points -> Points to Volume -> Volume to Mesh -> Blur Attribute smoothing -> smooth
shading -> Set Material. Radius, voxel size, threshold, smoothing and material are modifier inputs. The node and
socket names were probed on Blender 5.2: Points to Volume and Volume to Mesh take their resolution as a *Menu* socket named "Resolution Mode"
(items "Amount" / "Size"), Set Shade Smooth's geometry socket is named "Mesh" with identifier "Geometry".
"""
from __future__ import annotations

from pathlib import Path

import bpy

from . import output

MESH_PROP = "warp_sph_cache_mesh"      # custom property that marks the meshing object
MESH_NAME = "WarpSPH Mesh"
GROUP_NAME = "WarpSPH Meshing"
MODIFIER_NAME = "WarpSPH Meshing"
GROUP_VERSION = 2                      # bump when the node graph changes: older groups are rebuilt
_VERSION_PROP = "warp_sph_graph_version"

# (modifier input name, socket type, subtype, default attribute of MeshingDefaults or constant, minimum)
_INPUTS = (
    ("Radius", "NodeSocketFloat", 'DISTANCE', "radius", 0.0001),
    ("Voxel Size", "NodeSocketFloat", 'DISTANCE', "voxel_size", 0.0001),
    ("Threshold", "NodeSocketFloat", 'NONE', "threshold", 0.0),
    ("Smooth Iterations", "NodeSocketInt", 'NONE', "smooth_iterations", 0),
    ("Smooth Weight", "NodeSocketFloat", 'FACTOR', "smooth_weight", 0.0),
    ("Shade Smooth", "NodeSocketBool", 'NONE', True, None),
)
MATERIAL_INPUT = "Material"            # optional: material assigned to the mesh (Geometry Nodes meshes start without)


def find_mesh_object(scene):
    for obj in scene.objects:
        if obj.get(MESH_PROP) and obj.type == 'MESH':
            return obj
    return None


def _link(tree, source, target):
    tree.links.new(source, target)


def build_meshing_group(defaults: output.MeshingDefaults):
    """The meshing node group (created, or rebuilt when an older graph version is found)."""
    tree = bpy.data.node_groups.get(GROUP_NAME)
    if tree is not None and tree.get(_VERSION_PROP) == GROUP_VERSION:
        return tree
    if tree is not None:
        bpy.data.node_groups.remove(tree)
    tree = bpy.data.node_groups.new(GROUP_NAME, 'GeometryNodeTree')
    tree[_VERSION_PROP] = GROUP_VERSION
    interface = tree.interface
    interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    cloud = interface.new_socket("Point Cloud", in_out='INPUT', socket_type='NodeSocketObject')
    cloud.description = "The cache point cloud to mesh"
    for name, socket_type, subtype, default, minimum in _INPUTS:
        item = interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
        if subtype != 'NONE':
            item.subtype = subtype
        item.default_value = getattr(defaults, default) if isinstance(default, str) else default
        if minimum is not None:
            item.min_value = minimum
    interface.new_socket(MATERIAL_INPUT, in_out='INPUT', socket_type='NodeSocketMaterial')
    interface.items_tree["Radius"].description = "Radius of the sphere each particle adds to the density field"
    interface.items_tree["Voxel Size"].description = "Grid cell size of the density field and of the output mesh"
    interface.items_tree["Threshold"].description = "Density at which the surface is drawn (lower = fatter)"
    interface.items_tree["Smooth Iterations"].description = "Passes that average each vertex with its neighbours"

    nodes = tree.nodes
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    info.transform_space = 'RELATIVE'
    info.inputs["As Instance"].default_value = False
    to_volume = nodes.new("GeometryNodePointsToVolume")
    to_volume.inputs["Density"].default_value = 1.0
    to_volume.inputs["Resolution Mode"].default_value = 'Size'
    to_mesh = nodes.new("GeometryNodeVolumeToMesh")
    to_mesh.inputs["Resolution Mode"].default_value = 'Size'
    to_mesh.inputs["Adaptivity"].default_value = 0.0
    position = nodes.new("GeometryNodeInputPosition")
    blur = nodes.new("GeometryNodeBlurAttribute")
    blur.data_type = 'FLOAT_VECTOR'
    set_position = nodes.new("GeometryNodeSetPosition")
    shade = nodes.new("GeometryNodeSetShadeSmooth")
    shade.domain = 'FACE'
    material = nodes.new("GeometryNodeSetMaterial")
    for node, x in ((group_in, -900), (info, -700), (to_volume, -450), (to_mesh, -200), (position, -200),
                    (blur, 50), (set_position, 300), (shade, 550), (material, 800), (group_out, 1050)):
        node.location = (x, 0)
    position.location, blur.location = (-200, -250), (50, -250)

    socket_in = group_in.outputs
    _link(tree, socket_in["Point Cloud"], info.inputs["Object"])
    _link(tree, info.outputs["Geometry"], to_volume.inputs["Points"])
    _link(tree, socket_in["Voxel Size"], to_volume.inputs["Voxel Size"])
    _link(tree, socket_in["Radius"], to_volume.inputs["Radius"])
    _link(tree, to_volume.outputs["Volume"], to_mesh.inputs["Volume"])
    _link(tree, socket_in["Voxel Size"], to_mesh.inputs["Voxel Size"])
    _link(tree, socket_in["Threshold"], to_mesh.inputs["Threshold"])
    _link(tree, to_mesh.outputs["Mesh"], set_position.inputs["Geometry"])
    _link(tree, position.outputs["Position"], blur.inputs["Value"])
    _link(tree, socket_in["Smooth Iterations"], blur.inputs["Iterations"])
    _link(tree, socket_in["Smooth Weight"], blur.inputs["Weight"])
    _link(tree, blur.outputs["Value"], set_position.inputs["Position"])
    _link(tree, set_position.outputs["Geometry"], shade.inputs["Geometry"])
    _link(tree, socket_in["Shade Smooth"], shade.inputs["Shade Smooth"])
    _link(tree, shade.outputs["Geometry"], material.inputs["Geometry"])
    _link(tree, socket_in[MATERIAL_INPUT], material.inputs["Material"])
    _link(tree, material.outputs["Geometry"], group_out.inputs["Geometry"])
    return tree


def _get_input(modifier, identifier: str):
    """A modifier input by socket identifier. Blender 5.2 keeps them in ``modifier.properties.inputs`` (each a
    struct with ``value``); earlier versions used ID properties on the modifier."""
    inputs = getattr(modifier, "properties", None)
    if inputs is not None:
        return getattr(inputs.inputs, identifier).value
    return modifier[identifier]


def _set_input(modifier, identifier: str, value) -> None:
    inputs = getattr(modifier, "properties", None)
    if inputs is not None:
        getattr(inputs.inputs, identifier).value = value
    else:
        modifier[identifier] = value


def _set_inputs(modifier, tree, cloud_object, defaults: output.MeshingDefaults) -> None:
    for item in tree.interface.items_tree:
        if item.item_type != 'SOCKET' or item.in_out != 'INPUT':
            continue
        if item.name == "Point Cloud":
            _set_input(modifier, item.identifier, cloud_object)
        else:
            for name, _type, _subtype, default, _minimum in _INPUTS:
                if name == item.name:
                    _set_input(modifier, item.identifier, getattr(defaults, default) if isinstance(default, str) else default)


def ensure_mesh_object(scene, cloud_object, defaults: output.MeshingDefaults):
    """The meshing object of ``scene`` (created if missing) pointing at ``cloud_object``. Returns (object, created)."""
    obj = find_mesh_object(scene)
    created = obj is None
    tree = build_meshing_group(defaults)
    if created:
        mesh = bpy.data.meshes.new(MESH_NAME)
        obj = bpy.data.objects.new(MESH_NAME, mesh)
        obj[MESH_PROP] = True
        scene.collection.objects.link(obj)
    modifier = obj.modifiers.get(MODIFIER_NAME)
    if modifier is None:
        modifier = obj.modifiers.new(MODIFIER_NAME, 'NODES')
        modifier.node_group = tree
        _set_inputs(modifier, tree, cloud_object, defaults)
    elif modifier.node_group is not tree:  # a rebuilt group: re-point and reset the inputs
        modifier.node_group = tree
        _set_inputs(modifier, tree, cloud_object, defaults)
    obj.update_tag()  # inputs set from Python do not tag the object for re-evaluation by themselves
    cloud_object.hide_render = True  # the mesh replaces the points in renders; hiding is reversible by the user
    return obj, created


def remove_mesh_object(scene) -> bool:
    obj = find_mesh_object(scene)
    if obj is None:
        return False
    mesh = obj.data
    bpy.data.objects.remove(obj)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    return True


def _identifier(modifier, name: str) -> str | None:
    if modifier is None or modifier.node_group is None:
        return None
    for item in modifier.node_group.interface.items_tree:
        if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == name:
            return item.identifier
    return None


def modifier_input(obj, name: str):
    """The meshing modifier's input called ``name`` (its value, or None if there is no such input)."""
    modifier = obj.modifiers.get(MODIFIER_NAME)
    identifier = _identifier(modifier, name)
    return None if identifier is None else _get_input(modifier, identifier)


def input_socket(obj, name: str):
    """The Blender 5.2 input struct (draw it with ``layout.prop(struct, "value")``), or None."""
    modifier = obj.modifiers.get(MODIFIER_NAME)
    identifier = _identifier(modifier, name)
    properties = getattr(modifier, "properties", None)
    if identifier is None or properties is None:
        return None
    return getattr(properties.inputs, identifier, None)


def export_sequence(context, filepath: str, fmt: str, obj, frames: tuple[int, int]) -> Path:
    """Export ``obj`` over the baked ``frames`` as USD or Alembic. Selection and the current frame are restored.

    The scene's frame handlers run for each exported frame, so the playback point cloud (and the meshing object that
    reads it) hold that frame's cache. Returns the file written; raises RuntimeError if the exporter fails."""
    scene = context.scene
    target = output.export_path(filepath, fmt)
    target.parent.mkdir(parents=True, exist_ok=True)
    view_layer = context.view_layer
    selected = [o for o in view_layer.objects if o.select_get()]
    active, frame = view_layer.objects.active, scene.frame_current
    start, end = scene.frame_start, scene.frame_end
    try:
        for other in selected:
            other.select_set(False)
        obj.select_set(True)
        view_layer.objects.active = obj
        scene.frame_start, scene.frame_end = frames
        if fmt == "USD":
            result = bpy.ops.wm.usd_export(filepath=str(target), selected_objects_only=True, export_animation=True,
                                           export_materials=False, export_uvmaps=False, export_normals=True)
        else:
            result = bpy.ops.wm.alembic_export(filepath=str(target), selected=True, start=frames[0], end=frames[1],
                                               export_hair=False, export_particles=False)
        if result != {'FINISHED'}:
            raise RuntimeError(f"{fmt} export returned {sorted(result)}")
    finally:
        scene.frame_start, scene.frame_end = start, end
        obj.select_set(False)
        for other in selected:
            other.select_set(True)
        view_layer.objects.active = active
        scene.frame_set(frame)
    return target
