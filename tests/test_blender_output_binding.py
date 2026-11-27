import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class _Modifier(dict):
    def __init__(self, tree):
        super().__init__()
        self.node_group = tree


def test_existing_mesh_preset_rebinds_replaced_point_cloud(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "warp_sph_realtime" / "cache" / "blender_output.py"
    name = "warp_sph_realtime.cache._blender_output_binding_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"bpy": ModuleType("bpy")}):
        spec.loader.exec_module(module)

    tree = SimpleNamespace(interface=SimpleNamespace(items_tree=[SimpleNamespace(
        item_type="SOCKET", in_out="INPUT", name="Point Cloud", identifier="cloud_input"
    )]))
    modifier = _Modifier(tree)
    modifier["cloud_input"] = object()
    obj = SimpleNamespace(modifiers={module.MODIFIER_NAME: modifier}, update_tag=lambda: None)
    scene = SimpleNamespace(objects=[obj])
    replacement = SimpleNamespace(hide_render=False)
    monkeypatch.setattr(module, "find_mesh_object", lambda _scene: obj)
    monkeypatch.setattr(module, "build_meshing_group", lambda _defaults: tree)

    module.ensure_mesh_object(scene, replacement, defaults=None)

    assert modifier["cloud_input"] is replacement
    assert replacement.hide_render is True
