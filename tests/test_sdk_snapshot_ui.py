"""Snapshot/native-GN transaction rollback and UI integration contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from warp_sph_realtime import sdk_snapshot as module


class Staged:

    def __init__(self):
        self.committed = False
        self.rolled = False
        self.final = False

    def commit(self):
        self.committed = True
        return (NS(kind='OBJECT', name='out'),)

    def rollback(self):
        self.rolled = True
        self.committed = False

    def finalize(self):
        self.final = True

def test_geometry_nodes_failure_keeps_native_commit_reversible():
    assert hasattr(module, '_NodeOutput'), 'native/GN atomic output wrapper missing'
    staged = Staged()
    obj = NS(modifiers=[])
    bpy = NS(data=NS(objects={'out': obj}))

    def fail(_obj):
        raise RuntimeError('GN allocation failed')
    wrapped = module._NodeOutput(staged, bpy, fail)
    with pytest.raises(RuntimeError):
        wrapped.commit()
    wrapped.rollback()
    wrapped.rollback()
    assert staged.rolled and (not staged.final)

def test_existing_modifier_is_reused_not_duplicated():
    assert hasattr(module, '_NodeOutput'), 'native/GN atomic output wrapper missing'
    staged = Staged()
    group = {'interop_generated': True}
    obj = NS(modifiers=[NS(type='NODES', node_group=group)])
    wrapped = module._NodeOutput(staged, NS(data=NS(objects={'out': obj})), lambda obj: pytest.fail('existing GN rebuilt'))
    refs = wrapped.commit()
    wrapped.finalize()
    assert refs[0].name == 'out' and staged.final

def test_foreign_snapshot_target_rejected_before_replacement():
    assert hasattr(module, 'validate_target_owner'), 'consumer-owned replacement validation missing'
    obj = NS(type='MESH', data={'interop_snapshot': True, 'interop_source': 'another-producer'})
    with pytest.raises(ValueError):
        module.validate_target_owner(obj, 'warpsph:44')
    obj.data['interop_source'] = 'warpsph:44'
    module.validate_target_owner(obj, 'warpsph:44')
    obj.data['interop_snapshot'] = False
    with pytest.raises(ValueError):
        module.validate_target_owner(obj, 'warpsph:44')

def test_ui_module_and_registration_are_present_without_live_tick_modification():
    root = Path(__file__).resolve().parents[1] / 'warp_sph_realtime'
    ui = root / 'sdk_snapshot_ui.py'
    assert ui.exists(), 'native snapshot operator missing'
    source = ui.read_text()
    tree = ast.parse(source)
    assert 'wm.warp_sph_snapshot_native' in source
    assert any(isinstance(n, ast.ClassDef) and n.name == 'WM_OT_warp_sph_snapshot_native' for n in tree.body)
    assert 'sdk_snapshot_ui.register()' in (root / 'cache_ui.py').read_text()
    assert 'sdk_snapshot_ui.unregister()' in (root / 'cache_ui.py').read_text()
    assert 'sdk_snapshot' not in (root / 'sim_session.py').read_text(), 'capture added to live loop'
