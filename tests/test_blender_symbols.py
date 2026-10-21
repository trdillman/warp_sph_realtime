"""CPU-only preflight tests: no Blender, native function binding or pointer reads."""
import copy
import hashlib
import json
import struct
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from warp_sph_realtime.render import blender_symbols as bs


def pe_bytes(profile):
    data = bytearray(2048)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 60, 128)
    data[128:132] = b'PE\0\0'
    ident = profile['identity']
    struct.pack_into('<HHIIIHH', data, 132, 0x8664, 2, ident['pe_timestamp'], 0, 0, 240, 0)
    struct.pack_into('<H', data, 152, 0x20b)
    struct.pack_into('<I', data, 208, ident['image_size'])
    struct.pack_into('<I', data, 260, 16)
    struct.pack_into('<II', data, 312, 0x3000, 28)
    struct.pack_into('<8sIIIIIIHHI', data, 392, b'.text', 0x2000, 0x1000, 512, 512, 0, 0, 0, 0, 0x60000020)
    struct.pack_into('<8sIIIIIIHHI', data, 432, b'.data', 0x2000, 0x3000, 1024, 1024, 0, 0, 0, 0, 0xc0000040)
    record = b'RSDS' + uuid.UUID(ident['pdb_guid']).bytes_le + struct.pack('<I', ident['pdb_age']) + b'blender.pdb\0'
    struct.pack_into('<IIHHIIII', data, 1024, 0, 0, 0, 0, 2, len(record), 0x3040, 1088)
    data[1088:1088 + len(record)] = record
    return bytes(data)


def pdb_bytes(guid, age):
    data = bytearray(2560)
    data[:32] = b'Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0'
    struct.pack_into('<6I', data, 32, 512, 1, 5, 16, 0, 2)
    struct.pack_into('<I', data, 1024, 3)
    struct.pack_into('<4I', data, 1536, 2, 0xffffffff, 28, 4)
    struct.pack_into('<3I', data, 2048, 20000404, 0, age)
    data[2060:2076] = uuid.UUID(guid).bytes_le
    return bytes(data)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    profile = json.loads(bs.PROFILE_PATH.read_text())
    # Synthetic PE image keeps the full production profile schema but uses small RVAs.
    for index, symbol in enumerate(profile['symbols'].values()):
        symbol['rva'] = (0x1000 if symbol['kind'] == 'callable' else 0x3100) + index * 8
    exe = tmp_path / 'blender.exe'
    pdb = tmp_path / 'blender.pdb'
    exe.write_bytes(pe_bytes(profile))
    pdb.write_bytes(pdb_bytes(profile['identity']['pdb_guid'], profile['identity']['pdb_age']))
    profile['identity']['exe_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    profile['identity']['pdb_sha256'] = hashlib.sha256(pdb.read_bytes()).hexdigest()
    path = tmp_path / 'profile.json'
    raw = json.dumps(profile).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(bs, 'PROFILE_PATH', path)
    monkeypatch.setattr(bs, '_PROFILE_SHA256', hashlib.sha256(raw).hexdigest())
    calls = []

    def runner(request, timeout):
        calls.append((request, timeout))
        return {'schema_version': 1, 'symbols': {s['name']: s['rva'] for s in profile['symbols'].values()}}

    def resolve(**kwargs):
        kwargs.setdefault('verify_debug_symbols', True)
        return bs.resolve_symbols(exe, pdb, module_base=0x180000000,
                                  build_hash=profile['identity']['build_hash'], python_abi='3.13',
                                  helper_runner=runner, **kwargs)
    return profile, path, exe, pdb, calls, resolve


def test_rebase_and_immutable_result(candidate):
    profile, _, _, _, calls, resolve = candidate
    result = resolve()
    assert result.available, result
    assert len(calls) == 1
    resolved = result.resolved
    assert resolved.callables['vkWaitSemaphores'] == 0x180000000 + profile['symbols']['vkWaitSemaphores']['rva']
    assert bs.rebase_rvas({'a': 0x123}, 0x70000000)['a'] == 0x70000123
    with pytest.raises(TypeError):
        resolved.layouts['BPyGPUTexture']['offsets']['texture'] = 24
    with pytest.raises(TypeError):
        resolved.callables['wrong'] = 1
    assert resolved.metadata['abi_version'] == profile['abi_version']
    assert resolved.metadata['calling_convention'] == profile['calling_convention']
    assert dict(resolved.metadata['allocation']) == profile['allocation']
    assert resolved.metadata['provenance']['source_commit'] == profile['provenance']['source_commit']
    with pytest.raises(TypeError):
        resolved.metadata['allocation']['map_offset'] = 1
    with pytest.raises(TypeError):
        resolved.metadata['provenance']['evidence'][0] = 'changed'


@pytest.mark.parametrize('field,value', [('layouts', {}), ('enums', {}), ('signatures', {}), ('schema_version', 2)])
def test_profile_mutation_rejected_before_helper(candidate, field, value):
    profile, path, _, _, calls, resolve = candidate
    profile[field] = value
    path.write_text(json.dumps(profile))
    assert resolve().reason == 'profile_integrity'
    assert calls == []


def test_unknown_binary_and_build_rejected(candidate):
    _, _, exe, pdb, calls, resolve = candidate
    assert bs.resolve_symbols(exe, pdb, module_base=1, build_hash='unknown', python_abi='3.13').reason == 'unknown_profile'
    exe.write_bytes(exe.read_bytes() + b'changed')
    assert resolve().reason == 'unknown_profile'
    assert calls == []


@pytest.mark.parametrize('change', ['missing', 'guid', 'age', 'malformed'])
def test_pdb_rejection_precedes_helper(candidate, change):
    profile, _, _, pdb, calls, resolve = candidate
    if change == 'missing':
        pdb.unlink()
    elif change == 'malformed':
        pdb.write_bytes(b'broken')
    else:
        pdb.write_bytes(pdb_bytes(str(uuid.uuid4()) if change == 'guid' else profile['identity']['pdb_guid'],
                                  2 if change == 'age' else 1))
    assert not resolve().available
    assert calls == []


@pytest.mark.parametrize('mutation', ['schema', 'bool', 'out_of_bounds', 'wrong_in_range', 'stale', 'absolute', 'extra'])
def test_corrupt_cache_resolves_once_and_never_trusts_wrong_rva(candidate, tmp_path, mutation):
    profile, _, _, _, calls, resolve = candidate
    cache = tmp_path / 'cache.json'
    assert resolve(cache_path=cache).available
    original = json.loads(cache.read_text())
    poisoned = copy.deepcopy(original)
    key = next(iter(poisoned['rvas']))
    if mutation == 'schema':
        poisoned['schema_version'] = 999
    elif mutation == 'bool':
        poisoned['rvas'][key] = True
    elif mutation == 'out_of_bounds':
        poisoned['rvas'][key] = profile['identity']['image_size'] + 4
    elif mutation == 'wrong_in_range':
        poisoned['rvas'][key] += 1
    elif mutation == 'stale':
        poisoned['identity']['build_hash'] = 'stale'
    elif mutation == 'absolute':
        poisoned['rvas'][key] += 0x180000000
    else:
        poisoned['address'] = 0x180001000
    cache.write_text(json.dumps(poisoned))
    calls.clear()
    assert resolve(cache_path=cache).available
    assert len(calls) == 1
    assert json.loads(cache.read_text()) == original


def test_cache_rebases_without_helper_but_requires_pdb(candidate, tmp_path):
    profile, _, exe, pdb, calls, resolve = candidate
    cache = tmp_path / 'cache.json'
    assert resolve(cache_path=cache).available
    calls.clear()
    result = bs.resolve_symbols(exe, pdb, module_base=0x220000000, verify_debug_symbols=True,
                                build_hash=profile['identity']['build_hash'], python_abi='3.13', cache_path=cache)
    assert result.available
    assert result.resolved.module_base == 0x220000000
    assert calls == []
    assert 'module_base' not in cache.read_text()
    pdb.unlink()
    assert not resolve(cache_path=cache).available


def test_helper_timeout_and_wrong_symbol_fail_closed(candidate):
    profile, _, exe, pdb, _, _ = candidate
    kwargs = dict(module_base=0x180000000, build_hash=profile['identity']['build_hash'], python_abi='3.13',
                  verify_debug_symbols=True)
    def timeout(request, seconds):
        raise subprocess.TimeoutExpired('isolated-helper', seconds)
    assert bs.resolve_symbols(exe, pdb, helper_runner=timeout, **kwargs).reason == 'helper_timeout'
    def wrong(request, seconds):
        return {'schema_version': 1, 'symbols': {s['name']: s['rva'] + 1 for s in profile['symbols'].values()}}
    assert bs.resolve_symbols(exe, pdb, helper_runner=wrong, **kwargs).reason == 'helper_result'


def test_linux_stub_does_not_import_blender(monkeypatch):
    monkeypatch.setattr(bs.sys, 'platform', 'linux')
    assert bs.resolve_current_blender().reason == 'platform_unavailable'


def test_no_gpu_binding_or_pointer_extraction_in_resolver():
    import ast
    tree = ast.parse(Path(bs.__file__).read_text())
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert names.isdisjoint({'from_address', 'CFUNCTYPE', 'WINFUNCTYPE', 'backend_type_get', 'GPUTexture'})


def test_cache_identity_bool_is_not_an_integer(candidate, tmp_path):
    _, _, _, _, calls, resolve = candidate
    cache = tmp_path / 'cache.json'
    assert resolve(cache_path=cache).available
    value = json.loads(cache.read_text())
    value['identity']['pdb_age'] = True
    cache.write_text(json.dumps(value))
    calls.clear()
    assert resolve(cache_path=cache).available
    assert len(calls) == 1


@pytest.mark.parametrize('verify', [True, False])
@pytest.mark.parametrize('kind', ['nonexec_callable', 'executable_data', 'duplicate_rsds', 'truncated', 'wrong_machine'])
def test_pe_validation_before_helper(candidate, monkeypatch, kind, verify):
    profile, path, exe, _, calls, resolve = candidate
    data = bytearray(exe.read_bytes())
    if kind == 'nonexec_callable':
        struct.pack_into('<I', data, 428, 0x40000040)
    elif kind == 'executable_data':
        struct.pack_into('<I', data, 468, 0x60000020)
    elif kind == 'duplicate_rsds':
        struct.pack_into('<I', data, 316, 56)
        data[1052:1080] = data[1024:1052]
    elif kind == 'truncated':
        data = data[:130]
    else:
        struct.pack_into('<H', data, 132, 0x14c)
    exe.write_bytes(data)
    profile['identity']['exe_sha256'] = hashlib.sha256(data).hexdigest()
    raw = json.dumps(profile).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(bs, '_PROFILE_SHA256', hashlib.sha256(raw).hexdigest())
    assert not resolve(verify_debug_symbols=verify).available
    assert calls == []


def test_profile_authority_needs_no_pdb_helper_or_cache(candidate, tmp_path):
    profile, _, exe, pdb, calls, _ = candidate
    pdb.unlink()
    cache = tmp_path / 'cache.json'
    result = bs.resolve_symbols(exe, module_base=0x7ff700000000, build_hash=profile['identity']['build_hash'],
                                python_abi='3.13', cache_path=cache, helper_runner=lambda *a: calls.append(a))
    assert result.available and result.detail == 'profile'
    assert calls == [] and not cache.exists()
    rva = profile['symbols']['GPU_pixel_buffer_create']['rva']
    assert result.resolved.callables['GPU_pixel_buffer_create'] == 0x7ff700000000 + rva
    assert result.resolved.data.keys() == {k for k, v in profile['symbols'].items() if v['kind'] == 'data'}


@pytest.mark.parametrize('change', ['exe', 'pe_identity', 'module_base'])
def test_profile_authority_still_rejects_identity(candidate, monkeypatch, change):
    profile, path, exe, _, calls, resolve = candidate
    kwargs = {}
    if change == 'exe':
        exe.write_bytes(exe.read_bytes() + b'x')
    elif change == 'pe_identity':
        profile['identity']['pe_timestamp'] += 1
        raw = json.dumps(profile).encode()
        path.write_bytes(raw)
        monkeypatch.setattr(bs, '_PROFILE_SHA256', hashlib.sha256(raw).hexdigest())
    else:
        kwargs['module_base'] = 0
    result = bs.resolve_symbols(exe, build_hash=profile['identity']['build_hash'], python_abi='3.13',
                                **{'module_base': 0x180000000, **kwargs})
    assert not result.available and result.resolved is None
    assert calls == []


def test_verification_without_pdb_path_is_missing(candidate):
    profile, _, exe, _, calls, _ = candidate
    result = bs.resolve_symbols(exe, module_base=0x180000000, build_hash=profile['identity']['build_hash'],
                                python_abi='3.13', verify_debug_symbols=True)
    assert result.reason == 'pdb_missing'


def test_helper_has_isolated_local_environment(candidate, monkeypatch):
    _, _, exe, _, _, _ = candidate
    seen = {}
    monkeypatch.setenv('_NT_SYMBOL_PATH', 'srv*https://invalid.example')
    monkeypatch.setenv('_NT_ALT_SYMBOL_PATH', 'srv*https://invalid.example')
    monkeypatch.setenv('PYTHONPATH', 'untrusted')
    monkeypatch.setenv('DBGHELP_HOMEDIR', 'untrusted')
    monkeypatch.setattr(bs, '_sha256', lambda path: bs.load_profile()['helper']['python_sha256'])
    def run(args, **kwargs):
        seen.update(args=args, **kwargs)
        return SimpleNamespace(returncode=0, stdout='{}', stderr='')
    monkeypatch.setattr(bs.subprocess, 'run', run)
    bs._run_helper({'executable': str(exe)}, 90.0)
    assert '-I' in seen['args']
    assert Path(seen['args'][0]).name == 'python.exe'
    assert seen['timeout'] == 90.0
    if sys.platform == 'win32':  # the console-suppression flag only exists on Windows
        assert seen['creationflags'] == subprocess.CREATE_NO_WINDOW
    assert not any(key in seen['env'] for key in ['_NT_SYMBOL_PATH', '_NT_ALT_SYMBOL_PATH', 'PYTHONPATH', 'DBGHELP_HOMEDIR'])


def test_cache_and_helper_source_are_reported(candidate, tmp_path):
    _, _, _, _, _, resolve = candidate
    cache = tmp_path / 'cache.json'
    assert resolve(cache_path=cache).detail == 'helper'
    assert resolve(cache_path=cache).detail == 'cache'


def test_oversized_cache_is_discarded(candidate, tmp_path):
    _, _, _, _, calls, resolve = candidate
    cache = tmp_path / 'cache.json'
    cache.write_bytes(b' ' * (1024 * 1024 + 1))
    assert resolve(cache_path=cache).available
    assert len(calls) == 1


def test_dbghelp_name_length_can_include_terminating_nul():
    from warp_sph_realtime.render import blender_symbols_helper as helper
    name = '?GPU_pixel_buffer_create@blender@@YAPEAUGPUPixelBuffer@1@_K@Z'
    assert helper._decorated_name(name + '\0') == name
    assert helper._decorated_name(name) == name
    with pytest.raises(ValueError):
        helper._decorated_name(name + '\0unexpected')


def test_deeply_nested_cache_resolves_once(candidate, tmp_path):
    _, _, _, _, calls, resolve = candidate
    cache = tmp_path / 'cache.json'
    cache.write_bytes(b'[' * 20000 + b'0' + b']' * 20000)
    assert resolve(cache_path=cache).available
    assert len(calls) == 1


def test_deeply_nested_helper_output_is_unavailable(candidate):
    profile, _, exe, pdb, _, _ = candidate
    calls = []
    def helper(request, timeout):
        calls.append(request)
        return bs._json_bytes(b'[' * 20000 + b'0' + b']' * 20000)
    result = bs.resolve_symbols(exe, pdb, module_base=0x180000000, verify_debug_symbols=True,
                                build_hash=profile['identity']['build_hash'], python_abi='3.13',
                                helper_runner=helper)
    assert not result.available
    assert result.reason == 'invalid_inputs'
    assert len(calls) == 1
