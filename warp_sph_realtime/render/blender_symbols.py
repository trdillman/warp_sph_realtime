"""Fail-closed, CPU-only exact-build symbol preflight.

This module returns addresses, never binds/calls Blender functions or extracts GPU
objects. The shipped profile digest below is independent of the mutable RVA cache.
The executable SHA-256 pins the profile's RVAs; PDB/DbgHelp checks are optional.
Changing a profile requires review and a source-code anchor update.
"""
from collections.abc import Callable, Mapping
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import MappingProxyType
from typing import Any
import uuid

PROFILE_PATH = Path(__file__).with_name('profiles') / 'blender-5.2.0-fbe6228777e7-windows-amd64-cp313-v1.json'
_PROFILE_SHA256 = 'ad33ae7ce58b1744a4af821d6dbe462ad9518ffce7b96cab7cb3a6d819577ad3'
_MAX_JSON = 1024 * 1024
_MSF_MAGIC = b'Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0'


class PreflightError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class ResolvedProfile:
    profile_id: str
    identity: Mapping[str, Any]
    module_base: int
    callables: Mapping[str, int]
    data: Mapping[str, int]
    layouts: Mapping[str, Any]
    enums: Mapping[str, int]
    signatures: Mapping[str, str]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Resolution:
    available: bool
    reason: str
    detail: str = ''
    resolved: ResolvedProfile | None = None


@dataclass(frozen=True)
class PEIdentity:
    pe_timestamp: int
    image_size: int
    pdb_guid: str
    pdb_age: int
    pdb_name: str
    sections: tuple[tuple[int, int, bool], ...]


def _integer(value: Any, minimum: int = 0, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Expected bounded integer (booleans are invalid)')
    return value


def _sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _json_bytes(raw: bytes) -> Any:
    if len(raw) > _MAX_JSON:
        raise ValueError('JSON exceeds size limit')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique)
    except RecursionError as exc:
        raise ValueError('JSON exceeds supported nesting depth') from exc


def _read_json_file(path: Path) -> Any:
    with path.open('rb') as file:
        return _json_bytes(file.read(_MAX_JSON + 1))


def _same_json(actual: Any, expected: Any) -> bool:
    """Python equality alone incorrectly accepts True/1 and 1.0/1."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_same_json(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def load_profile() -> dict[str, Any]:
    raw = PROFILE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _PROFILE_SHA256:
        raise PreflightError('profile_integrity', 'Packaged ABI profile differs from reviewed code anchor')
    profile = _json_bytes(raw)
    if profile['schema_version'] != 1 or profile['abi_version'] != 1:
        raise PreflightError('profile_integrity', 'Unsupported packaged profile schema')
    return profile


def read_pe_identity(path: Path) -> PEIdentity:
    """Read AMD64 PE32+ sections and the unique CodeView RSDS record, without loading PE."""
    with path.open('rb') as file:
        size = os.fstat(file.fileno()).st_size
        def read(offset, length):
            if offset < 0 or length < 0 or offset + length > size:
                raise ValueError('PE field outside file')
            file.seek(offset)
            result = file.read(length)
            if len(result) != length:
                raise ValueError('Truncated PE')
            return result
        dos = read(0, 64)
        if dos[:2] != b'MZ':
            raise ValueError('Missing DOS signature')
        pe_offset = struct.unpack_from('<I', dos, 60)[0]
        header = read(pe_offset, 24)
        if header[:4] != b'PE\0\0':
            raise ValueError('Missing PE signature')
        machine, count, timestamp, _, _, optional_size, _ = struct.unpack_from('<HHIIIHH', header, 4)
        if machine != 0x8664 or not 1 <= count <= 96 or optional_size < 168:
            raise ValueError('Unsupported PE architecture or header')
        optional = read(pe_offset + 24, optional_size)
        if struct.unpack_from('<H', optional)[0] != 0x20b:
            raise ValueError('Expected PE32+')
        image_size = struct.unpack_from('<I', optional, 56)[0]
        if not image_size or struct.unpack_from('<I', optional, 108)[0] < 7:
            raise ValueError('Missing PE debug directory')
        directory_rva, directory_size = struct.unpack_from('<II', optional, 160)
        sections = []
        raw_sections = []
        for index in range(count):
            row = read(pe_offset + 24 + optional_size + index * 40, 40)
            virtual_size, rva, raw_size, raw_offset = struct.unpack_from('<4I', row, 8)
            flags = struct.unpack_from('<I', row, 36)[0]
            end = rva + max(virtual_size, raw_size)
            if rva >= image_size or end > image_size or raw_offset + raw_size > size:
                raise ValueError('Invalid PE section bounds')
            if any(rva < b and end > a for a, b, _ in sections):
                raise ValueError('Overlapping PE sections')
            sections.append((rva, end, bool(flags & 0x20000000)))
            raw_sections.append((rva, raw_size, raw_offset))
        def file_offset(rva, length):
            for start, raw_size, offset in raw_sections:
                if start <= rva and rva + length <= start + raw_size:
                    return offset + rva - start
            raise ValueError('PE RVA not backed by file bytes')
        if not directory_size or directory_size % 28 or directory_size > 28 * 128:
            raise ValueError('Invalid PE debug directory size')
        debug = read(file_offset(directory_rva, directory_size), directory_size)
        records = []
        for offset in range(0, directory_size, 28):
            _, _, _, _, kind, length, address, pointer = struct.unpack_from('<IIHHIIII', debug, offset)
            if kind != 2:
                continue
            if not 25 <= length <= 4096 or file_offset(address, length) != pointer:
                raise ValueError('Invalid CodeView record')
            record = read(pointer, length)
            if record[:4] != b'RSDS' or record[-1:] != b'\0':
                raise ValueError('Expected terminated RSDS CodeView record')
            guid = str(uuid.UUID(bytes_le=record[4:20]))
            age = struct.unpack_from('<I', record, 20)[0]
            records.append((guid, age, record[24:-1].decode('utf-8', errors='strict')))
        if len(records) != 1:
            raise ValueError('Expected exactly one RSDS identity')
        return PEIdentity(timestamp, image_size, *records[0], tuple(sections))


def read_pdb_identity(path: Path) -> tuple[str, int]:
    """Read MSF 7 PDB info stream only; bounded seeks avoid loading multi-GB debug data."""
    with path.open('rb') as file:
        size = os.fstat(file.fileno()).st_size
        def read(offset, length):
            if offset < 0 or length < 0 or offset + length > size:
                raise ValueError('PDB field outside file')
            file.seek(offset)
            result = file.read(length)
            if len(result) != length:
                raise ValueError('Truncated PDB')
            return result
        header = read(0, 56)
        if header[:32] != _MSF_MAGIC:
            raise ValueError('Expected MSF 7 PDB')
        block, _, blocks, directory_size, reserved, block_map = struct.unpack_from('<6I', header, 32)
        if block not in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
            raise ValueError('Unsupported PDB block size')
        if blocks * block != size or reserved or not 4 <= directory_size <= 16 * 1024 * 1024:
            raise ValueError('Invalid PDB superblock')
        count = (directory_size + block - 1) // block
        if count * 4 > block:
            raise ValueError('Unsupported PDB directory block map')
        ids = struct.unpack('<' + 'I' * count, read(block_map * block, count * 4))
        directory = b''.join(read(index * block, block) for index in ids)[:directory_size]
        streams = struct.unpack_from('<I', directory)[0]
        if not 2 <= streams <= (directory_size - 4) // 4:
            raise ValueError('Invalid PDB stream directory')
        sizes = struct.unpack_from('<' + 'I' * streams, directory, 4)
        cursor = 4 + streams * 4
        info = None
        for index, stream_size in enumerate(sizes):
            count = 0 if stream_size == 0xffffffff else (stream_size + block - 1) // block
            if cursor + count * 4 > directory_size:
                raise ValueError('Invalid PDB stream block list')
            ids = struct.unpack_from('<' + 'I' * count, directory, cursor)
            cursor += count * 4
            if any(value >= blocks for value in ids):
                raise ValueError('Invalid PDB stream block')
            if index == 1:
                if not 28 <= stream_size <= 16 * 1024 * 1024:
                    raise ValueError('Invalid PDB info stream size')
                info = b''.join(read(value * block, block) for value in ids)[:stream_size]
        if info is None or struct.unpack_from('<I', info)[0] != 20000404:
            raise ValueError('Unsupported PDB info stream version')
        return str(uuid.UUID(bytes_le=info[12:28])), struct.unpack_from('<I', info, 8)[0]


def rebase_rvas(rvas: Mapping[str, int], module_base: int) -> Mapping[str, int]:
    base = _integer(module_base, 1)
    return MappingProxyType({key: _integer(base + _integer(rva)) for key, rva in rvas.items()})


def _validate_rvas(rvas: Any, profile: dict, pe: PEIdentity) -> dict[str, int]:
    symbols = profile['symbols']
    if type(rvas) is not dict or set(rvas) != set(symbols):
        raise ValueError('Unexpected RVA schema or missing symbols')
    result = {}
    for key, symbol in symbols.items():
        rva = _integer(rvas[key], 1, pe.image_size - 1)
        if rva != symbol['rva']:
            raise ValueError(f'RVA differs from independently reviewed manifest: {key}')
        sections = [(a, b, executable) for a, b, executable in pe.sections if a <= rva < b]
        if len(sections) != 1 or sections[0][2] != (symbol['kind'] == 'callable'):
            raise ValueError(f'Wrong section permissions: {key}')
        result[key] = rva
    return result


def _run_helper(request: dict, timeout: float) -> dict:
    profile = load_profile()
    executable = Path(request['executable'])
    python = executable.parent / profile['helper']['python_relative']
    if _sha256(python) != profile['helper']['python_sha256']:
        raise PreflightError('helper_unavailable', 'Bundled helper Python differs from reviewed candidate')
    script = Path(__file__).with_name('blender_symbols_helper.py')
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('PYTHON', '_NT_SYMBOL', '_NT_ALT_SYMBOL', 'DBGHELP', 'SYMSRV',
                                                   '_NT_DEBUGGER'))}
    result = subprocess.run([str(python), '-I', str(script)], input=json.dumps(request),
                            capture_output=True, text=True, timeout=timeout, check=False, env=environment, **kwargs)
    if result.returncode:
        raise PreflightError('helper_failed', f'Isolated DbgHelp helper exit {result.returncode}: {result.stderr[:500]}')
    return _json_bytes(result.stdout.encode('utf-8'))


def _resolution(profile: dict, rvas: Mapping[str, int], module_base: int, source: str) -> Resolution:
    addresses = rebase_rvas(rvas, module_base)
    kinds = {key: symbol['kind'] for key, symbol in profile['symbols'].items()}
    resolved = ResolvedProfile(profile['profile_id'], _freeze(profile['identity']), module_base,
                               _freeze({k: v for k, v in addresses.items() if kinds[k] == 'callable'}),
                               _freeze({k: v for k, v in addresses.items() if kinds[k] == 'data'}),
                               _freeze(profile['layouts']), _freeze(profile['enums']), _freeze(profile['signatures']),
                               _freeze(profile))
    return Resolution(True, 'resolved', source, resolved=resolved)


def resolve_symbols(executable: Path | str, pdb_path: Path | str | None = None, *, module_base: int,
                    build_hash: str, python_abi: str, architecture: str = 'AMD64',
                    verify_debug_symbols: bool = False,
                    cache_path: Path | str | None = None,
                    helper_runner: Callable[[dict, float], dict] | None = None,
                    helper_timeout: float = 60.0) -> Resolution:
    """Resolve exact reviewed symbols. Injected helpers are a headless-test seam.

    The returned addresses are only valid for the supplied loaded module base.
    Production callers use resolve_current_blender, which discovers that base from OS.
    By default the shipped profile is the RVA authority once the executable digest and PE
    identity match it (source 'profile'). verify_debug_symbols additionally requires the
    matched PDB and re-resolves through the cache/isolated helper; then invalid caches fall
    back to exactly one isolated resolution and missing PDBs always fail.
    """
    try:
        profile = load_profile()
        identity = profile['identity']
        exe = Path(executable).resolve()
        if (build_hash != identity['build_hash'] or python_abi != identity['python_abi']
                or architecture != identity['architecture'] or _sha256(exe) != identity['exe_sha256']):
            raise PreflightError('unknown_profile', 'Executable, build, architecture or Python ABI is not approved')
        pe = read_pe_identity(exe)
        if (pe.pe_timestamp != identity['pe_timestamp'] or pe.image_size != identity['image_size']
                or pe.pdb_guid != identity['pdb_guid'] or pe.pdb_age != identity['pdb_age']):
            raise PreflightError('pe_identity', 'PE headers or RSDS differ from approved profile')
        rvas = _validate_rvas({key: symbol['rva'] for key, symbol in profile['symbols'].items()}, profile, pe)
        _integer(module_base, 1, (1 << 64) - pe.image_size)
        if not verify_debug_symbols:
            return _resolution(profile, rvas, module_base, 'profile')
        if pdb_path is None:
            raise PreflightError('pdb_missing', 'Debug-symbol verification requires the matching local PDB')
        pdb = Path(pdb_path).resolve()
        if not pdb.is_file():
            raise PreflightError('pdb_missing', 'Matching local PDB is required even with a valid cache')
        if read_pdb_identity(pdb) != (identity['pdb_guid'], identity['pdb_age']):
            raise PreflightError('pdb_identity', 'PDB GUID/age differ from approved profile')
        if _sha256(pdb) != identity['pdb_sha256']:
            raise PreflightError('pdb_identity', 'PDB digest differs from approved candidate')
        if isinstance(helper_timeout, bool) or not isinstance(helper_timeout, (int, float)) or not 0 < helper_timeout <= 120:
            raise PreflightError('invalid_request', 'Helper timeout must be positive and at most 120 seconds')
        cache_identity = {**identity, 'profile_id': profile['profile_id'], 'abi_version': profile['abi_version']}
        cache = Path(cache_path) if cache_path is not None else None
        rvas = None
        source = 'helper'
        if cache is not None:
            try:
                payload = _read_json_file(cache)
                if (type(payload) is not dict or set(payload) != {'schema_version', 'identity', 'rvas'}
                        or type(payload['schema_version']) is not int or payload['schema_version'] != 1
                        or not _same_json(payload['identity'], cache_identity)):
                    raise ValueError('Stale or malformed cache identity')
                rvas = _validate_rvas(payload['rvas'], profile, pe)
                source = 'cache'
            except (OSError, ValueError, KeyError, TypeError):
                rvas = None
        if rvas is None:
            request = {'schema_version': 1, 'executable': str(exe), 'pdb_path': str(pdb),
                       'image_size': pe.image_size, 'names': [s['name'] for s in profile['symbols'].values()]}
            payload = (helper_runner or _run_helper)(request, float(helper_timeout))
            try:
                if (type(payload) is not dict or set(payload) != {'schema_version', 'symbols'}
                        or type(payload['schema_version']) is not int or payload['schema_version'] != 1
                        or type(payload['symbols']) is not dict or set(payload['symbols']) != set(request['names'])):
                    raise ValueError('Wrong helper result schema or decorated symbol names')
                rvas = _validate_rvas({key: payload['symbols'][symbol['name']]
                                       for key, symbol in profile['symbols'].items()}, profile, pe)
            except (ValueError, KeyError, TypeError) as exc:
                raise PreflightError('helper_result', str(exc)) from exc
            if cache is not None:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    # Only identity and module-relative RVAs enter the cache.
                    import tempfile
                    with tempfile.NamedTemporaryFile('w', dir=cache.parent, encoding='utf-8', delete=False) as file:
                        temporary = Path(file.name)
                        json.dump({'schema_version': 1, 'identity': cache_identity, 'rvas': rvas}, file)
                    try:
                        temporary.replace(cache)
                    finally:
                        temporary.unlink(missing_ok=True)
                except OSError:
                    pass  # Cache is optional; validated resolution remains usable.
        return _resolution(profile, rvas, module_base, source)
    except subprocess.TimeoutExpired:
        return Resolution(False, 'helper_timeout', 'Isolated helper exceeded its bounded deadline and was terminated')
    except PreflightError as exc:
        return Resolution(False, exc.reason, str(exc))
    except (OSError, ValueError, KeyError, TypeError, struct.error, OverflowError) as exc:
        return Resolution(False, 'invalid_inputs', f'{type(exc).__name__}: {exc}')


def resolve_current_blender(*, verify_debug_symbols: bool = False, cache_path: Path | str | None = None,
                            helper_timeout: float = 60.0) -> Resolution:
    """Discover the current loaded Windows executable/base; perform no GPU operations."""
    if sys.platform != 'win32':
        return Resolution(False, 'platform_unavailable', 'Linux profiles remain unavailable until M6')
    try:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint32
        base = kernel.GetModuleHandleW(None)
        buffer = ctypes.create_unicode_buffer(32768)
        count = kernel.GetModuleFileNameW(None, buffer, len(buffer))
        if not base or not count or count >= len(buffer):
            return Resolution(False, 'module_unavailable', 'OS could not identify loaded main module')
        import bpy  # Lazy metadata only; headless module import remains independent of Blender.
        build_hash = bpy.app.build_hash.decode('ascii')
        architecture = 'AMD64' if struct.calcsize('P') == 8 else 'unsupported'
        return resolve_symbols(buffer.value, Path(buffer.value).with_suffix('.pdb'), module_base=base,
                               build_hash=build_hash, python_abi=f'{sys.version_info.major}.{sys.version_info.minor}',
                               architecture=architecture, verify_debug_symbols=verify_debug_symbols,
                               cache_path=cache_path, helper_timeout=helper_timeout)
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        return Resolution(False, 'module_unavailable', f'{type(exc).__name__}: {exc}')
