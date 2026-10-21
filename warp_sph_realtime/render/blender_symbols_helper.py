"""Isolated DbgHelp-only worker. No imports of Blender, GPU, or add-on runtime.

The parent validates PE/PDB identity and independently compares every returned RVA.
Run under the vetted bundled Python with -I; stdin/stdout are bounded JSON protocol.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys


def _decorated_name(value: str) -> str:
    # DbgHelp's SYMBOL_INFOW.NameLen includes the terminator on this candidate.
    # Remove at most that one terminator; never relax decorated-name equality.
    value = value.removesuffix('\0')
    if '\0' in value:
        raise ValueError('Embedded NUL in DbgHelp decorated symbol')
    return value


def resolve(request):
    if sys.platform != 'win32' or request['schema_version'] != 1:
        raise ValueError('Unsupported helper platform/protocol')
    pdb = Path(request['pdb_path']).resolve(strict=True)
    size = request['image_size']
    names = request['names']
    if (type(size) is not int or not 0 < size < 1 << 32 or type(names) is not list
            or not names or len(names) > 128 or any(type(n) is not str or len(n) > 4096 for n in names)):
        raise ValueError('Invalid helper request')
    # Explicit system path avoids DLL search through the Blender/add-on directory.
    dbg = ctypes.WinDLL(str(Path(os.environ['SystemRoot']) / 'System32' / 'dbghelp.dll'), use_last_error=True)
    handle = wintypes.HANDLE(0x5821)
    dbg.SymSetOptions.argtypes = [wintypes.DWORD]
    dbg.SymSetOptions.restype = wintypes.DWORD
    # Ignore symbol-server environment paths and disable symbol-server autodetection.
    dbg.SymSetOptions(0x4 | 0x10 | 0x200 | 0x80000 | 0x1000 | 0x40000 | 0x02000000)
    dbg.SymInitializeW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.BOOL]
    dbg.SymInitializeW.restype = wintypes.BOOL
    dbg.SymCleanup.argtypes = [wintypes.HANDLE]
    dbg.SymCleanup.restype = wintypes.BOOL
    dbg.SymLoadModuleExW.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                   ctypes.c_uint64, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dbg.SymLoadModuleExW.restype = ctypes.c_uint64
    class Symbol(ctypes.Structure):
        _fields_ = [('SizeOfStruct', wintypes.ULONG), ('TypeIndex', wintypes.ULONG),
                    ('Reserved', ctypes.c_uint64 * 2), ('Index', wintypes.ULONG), ('Size', wintypes.ULONG),
                    ('ModBase', ctypes.c_uint64), ('Flags', wintypes.ULONG), ('Value', ctypes.c_uint64),
                    ('Address', ctypes.c_uint64), ('Register', wintypes.ULONG), ('Scope', wintypes.ULONG),
                    ('Tag', wintypes.ULONG), ('NameLen', wintypes.ULONG), ('MaxNameLen', wintypes.ULONG),
                    ('Name', wintypes.WCHAR * 1)]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.POINTER(Symbol), wintypes.ULONG, ctypes.c_void_p)
    dbg.SymEnumSymbolsW.argtypes = [wintypes.HANDLE, ctypes.c_uint64, wintypes.LPCWSTR, callback_type, ctypes.c_void_p]
    dbg.SymEnumSymbolsW.restype = wintypes.BOOL
    if not dbg.SymInitializeW(handle, str(pdb.parent), False):
        raise OSError(ctypes.get_last_error(), 'SymInitializeW')
    try:
        # Load the exact verified PDB directly, not an auto-selected debug file for an EXE.
        base = dbg.SymLoadModuleExW(handle, None, str(pdb), None, 0x140000000, size, None, 0)
        if not base:
            raise OSError(ctypes.get_last_error(), 'SymLoadModuleExW exact PDB')
        output = {}
        for expected in names:
            hits = []
            def callback(pointer, _size, _context):
                symbol = pointer.contents
                name = _decorated_name(ctypes.wstring_at(ctypes.addressof(symbol) + Symbol.Name.offset, symbol.NameLen))
                if name == expected:
                    hits.append(int(symbol.Address - base))
                return True
            if not dbg.SymEnumSymbolsW(handle, base, expected, callback_type(callback), None):
                raise OSError(ctypes.get_last_error(), 'SymEnumSymbolsW')
            if len(hits) != 1:
                raise ValueError(f'Expected unique exact decorated symbol: {expected}')
            output[expected] = hits[0]
        return {'schema_version': 1, 'symbols': output}
    finally:
        dbg.SymCleanup(handle)


def main():
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('Request too large')
    print(json.dumps(resolve(json.loads(raw))))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
