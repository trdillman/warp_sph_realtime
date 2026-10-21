"""List blender.exe symbols matching masks via dbghelp + shipped blender.pdb."""
import ctypes, sys
from ctypes import wintypes

exe = sys.argv[1]
masks = sys.argv[2:]
dbg = ctypes.WinDLL("dbghelp")
proc = wintypes.HANDLE(0x1234)  # pseudo handle; no live process needed
dbg.SymSetOptions(0x2 | 0x4 | 0x10)  # UNDNAME | DEFERRED_LOADS | LOAD_LINES
dbg.SymInitializeW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.BOOL]
import os
assert dbg.SymInitializeW(proc, os.environ["SYMDIR"], False)
dbg.SymLoadModuleExW.restype = ctypes.c_uint64
dbg.SymLoadModuleExW.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                 ctypes.c_uint64, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
BASE = 0x140000000
base = dbg.SymLoadModuleExW(proc, None, exe, None, BASE, 0, None, 0)
assert base, ctypes.GetLastError()


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wintypes.ULONG), ("TypeIndex", wintypes.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                ("Index", wintypes.ULONG), ("Size", wintypes.ULONG), ("ModBase", ctypes.c_uint64),
                ("Flags", wintypes.ULONG), ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64),
                ("Register", wintypes.ULONG), ("Scope", wintypes.ULONG), ("Tag", wintypes.ULONG),
                ("NameLen", wintypes.ULONG), ("MaxNameLen", wintypes.ULONG), ("Name", wintypes.WCHAR * 1)]


CB = ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.POINTER(SYMBOL_INFOW), wintypes.ULONG, ctypes.c_void_p)
dbg.SymEnumSymbolsW.argtypes = [wintypes.HANDLE, ctypes.c_uint64, wintypes.LPCWSTR, CB, ctypes.c_void_p]
for mask in masks:
    hits = []

    def cb(p, size, ctx):
        s = p.contents
        name = ctypes.wstring_at(ctypes.addressof(s) + SYMBOL_INFOW.Name.offset, s.NameLen)
        hits.append((s.Address - base, name))
        return True

    dbg.SymEnumSymbolsW(proc, base, mask, CB(cb), None)
    print(f"== {mask}: {len(hits)}")
    for rva, name in sorted(set(hits), key=lambda x: x[1])[:40]:
        print(f"   rva=0x{rva:08x} {name}")

