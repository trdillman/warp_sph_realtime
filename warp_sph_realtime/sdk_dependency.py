"""Explicit, lazy admission of the byte-pinned private shared SDK candidate.

This does not install packages, alter sys.path, or activate a GPU backend. The
normal WarpSPH route never imports this dependency unless explicitly selected.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

_LOCK = Path(__file__).with_name("interop_sdk_lock.json")


def read_lock():
    return json.loads(_LOCK.read_text(encoding="utf-8"))


def verify_payload(package_root, expected):
    root = Path(package_root).resolve()
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Shared SDK candidate identity mismatch: missing {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Shared SDK candidate hash mismatch: {relative}")


def require_sdk():
    lock = read_lock()
    spec = importlib.util.find_spec("blender_vulkan_interop")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "Shared SDK is not installed. This opt-in requires the pinned development wheel; "
            "disable Shared SDK Output to use the existing WarpSPH renderer."
        )
    root = Path(spec.origin).resolve().parent
    verify_payload(root, lock["payload_sha256"])
    module = importlib.import_module("blender_vulkan_interop")
    if Path(module.__file__).resolve().parent != root:
        raise RuntimeError("Shared SDK import identity changed during activation")
    return module


def validate_environment():
    require_sdk()
    import bpy
    import gpu

    lock = read_lock()
    if bpy.app.build_hash.decode() != lock["blender_build"] or gpu.platform.backend_type_get() != lock["backend"]:
        raise RuntimeError("Shared SDK Output currently requires the qualified Blender build and Vulkan backend")
