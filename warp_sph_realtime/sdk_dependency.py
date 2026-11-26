"""Explicit, lazy admission of the byte-pinned private shared SDK candidate.

This does not install packages, alter sys.path, or activate a GPU backend. The
normal WarpSPH route never imports this dependency unless explicitly selected.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import sdk_bundle

_LOCK = Path(__file__).with_name("interop_sdk_lock.json")


def read_lock():
    return json.loads(_LOCK.read_text(encoding="utf-8"))


def verify_payload(package_root, expected):
    root = Path(package_root).resolve()
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Shared SDK candidate identity mismatch: missing {relative}")
        if sdk_bundle.payload_sha256(path) != digest:
            raise RuntimeError(f"Shared SDK candidate hash mismatch: {relative}")


def require_sdk():
    lock = read_lock()
    root = sdk_bundle.locate()
    if root is None:
        raise RuntimeError(
            "Shared SDK is not installed. This opt-in requires the pinned SDK, bundled in a private add-on build or installed separately; "
            "disable Shared SDK Output to use the existing WarpSPH renderer."
        )
    verify_payload(root, lock["payload_sha256"])
    return sdk_bundle.load(root)


def validate_environment():
    require_sdk()
    import bpy
    import gpu

    lock = read_lock()
    if bpy.app.build_hash.decode() != lock["blender_build"] or gpu.platform.backend_type_get() != lock["backend"]:
        raise RuntimeError("Shared SDK Output currently requires the qualified Blender build and Vulkan backend")
