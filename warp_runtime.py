"""Fail-closed runtime policy for WarpSPH's CUDA 12.9 qualification lane.

Importing this module never imports NVIDIA Warp and never touches Blender's Python.
The Python wheel and CUDA/OpenGL context must pass separate gates.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

CERTIFIED_WARP_VERSION = "1.17.0"
CERTIFIED_CUDA_MAJOR = 12


@dataclass(frozen=True)
class RuntimeStatus:
    available: bool
    message: str
    version: str = ""
    restart_required: bool = False


def warp_version_support(version: str) -> tuple[str, str]:
    """No floating minimum: newer releases (notably CUDA 13) are not qualified."""
    if str(version) == CERTIFIED_WARP_VERSION:
        return "certified", ""
    return "unsupported", (
        f"Warp {version or 'unknown'} is not certified for WarpSPH. "
        f"Use the locked Warp {CERTIFIED_WARP_VERSION} CUDA {CERTIFIED_CUDA_MAJOR} wheel."
    )


def _installer():
    try:
        from . import warp_installer
    except ImportError:
        import warp_installer
    return warp_installer


def status() -> RuntimeStatus:
    """Read-only UI state; does not load any native library."""
    installer = _installer()
    try:
        blocked = installer.foreign_warp_block_reason()
        if blocked:
            return RuntimeStatus(False, blocked)
        target = installer._default_install_target(create=False)
        pending = installer._pending_repair_path(target)
        if pending.is_file():
            return RuntimeStatus(False, "Warp repair staged; restart Blender to apply it", restart_required=True)
        version = installer._owned_distribution_version(target)
        if version is None:
            return RuntimeStatus(False, "NVIDIA Warp is not installed for WarpSPH")
        support, reason = warp_version_support(version)
        if support != "certified":
            return RuntimeStatus(False, reason, version=version)
        if not installer._is_managed(target):
            return RuntimeStatus(False, "Unmanaged Warp files; preserved but unavailable", version=version)
        return RuntimeStatus(True, f"Warp {version} CUDA 12 installed; CUDA/OpenGL checked on Start", version=version)
    except Exception as exc:
        return RuntimeStatus(False, f"WarpSPH runtime inspection failed: {exc}")


def require_cuda_runtime():
    """Verify owned distribution, imported module origin and CUDA availability.

    Deliberately runs only when Start is explicitly requested. In particular, an
    arbitrary 'warp' installed into Blender Python is never silently accepted.
    """
    installer = _installer()
    blocked = installer.foreign_warp_block_reason()
    if blocked:
        raise RuntimeError(blocked)
    installer.ensure_installed_warp_on_path()
    state = status()
    if not state.available:
        raise RuntimeError(state.message)
    importlib.invalidate_caches()
    wp = importlib.import_module("warp")
    file_name = getattr(wp, "__file__", None)
    site = installer._default_install_target(create=False)
    if not file_name or not Path(file_name).resolve().is_relative_to(site.resolve()):
        raise RuntimeError("Warp was imported from outside WarpSPH's owned directory; restart Blender")
    version = str(getattr(wp, "__version__", getattr(getattr(wp, "config", None), "version", "")))
    support, reason = warp_version_support(version)
    if support != "certified":
        raise RuntimeError(reason)
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA is unavailable: check NVIDIA GPU, driver and Warp CUDA 12.9 runtime")
    try:
        device = wp.get_device("cuda:0")
    except Exception as exc:
        raise RuntimeError(f"CUDA device cuda:0 is unavailable: {exc}") from exc
    if not device.is_cuda:
        raise RuntimeError("WarpSPH requires CUDA device cuda:0")
    return wp
