"""Locate and load the shared Blender Vulkan Interop SDK without editing sys.path.

A private add-on build (``scripts/build_extension.py``) copies the pinned SDK package into
``_vendor/blender_vulkan_interop`` inside the add-on; that bundled copy wins. Without it (public
releases, development checkouts) a separately installed ``blender_vulkan_interop`` is used.

``locate()`` imports nothing, so callers verify the byte lock before ``load()`` executes SDK code.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

NAME = "blender_vulkan_interop"
BUNDLED_ROOT = Path(__file__).with_name("_vendor") / NAME


def bundled_root() -> Path | None:
    root = BUNDLED_ROOT
    return root.resolve() if (root / "__init__.py").is_file() else None


def locate() -> Path | None:
    """Package directory of the SDK candidate: bundled first, then installed. Imports nothing."""
    root = bundled_root()
    if root is not None:
        return root
    spec = importlib.util.find_spec(NAME)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def load(root: Path):
    """Import the SDK package found at ``root`` (already verified by the caller)."""
    root = Path(root).resolve()
    loaded = sys.modules.get(NAME)
    if loaded is not None:
        if Path(loaded.__file__).resolve().parent != root:
            raise RuntimeError("Loaded SDK identity differs from the verified candidate; restart Blender")
        return loaded
    if root == bundled_root():
        spec = importlib.util.spec_from_file_location(NAME, root / "__init__.py", submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None:
            raise RuntimeError("Bundled Shared SDK could not be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in [k for k in sys.modules if k == NAME or k.startswith(NAME + ".")]:
                del sys.modules[key]
            raise
    else:
        module = importlib.import_module(NAME)
    if Path(module.__file__).resolve().parent != root:
        raise RuntimeError("Shared SDK import identity changed during activation")
    return module
