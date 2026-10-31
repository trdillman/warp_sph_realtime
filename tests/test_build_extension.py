"""The built extension must not ship the experimental package (no Blender required)."""

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_extension", ROOT / "scripts" / "build_extension.py")
assert _spec is not None and _spec.loader is not None
build_extension = importlib.util.module_from_spec(_spec)
sys.modules["build_extension"] = build_extension
_spec.loader.exec_module(build_extension)


def test_experimental_is_not_shipped():
    files = build_extension.shipped_files()
    assert not [f for f in files if f.startswith("experimental/")]
    assert (ROOT / "warp_sph_realtime" / "experimental" / "__init__.py").exists()


def test_runtime_modules_are_shipped():
    files = build_extension.shipped_files()
    for required in ("blender_manifest.toml", "__init__.py", "addon.py", "sim/sim_core.py", "render/gpu_points.py"):
        assert required in files


def test_exclusion_rules():
    patterns = ["__pycache__/", ".*", "/experimental/"]
    assert build_extension.is_excluded("experimental/ipbf_solver.py", patterns)
    assert build_extension.is_excluded("sim/__pycache__/x.pyc", patterns)
    assert build_extension.is_excluded(".hidden", patterns)
    assert not build_extension.is_excluded("sim/experimental_note.py", patterns)
    assert not build_extension.is_excluded("sim/experimental/x.py", patterns)


def test_no_runtime_module_imports_experimental():
    for path in (ROOT / "warp_sph_realtime").rglob("*.py"):
        if "experimental" in path.relative_to(ROOT / "warp_sph_realtime").parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            else:
                continue
            assert not any("experimental" in name.split(".") for name in names), (path, node.lineno)


def test_licence_files_are_shipped():
    files = build_extension.shipped_files()
    assert "LICENSE" in files and "THIRD_PARTY_NOTICES.md" in files
    manifest = build_extension.tomllib.loads((ROOT / "warp_sph_realtime" / "blender_manifest.toml").read_text())
    assert manifest["license"] == ["SPDX:GPL-3.0-or-later"]


def test_release_requires_a_license_file(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="LICENSE"):
        build_extension.check_license(tmp_path)
    (tmp_path / "LICENSE").write_text("terms")
    build_extension.check_license(tmp_path)  # no exit
