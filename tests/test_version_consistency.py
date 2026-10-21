"""The extension version lives only in blender_manifest.toml; everything else reads or is checked against it."""

import ast
import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "warp_sph_realtime"
MANIFEST = PACKAGE / "blender_manifest.toml"
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$"
)

_spec = importlib.util.spec_from_file_location("build_extension", ROOT / "scripts" / "build_extension.py")
assert _spec is not None and _spec.loader is not None
build_extension = importlib.util.module_from_spec(_spec)
sys.modules["build_extension"] = build_extension
_spec.loader.exec_module(build_extension)


def manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_version_is_semver():
    assert SEMVER.match(manifest()["version"]), manifest()["version"]


def test_addon_reports_manifest_version():
    """addon imports bpy, so execute only its ``_addon_version`` function, against the real file."""
    source = (PACKAGE / "addon.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_addon_version")
    namespace = {"__file__": str(PACKAGE / "addon.py")}
    exec(ast.get_source_segment(source, node) or "", namespace)
    assert namespace["_addon_version"]() == manifest()["version"]


def test_no_second_version_definition_in_package():
    """Blender 5.x extensions take their version from the manifest; a ``bl_info`` or ``__version__`` would drift."""
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigned = {
            target.id
            for stmt in tree.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        assert not assigned & {"bl_info", "__version__"}, path


def test_no_hardcoded_package_file_names():
    """Anything that names the built zip must derive the version from the manifest."""
    pattern = re.compile(r"warp_sph_realtime-\d+\.\d+\.\d+")
    roots = [ROOT / "tests", ROOT / "scripts", ROOT / ".github", PACKAGE]
    offenders = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and path != Path(__file__).resolve()
        and path.suffix in {".py", ".yml", ".yaml", ".toml", ".md", ".json"}
        and pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not offenders, offenders


def test_zip_name_follows_manifest():
    data = manifest()
    assert build_extension.zip_name() == f"{data['id']}-{data['version']}.zip"


def test_release_tag_must_match_manifest():
    version = manifest()["version"]
    build_extension.check_tag(f"v{version}")
    build_extension.check_tag(f"refs/tags/v{version}")
    for bad in (version, f"v{version}.1", "v9.9.9-nope", ""):
        with pytest.raises(SystemExit):
            build_extension.check_tag(bad)
