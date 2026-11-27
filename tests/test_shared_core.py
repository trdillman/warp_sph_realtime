"""The shared Vulkan transport core must stay identical in the add-on and the private SDK."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = next(p for p in (ROOT / "tools" / "check_shared_core.py", ROOT / "scripts" / "check_shared_core.py") if p.is_file())
_spec = importlib.util.spec_from_file_location("check_shared_core", CHECKER)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def test_this_repository_matches_the_shared_core_manifest():
    assert checker.check(ROOT, None) == []


def test_manifest_names_a_version_and_its_history():
    manifest = checker.load_manifest(ROOT)
    assert manifest["history"][0]["version"] == manifest["version"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_windows_line_endings_do_not_count_as_drift(tmp_path):
    copy = _copy_repo_files(tmp_path)
    for entry in checker.load_manifest(copy)["files"]:
        path = copy / entry[checker.side_of(copy)]
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert checker.check(copy, None) == []


def test_editing_a_shared_file_in_one_repository_is_reported(tmp_path):
    copy = _copy_repo_files(tmp_path)
    entry = checker.load_manifest(copy)["files"][0]
    path = copy / entry[checker.side_of(copy)]
    path.write_bytes(path.read_bytes() + b"# drift\n")
    assert any("differs from SHARED_CORE.json" in line for line in checker.check(copy, None))


def test_repositories_on_different_shared_versions_are_reported(tmp_path):
    here, there = _copy_repo_files(tmp_path / "a"), _copy_repo_files(tmp_path / "b")
    manifest = json.loads((there / checker.MANIFEST).read_text(encoding="utf-8"))
    manifest["version"] = "9.9"
    (there / checker.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert any("update both repositories together" in line for line in checker.check(here, there))


def _copy_repo_files(dest: Path) -> Path:
    """Minimal copy of this repository holding only the shared files and the manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest = checker.load_manifest(ROOT)
    side = checker.side_of(ROOT)
    for relative in [checker.MANIFEST, *(e[side] for e in manifest["files"])]:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    # side_of() needs the marker file of this repository kind
    marker = "src/blender_vulkan_interop/__init__.py" if side == "sdk" else "warp_sph_realtime/render/transport.py"
    (dest / marker).parent.mkdir(parents=True, exist_ok=True)
    if not (dest / marker).exists():
        (dest / marker).write_text("", encoding="utf-8")
    return dest


@pytest.mark.private_sdk
def test_pinned_sdk_is_on_the_same_shared_core():
    """The submodule pin must carry the same shared core version and bytes as this repository."""
    sdk = ROOT / "vendor" / "blender-vulkan-interop"
    assert checker.check(ROOT, sdk) == []
