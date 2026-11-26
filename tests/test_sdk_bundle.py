"""The pinned shared SDK is bundled into private builds and loaded from there without sys.path edits."""

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "warp_sph_realtime"
_spec = importlib.util.spec_from_file_location("build_extension_sdk", ROOT / "scripts" / "build_extension.py")
assert _spec is not None and _spec.loader is not None
build_extension = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_extension)


@pytest.mark.private_sdk
def test_vendored_sdk_matches_every_lock_exactly():
    payload = build_extension.check_sdk()
    for name in build_extension.SDK_LOCKS:
        assert payload == json.loads((PACKAGE / name).read_text(encoding="utf-8"))["payload_sha256"]


@pytest.mark.private_sdk
def test_windows_crlf_submodule_is_normalized_to_locked_lf_bytes(tmp_path):
    source = tmp_path / "crlf-sdk"
    import shutil

    shutil.copytree(build_extension.SDK_PACKAGE, source)
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".json"):
            canonical = path.read_bytes().replace(b"\r\n", b"\n")
            path.write_bytes(canonical.replace(b"\n", b"\r\n"))

    expected = build_extension.check_sdk(source)
    staged = build_extension.stage_source(tmp_path / "stage", source)
    bundled = staged / "_vendor" / source.name
    assert build_extension.sdk_payload(bundled) == expected
    assert all(b"\r\n" not in path.read_bytes() for path in bundled.rglob("*")
               if path.is_file() and path.suffix in (".py", ".json"))


def test_submodule_is_pinned_to_the_reconciled_sdk_commit():
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert "path = vendor/blender-vulkan-interop" in gitmodules
    assert "branch = interop/reconcile-20261102" in gitmodules
    staged = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s", "vendor/blender-vulkan-interop"],
                            capture_output=True, text=True)
    if staged.returncode != 0 or not staged.stdout:
        pytest.skip("not a git checkout")
    assert staged.stdout.split()[1].startswith("1b3e4be"), "SDK submodule moved off the locked commit"


@pytest.mark.private_sdk
def test_staged_build_bundles_sdk_and_no_sdk_build_does_not(tmp_path):
    with_sdk = build_extension.stage_source(tmp_path / "a", build_extension.SDK_PACKAGE)
    bundled = with_sdk / "_vendor" / "blender_vulkan_interop"
    assert build_extension.sdk_payload(bundled) == build_extension.check_sdk()
    without = build_extension.stage_source(tmp_path / "b", None)
    assert not (without / "_vendor").exists()


def test_archive_check_rejects_sdk_in_public_build_and_incomplete_bundle(tmp_path):
    expected = json.loads((PACKAGE / build_extension.SDK_LOCKS[0]).read_text(encoding="utf-8"))["payload_sha256"]
    full = [f"_vendor/blender_vulkan_interop/{name}" for name in expected]
    build_extension.check_archive_sdk(full, bundled=True)
    build_extension.check_archive_sdk(["__init__.py"], bundled=False)
    with pytest.raises(SystemExit, match="must not ship"):
        build_extension.check_archive_sdk(full, bundled=False)
    with pytest.raises(SystemExit, match="missing bundled SDK"):
        build_extension.check_archive_sdk(full[1:], bundled=True)


def test_release_build_refuses_to_bundle_the_private_sdk(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_extension.py", "--tag", f"v{build_extension.manifest_version()}"])
    with pytest.raises(SystemExit, match="--no-sdk"):
        build_extension.main()


@pytest.mark.private_sdk
def test_addon_loads_the_bundled_copy_without_path_changes(tmp_path):
    staged = build_extension.stage_source(tmp_path, build_extension.SDK_PACKAGE)
    with zipfile.ZipFile(tmp_path / "addon.zip", "w") as archive:  # what Blender installs: a package folder
        for path in staged.rglob("*"):
            archive.write(path, path.relative_to(tmp_path).as_posix())
    install = tmp_path / "installed"
    zipfile.ZipFile(tmp_path / "addon.zip").extractall(install)
    code = (
        "import sys;sys.path.insert(0," + repr(str(install)) + ");before=list(sys.path)\n"
        "from warp_sph_realtime import sdk_snapshot, sdk_dependency\n"
        "m=sdk_snapshot.require_sdk();assert sdk_dependency.require_sdk() is m\n"
        "from blender_vulkan_interop.output.contracts import SnapshotPayload\n"
        "assert sys.path==before, 'sys.path changed'\n"
        "print(m.__file__);print(sys.modules[SnapshotPayload.__module__].__file__)"
    )
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    bundled = str(install / "warp_sph_realtime" / "_vendor" / "blender_vulkan_interop")
    assert all(line.startswith(bundled) for line in run.stdout.splitlines()), run.stdout
