"""Validate and build the Blender extension zip into dist/, optionally installing it.

Usage:
    python scripts/build_extension.py [--blender PATH] [--install] [--tag vX.Y.Z] [--no-sdk]

By default the pinned private Blender Vulkan Interop SDK (git submodule ``vendor/blender-vulkan-interop``) is
checked against the add-on's byte locks and bundled as ``_vendor/blender_vulkan_interop`` in the zip. That SDK may
not be redistributed publicly, so public release builds must pass ``--no-sdk``; ``--tag`` enforces this.

``--tag`` (used by the release workflow) refuses to build unless the tag equals ``v`` plus the manifest version
and the repository has a LICENSE file.

Blender is located from --blender, then the BLENDER environment variable, then PATH.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "warp_sph_realtime"
DIST_DIR = REPO_ROOT / "dist"
SDK_CHECKOUT = REPO_ROOT / "vendor" / "blender-vulkan-interop"
SDK_PACKAGE = SDK_CHECKOUT / "src" / "blender_vulkan_interop"
SDK_LOCKS = ("interop_sdk_lock.json", "sdk_snapshot.lock.json")
VENDOR_DIR = "_vendor"


def manifest_version(source_dir: Path = SOURCE_DIR) -> str:
    """The single source of truth for the extension version."""
    with (source_dir / "blender_manifest.toml").open("rb") as handle:
        return str(tomllib.load(handle)["version"])


def zip_name(source_dir: Path = SOURCE_DIR) -> str:
    """File name ``blender extension build`` gives the package (``<id>-<version>.zip``)."""
    with (source_dir / "blender_manifest.toml").open("rb") as handle:
        return f"{tomllib.load(handle)['id']}-{manifest_version(source_dir)}.zip"


def check_tag(tag: str, source_dir: Path = SOURCE_DIR) -> None:
    """Exit unless ``tag`` is ``v`` plus the manifest version."""
    expected = f"v{manifest_version(source_dir)}"
    if tag.removeprefix("refs/tags/") != expected:
        sys.exit(f"Tag {tag!r} does not match blender_manifest.toml version: expected {expected!r}.")


LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")


def check_license(repo_root: Path = REPO_ROOT) -> None:
    """Exit unless the repository carries a licence file: the manifest declares a licence, so a release must ship it."""
    if not any((repo_root / name).is_file() for name in LICENSE_NAMES):
        sys.exit("Release refused: no LICENSE file at the repository root (docs/RELEASE_PLAN.md decision D2).")


def check_archive_license(names: list[str]) -> None:
    """Exit unless the built zip carries a licence file; Blender packages SOURCE_DIR, not the repository root."""
    if not any(name in LICENSE_NAMES for name in names):
        sys.exit("Release refused: the built zip has no LICENSE file (add it inside warp_sph_realtime/).")


def exclude_patterns(source_dir: Path = SOURCE_DIR) -> list[str]:
    """Return the manifest's ``build.paths_exclude_pattern`` list."""
    with (source_dir / "blender_manifest.toml").open("rb") as handle:
        return list(tomllib.load(handle).get("build", {}).get("paths_exclude_pattern", []))


def is_excluded(relative: str, patterns: list[str]) -> bool:
    """Gitignore-style subset used by the manifest: ``dir/``, ``/anchored``, ``*glob``."""
    parts = relative.split("/")
    for pattern in patterns:
        anchored = pattern.startswith("/")
        body = pattern.strip("/")
        if pattern.endswith("/"):
            candidates = parts[:-1]
            if anchored:
                candidates = candidates[:1]
        else:
            candidates = parts[:1] if anchored else parts
        if any(fnmatch(part, body) for part in candidates):
            return True
    return False


def shipped_files(source_dir: Path = SOURCE_DIR) -> list[str]:
    """Package-relative files that the extension build includes, per the manifest exclusions."""
    patterns = exclude_patterns(source_dir)
    names = (path.relative_to(source_dir).as_posix() for path in sorted(source_dir.rglob("*")) if path.is_file())
    return [name for name in names if not is_excluded(name, patterns)]


def sdk_payload(package: Path) -> dict[str, str]:
    """SHA-256 of every .py/.json file in the SDK package, keyed like the add-on's lock files.

    Git for Windows may check the pinned submodule out with CRLF. Hash its canonical LF text so
    the same pinned payload is accepted on every platform; the staged package is normalized too.
    """
    return {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts
    }


def check_sdk(package: Path = SDK_PACKAGE, source_dir: Path = SOURCE_DIR) -> dict[str, str]:
    """Exit unless ``package`` matches every SDK lock exactly (same files, same bytes)."""
    if not (package / "__init__.py").is_file():
        sys.exit(
            f"Shared SDK not found at {package}. Run `git submodule update --init vendor/blender-vulkan-interop` "
            "(needs access to the private repository), or pass --no-sdk to build without it."
        )
    actual = sdk_payload(package)
    for name in SDK_LOCKS:
        expected = json.loads((source_dir / name).read_text(encoding="utf-8"))["payload_sha256"]
        if actual != expected:
            wrong = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
            sys.exit(f"Shared SDK at {package} does not match {name}: {wrong[:5]}")
    return actual


def stage_source(dest: Path, sdk_package: Path | None, source_dir: Path = SOURCE_DIR) -> Path:
    """Copy the add-on into ``dest`` and, unless ``sdk_package`` is None, bundle the verified SDK beside it."""
    staged = dest / source_dir.name
    shutil.copytree(source_dir, staged, ignore=shutil.ignore_patterns("__pycache__", VENDOR_DIR))
    if sdk_package is not None:
        check_sdk(sdk_package, source_dir)
        bundled = staged / VENDOR_DIR / sdk_package.name
        shutil.copytree(sdk_package, bundled, ignore=shutil.ignore_patterns("__pycache__"))
        for path in bundled.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json"):
                content = path.read_bytes()
                normalized = content.replace(b"\r\n", b"\n")
                if normalized != content:
                    path.write_bytes(normalized)
        notice = sdk_package.parents[1] / "DISTRIBUTION.md"
        if notice.is_file():
            shutil.copy2(notice, staged / VENDOR_DIR / "SDK_DISTRIBUTION.md")
    return staged


def check_archive_sdk(names: list[str], bundled: bool, source_dir: Path = SOURCE_DIR) -> None:
    """Exit unless the zip carries exactly the locked SDK files when bundled, and no SDK files otherwise."""
    prefix = f"{VENDOR_DIR}/blender_vulkan_interop/"
    present = {n.removeprefix(prefix) for n in names if n.startswith(prefix) and not n.endswith("/")}
    if not bundled:
        if any(n.startswith(f"{VENDOR_DIR}/") for n in names):
            sys.exit("Built zip must not ship the private shared SDK in a --no-sdk build.")
        return
    expected = json.loads((source_dir / SDK_LOCKS[0]).read_text(encoding="utf-8"))["payload_sha256"]
    missing = sorted(set(expected) - present)
    if missing:
        sys.exit(f"Built zip is missing bundled SDK files: {missing[:5]}")


def find_blender(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("BLENDER") or shutil.which("blender")
    if not candidate:
        sys.exit("Blender not found: pass --blender PATH or set the BLENDER environment variable.")
    return candidate


def run_blender(blender: str, *args: str) -> None:
    cmd = [blender, "--factory-startup", "--command", "extension", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--blender", help="Path to the Blender executable")
    parser.add_argument("--install", action="store_true", help="Install the built zip into the user_default repository")
    parser.add_argument("--tag", help="Release tag; must equal v<manifest version>")
    parser.add_argument("--no-sdk", action="store_true", help="Leave out the private shared SDK (public releases)")
    args = parser.parse_args()

    if args.tag:
        check_tag(args.tag)
        check_license()
        if not args.no_sdk:
            sys.exit("Public release builds must pass --no-sdk: the shared SDK is private (see its DISTRIBUTION.md).")
    blender = find_blender(args.blender)
    DIST_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        source = stage_source(Path(staging), None if args.no_sdk else SDK_PACKAGE)
        run_blender(blender, "validate", str(source))
        run_blender(blender, "build", "--source-dir", str(source), "--output-dir", str(DIST_DIR))

    package = DIST_DIR / zip_name()
    if not package.is_file():
        sys.exit(f"Build finished but {package.name} was not produced in dist/.")
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        leaked = [n for n in names if n.startswith("experimental/")]
        packaged = tomllib.loads(archive.read("blender_manifest.toml").decode("utf-8"))["version"]
    if leaked:
        sys.exit(f"Built zip must not ship experimental/ modules, found: {leaked[:3]}")
    check_archive_sdk(names, bundled=not args.no_sdk)
    if args.tag:
        check_archive_license(names)
    if packaged != manifest_version():
        sys.exit(f"Packaged manifest version {packaged!r} differs from the source manifest.")
    print(f"Built {package} sha256={hashlib.sha256(package.read_bytes()).hexdigest()}")
    if args.install:
        run_blender(blender, "install-file", "-r", "user_default", "-e", str(package))


if __name__ == "__main__":
    main()
