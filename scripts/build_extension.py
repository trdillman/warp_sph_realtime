"""Validate and build the Blender extension zip into dist/, optionally installing it.

Usage:
    python scripts/build_extension.py [--blender PATH] [--install] [--tag vX.Y.Z]

``--tag`` (used by the release workflow) refuses to build unless the tag equals ``v`` plus the manifest version.

Blender is located from --blender, then the BLENDER environment variable, then PATH.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "warp_sph_realtime"
DIST_DIR = REPO_ROOT / "dist"


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
    args = parser.parse_args()

    if args.tag:
        check_tag(args.tag)
    blender = find_blender(args.blender)
    DIST_DIR.mkdir(exist_ok=True)
    run_blender(blender, "validate", str(SOURCE_DIR))
    run_blender(blender, "build", "--source-dir", str(SOURCE_DIR), "--output-dir", str(DIST_DIR))

    package = DIST_DIR / zip_name()
    if not package.is_file():
        sys.exit(f"Build finished but {package.name} was not produced in dist/.")
    with zipfile.ZipFile(package) as archive:
        leaked = [n for n in archive.namelist() if n.startswith("experimental/")]
        packaged = tomllib.loads(archive.read("blender_manifest.toml").decode("utf-8"))["version"]
    if leaked:
        sys.exit(f"Built zip must not ship experimental/ modules, found: {leaked[:3]}")
    if packaged != manifest_version():
        sys.exit(f"Packaged manifest version {packaged!r} differs from the source manifest.")
    print(f"Built {package} sha256={hashlib.sha256(package.read_bytes()).hexdigest()}")
    if args.install:
        run_blender(blender, "install-file", "-r", "user_default", "-e", str(package))


if __name__ == "__main__":
    main()
