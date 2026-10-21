"""Validate and build the Blender extension zip into dist/, optionally installing it.

Usage:
    python scripts/build_extension.py [--blender PATH] [--install]

Blender is located from --blender, then the BLENDER environment variable, then PATH.
"""

import argparse
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
    args = parser.parse_args()

    blender = find_blender(args.blender)
    DIST_DIR.mkdir(exist_ok=True)
    run_blender(blender, "validate", str(SOURCE_DIR))
    run_blender(blender, "build", "--source-dir", str(SOURCE_DIR), "--output-dir", str(DIST_DIR))

    zips = sorted(DIST_DIR.glob("warp_sph_realtime-*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        sys.exit("Build finished but no zip was produced in dist/.")
    leaked = [n for n in zipfile.ZipFile(zips[-1]).namelist() if n.startswith("experimental/")]
    if leaked:
        sys.exit(f"Built zip must not ship experimental/ modules, found: {leaked[:3]}")
    print(f"Built {zips[-1]}")
    if args.install:
        run_blender(blender, "install-file", "-r", "user_default", "-e", str(zips[-1]))


if __name__ == "__main__":
    main()
