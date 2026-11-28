#!/usr/bin/env python3
"""Keep the shared Vulkan transport core identical in the add-on and the private SDK.

``warp_sph_realtime`` (public add-on) and ``blender-vulkan-interop`` (private SDK) each carry
their own copy of the same Blender Vulkan transport files, because the public add-on must work
and ship without the private SDK. ``SHARED_CORE.json`` is the single record of which files are
shared and which version of them both repositories agree on. This script, identical in both
repositories, enforces it:

    python check_shared_core.py                 # this repository matches SHARED_CORE.json
    python check_shared_core.py --other PATH    # ...and so does the other repository, with the same manifest
    python check_shared_core.py --update        # after editing shared files: refresh hashes
    python check_shared_core.py --update --bump "what changed"   # also raise the shared version

Hashes use LF text so Windows checkouts agree. Classes the manifest lists under ``addon_only`` exist
only in the add-on (for example the CPU fallback ``ReadbackTransport``) and are removed before
hashing, so both sides still compare equal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

MANIFEST = "SHARED_CORE.json"
SELF = Path(__file__).resolve()


def side_of(root: Path) -> str:
    if (root / "src" / "blender_vulkan_interop" / "__init__.py").is_file():
        return "sdk"
    if (root / "warp_sph_realtime" / "render" / "transport.py").is_file():
        return "addon"
    raise SystemExit(f"{root} is neither the add-on nor the SDK repository")


def canonical(path: Path, drop_classes: list[str]) -> bytes:
    """LF-normalized bytes with add-on-only classes removed and blank-line runs collapsed."""
    text = path.read_bytes().replace(b"\r\n", b"\n")
    if path.suffix == ".py" and drop_classes:
        source = text.decode("utf-8")
        lines = source.split("\n")
        spans = []
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name in drop_classes:
                first = min([node.lineno, *(d.lineno for d in node.decorator_list)]) - 1
                spans.append((first, node.end_lineno))
        for first, last in sorted(spans, reverse=True):
            del lines[first:last]
        text = "\n".join(lines).encode("utf-8")
    return re.sub(rb"\n{4,}", b"\n\n\n", text).rstrip(b"\n") + b"\n"


def digest(root: Path, entry: dict, side: str) -> str:
    drop = entry.get("addon_only", [])
    return hashlib.sha256(canonical(root / entry[side], drop)).hexdigest()


def load_manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST).read_text(encoding="utf-8"))


def problems(root: Path, manifest: dict) -> list[str]:
    side = side_of(root)
    found = []
    for entry in manifest["files"]:
        path = root / entry[side]
        if not path.is_file():
            found.append(f"missing shared file: {entry[side]}")
        elif digest(root, entry, side) != entry["sha256"]:
            found.append(f"{entry[side]} differs from SHARED_CORE.json {manifest['version']}")
    return found


def check(root: Path, other: Path | None) -> list[str]:
    manifest = load_manifest(root)
    found = problems(root, manifest)
    if other is not None:
        other_manifest = load_manifest(other)
        if other_manifest != manifest:
            found.append(
                f"{side_of(other)} repository is on shared core {other_manifest['version']}, "
                f"this one on {manifest['version']}: update both repositories together"
            )
        found += [f"[{side_of(other)}] {p}" for p in problems(other, other_manifest)]
    return found


def update(root: Path, bump: str | None) -> None:
    manifest = load_manifest(root)
    side = side_of(root)
    for entry in manifest["files"]:
        entry["sha256"] = digest(root, entry, side)
    if bump:
        major, minor = manifest["version"].split(".")[:2]
        manifest["version"] = f"{major}.{int(minor) + 1}"
        manifest["history"].insert(0, {"version": manifest["version"], "change": bump})
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{MANIFEST}: version {manifest['version']}, hashes refreshed from the {side} copy")
    print("Now run the same command in the other repository, or copy this manifest there and run --update without --bump.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=SELF.parents[1], help="repository root (default: this one)")
    parser.add_argument("--other", type=Path, help="root of the other repository (the SDK checkout or the add-on)")
    parser.add_argument("--update", action="store_true", help="rewrite the hashes in SHARED_CORE.json")
    parser.add_argument("--bump", metavar="CHANGE", help="with --update: raise the shared version and record CHANGE")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.update:
        update(root, args.bump)
        return 0
    found = check(root, args.other.resolve() if args.other else None)
    for line in found:
        print(f"shared core: {line}", file=sys.stderr)
    if found:
        print("Fix: make the same change in both repositories, then run "
              "`check_shared_core.py --update --bump '<change>'` in each.", file=sys.stderr)
        return 1
    print(f"shared core {load_manifest(root)['version']} OK ({side_of(root)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
