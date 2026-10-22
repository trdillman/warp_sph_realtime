"""On-disk frame cache (P1.6, P2.2): one ``.npz`` per frame plus a ``manifest.json``.

Pure NumPy and standard library: no ``bpy``, no Warp. A cache directory holds one simulation timeline,
the frames ``frame_000001.npz`` ... named by scene frame number, each with ``position`` (N, 3) float32,
``velocity`` (N, 3) float32 and ``id`` (N,) int32 (the stable particle id; solver array order changes
every step).

Consistency rules, which resume and cancel rely on:

* A frame file is written to ``<name>.tmp`` and renamed into place, so a file that exists is complete
  (a killed process leaves at most one ``.tmp``).
* Frames are written in order by one writer, so the files always form a prefix of the timeline.
  ``cached_range`` is the contiguous run from the manifest's ``start``.
* The manifest stores the bake signature (particle count, solver settings, ...). A cache whose signature
  differs from the current one is never extended; it must be invalidated first.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
_FRAME_RE = re.compile(r"^frame_(-?\d+)\.npz$")

# position + velocity (3 x float32) + id (int32)
BYTES_PER_PARTICLE = 12 + 12 + 4
# Container bytes of one plain .npz of these three arrays (zip headers + .npy headers), measured; the
# size-estimate test keeps it exact.
FRAME_OVERHEAD_BYTES = 754


class CacheError(Exception):
    """The cache cannot be read or written as asked."""


class CacheMismatchError(CacheError):
    """The directory holds a cache baked with a different signature (settings or range start)."""


@dataclass(frozen=True)
class Frame:
    frame: int
    position: np.ndarray  # (N, 3) float32
    velocity: np.ndarray  # (N, 3) float32
    id: np.ndarray        # (N,) int32

    @property
    def count(self) -> int:
        return int(self.position.shape[0])


def frame_name(frame: int) -> str:
    return f"frame_{frame:06d}.npz"


def frame_bytes(particles: int) -> int:
    """Size on disk of one frame of ``particles`` particles."""
    return FRAME_OVERHEAD_BYTES + BYTES_PER_PARTICLE * max(int(particles), 0)


def estimate_bytes(particles: int, frames: int) -> int:
    """Size on disk of ``frames`` frames; exact for fixed-count caches (plain, uncompressed ``.npz``)."""
    return frame_bytes(particles) * max(int(frames), 0)


def free_bytes(directory: str | os.PathLike) -> int | None:
    """Free space of the drive that holds ``directory`` (which need not exist yet); None if unknown."""
    path = Path(directory)
    while not path.exists():
        if path.parent == path:
            return None
        path = path.parent
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "scene"


def default_cache_dir(blend_path: str, scene_name: str) -> Path:
    """Default cache directory of a scene: ``<blend folder>/<blend name>_warp_sph_cache/<scene>`` next to the
    saved .blend file, or ``<temp>/warp_sph_cache/unsaved/<scene>`` while the file is unsaved."""
    scene = _safe_name(scene_name)
    if blend_path:
        blend = Path(blend_path)
        return blend.parent / f"{_safe_name(blend.stem)}_warp_sph_cache" / scene
    return Path(tempfile.gettempdir()) / "warp_sph_cache" / "unsaved" / scene


def resolve_cache_dir(configured: str, blend_path: str, scene_name: str) -> Path:
    """The scene's cache directory: the configured one (already made absolute), else the default."""
    return Path(configured) if configured.strip() else default_cache_dir(blend_path, scene_name)


class DiskCache:
    """One cache directory. All methods are safe to call with the directory missing or empty."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)

    # -- manifest -------------------------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    def manifest(self) -> dict | None:
        """The manifest, or None when the directory has none. Raises CacheError if it is unreadable."""
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise CacheError(f"cannot read {self.manifest_path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("format") != FORMAT_VERSION or "start" not in data:
            raise CacheError(f"{self.manifest_path} is not a format {FORMAT_VERSION} WarpSPH cache manifest")
        return data

    def begin(self, start: int, signature: dict) -> None:
        """Create the directory and manifest, or confirm an existing manifest matches (CacheMismatchError)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.remove_temporary()
        signature = json.loads(json.dumps(signature))  # compare as JSON, not as Python objects
        existing = self.manifest()
        if existing is not None:
            if "gravity" in signature:
                old_signature = existing.get("signature") or {}
                if "gravity" not in old_signature:
                    raise CacheMismatchError(
                        "cached gravity is unknown (older cache); use Bake Cache to start a fresh bake "
                        "or choose a new cache folder")
                if old_signature["gravity"] != signature["gravity"]:
                    raise CacheMismatchError(
                        "solver gravity changed; use Bake Cache to start a fresh bake "
                        "or choose a new cache folder")
            if existing["start"] != start or existing.get("signature") != signature:
                raise CacheMismatchError(
                    f"{self.directory} was baked with different settings or start frame; invalidate it first")
            return
        document = {"format": FORMAT_VERSION, "start": int(start), "signature": signature}
        tmp = self.manifest_path.with_name(MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    # -- frames ---------------------------------------------------------------------------------------

    def path(self, frame: int) -> Path:
        return self.directory / frame_name(frame)

    def write_frame(self, frame: int, position: np.ndarray, velocity: np.ndarray, ids: np.ndarray) -> int:
        """Write one frame atomically; returns its size in bytes. Raises CacheError on bad data or I/O failure."""
        count = len(ids)
        if position.shape != (count, 3) or velocity.shape != (count, 3):
            raise CacheError(f"frame {frame}: position/velocity/id lengths differ")
        target = self.path(frame)
        tmp = target.with_name(target.name + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as handle:
                np.savez(handle,
                         position=np.ascontiguousarray(position, dtype=np.float32),
                         velocity=np.ascontiguousarray(velocity, dtype=np.float32),
                         id=np.ascontiguousarray(ids, dtype=np.int32))
            size = tmp.stat().st_size
            os.replace(tmp, target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise CacheError(f"cannot write frame {frame} to {self.directory}: {exc}") from exc
        return size

    def read_frame(self, frame: int) -> Frame:
        """Read one frame; the arrays are copies owned by the caller. Raises CacheError if missing or corrupt."""
        path = self.path(frame)
        try:
            with open(path, "rb") as handle, np.load(handle, allow_pickle=False) as data:  # own the handle: Windows
                position, velocity, ids = data["position"], data["velocity"], data["id"]  # cannot delete open files
        except FileNotFoundError as exc:
            raise CacheError(f"frame {frame} is not cached") from exc
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:  # truncated, missing member
            raise CacheError(f"frame {frame} is unreadable: {exc}") from exc
        count = len(ids)
        if (position.dtype != np.float32 or velocity.dtype != np.float32 or ids.dtype != np.int32
                or position.shape != (count, 3) or velocity.shape != (count, 3)):
            raise CacheError(f"frame {frame} has unexpected array shapes or types")
        return Frame(frame, position, velocity, ids)

    def read_ids(self, frame: int) -> np.ndarray:
        """Only the particle ids of one frame (the other members of the ``.npz`` are not read)."""
        try:
            with open(self.path(frame), "rb") as handle, np.load(handle, allow_pickle=False) as data:
                ids = data["id"]
        except FileNotFoundError as exc:
            raise CacheError(f"frame {frame} is not cached") from exc
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise CacheError(f"frame {frame} is unreadable: {exc}") from exc
        if ids.dtype != np.int32 or ids.ndim != 1:
            raise CacheError(f"frame {frame} has unexpected id array")
        return ids

    def frames(self) -> list[int]:
        """Cached frame numbers, sorted."""
        try:
            entries = os.listdir(self.directory)
        except FileNotFoundError:
            return []
        found = (_FRAME_RE.match(name) for name in entries)
        return sorted(int(m.group(1)) for m in found if m)

    def frame_stamps(self) -> dict[int, int]:
        """Modification time (ns) of every cached frame file, from one directory listing."""
        stamps: dict[int, int] = {}
        try:
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    found = _FRAME_RE.match(entry.name)
                    if found:
                        try:
                            stamps[int(found.group(1))] = entry.stat().st_mtime_ns
                        except OSError:
                            pass  # removed while listing
        except FileNotFoundError:
            pass
        return stamps

    def cached_range(self) -> tuple[int, int] | None:
        """(start, last) of the contiguous run of frames beginning at the manifest's start, or None."""
        manifest = self.manifest()
        if manifest is None:
            return None
        present = set(self.frames())
        start = int(manifest["start"])
        if start not in present:
            return None
        last = start
        while last + 1 in present:
            last += 1
        return start, last

    def cached_count(self) -> int:
        rng = self.cached_range()
        return 0 if rng is None else rng[1] - rng[0] + 1

    def size_bytes(self) -> int:
        total = 0
        for frame in self.frames():
            try:
                total += self.path(frame).stat().st_size
            except OSError:
                pass
        return total

    # -- invalidation ---------------------------------------------------------------------------------

    def remove_temporary(self) -> None:
        """Delete leftovers of an interrupted write."""
        if self.directory.is_dir():
            for tmp in self.directory.glob("*.tmp"):
                tmp.unlink(missing_ok=True)

    def invalidate_from(self, frame: int) -> int:
        """Delete every cached frame >= ``frame`` (later frames depend on it); returns how many."""
        removed = 0
        for number in self.frames():
            if number >= frame:
                try:
                    self.path(number).unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise CacheError(f"cannot delete frame {number}: {exc}") from exc
        self.remove_temporary()
        return removed

    def clear(self) -> int:
        """Delete all frames and the manifest; returns the number of frames removed."""
        removed = self.invalidate_from(min(self.frames(), default=0))
        self.manifest_path.unlink(missing_ok=True)
        return removed
