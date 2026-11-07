"""Versioned, bounded, exact rich-particle animation cache; no bpy/Warp/SDK.

Legacy position/velocity/int32 caches remain unchanged. This format stores the
complete native-output frame, with uint64 identity, canonical metric units and
an atomic committed-prefix manifest. No pickle, implicit frame skipping or overwrite.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

MAX_FRAME_BYTES = 256 * 1024 * 1024
_REQUIRED = {
    "position": (np.dtype("float32"), 3),
    "velocity": (np.dtype("float32"), 3),
    "radius": (np.dtype("float32"), 1),
    "color": (np.dtype("float32"), 4),
    "density": (np.dtype("float32"), 1),
    "id": (np.dtype("uint64"), 1),
}
_FORMAT = "warpsph-rich-particles-1"


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _atomic_json(path, data):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class NativeFrame:
    fields: object
    stamp: object

    def __post_init__(self):
        if set(self.fields) != set(_REQUIRED):
            raise ValueError("rich frame requires position/velocity/radius/color/density/id")
        ids = np.asarray(self.fields["id"])
        if ids.ndim != 1 or ids.dtype != np.uint64:
            raise ValueError("particle IDs require an exact uint64 vector")
        count = len(ids)
        if len(np.unique(ids)) != count:
            raise ValueError("duplicate particle identities")
        copied = {}
        for name, (dtype, width) in _REQUIRED.items():
            values = np.asarray(self.fields[name])
            shape = (count,) if width == 1 else (count, width)
            if values.dtype != dtype or values.shape != shape:
                raise ValueError(f"bad {name} dtype/shape")
            if name != "id" and not np.isfinite(values).all():
                raise ValueError(f"nonfinite {name}")
            if name == "radius" and np.any(values < 0):
                raise ValueError("negative radius")
            # Bytes-backed copies cannot be made writable by a downstream owner.
            owned = np.frombuffer(values.tobytes(order="C"), dtype=dtype).reshape(shape)
            copied[name] = owned
        stamp = dict(self.stamp)
        if set(stamp) != {"source_id", "epoch", "frame_id", "simulation_time_s", "scene_revision"}:
            raise ValueError("invalid frame stamp schema")
        if not isinstance(stamp["source_id"], str) or not stamp["source_id"]:
            raise ValueError("frame source is required")
        for name in ("epoch", "frame_id", "scene_revision"):
            _int(stamp[name], name)
        seconds = stamp["simulation_time_s"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid simulation time")
        object.__setattr__(self, "fields", MappingProxyType(copied))
        object.__setattr__(self, "stamp", MappingProxyType(stamp))

    @property
    def count(self):
        return len(self.fields["id"])

    @property
    def nbytes(self):
        return sum(array.nbytes for array in self.fields.values())


def from_payload(payload):
    """SDK frame positions/radii use spatial scale; velocities already mean m/s."""
    arrays = {
        a.name: np.frombuffer(a.data, dtype="<u8" if a.scalar_type == "uint64" else "<f4").reshape(a.shape)
        for a in payload.arrays
    }
    if set(arrays) != set(_REQUIRED) or payload.spatial is None:
        raise ValueError("unsupported rich particle payload")
    scale = float(payload.spatial.meters_per_unit)
    origin = np.asarray(payload.spatial.origin_m, dtype=np.float64)
    arrays["position"] = (arrays["position"].astype(np.float64) * scale + origin).astype(np.float32)
    arrays["radius"] = (arrays["radius"].astype(np.float64) * scale).astype(np.float32)
    stamp = {
        name: getattr(payload.stamp, name)
        for name in ("source_id", "epoch", "frame_id", "simulation_time_s", "scene_revision")
    }
    return NativeFrame(arrays, stamp)


class SequenceWriter:
    def __init__(self, directory, *, first, last, fps, signature, max_frame_bytes=MAX_FRAME_BYTES):
        _int(first, "first", -1048574)
        _int(last, "last", first)
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("positive finite fps required")
        _int(max_frame_bytes, "frame budget")
        checked_signature = json.loads(json.dumps(signature, allow_nan=False))
        self.directory = Path(directory).resolve()
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError("animation directory must be empty; existing files are never replaced")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = self.directory / ".writer.lock"
        self.max_frame_bytes = max_frame_bytes
        self.closed = False
        self.document = {
            "format": _FORMAT,
            "first": first,
            "last": last,
            "fps": float(fps),
            "signature": checked_signature,
            "status": "RECORDING",
            "frames": [],
            "units": "metres;velocity=m/s",
        }
        self.previous_stamp = None
        # Opening exclusively establishes ownership. A failed open must never
        # remove a competing writer's lock; all later I/O failures release ours.
        lock_handle = self.lock.open("x", encoding="ascii")
        try:
            with lock_handle as handle:
                identity = str(os.getpid())
                if handle.write(identity) != len(identity):
                    raise OSError("incomplete animation writer lock")
            _atomic_json(self.directory / "sequence.json", self.document)
        except Exception:
            self.lock.unlink(missing_ok=True)
            raise

    def append(self, number, frame):
        if self.closed:
            raise ValueError("animation writer closed")
        expected = self.document["first"] + len(self.document["frames"])
        if type(number) is not int or number != expected or number > self.document["last"]:
            raise ValueError("animation frames must be contiguous and never overwritten")
        if not isinstance(frame, NativeFrame) or frame.nbytes > self.max_frame_bytes:
            raise ValueError("native animation frame exceeds budget")
        stamp = dict(frame.stamp)
        if self.previous_stamp is not None:
            previous = self.previous_stamp
            if (stamp["source_id"], stamp["epoch"]) != (previous["source_id"], previous["epoch"]):
                raise ValueError("source/epoch changed within animation")
            if stamp["frame_id"] <= previous["frame_id"] or stamp["simulation_time_s"] < previous["simulation_time_s"]:
                raise ValueError("stale animation frame stamp")
        name = f"frame_{number:06d}.npz"
        target = self.directory / name
        if target.exists():
            raise ValueError("uncommitted/existing frame must not be overwritten")
        temporary = self.directory / (name + "." + uuid.uuid4().hex + ".tmp")
        metadata = np.frombuffer(json.dumps(stamp, allow_nan=False).encode("utf-8"), dtype=np.uint8)
        if metadata.nbytes > 8192:
            raise ValueError("frame metadata exceeds budget")
        try:
            with temporary.open("xb") as handle:
                np.savez(handle, **frame.fields, metadata=metadata)
                handle.flush()
                os.fsync(handle.fileno())
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            size = temporary.stat().st_size
            os.replace(temporary, target)
            entry = {
                "frame": number,
                "file": name,
                "count": frame.count,
                "bytes": size,
                "payload_bytes": frame.nbytes,
                "sha256": digest,
                "simulation_time_s": stamp["simulation_time_s"],
            }
            updated = dict(self.document, frames=self.document["frames"] + [entry])
            _atomic_json(self.directory / "sequence.json", updated)
            self.document = updated
            self.previous_stamp = stamp
            return entry
        finally:
            temporary.unlink(missing_ok=True)

    def close(self, status):
        if self.closed:
            return
        if status not in ("COMPLETE", "CANCELLED", "FAILED"):
            raise ValueError("invalid animation completion state")
        if status == "COMPLETE" and len(self.document["frames"]) != self.document["last"] - self.document["first"] + 1:
            raise ValueError("cannot mark an incomplete animation complete")
        updated = dict(self.document, status=status)
        _atomic_json(self.directory / "sequence.json", updated)
        self.document = updated
        self.closed = True
        self.lock.unlink(missing_ok=True)


class SequenceReader:
    def __init__(self, directory, *, max_frame_bytes=MAX_FRAME_BYTES):
        self.directory = Path(directory).resolve()
        _int(max_frame_bytes, "frame budget")
        self.max_frame_bytes = max_frame_bytes
        path = self.directory / "sequence.json"
        limit = 16 * 1024 * 1024
        with path.open('rb') as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("animation manifest exceeds budget")
        self.document = json.loads(data.decode('utf-8'))
        d = self.document
        if d.get("format") != _FORMAT or d.get("units") != "metres;velocity=m/s":
            raise ValueError("unsupported native animation format")
        if type(d.get("fps")) not in (int, float) or not math.isfinite(d["fps"]) or d["fps"] <= 0:
            raise ValueError("invalid animation frame rate")
        _int(d["first"], "first", -1048574)
        _int(d["last"], "last", d["first"])
        if d.get("status") not in ("RECORDING", "COMPLETE", "CANCELLED", "FAILED") or not isinstance(
            d.get("frames"), list
        ):
            raise ValueError("invalid animation manifest")
        for index, item in enumerate(d["frames"]):
            number = d["first"] + index
            if item["frame"] != number or item["file"] != f"frame_{number:06d}.npz" or number > d["last"]:
                raise ValueError("animation manifest is not a contiguous safe prefix")
            _int(item["count"], "particle count")
            _int(item["bytes"], "file bytes")
            _int(item["payload_bytes"], "payload bytes")
        if d["status"] == "COMPLETE" and len(d["frames"]) != d["last"] - d["first"] + 1:
            raise ValueError("incomplete animation marked COMPLETE")

    @property
    def span(self):
        return (
            (self.document["first"], self.document["first"] + len(self.document["frames"]) - 1)
            if self.document["frames"]
            else None
        )

    @property
    def status(self):
        return self.document["status"]

    def clamp(self, number):
        if self.span is None:
            raise ValueError("animation has no committed frames")
        return min(max(int(number), self.span[0]), self.span[1])

    def read(self, number):
        if self.span is None or not self.span[0] <= number <= self.span[1]:
            raise ValueError("animation frame not committed")
        item = self.document["frames"][number - self.span[0]]
        if item["payload_bytes"] > self.max_frame_bytes or item["bytes"] > self.max_frame_bytes + 16384:
            raise ValueError("animation frame exceeds readback budget")
        path = self.directory / item["file"]
        try:
            with path.open("rb") as stream:
                data = stream.read(item["bytes"] + 1)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("animation frame integrity/hash mismatch")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                infos = archive.infolist()
                expected = {name + '.npy' for name in (*_REQUIRED, 'metadata')}
                if len(infos) != len(expected) or {info.filename for info in infos} != expected:
                    raise ValueError('unexpected or duplicate animation fields')
                if sum(info.file_size for info in infos) > self.max_frame_bytes + 8192:
                    raise ValueError("expanded animation frame exceeds budget")
                # NumPy allocates from the NPY header before discovering truncated
                # data. Validate declarations before any array deserialization.
                for info in infos:
                    name = info.filename[:-4]
                    with archive.open(info) as member:
                        version = np.lib.format.read_magic(member)
                        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                                   (2, 0): np.lib.format.read_array_header_2_0}
                        if version not in readers:
                            raise ValueError('unsupported animation array format')
                        shape, fortran, dtype = readers[version](member, max_header_size=8192)
                        if name == 'metadata':
                            valid = (dtype == np.dtype('uint8') and len(shape) == 1
                                     and 0 <= shape[0] <= 8192)
                        else:
                            wanted_dtype, width = _REQUIRED[name]
                            wanted_shape = ((item['count'], width) if width > 1 else (item['count'],))
                            valid = dtype == wanted_dtype and shape == wanted_shape
                        if not valid or fortran:
                            raise ValueError('invalid animation array declaration')
                        size = math.prod(shape) * dtype.itemsize
                        if size > self.max_frame_bytes or member.tell() + size != info.file_size:
                            raise ValueError('animation array size/budget mismatch')
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                if set(archive.files) != set(_REQUIRED) | {"metadata"}:
                    raise ValueError("unexpected animation fields")
                metadata = archive["metadata"]
                if metadata.dtype != np.uint8 or metadata.ndim != 1 or metadata.nbytes > 8192:
                    raise ValueError("invalid frame metadata")
                frame = NativeFrame({name: archive[name] for name in _REQUIRED}, json.loads(metadata.tobytes()))
            if frame.stamp["simulation_time_s"] != item["simulation_time_s"]:
                raise ValueError("simulation time does not match committed manifest")
            if frame.count != item["count"] or frame.nbytes != item["payload_bytes"]:
                raise ValueError("animation count/byte integrity mismatch")
            return frame
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"animation frame integrity/read failure: {exc}") from exc

    def validate_all(self):
        if self.span is None:
            raise ValueError("animation has no committed frames")
        previous = None
        for number in range(self.span[0], self.span[1] + 1):
            current = self.read(number).stamp
            if previous is not None:
                if (current["source_id"], current["epoch"]) != (previous["source_id"], previous["epoch"]):
                    raise ValueError("animation mixes source/epoch identity")
                if (
                    current["frame_id"] <= previous["frame_id"]
                    or current["simulation_time_s"] < previous["simulation_time_s"]
                ):
                    raise ValueError("animation contains regressed frame/time")
            previous = current
        return self.span
