"""Rich animation cache correctness, admission and atomic prefix tests."""

from collections.abc import Mapping
from importlib.util import find_spec

import numpy as np
import pytest


def api():
    assert find_spec("warp_sph_realtime.cache.native_sequence"), "rich animation storage missing"
    from warp_sph_realtime.cache import native_sequence

    return native_sequence


def frame(n=3, step=1):
    m = api()
    fields = {
        "position": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        "velocity": np.ones((n, 3), dtype=np.float32),
        "radius": np.full(n, 0.1, dtype=np.float32),
        "color": np.ones((n, 4), dtype=np.float32),
        "density": np.full(n, 1000, dtype=np.float32),
        "id": np.arange(n, dtype=np.uint64) + np.uint64(2**63),
    }
    return m.NativeFrame(
        fields, {"source_id": "s", "epoch": 0, "frame_id": step, "simulation_time_s": step * 0.01, "scene_revision": 0}
    )


def test_roundtrip_keeps_uint64_attributes_empty_and_prefix(tmp_path):
    m = api()
    writer = m.SequenceWriter(tmp_path / "anim", first=7, last=9, fps=24, signature={"test": 1})
    for index, n in enumerate((3, 0, 5), 7):
        writer.append(index, frame(n, index))
    writer.close("COMPLETE")
    reader = m.SequenceReader(tmp_path / "anim")
    assert reader.span == (7, 9) and reader.status == "COMPLETE"
    for index, n in enumerate((3, 0, 5), 7):
        got = reader.read(index)
        expected = frame(n, index)
        assert isinstance(expected.fields, Mapping)
        for key in expected.fields:
            np.testing.assert_array_equal(got.fields[key], expected.fields[key])
    assert reader.clamp(0) == 7 and reader.clamp(99) == 9


def test_existing_directory_never_overwritten(tmp_path):
    m = api()
    (tmp_path / "artist.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty|existing"):
        m.SequenceWriter(tmp_path, first=1, last=2, fps=24, signature={})
    assert (tmp_path / "artist.txt").read_text() == "keep"


def test_gap_and_duplicate_append_reject_without_advancing(tmp_path):
    m = api()
    w = m.SequenceWriter(tmp_path / "a", first=1, last=3, fps=24, signature={})
    with pytest.raises(ValueError):
        w.append(2, frame())
    w.append(1, frame())
    with pytest.raises(ValueError):
        w.append(1, frame())
    w.close("CANCELLED")
    assert m.SequenceReader(tmp_path / "a").span == (1, 1)


def test_corrupt_or_missing_frame_is_never_last_good_frame(tmp_path):
    m = api()
    w = m.SequenceWriter(tmp_path / "a", first=1, last=2, fps=24, signature={})
    w.append(1, frame())
    w.append(2, frame(step=2))
    w.close("COMPLETE")
    reader = m.SequenceReader(tmp_path / "a")
    reader.read(1)
    (tmp_path / "a" / "frame_000002.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity|hash"):
        reader.read(2)


def test_budgets_ids_and_metadata_fail_before_write(tmp_path):
    m = api()
    f = frame()
    w = m.SequenceWriter(tmp_path / "a", first=1, last=1, fps=24, signature={}, max_frame_bytes=10)
    with pytest.raises(ValueError, match="budget"):
        w.append(1, f)
    assert not list((tmp_path / "a").glob("*.npz"))
    w.close("CANCELLED")
    bad = {k: v.copy() for k, v in f.fields.items()}
    bad["id"][:] = 1
    with pytest.raises(ValueError, match="identit|ID"):
        m.NativeFrame(bad, f.stamp)
    bad = {k: v.copy() for k, v in f.fields.items()}
    bad["position"][0, 0] = np.nan
    with pytest.raises(ValueError):
        m.NativeFrame(bad, f.stamp)


def test_capture_is_immutable_and_ownership_epoch_cannot_mix(tmp_path):
    m = api()
    f = frame()
    with pytest.raises(ValueError):
        f.fields["id"][0] = 4
    w = m.SequenceWriter(tmp_path / "a", first=1, last=2, fps=24, signature={})
    w.append(1, f)
    bad = m.NativeFrame(dict(f.fields), dict(f.stamp, epoch=1, frame_id=2))
    with pytest.raises(ValueError, match="epoch|source"):
        w.append(2, bad)
    w.close("CANCELLED")


def test_live_throttle_uses_epoch_and_does_not_queue_duplicates():
    from warp_sph_realtime import native_output as m

    gate = m.LiveGate()
    stamp = ("source", 0, 1)
    assert gate.due(stamp, 10, 1.0, False)
    gate.queued(stamp, 1.0)
    assert not gate.due(stamp, 10, 4.0, False)
    assert not gate.due(("source", 0, 2), 10, 1.05, False)
    assert gate.due(("source", 0, 2), 10, 1.11, False)
    assert not gate.due(("source", 0, 2), 10, 1.11, True)
    assert gate.due(("source", 1, 1), 10, 2.0, False)
