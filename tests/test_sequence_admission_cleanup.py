"""Cross-platform writer admission leaves no false ownership after rejection."""

from pathlib import Path

import pytest

from warp_sph_realtime.cache.native_sequence import SequenceWriter


@pytest.mark.parametrize("signature", [{"bad": float("nan")}, {"bad": object()}])
def test_invalid_signature_does_not_leave_active_writer(tmp_path, signature):
    directory = tmp_path / "failed-signature"
    with pytest.raises((TypeError, ValueError)):
        SequenceWriter(directory, first=1, last=2, fps=24, signature=signature)
    assert not (directory / ".writer.lock").exists()
    assert not directory.exists() or not list(directory.iterdir())
    retry = SequenceWriter(directory, first=1, last=2, fps=24, signature={})
    retry.close("CANCELLED")


@pytest.mark.parametrize("failure", ["write", "close", "short_write"])
def test_lock_io_failure_releases_only_owned_reservation(tmp_path, monkeypatch, failure):
    directory = tmp_path / "failed-lock"
    original_open = Path.open

    class FailingLock:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def write(self, value):
            if failure == "write":
                raise OSError("injected lock write failure")
            if failure == "short_write":
                return 0
            return self.handle.write(value)

        def __exit__(self, *args):
            self.handle.close()
            if failure == "close":
                raise OSError("injected lock close failure")

    def fail_lock(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return FailingLock(handle) if path.name == ".writer.lock" else handle

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_lock)
        with pytest.raises(OSError):
            SequenceWriter(directory, first=1, last=2, fps=24, signature={})
    assert not list(directory.iterdir())
    retry = SequenceWriter(directory, first=1, last=2, fps=24, signature={})
    retry.close("CANCELLED")


def test_exclusive_lock_race_preserves_other_writer(tmp_path, monkeypatch):
    directory = tmp_path / "racing-writer"
    original_open = Path.open

    def race_lock(path, *args, **kwargs):
        if path.name == ".writer.lock":
            with original_open(path, "w", encoding="ascii") as handle:
                handle.write("other writer")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", race_lock)
        with pytest.raises(FileExistsError):
            SequenceWriter(directory, first=1, last=2, fps=24, signature={})
    assert (directory / ".writer.lock").read_text() == "other writer"
