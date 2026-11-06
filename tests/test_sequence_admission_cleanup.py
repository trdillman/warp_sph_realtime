"""Cross-platform writer admission leaves no false ownership after rejection."""

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
