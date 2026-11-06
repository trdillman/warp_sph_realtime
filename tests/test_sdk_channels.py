"""Only producer packing, never solver physics, is introduced by ADOPT-01."""

import importlib.util

import numpy as np
import pytest
import warp as wp


def packer():
    assert importlib.util.find_spec("warp_sph_realtime.sdk_channels"), "GPU frame-field adapter missing"
    from warp_sph_realtime.sdk_channels import pack_fields

    return pack_fields


def test_solver_sorted_ids_widen_exactly_and_unused_slots_not_consumed():
    pack = packer()
    source = wp.array(np.array([7, 0, 2147483647, 99], dtype=np.int32), dtype=wp.int32, device="cpu")
    ids = wp.full(4, wp.uint64(123), dtype=wp.uint64, device="cpu")
    radius = wp.full(4, 9.0, dtype=wp.float32, device="cpu")
    color = wp.zeros(4, dtype=wp.vec4, device="cpu")
    pack(source, ids, radius, color, 3, 0.02, (0.2, 0.4, 0.6, 1.0))
    np.testing.assert_array_equal(ids.numpy(), [7, 0, 2147483647, 123])
    np.testing.assert_allclose(radius.numpy()[:3], 0.02)
    np.testing.assert_allclose(color.numpy()[:3], np.tile([0.2, 0.4, 0.6, 1.0], (3, 1)))
    np.testing.assert_array_equal(source.numpy(), [7, 0, 2147483647, 99])


def test_zero_active_count_does_not_launch_or_mutate_sources():
    pack = packer()
    source = wp.array(np.array([8], dtype=np.int32), dtype=wp.int32, device="cpu")
    ids = wp.full(1, wp.uint64(55), dtype=wp.uint64, device="cpu")
    r = wp.zeros(1, dtype=wp.float32, device="cpu")
    c = wp.zeros(1, dtype=wp.vec4, device="cpu")
    pack(source, ids, r, c, 0, 0.1, (1.0, 1.0, 1.0, 1.0))
    assert ids.numpy()[0] == 55


def test_invalid_radius_fails_before_gpu_work():
    pack = packer()
    source = wp.zeros(1, dtype=wp.int32, device="cpu")
    ids = wp.zeros(1, dtype=wp.uint64, device="cpu")
    r = wp.zeros(1, dtype=wp.float32, device="cpu")
    c = wp.zeros(1, dtype=wp.vec4, device="cpu")
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pack(source, ids, r, c, 1, bad, (1.0, 1.0, 1.0, 1.0))
