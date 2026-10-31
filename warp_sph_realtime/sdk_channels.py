"""GPU-only adaptation of existing solver IDs and artist display fields.

No solver buffers or physics kernels are modified. Density and velocity retain
solver units. Inactive capacity entries are never published as live particles.
"""

import math

import warp as wp


@wp.kernel
def _fields(
    source: wp.array[wp.int32],
    ids: wp.array[wp.uint64],
    radius: wp.array[wp.float32],
    color: wp.array[wp.vec4],
    radius_value: float,
    color_value: wp.vec4,
):
    i = wp.tid()
    ids[i] = wp.uint64(source[i])
    radius[i] = radius_value
    color[i] = color_value


def pack_fields(source, ids, radius, color, count, radius_value, color_value):
    if type(count) is not int or count < 0 or count > source.shape[0]:
        raise ValueError("invalid logical particle count")
    if not math.isfinite(radius_value) or radius_value < 0:
        raise ValueError("particle radius must be finite and nonnegative")
    if len(color_value) != 4 or not all(math.isfinite(x) for x in color_value):
        raise ValueError("particle colour must contain four finite values")
    if source.dtype != wp.int32 or ids.dtype != wp.uint64:
        raise ValueError("stable solver int32 IDs and uint64 output required")
    if any(array.shape[0] < count or array.device != source.device for array in (ids, radius, color)):
        raise ValueError("output capacity or device mismatch")
    if count:
        wp.launch(
            _fields,
            dim=count,
            inputs=[source, ids, radius, color, float(radius_value), wp.vec4(*color_value)],
            device=source.device,
        )
