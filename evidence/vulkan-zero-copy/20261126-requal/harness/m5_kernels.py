"""Deterministic synthetic M5 producer (no solver physics): moves each position by a small offset."""
import warp as wp


@wp.kernel
def animate(base: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3), t: float):
    i = wp.tid()
    s = t + float(i) * 0.001
    out[i] = base[i] + wp.vec3(0.01 * wp.sin(s), 0.01 * wp.cos(s), 0.0)
