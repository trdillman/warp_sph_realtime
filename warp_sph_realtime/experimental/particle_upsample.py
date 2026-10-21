import warp as wp


@wp.kernel
def k_flag_surface_particles_for_split(
    density_sorted: wp.array[float],
    split_flags: wp.array[int],
    rest_density: float
):
    """Flags low-density free-surface particles for dynamic sub-particle splitting."""
    i = int(wp.tid())
    if density_sorted[i] < 0.85 * rest_density:
        split_flags[i] = 1
    else:
        split_flags[i] = 0
