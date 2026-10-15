import warp as wp

# Secondary particle classification constants based on local fluid neighbor count (n_f):
# Spray: n_f <= 5 | Foam: 5 < n_f < 15 | Bubble: n_f >= 15
SPRAY_MAX_NEIGHBORS = 5
BUBBLE_MIN_NEIGHBORS = 15

@wp.kernel
def k_classify_and_advect_whitewater(
    grid: wp.uint64,
    fluid_pos: wp.array[wp.vec3],
    fluid_vel: wp.array[wp.vec3],
    ww_pos: wp.array[wp.vec3],
    ww_vel: wp.array[wp.vec3],
    ww_life: wp.array[float],
    ww_type: wp.array[int],
    h: float,
    dt: float
):
    """Classifies secondary particles by fluid neighbor count (n_f) and updates physics."""
    i = int(wp.tid())
    if ww_life[i] <= 0.0:
        return

    xi = ww_pos[i]
    n_f = int(0)
    v_avg = wp.vec3(0.0, 0.0, 0.0)

    neighbors = wp.hash_grid_query(grid, xi, h)
    for j in neighbors:
        if wp.length(xi - fluid_pos[j]) < h:
            n_f += 1
            v_avg += fluid_vel[j]

    if n_f <= SPRAY_MAX_NEIGHBORS:
        ww_type[i] = 0  # Ballistic Spray
        ww_vel[i] = ww_vel[i] + wp.vec3(0.0, 0.0, -9.81) * dt
    elif n_f < BUBBLE_MIN_NEIGHBORS:
        ww_type[i] = 1  # Surface-Advected Foam
        ww_vel[i] = v_avg / float(n_f)
    else:
        ww_type[i] = 2  # Submerged Buoyant Bubble
        v_fluid = v_avg / float(n_f)
        ww_vel[i] = v_fluid + wp.vec3(0.0, 0.0, 2.5) * dt

    ww_pos[i] = xi + ww_vel[i] * dt
    ww_life[i] = ww_life[i] - dt
