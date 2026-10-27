import logging
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..scene.description import SceneDescription, rigid_key
from .colliders import MeshSet, mesh_deep_signed_distance, mesh_signed_distance

wp.init()

log = logging.getLogger(__name__)

MAX_PARTICLES = 4_000_000

ARTIFICIAL_VISCOSITY_ALPHA = wp.constant(0.1)

# The viscous force is explicit: each substep relaxes a particle's velocity toward its neighbours' by
# (viscosity / mass) * dt. Above 1 that overshoots and above 2 it diverges, so the coefficient is capped
# where the relaxation per substep reaches this fraction.
# `viscosity` is per particle (kg/s), not a material constant: the relaxation rate viscosity / mass grows as
# 1 / spacing^3, so the same value is thicker at finer spacing, and the cap (proportional to mass / dt) is
# lower there. The cap is reported, not silent: rebuild_cuda_graph() logs a warning and the UI shows it.
VISCOSITY_RELAXATION_LIMIT = 0.9

@wp.func
def cubic_spline_kernel(r: float, h: float) -> float:
    q = r / h
    if q >= 1.0:
        return 0.0
    sigma = 8.0 / (3.14159265 * h * h * h)
    if q <= 0.5:
        return sigma * (6.0 * (q * q * q - q * q) + 1.0)
    else:
        term = 1.0 - q
        return sigma * 2.0 * (term * term * term)

def lattice_rest_mass(spacing: float, h: float, rest_density: float) -> float:
    """Particle mass that makes a filled cubic lattice at `spacing` sum to exactly rest_density.

    Any mismatch leaves the resting fluid pre-compressed, where the gamma=7 Tait EOS is far
    stiffer than the nominal sound speed assumed by the time step.
    """
    reach = int(h / spacing) + 1
    w_sum = 0.0
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                q = spacing * (i * i + j * j + k * k) ** 0.5 / h
                if q < 1.0:
                    w = 6.0 * (q ** 3 - q ** 2) + 1.0 if q <= 0.5 else 2.0 * (1.0 - q) ** 3
                    w_sum += w * 8.0 / (3.14159265 * h ** 3)
    return rest_density / w_sum


@wp.func
def cubic_spline_grad(diff: wp.vec3, r: float, h: float) -> wp.vec3:
    q = r / h
    if q <= 1e-6 or q >= 1.0:
        return wp.vec3(0.0, 0.0, 0.0)
    sigma = 48.0 / (3.14159265 * h * h * h * h)
    if q <= 0.5:
        grad_mag = sigma * q * (3.0 * q - 2.0)
    else:
        term = 1.0 - q
        grad_mag = -sigma * (term * term)
    return (diff / r) * grad_mag

@wp.func
def wall_accel(d: float, v_into: float, r0: float, sound_speed: float, dt: float) -> float:
    """Repulsive acceleration away from a wall at distance d (soft boundary layer of width r0).

    Stiffness scales with c^2 so the layer is as stiff as the fluid; the damping term only acts on
    motion into the wall, keeping hard position clamps as a rarely used last resort.

    The damping is explicit: one step removes omega * overlap * dt of the approach speed, and
    omega * dt = c / r0 * dt = 2 * cfl_factor, so above cfl 0.5 it would remove more than all of it and
    flip the velocity. It is capped at stopping the approach in one step, which never binds at
    cfl_factor <= 0.5 (the default is 0.4, bit-identical to the uncapped law).
    """
    if d >= r0:
        return 0.0
    overlap = 1.0 - wp.max(d, 0.0) / r0
    omega = sound_speed / r0
    a = 0.5 * sound_speed * sound_speed / r0 * overlap
    if v_into > 0.0:
        a += wp.min(omega * v_into * overlap, v_into / dt)
    return a


@wp.func
def compliant_contact_accel(d: float, v_into: float, r0: float, sound_speed: float) -> float:
    """C1 repulsion in a boundary layer, with smooth approach-only damping.

    Unlike hard restitution, force goes to zero continuously at first contact.
    Squaring the compact overlap also makes its first derivative zero there.
    Mode 2 opts into this forward law; it is not a surrogate backward pass.
    """
    overlap = wp.max(1.0 - d / r0, 0.0)
    approach = 0.5 * (v_into + wp.sqrt(v_into * v_into + 0.0001))
    return (0.5 * sound_speed * sound_speed / r0 + sound_speed / r0 * approach) * overlap * overlap


@wp.func
def compliant_cylinder_accel(x: wp.vec3, v: wp.vec3, cx: float, cy: float,
                             r0: float, sound_speed: float) -> wp.vec3:
    diff = wp.vec3(x[0] - cx, x[1] - cy, 0.0)
    radius = wp.length(diff)
    normal = diff / wp.max(radius, 1e-6)
    # C1 fade at the existing pillar top; the pouring case stays below it.
    t = wp.clamp((2.2 - x[2]) / r0, 0.0, 1.0)
    top_weight = t * t * (3.0 - 2.0 * t)
    return normal * top_weight * compliant_contact_accel(radius - 0.18, -wp.dot(v, normal), r0, sound_speed)


@wp.func
def collide_cylinder_2d(
    x: wp.vec3,
    v: wp.vec3,
    cx: float,
    cy: float,
    radius: float,
    height: float,
    restitution: float
) -> tuple[wp.vec3, wp.vec3]:
    """Analytical SDF collision against a vertical cylinder pillar."""
    if x[2] < height:
        dx = x[0] - cx
        dy = x[1] - cy
        dist_xy = wp.sqrt(dx * dx + dy * dy)
        if dist_xy < radius and dist_xy > 1e-6:
            nx = dx / dist_xy
            ny = dy / dist_xy
            x = wp.vec3(cx + nx * radius, cy + ny * radius, x[2])
            vn = v[0] * nx + v[1] * ny
            if vn < 0.0:
                v = wp.vec3(
                    v[0] - (1.0 + restitution) * vn * nx,
                    v[1] - (1.0 + restitution) * vn * ny,
                    v[2]
                )
    return x, v

@wp.kernel
def k_reorder_soa(
    grid: wp.uint64,
    pos_in: wp.array[wp.vec3],
    vel_in: wp.array[wp.vec3],
    pos_sorted: wp.array[wp.vec3],
    vel_sorted: wp.array[wp.vec3],
):
    """Reorders particle SoA data into spatial hash cell order for coalesced L2 cache reads."""
    tid = int(wp.tid())
    src_i = wp.hash_grid_point_id(grid, tid)
    pos_sorted[tid] = pos_in[src_i]
    vel_sorted[tid] = vel_in[src_i]

@wp.kernel
def k_reorder_soa_id(
    grid: wp.uint64,
    pos_in: wp.array[wp.vec3],
    vel_in: wp.array[wp.vec3],
    id_in: wp.array[wp.int32],
    pos_sorted: wp.array[wp.vec3],
    vel_sorted: wp.array[wp.vec3],
    id_sorted: wp.array[wp.int32],
):
    """k_reorder_soa plus the stable particle id (k_reorder_soa stays for the experimental callers)."""
    tid = int(wp.tid())
    src_i = wp.hash_grid_point_id(grid, tid)
    pos_sorted[tid] = pos_in[src_i]
    vel_sorted[tid] = vel_in[src_i]
    id_sorted[tid] = id_in[src_i]


@wp.func
def particle_density(grid: wp.uint64, pos_sorted: wp.array[wp.vec3], xi: wp.vec3,
                     mass: float, h: float) -> float:
    # Warp 1.17 must replay this dynamic accumulation before differentiating
    # nonlinear consumers. A function boundary makes that replay explicit.
    rho = float(0.0)

    neighbors = wp.hash_grid_query(grid, xi, h)
    for j in neighbors:
        r = wp.length(xi - pos_sorted[j])
        if r < h:
            rho += mass * cubic_spline_kernel(r, h)
    return rho


@wp.kernel
def k_compute_density_coalesced(
    grid: wp.uint64,
    pos_sorted: wp.array[wp.vec3],
    density_sorted: wp.array[float],
    mass: float,
    h: float,
    rest_density: float,
    contact_mode: int,
):
    tid = int(wp.tid())
    rho = particle_density(grid, pos_sorted, pos_sorted[tid], mass, h)

    if contact_mode == 2:
        # C1 transition across 1% of rest density. Outside this narrow band the
        # original constitutive law is exact; no straight-through derivative.
        width = 0.01 * rest_density
        delta = rho - rest_density
        if delta <= -width:
            density_sorted[tid] = rest_density
        elif delta >= width:
            density_sorted[tid] = rho
        else:
            density_sorted[tid] = rest_density + (delta + width) * (delta + width) / (4.0 * width)
    else:
        density_sorted[tid] = wp.max(rho, rest_density)

@wp.func
def particle_forces(
    grid: wp.uint64,
    pos_sorted: wp.array[wp.vec3],
    vel_sorted: wp.array[wp.vec3],
    density_sorted: wp.array[float],
    mass: float,
    h: float,
    rest_density: float,
    stiffness: float,
    viscosity: float,
    enable_obstacles: int,
    tid: int,
) -> tuple[wp.vec3, wp.vec3]:
    xi = pos_sorted[tid]
    vi = vel_sorted[tid]
    rho_i = density_sorted[tid]
    p_i = stiffness * (wp.pow(rho_i / rest_density, 7.0) - 1.0)
    sound_speed = wp.sqrt(7.0 * stiffness / rest_density)

    f_pressure = wp.vec3(0.0, 0.0, 0.0)
    f_visc = wp.vec3(0.0, 0.0, 0.0)

    neighbors = wp.hash_grid_query(grid, xi, h)
    for j in neighbors:
        if tid == j:
            continue
        xj = pos_sorted[j]
        diff = xi - xj
        r = wp.length(diff)
        if r < h and r > 1e-6:
            rho_j = density_sorted[j]
            p_j = stiffness * (wp.pow(rho_j / rest_density, 7.0) - 1.0)
            grad_w = cubic_spline_grad(diff, r, h)

            vj = vel_sorted[j]
            f_pressure -= mass * mass * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad_w
            f_visc += viscosity * mass * ((vj - vi) / rho_j) * cubic_spline_kernel(r, h)

            # Monaghan artificial viscosity: damps approaching pairs so impact shocks
            # cannot pump energy into the stiff gamma=7 EOS.
            vr = wp.dot(vi - vj, diff)
            if vr < 0.0 or enable_obstacles == 2:
                compression = vr
                if enable_obstacles == 2:
                    compression = 0.5 * (vr - wp.sqrt(vr * vr + 1e-8))
                mu = h * compression / (r * r + 0.01 * h * h)
                pi_ij = -ARTIFICIAL_VISCOSITY_ALPHA * sound_speed * mu / (0.5 * (rho_i + rho_j))
                f_visc -= mass * mass * pi_ij * grad_w

    return f_pressure, f_visc


@wp.kernel
def k_wcsph_forces_and_copyback(
    grid: wp.uint64,
    pos_sorted: wp.array[wp.vec3],
    vel_sorted: wp.array[wp.vec3],
    density_sorted: wp.array[float],
    pos_vbo_out: wp.array[wp.vec3],
    vel_out: wp.array[wp.vec3],
    mass: float,
    h: float,
    rest_density: float,
    stiffness: float,
    viscosity: float,
    dt: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    enable_obstacles: int,
):
    tid = int(wp.tid())
    xi = pos_sorted[tid]
    vi = vel_sorted[tid]
    # Keep loop-local adjoints separate from contact/integration adjoints.
    f_pressure, f_visc = particle_forces(grid, pos_sorted, vel_sorted, density_sorted,
                                       mass, h, rest_density, stiffness, viscosity, enable_obstacles, tid)
    sound_speed = wp.sqrt(7.0 * stiffness / rest_density)
    gravity = wp.vec3(0.0, 0.0, -9.81)
    acc = (f_pressure + f_visc) / mass + gravity

    # Soft domain walls (floor, ceiling, 4 sides) keep particles off the clamp planes.
    r0 = 0.5 * h
    acc += wp.vec3(
        wall_accel(xi[0] - domain_min[0], -vi[0], r0, sound_speed, dt)
        - wall_accel(domain_max[0] - xi[0], vi[0], r0, sound_speed, dt),
        wall_accel(xi[1] - domain_min[1], -vi[1], r0, sound_speed, dt)
        - wall_accel(domain_max[1] - xi[1], vi[1], r0, sound_speed, dt),
        wall_accel(xi[2] - domain_min[2], -vi[2], r0, sound_speed, dt)
        - wall_accel(domain_max[2] - xi[2], vi[2], r0, sound_speed, dt),
    )

    if enable_obstacles == 2:
        # Replace the original wall force in this opt-in mode with the same C1
        # law used by obstacles. Hard domain clamps below remain a safety net.
        acc = (f_pressure + f_visc) / mass + gravity
        acc += wp.vec3(
            compliant_contact_accel(xi[0] - domain_min[0], -vi[0], r0, sound_speed)
            - compliant_contact_accel(domain_max[0] - xi[0], vi[0], r0, sound_speed),
            compliant_contact_accel(xi[1] - domain_min[1], -vi[1], r0, sound_speed)
            - compliant_contact_accel(domain_max[1] - xi[1], vi[1], r0, sound_speed),
            compliant_contact_accel(xi[2] - domain_min[2], -vi[2], r0, sound_speed)
            - compliant_contact_accel(domain_max[2] - xi[2], vi[2], r0, sound_speed),
        )
        diff = xi - wp.vec3(0.0, 0.0, 0.55)
        radius = wp.length(diff)
        normal = diff / wp.max(radius, 1e-6)
        acc += normal * compliant_contact_accel(radius - 0.55, -wp.dot(vi, normal), r0, sound_speed)
        acc += compliant_cylinder_accel(xi, vi, -0.85, -0.55, r0, sound_speed)
        acc += compliant_cylinder_accel(xi, vi, -0.85, 0.55, r0, sound_speed)
        acc += compliant_cylinder_accel(xi, vi, 0.85, -0.55, r0, sound_speed)
        acc += compliant_cylinder_accel(xi, vi, 0.85, 0.55, r0, sound_speed)

    v_next = vi + acc * dt
    x_next = xi + v_next * dt

    # Optional Scene 1 GPU SDF Obstacles (Central Sphere + 4 Vertical Pillars)
    if enable_obstacles == 1:
        sph_c = wp.vec3(0.0, 0.0, 0.55)
        sph_r = 0.55
        diff_s = x_next - sph_c
        dist_s = wp.length(diff_s)
        if dist_s < sph_r and dist_s > 1e-6:
            n_s = diff_s / dist_s
            x_next = sph_c + n_s * sph_r
            vn_s = wp.dot(v_next, n_s)
            if vn_s < 0.0:
                v_next = v_next - 1.35 * vn_s * n_s

        x_next, v_next = collide_cylinder_2d(x_next, v_next, -0.85, -0.55, 0.18, 2.2, 0.35)
        x_next, v_next = collide_cylinder_2d(x_next, v_next, -0.85,  0.55, 0.18, 2.2, 0.35)
        x_next, v_next = collide_cylinder_2d(x_next, v_next,  0.85, -0.55, 0.18, 2.2, 0.35)
        x_next, v_next = collide_cylinder_2d(x_next, v_next,  0.85,  0.55, 0.18, 2.2, 0.35)

    # Domain Boundary Clamping & Damped Reflection
    damping = 0.5
    if x_next[0] < domain_min[0]:
        x_next = wp.vec3(domain_min[0], x_next[1], x_next[2])
        v_next = wp.vec3(-v_next[0] * damping, v_next[1], v_next[2])
    elif x_next[0] > domain_max[0]:
        x_next = wp.vec3(domain_max[0], x_next[1], x_next[2])
        v_next = wp.vec3(-v_next[0] * damping, v_next[1], v_next[2])

    if x_next[1] < domain_min[1]:
        x_next = wp.vec3(x_next[0], domain_min[1], x_next[2])
        v_next = wp.vec3(v_next[0], -v_next[1] * damping, v_next[2])
    elif x_next[1] > domain_max[1]:
        x_next = wp.vec3(x_next[0], domain_max[1], x_next[2])
        v_next = wp.vec3(v_next[0], -v_next[1] * damping, v_next[2])

    if x_next[2] < domain_min[2]:
        x_next = wp.vec3(x_next[0], x_next[1], domain_min[2])
        v_next = wp.vec3(v_next[0], v_next[1], -v_next[2] * damping)
    elif x_next[2] > domain_max[2]:
        x_next = wp.vec3(x_next[0], x_next[1], domain_max[2])
        v_next = wp.vec3(v_next[0], v_next[1], -v_next[2] * damping)

    vel_out[tid] = v_next
    pos_vbo_out[tid] = x_next


@wp.func
def clamp_to_domain(x: wp.vec3, v: wp.vec3, domain_min: wp.vec3, domain_max: wp.vec3,
                    damping: float) -> tuple[wp.vec3, wp.vec3]:
    """Hard safety clamp with damped reflection, identical to the legacy kernel's inline clamps."""
    px = x
    pv = v
    for axis in range(3):
        if px[axis] < domain_min[axis]:
            px[axis] = domain_min[axis]
            pv[axis] = -pv[axis] * damping
        elif px[axis] > domain_max[axis]:
            px[axis] = domain_max[axis]
            pv[axis] = -pv[axis] * damping
    return px, pv


@wp.func
def collider_distance(mesh_id: wp.uint64, xform: wp.transform, x: wp.vec3,
                      radius: float) -> tuple[float, wp.vec3, wp.vec3]:
    """mesh_signed_distance for a collider placed by a rigid transform (mesh stored in its local frame)."""
    d, n_local, cp_local = mesh_signed_distance(mesh_id, wp.transform_point(wp.transform_inverse(xform), x), radius)
    return d, wp.transform_vector(xform, n_local), wp.transform_point(xform, cp_local)


@wp.func
def collider_deep_distance(mesh_id: wp.uint64, xform: wp.transform, x: wp.vec3, d_near: float, radius: float,
                           lower: wp.vec3, upper: wp.vec3) -> tuple[float, wp.vec3, wp.vec3]:
    """Signed distance of a point that collider_distance found no face for within `radius` (d_near == radius).

    Such a point is either clear of the collider or deeper inside it than `radius`; the near query cannot
    tell, so a particle carried deep in within one substep (a fast collider, a large time step) would stay
    trapped. Points inside the collider's local bounding box are re-queried without the radius limit; all
    others keep d_near. Points with a face within `radius` never get here, so their result is unchanged.
    """
    d = d_near
    n = wp.vec3(0.0, 0.0, 0.0)
    cp = x
    xl = wp.transform_point(wp.transform_inverse(xform), x)
    if (xl[0] >= lower[0] and xl[0] <= upper[0] and xl[1] >= lower[1] and xl[1] <= upper[1]
            and xl[2] >= lower[2] and xl[2] <= upper[2]):
        d_l, n_l, cp_l = mesh_deep_signed_distance(mesh_id, xl, wp.length(upper - lower) + radius)
        d = d_l
        n = wp.transform_vector(xform, n_l)
        cp = wp.transform_point(xform, cp_l)
    return d, n, cp


@wp.func
def collider_velocity(lin: wp.vec3, ang: wp.vec3, xform: wp.transform, x: wp.vec3) -> wp.vec3:
    """Velocity of the collider's material point at x (rigid motion about its pivot)."""
    return lin + wp.cross(ang, x - wp.transform_get_translation(xform))


@wp.kernel
def k_scene_density(
    grid: wp.uint64,
    pos_sorted: wp.array[wp.vec3],
    density_sorted: wp.array[float],
    mass: float,
    h: float,
    rest_density: float,
    active: wp.array[wp.int32],
):
    """k_compute_density_coalesced (default contact mode) for the first active[0] particles only.

    Launched over the whole pool so the captured graph never changes; the live count is read on device.
    """
    tid = int(wp.tid())
    if tid >= active[0]:
        return
    density_sorted[tid] = wp.max(particle_density(grid, pos_sorted, pos_sorted[tid], mass, h), rest_density)


@wp.kernel
def k_scene_forces(
    grid: wp.uint64,
    pos_sorted: wp.array[wp.vec3],
    vel_sorted: wp.array[wp.vec3],
    density_sorted: wp.array[float],
    pos_vbo_out: wp.array[wp.vec3],
    vel_out: wp.array[wp.vec3],
    mass: float,
    h: float,
    rest_density: float,
    stiffness: float,
    viscosity: float,
    dt: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    gravity: wp.vec3,
    active: wp.array[wp.int32],
):
    """Scene-mode fluid forces, user gravity and soft domain walls, integrated into the output buffers.

    Like k_wcsph_forces_and_copyback without the analytic obstacles; k_scene_boundaries then applies
    the colliders and the hard domain clamp. Colliders are a separate kernel so their mesh-query code
    does not raise the register use (and lower the occupancy) of the neighbour loop. The legacy kernel
    is left untouched because the experimental differentiable code launches it.
    """
    tid = int(wp.tid())
    if tid >= active[0]:
        return
    xi = pos_sorted[tid]
    vi = vel_sorted[tid]
    f_pressure, f_visc = particle_forces(grid, pos_sorted, vel_sorted, density_sorted,
                                         mass, h, rest_density, stiffness, viscosity, 0, tid)
    sound_speed = wp.sqrt(7.0 * stiffness / rest_density)
    acc = (f_pressure + f_visc) / mass + gravity

    r0 = 0.5 * h
    acc += wp.vec3(
        wall_accel(xi[0] - domain_min[0], -vi[0], r0, sound_speed, dt)
        - wall_accel(domain_max[0] - xi[0], vi[0], r0, sound_speed, dt),
        wall_accel(xi[1] - domain_min[1], -vi[1], r0, sound_speed, dt)
        - wall_accel(domain_max[1] - xi[1], vi[1], r0, sound_speed, dt),
        wall_accel(xi[2] - domain_min[2], -vi[2], r0, sound_speed, dt)
        - wall_accel(domain_max[2] - xi[2], vi[2], r0, sound_speed, dt),
    )
    v_next = vi + acc * dt
    vel_out[tid] = v_next
    pos_vbo_out[tid] = xi + v_next * dt


@wp.kernel
def k_scene_boundaries(
    pos_sorted: wp.array[wp.vec3],
    vel_sorted: wp.array[wp.vec3],
    pos_out: wp.array[wp.vec3],
    vel_out: wp.array[wp.vec3],
    h: float,
    rest_density: float,
    stiffness: float,
    dt: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    collider_ids: wp.array[wp.uint64],
    num_colliders: int,
    collider_radius: float,
    collider_xform: wp.array[wp.transform],
    collider_lin: wp.array[wp.vec3],
    collider_ang: wp.array[wp.vec3],
    collider_lower: wp.array[wp.vec3],
    collider_upper: wp.array[wp.vec3],
    active: wp.array[wp.int32],
):
    """Rigid mesh colliders and the hard domain clamp, after k_scene_forces integrated the fluid.

    Colliders get the same soft wall law as the domain planes, driven by mesh signed distance within
    `collider_radius` of the substep's start position, plus a hard projection to the surface if a
    particle still ends up inside, also deeper than `collider_radius` (collider_deep_distance). The soft
    law's acceleration a enters exactly as if it had been part of the integration (v += a dt, x += a dt^2). Each collider is placed by `collider_xform` (identity
    for static ones) and both laws act on the velocity relative to the collider's surface, so a moving
    collider pushes and drags the fluid. Transforms, collider velocities and the active particle count
    are device buffers, so the captured graph reads their current values.
    """
    tid = int(wp.tid())
    if tid >= active[0]:
        return
    xi = pos_sorted[tid]
    vi = vel_sorted[tid]
    x_next = pos_out[tid]
    v_next = vel_out[tid]
    sound_speed = wp.sqrt(7.0 * stiffness / rest_density)
    r0 = 0.5 * h

    acc = wp.vec3(0.0, 0.0, 0.0)
    for c in range(num_colliders):
        xform = collider_xform[c]
        d, n, _cp = collider_distance(collider_ids[c], xform, xi, collider_radius)
        if d < r0:
            v_rel = vi - collider_velocity(collider_lin[c], collider_ang[c], xform, xi)
            acc += n * wall_accel(d, -wp.dot(v_rel, n), r0, sound_speed, dt)
    v_next = v_next + acc * dt
    x_next = x_next + acc * (dt * dt)

    for c in range(num_colliders):
        xform = collider_xform[c]
        d, n, cp = collider_distance(collider_ids[c], xform, x_next, collider_radius)
        if d >= collider_radius:
            # No face within the query radius. A particle that ended the previous substep outside can only
            # be deeper than the radius now if it moved at least that far relative to the collider (at the
            # default time step that takes ~30 m/s), so only such fast movers pay for the unbounded query.
            v_c = collider_velocity(collider_lin[c], collider_ang[c], xform, x_next)
            if wp.length(x_next - xi - v_c * dt) >= 0.5 * collider_radius:
                d, n, cp = collider_deep_distance(collider_ids[c], xform, x_next, d, collider_radius,
                                                  collider_lower[c], collider_upper[c])
        if d < 0.0:
            x_next = cp + n * (0.01 * h)
            vn = wp.dot(v_next - collider_velocity(collider_lin[c], collider_ang[c], xform, x_next), n)
            if vn < 0.0:
                v_next = v_next - 1.5 * vn * n

    x_next, v_next = clamp_to_domain(x_next, v_next, domain_min, domain_max, 0.5)

    vel_out[tid] = v_next
    pos_out[tid] = x_next


@wp.kernel
def k_set_active(active: wp.array[wp.int32], count: int):
    active[0] = count


@wp.kernel
def k_emit(
    nozzle_pos: wp.array[wp.vec3],
    nozzle_vel: wp.array[wp.vec3],
    offset: int,
    count: int,
    first_index: int,
    first_slot: int,
    rate: float,
    elapsed: float,
    first_id: int,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    pid: wp.array[wp.int32],
):
    """Activates pool slots [first_slot, first_slot + dim) with an emitter's particles first_index, ...

    Particle g leaves nozzle (g mod count) of the emitter's shuffled nozzles at (g + 1) / rate seconds
    after the emitter started; it is placed where it has travelled since then (`elapsed` is the time
    since the start), so a steady stream has no gaps at substep boundaries. The new particles take the
    fresh ids first_id, first_id + 1, ... (a free slot's id entry is stale after the reorder swap).
    """
    j = int(wp.tid())
    g = first_index + j
    k = offset + g % count
    v = nozzle_vel[k]
    age = wp.max(elapsed - float(g + 1) / rate, 0.0)
    pos[first_slot + j] = nozzle_pos[k] + v * age
    vel[first_slot + j] = v
    pid[first_slot + j] = first_id + j


@wp.kernel
def k_write_key(key_p: wp.array[wp.vec3], key_q: wp.array[wp.quat], index: int, p: wp.vec3, q: wp.quat):
    key_p[index] = p
    key_q[index] = q


@wp.kernel
def k_collider_motion(
    key_p: wp.array[wp.vec3],
    key_q: wp.array[wp.quat],
    motion_collider: wp.array[wp.int32],
    num_motion: int,
    slot_a: int,
    slot_b: int,
    alpha: float,
    key_rate: float,
    collider_xform: wp.array[wp.transform],
    collider_lin: wp.array[wp.vec3],
    collider_ang: wp.array[wp.vec3],
):
    """Pose and velocity of each animated collider between two frame keys (alpha in [0, 1]).

    Position is interpolated linearly and rotation by slerp; the velocities are the key differences
    times `key_rate` (keys per second; 0 holds the collider still).
    """
    m = int(wp.tid())
    pa = key_p[slot_a * num_motion + m]
    pb = key_p[slot_b * num_motion + m]
    qa = key_q[slot_a * num_motion + m]
    qb = key_q[slot_b * num_motion + m]
    if wp.dot(qa, qb) < 0.0:  # shortest arc
        qb = -qb
    c = motion_collider[m]
    collider_xform[c] = wp.transform(pa + (pb - pa) * alpha, wp.quat_slerp(qa, qb, alpha))
    collider_lin[c] = (pb - pa) * key_rate
    dq = qb * wp.quat_inverse(qa)  # world-frame rotation from key a to key b
    axis = wp.vec3(dq[0], dq[1], dq[2])
    w = dq[3]
    if w < 0.0:
        axis = -axis
        w = -w
    s = wp.length(axis)
    omega = axis * (2.0 / wp.max(w, 1e-6))  # small-angle limit of 2 * atan2(s, w) / s
    if s > 1e-6:
        omega = axis * (2.0 * wp.atan2(s, w) / s)
    collider_ang[c] = omega * key_rate


@wp.kernel
def k_scale_copy(src: wp.array[wp.vec3], dst: wp.array[wp.vec3], scale: float):
    tid = int(wp.tid())
    dst[tid] = src[tid] * scale


# Pool slots without a particle sit here: far outside any view, so they are clipped, never drawn.
PARKED_POSITION = 1.0e30

# Frame keys kept per animated collider (a ring indexed by frame number).
KEY_RING = 8


@dataclass
class _EmitterSchedule:
    """Host-side emission clock of one emitter (all times in simulation seconds)."""

    name: str
    offset: int        # first nozzle in the solver's nozzle arrays
    count: int         # nozzles
    rate: float        # particles per second
    t_start: float
    t_end: float
    emitted: int = 0


class RealtimeWCSPHSolver:
    def __init__(
        self,
        num_particles: int = 1_000_000,
        h: float = 0.04,
        substeps: int = 2,
        enable_obstacles: int = 1,
        device: str = "cuda:0",
    ):
        # The product always runs on cuda:0 (never silently changed). "cpu" exists for headless
        # validation in CI without a GPU: eager launches only, no CUDA graph.
        self.device = device
        self.num_particles = num_particles
        self.h = h
        self.substeps = substeps
        self.enable_obstacles = enable_obstacles
        self.rest_density = 1000.0
        self.mass = lattice_rest_mass(h * 0.5, h, self.rest_density)
        self.stiffness = 90000.0  # c ~= 25 m/s; lower values let impacts reach Mach 0.4+ and blow up
        self.viscosity = 0.15  # kg/s per particle; the force is viscosity * (smoothed neighbour velocity - v)
        self.cfl_factor = 0.4  # dt as a fraction of the sound-crossing time of one smoothing length
        self.max_dt = None
        self.dt = self.cfl_factor * h / (7.0 * self.stiffness / self.rest_density) ** 0.5
        self.domain_min = wp.vec3(-2.5, -2.5, 0.0)
        self.domain_max = wp.vec3(2.5, 2.5, 5.0)
        self.gravity = wp.vec3(0.0, 0.0, -9.81)
        # Scene mode (set_scene): a SceneDescription's gravity, mesh colliders and emitters. Off for the
        # raw solver that the experimental differentiable code drives.
        self.scene_mode = False
        self.colliders: MeshSet | None = None
        self.collider_radius = 0.0
        self.collider_xform = self.collider_lin = self.collider_ang = None
        self._motions = ()
        self._key_p = self._key_q = self._motion_collider = None
        self._key_frames: dict[int, int] = {}  # frame -> ring slot
        # Particle pool: slots [0, active_count) hold particles, the rest wait for emitters (parked).
        # num_particles is the pool size; the device copy of the count is what the captured graph reads.
        self.active_count = num_particles
        self.active = wp.array([num_particles], dtype=wp.int32, device=self.device)
        self._next_id = num_particles     # host counter: next fresh particle id (k_emit)
        self.sim_time = 0.0
        self.unit_scale = 1.0
        self.start_frame = 1.0
        self.fps = 24.0
        self._emitters: list[_EmitterSchedule] = []
        self.nozzle_pos = self.nozzle_vel = None
        self.emission_exhausted = False
        self._exhaustion_unreported = False

        # O(n) spatial hash grid (128^3 cells)
        self.grid = wp.HashGrid(128, 128, 128, device=self.device)

        # Primary SoA GPU Buffers
        self.pos_sim = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.vel = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)

        # Reordered SoA Buffers for Coalesced L2 Cache Neighbor Traversal
        self.pos_sorted = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.vel_sorted = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.density_sorted = wp.zeros(num_particles, dtype=float, device=self.device)

        # Stable particle identity. Every step rewrites pos_sim / vel in spatial-sort order, so the array
        # index is not an identity; `pid[i]` is the id of the particle now stored at index i.
        self.pid = wp.array(np.arange(num_particles, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.pid_sorted = wp.zeros(num_particles, dtype=wp.int32, device=self.device)

        # Warp 1.16+ CUDA Graph for zero-overhead physics launches
        self.cuda_graph = None
        self.rebuild_cuda_graph()

    def set_particle_spacing(self, spacing: float, max_dt: float | None = None):
        """Matches h and mass to the spawn spacing (h = 2*spacing) and sets dt to the CFL limit.

        Call rebuild_cuda_graph() afterwards so the captured kernels pick up the new values.
        """
        self.h = 2.0 * spacing
        self.mass = lattice_rest_mass(spacing, self.h, self.rest_density)
        self.max_dt = max_dt
        self.collider_radius = self.h
        self.update_time_step()

    def update_time_step(self):
        """dt = cfl_factor * h / c. Larger cfl_factor or lower stiffness = faster but less stable.

        Call rebuild_cuda_graph() afterwards so the captured kernels pick up the new values.
        """
        sound_speed = (7.0 * self.stiffness / self.rest_density) ** 0.5
        self.dt = self.cfl_factor * self.h / sound_speed
        if self.max_dt is not None:
            self.dt = min(self.dt, self.max_dt)

    @property
    def effective_viscosity(self) -> float:
        """`viscosity` capped to keep the explicit relaxation stable at the current mass and time step.

        Read when the kernels are launched (or captured), so call rebuild_cuda_graph() after changing
        `viscosity`, `dt` or `mass`.
        """
        return min(self.viscosity, VISCOSITY_RELAXATION_LIMIT * self.mass / self.dt)

    @property
    def viscosity_limited(self) -> bool:
        """True when `effective_viscosity` is below the requested `viscosity` (the stability cap applies)."""
        return self.effective_viscosity < self.viscosity

    def _report_viscosity_cap(self):
        key = (self.viscosity, self.effective_viscosity)
        if self.viscosity_limited and key != getattr(self, "_viscosity_cap_reported", None):
            log.warning("viscosity %g is limited to %g at particle spacing %g and time step %g s (explicit "
                        "stability cap)", self.viscosity,
                        self.effective_viscosity, 0.5 * self.h, self.dt)
            self._viscosity_cap_reported = key

    def rebuild_cuda_graph(self):
        """Captures the coalesced Density + WCSPH Force + SDF Obstacle + Copy-back kernels into a CUDA Graph.

        A no-op on the CPU device, where step() launches the kernels eagerly. Logs a warning (once per
        setting) when `viscosity` is capped.
        """
        self._report_viscosity_cap()
        if not wp.get_device(self.device).is_cuda:
            self.cuda_graph = None
            return
        with wp.ScopedCapture(device=self.device) as capture:
            self._launch_density_and_forces()
        self.cuda_graph = capture.graph

    def _launch_density_and_forces(self):
        """Density + force/integrate kernels on the sorted buffers; shared by eager steps and graph capture."""
        if self.scene_mode:
            assert self.colliders is not None
            wp.launch(
                kernel=k_scene_density,
                dim=self.num_particles,
                inputs=[self.grid.id, self.pos_sorted, self.density_sorted,
                        self.mass, self.h, self.rest_density, self.active],
                device=self.device,
            )
            wp.launch(
                kernel=k_scene_forces,
                dim=self.num_particles,
                inputs=[
                    self.grid.id, self.pos_sorted, self.vel_sorted, self.density_sorted,
                    self.pos_sim, self.vel, self.mass, self.h, self.rest_density,
                    self.stiffness, self.effective_viscosity, self.dt, self.domain_min, self.domain_max,
                    self.gravity, self.active,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=k_scene_boundaries,
                dim=self.num_particles,
                inputs=[
                    self.pos_sorted, self.vel_sorted, self.pos_sim, self.vel, self.h, self.rest_density,
                    self.stiffness, self.dt, self.domain_min, self.domain_max,
                    self.colliders.ids, self.colliders.count, self.collider_radius,
                    self.collider_xform, self.collider_lin, self.collider_ang,
                    self.colliders.lower, self.colliders.upper, self.active,
                ],
                device=self.device,
            )
            return
        wp.launch(
            kernel=k_compute_density_coalesced,
            dim=self.num_particles,
            inputs=[self.grid.id, self.pos_sorted, self.density_sorted,
                    self.mass, self.h, self.rest_density, self.enable_obstacles],
            device=self.device,
        )
        wp.launch(
            kernel=k_wcsph_forces_and_copyback,
            dim=self.num_particles,
            inputs=[
                self.grid.id, self.pos_sorted, self.vel_sorted, self.density_sorted,
                self.pos_sim, self.vel, self.mass, self.h, self.rest_density,
                self.stiffness, self.effective_viscosity, self.dt, self.domain_min, self.domain_max,
                self.enable_obstacles,
            ],
            device=self.device,
        )

    def set_scene(self, desc: SceneDescription, active_count: int | None = None):
        """Adopts a scene's domain, gravity, unit scale, mesh colliders (one wp.Mesh each, built once) and
        emitters. Call set_particle_spacing first (emitter nozzles use its spacing) and
        rebuild_cuda_graph() afterwards.

        `active_count` is how many pool slots already hold particles (default: all); the caller places
        them and parks the rest. Animated colliders start at their start-frame pose; later poses come
        from set_collider_keyframe. Disables the analytic demo obstacles.
        """
        device = self.device
        self.domain_min = wp.vec3(*desc.domain_min)
        self.domain_max = wp.vec3(*desc.domain_max)
        self.gravity = wp.vec3(*desc.gravity)
        self.enable_obstacles = 0
        self.unit_scale, self.start_frame, self.fps = desc.unit_scale, desc.start_frame, desc.fps
        self.sim_time = 0.0

        # Colliders: moving ones are stored relative to their pivot and placed by a transform.
        count = len(desc.colliders)
        origins = np.zeros((count, 3))
        poses = np.zeros((max(count, 1), 7), dtype=np.float32)
        poses[:, 6] = 1.0  # identity rotation (x, y, z, w)
        for motion in desc.collider_motion:
            origins[motion.collider] = motion.pivot
            poses[motion.collider, :3] = motion.pivot
        self.colliders = MeshSet(desc.colliders, device, origins)
        self.collider_xform = wp.array(poses, dtype=wp.transform, device=device)
        self.collider_lin = wp.zeros(max(count, 1), dtype=wp.vec3, device=device)
        self.collider_ang = wp.zeros(max(count, 1), dtype=wp.vec3, device=device)
        self._motions = desc.collider_motion
        self._key_frames = {}
        if self._motions:
            m = len(self._motions)
            self._motion_collider = wp.array([mo.collider for mo in self._motions], dtype=wp.int32, device=device)
            key_p = np.zeros((KEY_RING, m, 3), dtype=np.float32)
            key_q = np.zeros((KEY_RING, m, 4), dtype=np.float32)
            key_q[..., 3] = 1.0
            first = math.floor(self.start_frame)
            key_p[first % KEY_RING] = [mo.pivot for mo in self._motions]  # start pose: pivot, no rotation
            self._key_p = wp.array(key_p.reshape(-1, 3), dtype=wp.vec3, device=device)
            self._key_q = wp.array(key_q.reshape(-1, 4), dtype=wp.quat, device=device)
            self._key_frames[first] = first % KEY_RING

        self._set_active_count(self.num_particles if active_count is None else active_count)
        # The placed particles keep ids [0, active_count) from construction; emitted ones count on from there.
        self._next_id = self.active_count
        self._setup_emitters(desc)
        if self._emitters:
            self.grid.reserve(self.num_particles)  # the grid never grows while the pool fills
        self.scene_mode = True
        self.collider_radius = self.h  # re-derived in set_particle_spacing

    def _setup_emitters(self, desc: SceneDescription):
        self._emitters = []
        self.nozzle_pos = self.nozzle_vel = None
        self.emission_exhausted = False
        self._exhaustion_unreported = False
        if not desc.emitters:
            return
        from ..scene.fill import emitter_nozzle_arrays

        # A particle starts at most one substep of travel past its nozzle (k_emit), so nozzles keep one
        # spacing plus that travel from every wall and collider: no particle starts in a boundary layer.
        spacing = 0.5 * self.h
        speeds = [abs(e.speed) if e.velocity_mode == "NORMAL" else math.hypot(*e.vector) for e in desc.emitters]
        clearance = spacing + max(speeds) * self.dt
        positions, velocities, ranges = emitter_nozzle_arrays(desc, spacing, self.device, clearance)
        self.nozzle_pos = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.nozzle_vel = wp.array(velocities, dtype=wp.vec3, device=self.device)
        for emitter, (offset, count, natural) in zip(desc.emitters, ranges, strict=True):
            # Above the natural rate particles would start closer than one spacing: clamp to it.
            rate = natural if emitter.rate <= 0.0 else min(emitter.rate, natural)
            self._emitters.append(_EmitterSchedule(
                emitter.mesh.name, offset, count, rate,
                max((emitter.frame_start - desc.start_frame) / desc.fps, 0.0),
                (emitter.frame_end - desc.start_frame) / desc.fps))

    @property
    def emitter_rates(self) -> dict[str, float]:
        """Particles per second of each emitter while it is on (after the full-inflow clamp)."""
        return {e.name: e.rate for e in self._emitters}

    def _set_active_count(self, count: int):
        self.active_count = int(count)
        wp.launch(k_set_active, dim=1, inputs=[self.active, self.active_count], device=self.device)

    def _emit(self):
        """Activates the particles that the emitters released by the current simulation time.

        Host bookkeeping plus one small launch per emitting emitter, outside graph capture; nothing is
        allocated. Once the pool is full, emission stops for the rest of the run.
        """
        if not self._emitters or self.emission_exhausted:
            return
        n = self.active_count
        for e in self._emitters:
            if self.sim_time <= e.t_start:
                continue
            due = math.floor(e.rate * (min(self.sim_time, e.t_end) - e.t_start) + 1e-9) - e.emitted
            if due <= 0:
                continue
            if due > self.num_particles - n:
                due = self.num_particles - n
                self.emission_exhausted = self._exhaustion_unreported = True
            if due > 0:
                wp.launch(k_emit, dim=due, inputs=[
                    self.nozzle_pos, self.nozzle_vel, e.offset, e.count, e.emitted, n, e.rate,
                    self.sim_time - e.t_start, self._next_id, self.pos_sim, self.vel, self.pid],
                    device=self.device)
                e.emitted += due
                self._next_id += due
                n += due
            if self.emission_exhausted:
                break
        if n != self.active_count:
            self._set_active_count(n)

    @property
    def has_emitters(self) -> bool:
        """True when the pool fills while running (the particle count of a frame is not the pool size)."""
        return bool(self._emitters)

    def restore_state(self, position: np.ndarray, velocity: np.ndarray, ids: np.ndarray,
                      steps: int | None = None):
        """Loads a recorded state (bake resume): the first len(position) slots hold particles, the rest park.

        One-off host upload, never per frame. `steps` (substeps since the scene start, at the current dt)
        re-derives the clock (animated colliders) and the emitters' progress; None keeps the current clock. Raises ValueError
        if the count does not fit the pool (or differs from it without emitters).
        """
        n = len(position)
        if not 0 < n <= self.num_particles or (n != self.num_particles and not self._emitters):
            raise ValueError(f"cached frame has {n} particles, the scene has {self.num_particles}")
        pos = np.full((self.num_particles, 3), PARKED_POSITION, dtype=np.float32)
        vel = np.zeros((self.num_particles, 3), dtype=np.float32)
        pid = np.arange(self.num_particles, dtype=np.int32)
        pos[:n], vel[:n], pid[:n] = position, velocity, ids
        self.pos_sim.assign(pos)
        self.vel.assign(vel)
        self.pid.assign(pid)
        if steps is not None:
            # Summed substep by substep like step(), so the clock is bit-identical to an uninterrupted run;
            # the last emission happened at the start of the last substep (`emit_time`), as in _emit.
            t = emit_time = 0.0
            for _ in range(int(steps)):
                emit_time = t
                t += self.dt
            self.sim_time = t
            for e in self._emitters:
                due = math.floor(e.rate * (min(emit_time, e.t_end) - e.t_start) + 1e-9)
                e.emitted = max(due, 0) if steps and emit_time > e.t_start else 0
        self.emission_exhausted = bool(self._emitters) and n >= self.num_particles
        self._exhaustion_unreported = False
        self._set_active_count(n)
        self._next_id = max(int(np.max(ids)) + 1, n)

    def take_exhaustion_notice(self) -> bool:
        """True once, after emission stopped because the particle budget is used up."""
        notice, self._exhaustion_unreported = self._exhaustion_unreported, False
        return notice

    # -- animated colliders --------------------------------------------------------------------------

    def _frame_at(self, t: float) -> float:
        return self.start_frame + t * self.fps

    @property
    def current_frame(self) -> int:
        """The scene frame the simulation clock is in."""
        return math.floor(self._frame_at(self.sim_time))

    @property
    def animated_collider_names(self) -> list[str]:
        """Object names of the animated colliders, in the order set_collider_keyframe expects."""
        return [motion.name for motion in self._motions]

    def collider_frames_needed(self) -> list[int]:
        """Scene frames whose animated-collider transforms the next step() needs and does not have.

        The caller reads each frame's world matrices (once per frame) and passes them to
        set_collider_keyframe before stepping. Empty when no collider is animated.
        """
        if not self._motions:
            return []
        first = math.floor(self._frame_at(self.sim_time))
        last = math.floor(self._frame_at(self.sim_time + self.substeps * self.dt)) + 1
        frames = range(first, min(last, first + KEY_RING - 1) + 1)
        return [f for f in frames if f not in self._key_frames]

    def set_collider_keyframe(self, frame: int, matrices):
        """Stores the animated colliders' world matrices (4x4, scene units, in collider_motion order) at
        `frame`. Raises SceneValidationError if a collider's motion is not rigid."""
        frame = int(frame)
        if len(matrices) != len(self._motions):
            raise ValueError(f"expected {len(self._motions)} collider matrices, got {len(matrices)}")
        keys = [rigid_key(motion, matrix, self.unit_scale) for motion, matrix in zip(self._motions, matrices, strict=True)]
        slot = frame % KEY_RING
        for old in [f for f, s in self._key_frames.items() if s == slot]:
            del self._key_frames[old]
        for m, (origin, quat) in enumerate(keys):
            wp.launch(k_write_key, dim=1, inputs=[
                self._key_p, self._key_q, slot * len(self._motions) + m,
                wp.vec3(*origin.tolist()), wp.quat(*quat.tolist())], device=self.device)
        self._key_frames[frame] = slot

    def _update_colliders(self):
        """Poses and velocities of the animated colliders for the coming substep (end-of-substep time)."""
        if not self._motions:
            return
        f = self._frame_at(self.sim_time + self.dt)
        fa = math.floor(f)
        slot_a, slot_b = self._key_frames.get(fa), self._key_frames.get(fa + 1)
        alpha, key_rate = f - fa, self.fps
        if slot_a is None or slot_b is None:
            # A missing key holds the nearest known pose still (no caller fed the frames).
            known = slot_a if slot_a is not None else slot_b
            if known is None:
                before = [k for k in self._key_frames if k <= fa]
                known = self._key_frames[max(before) if before else min(self._key_frames)]
            slot_a = slot_b = known
            alpha, key_rate = 0.0, 0.0
        wp.launch(k_collider_motion, dim=len(self._motions), inputs=[
            self._key_p, self._key_q, self._motion_collider, len(self._motions), slot_a, slot_b,
            float(alpha), float(key_rate), self.collider_xform, self.collider_lin, self.collider_ang],
            device=self.device)

    @classmethod
    def from_scene(cls, desc: SceneDescription, spacing: float, substeps: int = 2,
                   stiffness: float | None = None, cfl_factor: float | None = None,
                   viscosity: float | None = None, max_particles: int = MAX_PARTICLES, device: str = "cuda:0") -> "RealtimeWCSPHSolver":
        """Builds a ready-to-step solver for an authored scene.

        Fills the fluid volumes on a cubic lattice at `spacing` (h = 2*spacing, mass from the same
        lattice, so the fill is exactly at rest density), starts at rest, and captures the CUDA graph.
        Without emitters the pool is exactly the fill. With emitters it is the scene's particle budget,
        allocated once: the fill takes the first slots and emitters activate the rest.
        Raises SceneFillError if the fill is empty (and there are no emitters), exceeds `max_particles`
        or the budget, or an emitter cannot emit.
        """
        from ..scene.fill import SceneFillError, fill_particles

        points = fill_particles(desc, spacing, device=device)
        if len(points) > max_particles:
            raise SceneFillError(
                f"scene fills {len(points)} particles at spacing {spacing}, above the {max_particles} limit; "
                "increase the particle spacing")
        capacity = len(points)
        if desc.emitters:
            capacity = desc.particle_budget
            if capacity > max_particles:
                raise SceneFillError(f"particle budget {capacity} is above the {max_particles} limit")
            if len(points) > capacity:
                raise SceneFillError(
                    f"scene fills {len(points)} particles at spacing {spacing}, above the particle budget "
                    f"{capacity}; raise Max Particles or increase the particle spacing")
        solver = cls(num_particles=capacity, h=2.0 * spacing, substeps=substeps, enable_obstacles=0, device=device)
        if stiffness is not None:
            solver.stiffness = stiffness
        if cfl_factor is not None:
            solver.cfl_factor = cfl_factor
        if viscosity is not None:
            solver.viscosity = viscosity
        solver.set_particle_spacing(spacing)
        solver.set_scene(desc, active_count=len(points))
        initial = np.full((capacity, 3), PARKED_POSITION, dtype=np.float32)
        initial[:len(points)] = points
        solver.pos_sim.assign(initial)
        solver.vel.zero_()
        solver.rebuild_cuda_graph()
        return solver

    def get_vram_usage_mb(self) -> float:
        bytes_per_particle = (5 * 12) + 4 + 12 + 2 * 4  # + pid and pid_sorted
        return (self.num_particles * bytes_per_particle) / (1024.0 * 1024.0)

    def copy_positions(self, out: wp.array):
        """Writes the particle positions to `out` in scene units (pool slots without a particle stay parked)."""
        if self.unit_scale == 1.0:
            wp.copy(out, self.pos_sim)
        else:
            wp.launch(k_scale_copy, dim=self.num_particles, inputs=[self.pos_sim, out, 1.0 / self.unit_scale],
                      device=self.device)

    def step(self, pos_vbo_wp: wp.array, use_graph: bool = True):
        for _ in range(self.substeps):
            n = self.num_particles
            if self.scene_mode:
                self._emit()
                self._update_colliders()
                n = self.active_count
            # Only the live particles are hashed and sorted; views share memory, nothing is allocated.
            pos = self.pos_sim if n == self.num_particles else self.pos_sim[:n]
            pos_sorted = self.pos_sorted if n == self.num_particles else self.pos_sorted[:n]
            if n > 0:
                self.grid.build(points=pos, radius=self.h)

                wp.launch(
                    kernel=k_reorder_soa_id,
                    dim=n,
                    inputs=[self.grid.id, self.pos_sim, self.vel, self.pid,
                            self.pos_sorted, self.vel_sorted, self.pid_sorted],
                    device=self.device,
                )
                # pos_sim / vel leave this substep in sorted order, so the sorted ids become the ids. A swap,
                # not a copy: the id arrays are never captured in the CUDA graph. Only [0, n) is meaningful
                # after the swap; k_emit writes the ids of the slots it activates.
                self.pid, self.pid_sorted = self.pid_sorted, self.pid

                self.grid.build(points=pos_sorted, radius=self.h)

                if use_graph and self.cuda_graph is not None:
                    wp.capture_launch(self.cuda_graph)
                else:
                    self._launch_density_and_forces()
            self.sim_time += self.dt

        self.copy_positions(pos_vbo_wp)
