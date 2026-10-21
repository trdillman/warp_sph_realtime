import warp as wp

wp.init()

ARTIFICIAL_VISCOSITY_ALPHA = wp.constant(0.1)

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
def wall_accel(d: float, v_into: float, r0: float, sound_speed: float) -> float:
    """Repulsive acceleration away from a wall at distance d (soft boundary layer of width r0).

    Stiffness scales with c^2 so the layer is as stiff as the fluid; the damping term only acts on
    motion into the wall, keeping hard position clamps as a rarely used last resort.
    """
    if d >= r0:
        return 0.0
    overlap = 1.0 - wp.max(d, 0.0) / r0
    omega = sound_speed / r0
    a = 0.5 * sound_speed * sound_speed / r0 * overlap
    if v_into > 0.0:
        a += omega * v_into * overlap
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
        wall_accel(xi[0] - domain_min[0], -vi[0], r0, sound_speed)
        - wall_accel(domain_max[0] - xi[0], vi[0], r0, sound_speed),
        wall_accel(xi[1] - domain_min[1], -vi[1], r0, sound_speed)
        - wall_accel(domain_max[1] - xi[1], vi[1], r0, sound_speed),
        wall_accel(xi[2] - domain_min[2], -vi[2], r0, sound_speed)
        - wall_accel(domain_max[2] - xi[2], vi[2], r0, sound_speed),
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


class RealtimeWCSPHSolver:
    def __init__(
        self,
        num_particles: int = 1_000_000,
        h: float = 0.04,
        substeps: int = 2,
        enable_obstacles: int = 1,
    ):
        self.num_particles = num_particles
        self.h = h
        self.substeps = substeps
        self.enable_obstacles = enable_obstacles
        self.rest_density = 1000.0
        self.mass = lattice_rest_mass(h * 0.5, h, self.rest_density)
        self.stiffness = 90000.0  # c ~= 25 m/s; lower values let impacts reach Mach 0.4+ and blow up
        self.viscosity = 0.15
        self.cfl_factor = 0.4  # dt as a fraction of the sound-crossing time of one smoothing length
        self.max_dt = None
        self.dt = self.cfl_factor * h / (7.0 * self.stiffness / self.rest_density) ** 0.5
        self.domain_min = wp.vec3(-2.5, -2.5, 0.0)
        self.domain_max = wp.vec3(2.5, 2.5, 5.0)

        # O(n) spatial hash grid (128^3 cells)
        self.grid = wp.HashGrid(128, 128, 128, device="cuda:0")

        # Primary SoA GPU Buffers
        self.pos_sim = wp.zeros(num_particles, dtype=wp.vec3, device="cuda:0")
        self.vel = wp.zeros(num_particles, dtype=wp.vec3, device="cuda:0")

        # Reordered SoA Buffers for Coalesced L2 Cache Neighbor Traversal
        self.pos_sorted = wp.zeros(num_particles, dtype=wp.vec3, device="cuda:0")
        self.vel_sorted = wp.zeros(num_particles, dtype=wp.vec3, device="cuda:0")
        self.density_sorted = wp.zeros(num_particles, dtype=float, device="cuda:0")

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
        self.update_time_step()

    def update_time_step(self):
        """dt = cfl_factor * h / c. Larger cfl_factor or lower stiffness = faster but less stable.

        Call rebuild_cuda_graph() afterwards so the captured kernels pick up the new values.
        """
        sound_speed = (7.0 * self.stiffness / self.rest_density) ** 0.5
        self.dt = self.cfl_factor * self.h / sound_speed
        if self.max_dt is not None:
            self.dt = min(self.dt, self.max_dt)

    def rebuild_cuda_graph(self):
        """Captures the coalesced Density + WCSPH Force + SDF Obstacle + Copy-back kernels into a CUDA Graph."""
        with wp.ScopedCapture(device="cuda:0") as capture:
            wp.launch(
                kernel=k_compute_density_coalesced,
                dim=self.num_particles,
                inputs=[
                    self.grid.id,
                    self.pos_sorted,
                    self.density_sorted,
                    self.mass,
                    self.h,
                    self.rest_density,
                    self.enable_obstacles,
                ],
                device="cuda:0",
            )
            wp.launch(
                kernel=k_wcsph_forces_and_copyback,
                dim=self.num_particles,
                inputs=[
                    self.grid.id,
                    self.pos_sorted,
                    self.vel_sorted,
                    self.density_sorted,
                    self.pos_sim,
                    self.vel,
                    self.mass,
                    self.h,
                    self.rest_density,
                    self.stiffness,
                    self.viscosity,
                    self.dt,
                    self.domain_min,
                    self.domain_max,
                    self.enable_obstacles,
                ],
                device="cuda:0",
            )
        self.cuda_graph = capture.graph

    def get_vram_usage_mb(self) -> float:
        bytes_per_particle = (5 * 12) + 4 + 12
        return (self.num_particles * bytes_per_particle) / (1024.0 * 1024.0)

    def step(self, pos_vbo_wp: wp.array, use_graph: bool = True):
        for _ in range(self.substeps):
            self.grid.build(points=self.pos_sim, radius=self.h)

            wp.launch(
                kernel=k_reorder_soa,
                dim=self.num_particles,
                inputs=[self.grid.id, self.pos_sim, self.vel, self.pos_sorted, self.vel_sorted],
                device="cuda:0",
            )

            self.grid.build(points=self.pos_sorted, radius=self.h)

            if use_graph and self.cuda_graph is not None:
                wp.capture_launch(self.cuda_graph)
            else:
                wp.launch(
                    kernel=k_compute_density_coalesced,
                    dim=self.num_particles,
                    inputs=[self.grid.id, self.pos_sorted, self.density_sorted,
                            self.mass, self.h, self.rest_density, self.enable_obstacles],
                    device="cuda:0",
                )
                wp.launch(
                    kernel=k_wcsph_forces_and_copyback,
                    dim=self.num_particles,
                    inputs=[
                        self.grid.id, self.pos_sorted, self.vel_sorted, self.density_sorted,
                        self.pos_sim, self.vel, self.mass, self.h, self.rest_density,
                        self.stiffness, self.viscosity, self.dt, self.domain_min, self.domain_max,
                        self.enable_obstacles
                    ],
                    device="cuda:0",
                )

        wp.copy(pos_vbo_wp, self.pos_sim)
