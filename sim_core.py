import warp as wp

wp.init()

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
    pos_in: wp.array(dtype=wp.vec3),
    vel_in: wp.array(dtype=wp.vec3),
    pos_sorted: wp.array(dtype=wp.vec3),
    vel_sorted: wp.array(dtype=wp.vec3),
):
    """Reorders particle SoA data into spatial hash cell order for coalesced L2 cache reads."""
    tid = wp.tid()
    src_i = wp.hash_grid_point_id(grid, tid)
    pos_sorted[tid] = pos_in[src_i]
    vel_sorted[tid] = vel_in[src_i]

@wp.kernel
def k_compute_density_coalesced(
    grid: wp.uint64,
    pos_sorted: wp.array(dtype=wp.vec3),
    density_sorted: wp.array(dtype=float),
    mass: float,
    h: float,
    rest_density: float
):
    tid = wp.tid()
    xi = pos_sorted[tid]
    rho = float(0.0)

    neighbors = wp.hash_grid_query(grid, xi, h)
    for j in neighbors:
        r = wp.length(xi - pos_sorted[j])
        if r < h:
            rho += mass * cubic_spline_kernel(r, h)

    density_sorted[tid] = wp.max(rho, rest_density)

@wp.kernel
def k_wcsph_forces_and_copyback(
    grid: wp.uint64,
    pos_sorted: wp.array(dtype=wp.vec3),
    vel_sorted: wp.array(dtype=wp.vec3),
    density_sorted: wp.array(dtype=float),
    pos_vbo_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
    mass: float,
    h: float,
    rest_density: float,
    stiffness: float,
    viscosity: float,
    dt: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    enable_obstacles: int
):
    tid = wp.tid()
    xi = pos_sorted[tid]
    vi = vel_sorted[tid]
    rho_i = density_sorted[tid]
    p_i = stiffness * (wp.pow(rho_i / rest_density, 7.0) - 1.0)

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

            f_pressure -= mass * mass * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad_w
            f_visc += viscosity * mass * ((vel_sorted[j] - vi) / rho_j) * cubic_spline_kernel(r, h)

    gravity = wp.vec3(0.0, 0.0, -9.81)
    acc = (f_pressure + f_visc) / mass + gravity

    v_next = vi + acc * dt
    x_next = xi + v_next * dt

    # Optional Scene 1 GPU SDF Obstacles (Central Sphere + 4 Vertical Pillars)
    if enable_obstacles != 0:
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
        self.mass = 0.8 * (h * 0.5) ** 3 * self.rest_density
        self.stiffness = 35000.0
        self.viscosity = 0.15
        self.dt = 0.0015
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
                    inputs=[self.grid.id, self.pos_sorted, self.density_sorted, self.mass, self.h, self.rest_density],
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
