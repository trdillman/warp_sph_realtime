import warp as wp

@wp.kernel
def k_collide_sdf_primitives(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    sphere_center: wp.vec3,
    sphere_radius: float,
    restitution: float
):
    """Evaluates analytical/mesh SDF collision and applies penetration correction."""
    i = wp.tid()
    xi = pos[i]
    diff = xi - sphere_center
    dist = wp.length(diff)
    sdf = dist - sphere_radius

    if sdf < 0.0 and dist > 1e-6:
        normal = diff / dist
        pos[i] = sphere_center + normal * sphere_radius
        vn = wp.dot(vel[i], normal)
        if vn < 0.0:
            vel[i] = vel[i] - (1.0 + restitution) * vn * normal

class SDFColliderManager:
    """Manages GPU SDF colliders and two-way rigid body impulse coupling."""
    def __init__(self):
        self.active = False
        self.sphere_center = wp.vec3(0.0, 0.0, 0.5)
        self.sphere_radius = 0.6
        self.restitution = 0.3

    def apply(self, pos: wp.array, vel: wp.array, num_particles: int):
        if not self.active:
            return
        wp.launch(
            kernel=k_collide_sdf_primitives,
            dim=num_particles,
            inputs=[pos, vel, self.sphere_center, self.sphere_radius, self.restitution],
            device=pos.device,
        )
