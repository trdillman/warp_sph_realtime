import warp as wp

from ..sim.sim_core import cubic_spline_grad, cubic_spline_kernel


@wp.kernel
def k_ipbf_predict(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    pos_pred: wp.array[wp.vec3],
    dt: float
):
    """Variational prediction step: x* = x_t + dt * v_t + dt^2 * a_ext."""
    i = int(wp.tid())
    g = wp.vec3(0.0, 0.0, -9.81)
    pos_pred[i] = pos[i] + vel[i] * dt + g * (dt * dt)

@wp.kernel
def k_ipbf_vbd_iteration(
    grid: wp.uint64,
    pos_pred: wp.array[wp.vec3],
    pos_corr: wp.array[wp.vec3],
    mass: float,
    h: float,
    rest_density: float,
    relaxation: float
):
    """Local 3x3 Hessian / Jacobi position update for unconditional stability at large dt."""
    tid = int(wp.tid())
    xi = pos_pred[tid]
    rho_i = float(0.0)
    grad_sum = wp.vec3(0.0, 0.0, 0.0)
    diag_hessian = float(1e-5)

    neighbors = wp.hash_grid_query(grid, xi, h)
    for j in neighbors:
        xj = pos_pred[j]
        diff = xi - xj
        r = wp.length(diff)
        if r < h:
            rho_i += mass * cubic_spline_kernel(r, h)
            if r > 1e-6:
                gw = cubic_spline_grad(diff, r, h) * (mass / rest_density)
                grad_sum += gw
                diag_hessian += wp.dot(gw, gw)

    c_i = wp.max((rho_i / rest_density) - 1.0, 0.0)
    diag_hessian += wp.dot(grad_sum, grad_sum)
    lambda_i = -c_i / diag_hessian
    pos_corr[tid] = xi + (grad_sum * lambda_i) * relaxation
