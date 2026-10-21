"""Isolate production-kernel adjoints at a retained failing-pour checkpoint.

Run once with ``python -m scripts.diagnose_contact_vjp``. This is an offline
diagnostic, never an interactive solver path. The spatial hash is deliberately
frozen during local differences, matching the grid retained by Warp's tape.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from warp_sph_realtime.experimental.differentiable_pour import PourExperiment
from warp_sph_realtime.sim.sim_core import (
    k_compute_density_coalesced,
    k_wcsph_forces_and_copyback,
)


@wp.kernel
def scalar_objective(values: wp.array[float], weights: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, values[i] * weights[i])


@wp.kernel
def vector_objective(
    positions: wp.array[wp.vec3], velocities: wp.array[wp.vec3],
    position_weights: wp.array[wp.vec3], velocity_weights: wp.array[wp.vec3],
    loss: wp.array[float],
):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(positions[i], position_weights[i])
                  + wp.dot(velocities[i], velocity_weights[i]))


def _dot64(values, weights):
    return float(np.sum(values.astype(np.float64) * weights.astype(np.float64)))


def _direction(rng, shape):
    direction = rng.normal(size=shape).astype(np.float32)
    return direction / np.sqrt(np.mean(direction * direction))


def _compare(base_inputs, gradients, forward, specifications, rng):
    """Independent central differences; record agreement without hiding errors."""
    rows = []
    for name, epsilons in specifications.items():
        for direction_index in range(2):
            direction = _direction(rng, base_inputs[name].shape)
            analytic = _dot64(gradients[name], direction)
            checks = []
            for epsilon in epsilons:
                plus = {key: value.copy() for key, value in base_inputs.items()}
                minus = {key: value.copy() for key, value in base_inputs.items()}
                plus[name] += epsilon * direction
                minus[name] -= epsilon * direction
                f_plus, f_minus = forward(plus), forward(minus)
                numerical = (f_plus - f_minus) / (2.0 * epsilon)
                checks.append({
                    "epsilon": epsilon, "finite_difference": numerical,
                    "absolute_error": abs(analytic - numerical),
                    "relative_error": abs(analytic - numerical) / max(
                        abs(analytic), abs(numerical), 1e-10),
                    "f_plus": f_plus, "f_minus": f_minus,
                })
            rows.append({"input": name, "direction": direction_index,
                         "analytic": analytic, "checks": checks})
    return rows


def density_vjp(solver, grid, checkpoint, rng):
    count = len(checkpoint)
    weights = rng.normal(size=count).astype(np.float32) / np.sqrt(count)
    positions = wp.array(checkpoint, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    densities = wp.zeros(count, dtype=float, device="cuda:0", requires_grad=True)
    gpu_weights = wp.array(weights, dtype=float, device="cuda:0")
    loss = wp.zeros(1, dtype=float, device="cuda:0", requires_grad=True)

    def launch(values):
        positions.assign(values["position"])
        wp.launch(k_compute_density_coalesced, count,
                  [grid.id, positions, densities, solver.mass, solver.h, solver.rest_density, solver.enable_obstacles],
                  device="cuda:0")

    tape = wp.Tape()
    with tape:
        launch({"position": checkpoint})
        wp.launch(scalar_objective, count, [densities, gpu_weights, loss], device="cuda:0")
    tape.backward(loss)
    gradients = {"position": positions.grad.numpy().copy()}
    baseline = densities.numpy().copy()
    tape.zero()

    def forward(values):
        launch(values)
        return _dot64(densities.numpy(), weights)

    return {"rows": _compare({"position": checkpoint}, gradients, forward,
                              {"position": [0.001, 0.0003, 0.0001]}, rng),
            "density_min": float(baseline.min()), "density_max": float(baseline.max()),
            "density_at_rest_count": int(np.sum(baseline == solver.rest_density))}


def force_vjp(solver, grid, checkpoint_p, checkpoint_v, checkpoint_rho, rng, chained=False):
    count = len(checkpoint_p)
    base = {"position": checkpoint_p, "velocity": checkpoint_v, "density": checkpoint_rho}
    arrays = {
        name: wp.array(values, dtype=float if name == "density" else wp.vec3,
                       device="cuda:0", requires_grad=True)
        for name, values in base.items()
    }
    output_p = wp.zeros(count, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    output_v = wp.zeros(count, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    weight_p = (rng.normal(size=checkpoint_p.shape) / np.sqrt(count)).astype(np.float32)
    weight_v = (rng.normal(size=checkpoint_v.shape) / np.sqrt(count)).astype(np.float32)
    gpu_weight_p = wp.array(weight_p, dtype=wp.vec3, device="cuda:0")
    gpu_weight_v = wp.array(weight_v, dtype=wp.vec3, device="cuda:0")
    loss = wp.zeros(1, dtype=float, device="cuda:0", requires_grad=True)

    def launch(values):
        for name, array in arrays.items():
            array.assign(values[name])
        if chained:
            wp.launch(k_compute_density_coalesced, count,
                      [grid.id, arrays["position"], arrays["density"],
                       solver.mass, solver.h, solver.rest_density, solver.enable_obstacles], device="cuda:0")
        wp.launch(k_wcsph_forces_and_copyback, count,
                  [grid.id, arrays["position"], arrays["velocity"], arrays["density"],
                   output_p, output_v, solver.mass, solver.h, solver.rest_density,
                   solver.stiffness, solver.viscosity, solver.dt, solver.domain_min,
                   solver.domain_max, solver.enable_obstacles], device="cuda:0")

    tape = wp.Tape()
    with tape:
        launch(base)
        wp.launch(vector_objective, count,
                  [output_p, output_v, gpu_weight_p, gpu_weight_v, loss], device="cuda:0")
    tape.backward(loss)
    gradients = {name: array.grad.numpy().copy() for name, array in arrays.items()}
    baseline_p, baseline_v = output_p.numpy(), output_v.numpy()
    tape.zero()

    def forward(values):
        launch(values)
        return _dot64(output_p.numpy(), weight_p) + _dot64(output_v.numpy(), weight_v)

    specifications = {"position": [0.001, 0.0003, 0.0001],
                      "velocity": [0.003, 0.001, 0.0003]}
    if not chained:
        specifications["density"] = [0.3, 0.1, 0.03]
    return {"rows": _compare(base, gradients, forward, specifications, rng),
            "baseline_position_max_abs": float(np.max(np.abs(baseline_p))),
            "baseline_velocity_max_abs": float(np.max(np.abs(baseline_v))),
            "finite_gradients": all(bool(np.isfinite(g).all()) for g in gradients.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/contact-correction-20261017/vjp.json"))
    args = parser.parse_args()
    # Exclusive creation protects any previous raw diagnostic, including failures.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "checkpoint": 100, "steps": 128,
              "fixture": "reachable", "controls": [1.4, 0.4], "rng_seed": 8127,
              "grid_policy": "frozen retained grid for local AD and finite differences",
              "objective_policy": "same random linear functional; CPU float64 FD reduction",
              "scope": "local diagnostic; agreement does not establish full-rollout validity"}
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            experiment = PourExperiment(128, "reachable")
            report["environment"] = {"python": sys.version, "warp": wp.__version__,
                                     "numpy": np.__version__, "device": str(wp.get_device("cuda:0"))}
            report["solver"] = {"h": experiment.solver.h, "mass": experiment.solver.mass,
                                "dt": experiment.solver.dt, "mode": experiment.solver.enable_obstacles,
                                "rest_density": experiment.solver.rest_density,
                                "stiffness": experiment.solver.stiffness,
                                "viscosity": experiment.solver.viscosity}
            paths = [Path(__file__), Path("warp_sph_realtime/sim/sim_core.py"),
                     Path("warp_sph_realtime/experimental/differentiable_pour.py")]
            report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in paths}
            experiment.evaluate(report["controls"])
            checkpoint = report["checkpoint"]
            positions = experiment.sorted_positions[checkpoint].numpy().copy()
            velocities = experiment.sorted_velocities[checkpoint].numpy().copy()
            densities = experiment.densities[checkpoint].numpy().copy()
            grid = experiment.neighbor_grids[checkpoint]
            rng = np.random.default_rng(report["rng_seed"])
            report["density"] = density_vjp(experiment.solver, grid, positions, rng)
            report["force"] = force_vjp(
                experiment.solver, grid, positions, velocities, densities, rng)
            report["density_and_force"] = force_vjp(
                experiment.solver, grid, positions, velocities, densities, rng, chained=True)
            # One particle approaching the sphere: pair pressure/viscosity absent.
            single_p = np.array([[-0.56, 0.0, 0.55]], dtype=np.float32)
            single_v = np.array([[1.0, 0.15, -0.1]], dtype=np.float32)
            single_rho = np.array([experiment.solver.rest_density], dtype=np.float32)
            single_grid = wp.HashGrid(16, 16, 16, device="cuda:0")
            single_gpu_p = wp.array(single_p, dtype=wp.vec3, device="cuda:0")
            single_grid.build(single_gpu_p, experiment.solver.h)
            report["contact_only_force"] = force_vjp(
                experiment.solver, single_grid, single_p, single_v, single_rho, rng)
            report["status"] = "complete"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stream.write(json.dumps(report, indent=2, allow_nan=False))
            stream.write("\n")
    print(f"Saved local VJP diagnostic: {args.output}", flush=True)


if __name__ == "__main__":
    main()
