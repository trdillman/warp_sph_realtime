"""Paired offline emitter optimization; never invoked by the Blender modal path."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import warp as wp

from warp_sph_realtime.experimental.keyframe_pour import KeyframePour, outcome_rank

METHODS = ("adam", "coordinate", "cem", "grouped_coordinate")
BUDGET = 64
TIME_LIMIT = 10.0


def compact(result, controls):
    return {key: result[key] for key in ("loss", "coverage", "spill", "success")} | {
        "controls": np.asarray(controls).tolist()}


def optimize(evaluate, start, method, seed, budget=BUDGET, time_limit=TIME_LIMIT):
    if method not in METHODS or budget < 2:
        raise ValueError("Invalid method or budget")
    rng = np.random.default_rng(seed + 719)
    x = np.asarray(start, dtype=float).copy()
    history = []
    best = None
    units = 0
    first, second = np.zeros(8), np.zeros(8)
    center, step, direction = x.copy(), 0.06, 0
    sweep_rank = None
    mean, sigma = x.copy(), np.full(8, 0.08)
    population = []
    began = time.perf_counter()
    charge = 2 if method == "adam" else 1
    while units + charge <= budget:
        result = evaluate(x, gradient=method == "adam")
        units += charge
        candidate = compact(result, x)
        if best is None or outcome_rank(candidate) < outcome_rank(best):
            best = candidate.copy()
        history.append({"units": units, "forwards": len(history) + 1,
                        "backwards": len(history) + 1 if method == "adam" else 0,
                        "completed_seconds": time.perf_counter() - began,
                        "evaluation_seconds": result["seconds"], "candidate": candidate,
                        "best": best.copy()})
        if method == "adam":
            derivative = np.asarray(result["gradient"])
            first = 0.9 * first + 0.1 * derivative
            second = 0.999 * second + 0.001 * derivative ** 2
            iteration = len(history)
            x = np.clip(x - 0.025 * (first / (1 - 0.9 ** iteration)) /
                        (np.sqrt(second / (1 - 0.999 ** iteration)) + 1e-8), -0.30, 0.30)
        elif method in ("coordinate", "grouped_coordinate"):
            directions = 4 if method == "grouped_coordinate" else 16
            if sweep_rank is None:
                sweep_rank = outcome_rank(best)
            elif direction == directions:
                if outcome_rank(best) >= sweep_rank:
                    step *= 0.5
                center = np.array(best["controls"])
                sweep_rank = outcome_rank(best)
                direction = 0
            delta = np.zeros(8)
            if method == "grouped_coordinate":
                axis = direction // 2
                shift = step * (1 if direction % 2 == 0 else -1)
                shift = np.clip(shift, -0.30 - center[axis::2].min(), 0.30 - center[axis::2].max())
                delta[axis::2] = shift
            else:
                delta[direction // 2] = step * (1 if direction % 2 == 0 else -1)
            x = np.clip(center + delta, -0.30, 0.30)
            direction += 1
        else:
            population.append(candidate)
            if len(population) == 12:
                elite = sorted(population, key=outcome_rank)[:4]
                elite_controls = np.array([r["controls"] for r in elite])
                mean = 0.3 * mean + 0.7 * elite_controls.mean(axis=0)
                sigma = np.maximum(0.012, 0.3 * sigma + 0.7 * elite_controls.std(axis=0))
                population = []
            x = np.clip(rng.normal(mean, sigma), -0.30, 0.30)
    eligible = [r for r in history if r["completed_seconds"] <= time_limit]
    return {"method": method, "seed": seed, "start": np.asarray(start).tolist(),
            "history": history, "best": best, "units": units,
            "forwards": len(history), "backwards": len(history) if method == "adam" else 0,
            "wall_seconds": time.perf_counter() - began,
            "time_cutoff_seconds": time_limit,
            "best_at_time_cutoff": eligible[-1]["best"] if eligible else None,
            "completed_units_at_time_cutoff": eligible[-1]["units"] if eligible else 0}


def starting_controls(seed):
    rng = np.random.default_rng(seed + 163)
    translation = np.array([rng.uniform(0.14, 0.22), rng.uniform(-0.09, 0.09)])
    return np.clip(np.tile(translation, (4, 1)) + rng.normal(0, 0.018, (4, 2)), -0.28, 0.28).ravel()


def source_identity(output):
    source = output / "source"
    source.mkdir(exist_ok=True)
    paths = [Path(__file__), Path("warp_sph_realtime/experimental/keyframe_pour.py"),
             Path("warp_sph_realtime/sim/sim_core.py"), Path("scripts/verify_keyframe_results.py"),
             Path("tests/test_keyframe_receipts.py"), Path("docs/KEYFRAME_EXPERIMENT_CONTRACT.md")]
    hashes = {}
    for path in paths:
        hashes[str(path.relative_to(Path.cwd()) if path.is_absolute() else path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source / path.name)
    return hashes


def validate_selected(experiment, candidate, retain_frames=False):
    result = experiment.evaluate(candidate["controls"])
    trajectory = experiment.trajectory()
    if (trajectory["recomputed_spill"] != candidate["spill"]
            or trajectory["recomputed_coverage"] != candidate["coverage"]
            or not trajectory["finite"] or trajectory["hard_floor_clamps"]
            or trajectory["max_solid_penetration"] > 1e-5
            or any(x < b - 1e-6 for x, b in zip(trajectory["minimum"], [-2.5, -2.5, 0], strict=True))
            or any(x > b + 1e-6 for x, b in zip(trajectory["maximum"], [2.5, 2.5, 5], strict=True))):
        raise RuntimeError("Selected trajectory does not pass independent safety/outcome checks")
    if (result["coverage"] != candidate["coverage"] or result["spill"] != candidate["spill"]
            or not np.isclose(result["loss"], candidate["loss"], rtol=1e-5, atol=1e-7)):
        raise RuntimeError("Selected outcome was not reproducible")
    replay = experiment.evaluate(candidate["controls"], graph=True)
    dp = float(np.abs(result["positions"] - replay["positions"]).max())
    dv = float(np.abs(result["velocities"] - replay["velocities"]).max())
    replay_passed = bool(np.allclose(result["positions"], replay["positions"], atol=1e-5, rtol=1e-5)
                         and np.allclose(result["velocities"], replay["velocities"], atol=1e-5, rtol=1e-5)
                         and result["coverage"] == replay["coverage"] and result["spill"] == replay["spill"])
    if not replay_passed:
        raise RuntimeError("Shared-kernel graph replay mismatch")
    return {"trajectory": {k: v for k, v in trajectory.items() if k != "frames"},
            "frames": trajectory["frames"] if retain_frames else [],
            "replay_position_max_abs": dp, "replay_velocity_max_abs": dv, "replay_passed": replay_passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "pilot.json" if args.pilot else args.output / "comparison.json"
    if path.exists():
        raise FileExistsError(path)
    report = {"status": "running", "created_utc": datetime.now(UTC).isoformat(),
              "source_sha256": source_identity(args.output), "pilot": args.pilot,
              "trials": [], "gradient_checks": [], "aa": []}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    save()
    try:
        setup = time.perf_counter()
        experiment = KeyframePour()
        report["environment"] = {"python": platform.python_version(), "warp": wp.__version__,
                                 "numpy": np.__version__, "platform": platform.platform(),
                                 "device": str(wp.get_device("cuda:0")),
                                 "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
        report["contract"] = {"steps": experiment.steps, "release_steps": experiment.release_steps,
                              "dt": experiment.solver.dt, "duration": experiment.steps * experiment.solver.dt,
                              "burst_final_ages": [(experiment.steps - t) * experiment.solver.dt for t in experiment.release_steps],
                              "dimensions": 8, "particles": 64, "budget": BUDGET,
                              "time_limit": TIME_LIMIT, "methods": METHODS, "seeds": list(range(8)),
                              "target_minimum": 0.90, "spill_maximum": 0.05,
                              "target_geometry": {"center": [-1.45, 0], "half_widths": [0.22, 0.22], "plane_z": 0.35},
                              "target_semantics": "first downward segment-interpolated aperture crossing, not final basin retention",
                              "spill_box": {"minimum": [-1.22, -0.40, 0.30], "maximum": [-0.68, 0.40, 0.40]},
                              "spill_semantics": "any segment intersection over full retained trajectory, unique persistent particle IDs",
                              "emission": "four sixteen-particle planar bursts, cubic Hermite keyframe path with centered/secant key velocities plus nozzle vz=-2m/s",
                              "emission_full_domain_minimum_clearance": 0.0840075756488817,
                              "timing_policy": "serialized warmed persistent process, rotating method order; descriptive time cutoff only; A/A envelope0.8..1.2",
                              "graph_policy": "shared SPH kernels captured per-step after grids; not production step() staged-emission integration",
                              "safety": {"solid_penetration_maximum": 1e-5, "hard_floor_clamps": 0,
                                         "finite_all_states": True, "domain_tolerance": 1e-6},
                              "work_unit": "one full staged forward or one full reverse pass",
                              "selection": "feasibility, then sum of target/spill violations, then loss"}
        warm = experiment.evaluate(np.zeros(8), gradient=True)
        report["setup_and_one_warmup_seconds"] = time.perf_counter() - setup
        report["witness"] = compact(warm, np.zeros(8))
        report["witness"].update(validate_selected(experiment, report["witness"]))
        print("witness", {k: v for k, v in report["witness"].items() if k != "trajectory"}, flush=True)
        save()
        if not warm["success"]:
            raise RuntimeError("Witness does not meet artist constraints")
        checks = [starting_controls(0), np.array([0.06, 0.01, 0.04, -0.01, 0.02, 0.01, 0, -0.01])]
        for controls in checks:
            check = experiment.check_gradient(controls)
            report["gradient_checks"].append(check)
            print("gradient", check, flush=True)
            save()
        if not all(r["passed"] for r in report["gradient_checks"]):
            raise RuntimeError("Gradient qualification failed")
        if args.pilot:
            for method in METHODS:
                trial = optimize(experiment.evaluate, starting_controls(0), method, 0, budget=32)
                report["trials"].append(trial)
                print(method, trial["best"], trial["wall_seconds"], flush=True)
                save()
        else:
            experiment.evaluate(starting_controls(0))
            report["initial_frames_seed0"] = experiment.trajectory()["frames"]
            for repeat in range(8):
                a = experiment.evaluate(starting_controls(0))["seconds"]
                b = experiment.evaluate(starting_controls(0))["seconds"]
                report["aa"].append({"repeat": repeat, "a_seconds": a, "b_seconds": b, "ratio": a / b})
            save()
            for seed in range(8):
                methods = METHODS[seed % 4:] + METHODS[:seed % 4]
                for method in methods:
                    trial = optimize(experiment.evaluate, starting_controls(seed), method, seed)
                    report["trials"].append(trial)
                    save()
                    best = trial["best"]
                    trial.update(validate_selected(experiment, best, retain_frames=seed == 0))
                    cutoff = trial["best_at_time_cutoff"]
                    if cutoff is None:
                        raise RuntimeError("No completed candidate within time budget")
                    trial["time_cutoff_validation"] = validate_selected(experiment, cutoff)
                    if method == "adam" and seed in (0, 4, 7):
                        check = experiment.check_gradient(best["controls"])
                        check["selected_seed"] = seed
                        report["gradient_checks"].append(check)
                        if not check["passed"]:
                            raise RuntimeError("Selected AD gradient check failed")
                    print(seed, method, "budget", best, "time", trial["best_at_time_cutoff"], trial["wall_seconds"], flush=True)
                    save()
        report["status"] = "complete"
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
