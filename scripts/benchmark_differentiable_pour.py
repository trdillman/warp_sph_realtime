"""Run the frozen differentiable-pour comparison and save machine-readable evidence.

Run from the repository root: python -m scripts.benchmark_differentiable_pour
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import warp as wp

from warp_sph_realtime.experimental.differentiable_pour import PourExperiment


def optimize(evaluate, start, method, seed, budget=24):
    """Common evaluator and bounds; charge each forward and backward separately."""
    if method not in ("adam", "fd_adam", "random", "coordinate") or budget < 2:
        raise ValueError("Unknown method or budget below two")
    rng = np.random.default_rng(seed)
    controls = np.asarray(start, dtype=float).copy()
    best_loss = float("inf")
    best = None
    history = []
    units = 0
    forwards = 0
    backwards = 0
    first_moment = np.zeros(2)
    second_moment = np.zeros(2)
    center = controls.copy()
    sweep_loss = float("inf")
    step = 0.5
    direction = 0
    began = time.perf_counter()
    charge = 2 if method == "adam" else 5 if method == "fd_adam" else 1
    while units + charge <= budget:
        result = evaluate(controls, gradient=method == "adam")
        forwards += 1
        backwards += int(method == "adam")
        if method == "fd_adam":
            derivative = np.zeros(2)
            for axis in range(2):
                plus, minus = controls.copy(), controls.copy()
                plus[axis] = min(2.0, plus[axis] + 0.001)
                minus[axis] = max(-2.0, minus[axis] - 0.001)
                positive = evaluate(plus, gradient=False)
                negative = evaluate(minus, gradient=False)
                derivative[axis] = (positive["loss"] - negative["loss"]) / (plus[axis] - minus[axis])
                forwards += 2
            result["gradient"] = derivative
        units = forwards + backwards
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            best = {"controls": controls.tolist(), "loss": best_loss, "coverage": result["coverage"]}
        history.append({"units": units, "controls": controls.tolist(), "loss": result["loss"],
                        "coverage": result["coverage"], "best_loss": best_loss,
                        "seconds": result["seconds"], "elapsed_seconds": time.perf_counter() - began})
        if method in ("adam", "fd_adam"):
            derivative = np.asarray(result["gradient"])
            first_moment = 0.9 * first_moment + 0.1 * derivative
            second_moment = 0.999 * second_moment + 0.001 * derivative ** 2
            iteration = len(history)
            m = first_moment / (1 - 0.9 ** iteration)
            v = second_moment / (1 - 0.999 ** iteration)
            controls = np.clip(controls - 0.25 * m / (np.sqrt(v) + 1e-8), -2, 2)
        elif method == "random":
            controls = rng.uniform(-2, 2, 2)
        else:
            if forwards == 1:
                sweep_loss = best_loss
            elif direction == 4:
                if best_loss >= sweep_loss:
                    step *= 0.5
                assert best is not None
                center = np.array(best["controls"])
                sweep_loss = best_loss
                direction = 0
            delta = np.zeros(2)
            delta[direction // 2] = step * (1 if direction % 2 == 0 else -1)
            controls = np.clip(center + delta, -2, 2)
            direction += 1
    return {"method": method, "seed": seed, "start": np.asarray(start).tolist(), "best": best,
            "selection_policy": "central_iterates" if method == "fd_adam" else "all_evaluated_controls",
            "history": history, "forward_evaluations": forwards, "backward_evaluations": backwards,
            "units": units, "wall_seconds": time.perf_counter() - began}


def source_identity():
    paths = [Path(__file__), Path("warp_sph_realtime/experimental/differentiable_pour.py"),
             Path("warp_sph_realtime/sim/sim_core.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/differentiable-pour-20261017/results.json"))
    parser.add_argument("--fixture", choices=("floor", "deflected", "reachable"), default="floor")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior evidence; choose a new output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_identity()
    snapshot = args.output.parent / (args.output.stem + "-source")
    snapshot.mkdir(exist_ok=False)
    for source in hashes:
        shutil.copy2(source, snapshot / Path(source).name)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "warp": wp.__version__,
        "gpu": str(wp.get_device("cuda:0").name), "source_sha256": hashes,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "workload": {"particles": 64, "steps": 256 if args.fixture == "deflected" else 192,
                     "fixture": args.fixture, "controls": "initial vx, vy in [-2,2]",
                     "target_xy": [0.3, 0.2], "half_width": 0.18, "coverage_max_z": 0.18,
                     "budget": 24, "seeds": list(range(5)), "adam_step_size": 0.25,
                     "coordinate_initial_step": 0.5,
                     "scope": "finite slug aimed at floor patch, not continuous pouring into a cup"},
        "trials": [],
    }
    reachable = args.fixture == "reachable"
    methods_all = ["adam", "random", "coordinate", "fd_adam"] if reachable else ["adam", "random", "coordinate"]
    if reachable:
        report["workload"].update(steps=384, target_xy=[-0.8, 0.0], half_width=0.22,
                                  budget=64, seeds=list(range(8)), success_coverage=0.75,
                                  contact_mode=2, objective="3D squared distance outside box + .05 xy center distance")

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    save()
    try:
        experiment = PourExperiment(steps=report["workload"]["steps"], fixture=args.fixture)
        report["initial_positions"] = experiment.initial.tolist()
        report["workload"]["dt"] = experiment.solver.dt
        report["workload"]["simulated_seconds"] = experiment.steps * experiment.solver.dt
        # Warm both kernel directions and all persistent grids before comparison timing.
        experiment.evaluate([0.7, -0.2], gradient=True)
        experiment.evaluate([0.7, -0.2])
        if reachable:
            witness = experiment.evaluate([0, 0])
            witness_p, _ = experiment.production_rollout([0, 0], use_graph=True)
            report["reachability_witness"] = {
                "controls": [0, 0], "loss": witness["loss"], "coverage": witness["coverage"],
                "trajectory": experiment.trajectory_diagnostics(),
                "replay_matches": bool(np.allclose(witness_p, witness["positions"], atol=2e-6, rtol=2e-5))}
            save()
            witness_trajectory = report["reachability_witness"]["trajectory"]
            if (witness["coverage"] < 0.75 or not report["reachability_witness"]["replay_matches"]
                    or witness_trajectory["max_solid_penetration"] > 0.001
                    or witness_trajectory["hard_floor_clamp_particle_steps"] != 0
                    or not witness_trajectory["finite_all_states"]
                    or witness_trajectory["max_z"] >= 2.14):
                raise RuntimeError("Target reachability witness failed")
            report["repeatability"] = []
            for _ in range(8):
                first = experiment.evaluate([0.7, -0.2])
                second = experiment.evaluate([0.7, -0.2])
                report["repeatability"].append({"loss_delta": abs(first["loss"] - second["loss"]),
                                               "seconds": [first["seconds"], second["seconds"]]})
        report["gradient_checks"] = []
        for controls in ([0.0, 0.0], [0.7, -0.2], [1.4, 0.4]):
            for epsilon in (0.002, 0.001):
                report["gradient_checks"].append(experiment.check_gradient(controls, epsilon))
        report["gradient_checks_passed"] = all(row["passed"] for row in report["gradient_checks"])
        print(f"Gradient checks passed: {report['gradient_checks_passed']}", flush=True)
        save()
        retained = experiment.evaluate([0.7, -0.2])
        report["parity"] = []
        for graph in (False, True):
            p, v = experiment.production_rollout([0.7, -0.2], graph)
            p_error = float(np.max(np.abs(p - retained["positions"])))
            v_error = float(np.max(np.abs(v - retained["velocities"])))
            passed = bool(np.allclose(p, retained["positions"], atol=2e-6, rtol=2e-5)
                          and np.allclose(v, retained["velocities"], atol=2e-6, rtol=2e-5))
            report["parity"].append({"graph": graph, "position_max_abs": p_error,
                                    "velocity_max_abs": v_error, "passed": passed})
        if not all(row["passed"] for row in report["parity"]):
            raise RuntimeError("Forward parity failed; comparison would not represent the current solver")
        for seed in report["workload"]["seeds"]:
            start = np.random.default_rng(seed + 1000).uniform(-0.5, 0.5, 2)
            if reachable:
                start[0] += 1.0
                report.setdefault("start_gradient_checks", []).extend(
                    [experiment.check_gradient(start, eps) for eps in (0.002, 0.001)])
            # Rotate order to reduce systematic timing-order bias.
            offset = seed % len(methods_all)
            methods = methods_all[offset:] + methods_all[:offset]
            for method in methods:
                trial = optimize(experiment.evaluate, start, method, seed, budget=report["workload"]["budget"])
                trial["initial_positions"] = experiment.evaluate(start)["positions"].tolist()
                if reachable and seed == 0:
                    trial["initial_frames"] = [state.numpy().tolist() for state in experiment.positions[::12]]
                p, _ = experiment.production_rollout(trial["best"]["controls"], use_graph=True)
                trial["final_positions"] = p.tolist()
                replay_loss = float(np.mean(np.sum((p[:, :2] - [0.3, 0.2]) ** 2, axis=1)))
                if reachable:
                    delta = p[:, :2] - [-0.8, 0.0]
                    outside = np.maximum(np.abs(delta) - 0.22, 0)
                    replay_loss = float(np.mean(np.sum(outside ** 2 + 0.05 * delta ** 2, axis=1)
                                                + np.maximum(p[:, 2] - 0.18, 0) ** 2))
                    trial["gradient_checks"] = [experiment.check_gradient(trial["best"]["controls"], eps)
                                                 for eps in (0.002, 0.001)]
                    selected = experiment.evaluate(trial["best"]["controls"])
                    trial["trajectory"] = experiment.trajectory_diagnostics()
                    if seed == 0:
                        trial["selected_frames"] = [state.numpy().tolist() for state in experiment.positions[::12]]
                    trial["production_coverage"] = float(((np.abs(delta) <= 0.22).all(axis=1)
                                                          & (p[:, 2] <= 0.18)).mean())
                    trial["production_positions_match"] = bool(np.allclose(
                        p, selected["positions"], atol=2e-6, rtol=2e-5))
                    successful = [r for r in trial["history"] if r["coverage"] >= 0.75]
                    trial["first_success_seconds"] = successful[0]["elapsed_seconds"] if successful else None
                trial["production_graph_replay_loss"] = replay_loss
                trial["replay_passed"] = bool(np.isclose(replay_loss, trial["best"]["loss"],
                                                       atol=2e-6, rtol=2e-5))
                report["trials"].append(trial)
                save()
                print(f"seed={seed} {method}: loss={trial['best']['loss']:.6f} "
                      f"coverage={trial['best']['coverage']:.3f} seconds={trial['wall_seconds']:.2f}", flush=True)
        report["summary"] = {}
        for method in methods_all:
            trials = [r for r in report["trials"] if r["method"] == method]
            report["summary"][method] = {
                "median_loss": float(np.median([r["best"]["loss"] for r in trials])),
                "median_coverage": float(np.median([r["best"]["coverage"] for r in trials])),
                "median_wall_seconds": float(np.median([r["wall_seconds"] for r in trials])),
                "successes": sum(r["best"]["coverage"] >= 0.75 for r in trials),
            }
        if reachable:
            report["gradient_checks_passed"] = (report["gradient_checks_passed"]
                and all(c["passed"] for c in report["start_gradient_checks"])
                and all(c["passed"] for t in report["trials"] for c in t["gradient_checks"]))
            report["trajectory_checks_passed"] = all(
                t["trajectory"]["max_solid_penetration"] <= 0.001
                and t["trajectory"]["hard_floor_clamp_particle_steps"] == 0
                and t["trajectory"]["finite_all_states"] and t["production_positions_match"]
                for t in report["trials"])
        report["all_replays_passed"] = all(row["replay_passed"] for row in report["trials"])
        report["status"] = ("complete" if report["gradient_checks_passed"] and report["all_replays_passed"]
                            and report.get("trajectory_checks_passed", True)
                            else "validation_failed")
        print(json.dumps(report["summary"], indent=2), flush=True)
        save()
        if report["status"] != "complete":
            # Preserve all diagnostic comparisons, but never signal validated success.
            raise SystemExit(2)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
