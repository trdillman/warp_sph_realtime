"""Independently audit saved keyframe comparison receipts without running CUDA."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def rank(candidate):
    violation = max(0.90 - candidate["coverage"], 0.0) + max(candidate["spill"] - 0.05, 0.0)
    return (violation > 0, violation, candidate["loss"])


def candidate_errors(candidate):
    errors = []
    for key in ("loss", "coverage", "spill"):
        if not math.isfinite(candidate[key]):
            errors.append(f"nonfinite {key}")
    if not 0 <= candidate["coverage"] <= 1 or not 0 <= candidate["spill"] <= 1:
        errors.append("invalid particle fraction")
    for key in ("coverage", "spill"):
        if abs(candidate[key] * 64 - round(candidate[key] * 64)) > 1e-6:
            errors.append(f"{key} is not a 64-particle fraction")
    expected = candidate["coverage"] >= 0.90 and candidate["spill"] <= 0.05
    if candidate["success"] != expected:
        errors.append("success contradicts hard constraints")
    values = candidate["controls"]
    if len(values) != 8 or any(not math.isfinite(x) or abs(x) > 0.300001 for x in values):
        errors.append("invalid controls")
    return errors


def trial_errors(trial, budget, time_limit):
    errors = []
    history = trial["history"]
    if not history:
        return ["empty history"]
    charge = 2 if trial["method"] == "adam" else 1
    best = None
    previous_time = 0.0
    eligible = None
    for i, row in enumerate(history, 1):
        candidate = row["candidate"]
        errors.extend(candidate_errors(candidate))
        if row["units"] != i * charge or row["forwards"] != i or row["backwards"] != i * (charge - 1):
            errors.append("history work accounting mismatch")
        if not math.isfinite(row["completed_seconds"]) or row["completed_seconds"] <= previous_time:
            errors.append("nonmonotonic completion time")
        previous_time = row["completed_seconds"]
        if best is None or rank(candidate) < rank(best):
            best = candidate
        if row["best"] != best:
            errors.append("selection is not best constraint-ranked observed candidate")
        if row["completed_seconds"] <= time_limit:
            eligible = row
    if history[0]["candidate"]["controls"] != trial["start"]:
        errors.append("trial did not evaluate paired start first")
    if trial["best"] != best or trial["units"] != history[-1]["units"] or trial["units"] > budget:
        errors.append("final selection or budget mismatch")
    if trial["units"] != budget:
        errors.append("incomplete nominal budget")
    if trial["forwards"] != len(history) or trial["backwards"] != len(history) * (charge - 1):
        errors.append("final evaluation counts mismatch")
    if trial["time_cutoff_seconds"] != time_limit:
        errors.append("time cutoff mismatch")
    if trial["best_at_time_cutoff"] != (eligible["best"] if eligible else None):
        errors.append("time cutoff includes unavailable or incorrectly selected candidate")
    if trial["completed_units_at_time_cutoff"] != (eligible["units"] if eligible else 0):
        errors.append("time cutoff work count mismatch")
    if not math.isfinite(trial["wall_seconds"]) or trial["wall_seconds"] < previous_time:
        errors.append("invalid total wall time")
    return errors


def trajectory_errors(trajectory, candidate):
    errors = []
    if not trajectory["finite"] or trajectory["hard_floor_clamps"] or trajectory["max_solid_penetration"] > 1e-5:
        errors.append("unsafe trajectory")
    if any(x < bound - 1e-6 for x, bound in zip(trajectory["minimum"], [-2.5, -2.5, 0], strict=True)) or any(
            x > bound + 1e-6 for x, bound in zip(trajectory["maximum"], [2.5, 2.5, 5], strict=True)):
        errors.append("domain violation")
    for metric in ("coverage", "spill"):
        if trajectory[f"recomputed_{metric}"] != candidate[metric]:
            errors.append(f"independent {metric} mismatch")
    return errors


def qualification_errors(report):
    """Bind prescribed derivative probes to the actual selected controls."""
    errors = []
    checks = report["gradient_checks"]
    initial = [row for row in checks if "selected_seed" not in row]
    selected = [row for row in checks if "selected_seed" in row]
    if len(checks) != 5 or len(initial) != 2 or sorted(row["selected_seed"] for row in selected) != [0, 4, 7]:
        errors.append("missing or duplicated prescribed gradient probes")
    expected_initial = [next(row["start"] for row in report["trials"] if row["seed"] == 0),
                        [0.06, 0.01, 0.04, -0.01, 0.02, 0.01, 0, -0.01]]

    def same_controls(a, b):
        return len(a) == len(b) == 8 and all(math.isfinite(x) and abs(x - y) <= 1e-7 for x, y in zip(a, b, strict=True))

    if len(initial) == 2 and not all(same_controls(row["controls"], expected)
                                     for row, expected in zip(initial, expected_initial, strict=True)):
        errors.append("initial derivative probes do not match fixed controls")
    for row in selected:
        trials = [trial for trial in report["trials"] if trial["method"] == "adam" and trial["seed"] == row["selected_seed"]]
        if len(trials) != 1 or not same_controls(row["controls"], trials[0]["best"]["controls"]):
            errors.append("selected derivative probe controls mismatch")
    for row in checks:
        if row["epsilon"] != 0.0005 or not 0 <= row["rtol"] <= 0.08 or not 0 <= row["atol"] <= 0.0005:
            errors.append("changed derivative epsilon or tolerance")
    witness = report["witness"]
    if witness["replay_passed"] is not True or any(
            not math.isfinite(witness[key]) or witness[key] < 0
            for key in ("replay_position_max_abs", "replay_velocity_max_abs")):
        errors.append("witness graph replay failed")
    return errors


def verify(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    contract = report["contract"]
    if report["status"] != "complete" or report["pilot"]:
        errors.append("not a complete measured comparison")
    if contract["dimensions"] != 8 or contract["particles"] != 64:
        errors.append("unexpected workload")
    if (contract["budget"] != 64 or contract["time_limit"] != 10.0 or contract["steps"] != 640
            or contract["release_steps"] != [0, 96, 192, 288]
            or not math.isclose(contract["dt"], 0.2 * 0.12 / math.sqrt(630), rel_tol=1e-10)):
        errors.append("changed frozen budget or simulation clock")
    if contract["target_minimum"] != 0.90 or contract["spill_maximum"] != 0.05:
        errors.append("changed artist constraints")
    if set(contract["methods"]) != {"adam", "coordinate", "cem", "grouped_coordinate"} or contract["seeds"] != list(range(8)):
        errors.append("changed method/seed matrix")
    if len(report["aa"]) != 8:
        errors.append("missing A/A pairs")
    for sample in report["aa"]:
        if min(sample["a_seconds"], sample["b_seconds"]) <= 0 or not math.isclose(
                sample["ratio"], sample["a_seconds"] / sample["b_seconds"], rel_tol=1e-9):
            errors.append("invalid A/A ratio")
    if not report["witness"]["success"]:
        errors.append("unreachable witness")
    errors.extend(candidate_errors(report["witness"]))
    errors.extend("witness: " + error for error in trajectory_errors(report["witness"]["trajectory"], report["witness"]))
    if len(report["gradient_checks"]) < 2:
        errors.append("missing full-dimensional gradient checks")
    for check in report["gradient_checks"]:
        if check["rtol"] > 0.08 or check["atol"] > 0.0005:
            errors.append("relaxed gradient tolerance")
        if len(check["analytic"]) != 8 or len(check["finite_difference"]) != 8:
            errors.append("incomplete gradient vector")
        if not check["passed"] or any(not math.isfinite(a) or not math.isfinite(b) or
                abs(a - b) > check["atol"] + check["rtol"] * abs(b)
                for a, b in zip(check["analytic"], check["finite_difference"], strict=True)):
            errors.append("numerical gradient mismatch")
    pairs = [(row["seed"], row["method"]) for row in report["trials"]]
    expected = {(seed, method) for seed in contract["seeds"] for method in contract["methods"]}
    if len(pairs) != len(expected) or set(pairs) != expected:
        errors.append("missing or duplicated trials")
    for seed in contract["seeds"]:
        starts = [row["start"] for row in report["trials"] if row["seed"] == seed]
        if starts and any(start != starts[0] for start in starts):
            errors.append(f"unpaired start seed {seed}")
    for trial in report["trials"]:
        prefix = f"{trial['seed']}/{trial['method']}: "
        errors.extend(prefix + error for error in trial_errors(trial, contract["budget"], contract["time_limit"]))
        if not trial["replay_passed"]:
            errors.append(prefix + "failed graph replay")
        errors.extend(prefix + error for error in trajectory_errors(trial["trajectory"], trial["best"]))
        if trial["best_at_time_cutoff"] is not None:
            proof = trial["time_cutoff_validation"]
            if not proof["replay_passed"]:
                errors.append(prefix + "failed time-cutoff graph replay")
            errors.extend(prefix + "time cutoff: " + error for error in trajectory_errors(
                proof["trajectory"], trial["best_at_time_cutoff"]))
    errors.extend(qualification_errors(report))
    for name, digest in report["source_sha256"].items():
        snapshot = path.parent / "source" / Path(name).name
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            errors.append(f"source snapshot mismatch: {name}")
    return {"passed": not errors, "errors": errors, "trial_count": len(pairs),
            "gradient_vectors": len(report["gradient_checks"]),
            "comparison_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.receipt)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
