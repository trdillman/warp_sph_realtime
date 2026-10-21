"""Independent acceptance checks over saved measurements; never runs GPU work."""
import hashlib
import json
from pathlib import Path


def validate(report):
    errors = []
    witness = report.get("reachability_witness", {})
    if witness.get("coverage", 0) < 0.75 or not witness.get("replay_matches", False):
        errors.append("reachability")
    trajectories = [witness.get("trajectory", {})]
    trials = report.get("trials", [])
    expected = {(seed, method) for seed in range(8) for method in ("adam", "random", "coordinate", "fd_adam")}
    actual = [(t["seed"], t["method"]) for t in trials]
    if set(actual) != expected or len(actual) != len(expected):
        errors.append("trial completeness")
    checks = report.get("gradient_checks", []) + report.get("start_gradient_checks", [])
    for trial in trials:
        trajectories.append(trial.get("trajectory", {}))
        checks.extend(trial.get("gradient_checks", []))
        expected_units = 60 if trial["method"] == "fd_adam" else 64
        if trial["units"] != expected_units or trial["units"] != (
                trial["forward_evaluations"] + trial["backward_evaluations"]):
            errors.append("work accounting")
        if not trial.get("replay_passed") or not trial.get("production_positions_match"):
            errors.append("production replay")
        if trial.get("production_coverage") != trial["best"]["coverage"]:
            errors.append("coverage replay")
    if len(checks) != 86 or not all(c.get("passed", False) for c in checks):
        errors.append("gradient checks")
    for trajectory in trajectories:
        if (trajectory.get("max_solid_penetration", float("inf")) > 0.001
                or trajectory.get("hard_floor_clamp_particle_steps", 1) != 0
                or not trajectory.get("finite_all_states", False)
                or trajectory.get("max_z", float("inf")) >= 2.14):
            errors.append("trajectory safety")
    if len(report.get("parity", [])) != 2 or not all(r["passed"] for r in report.get("parity", [])):
        errors.append("forward parity")
    if len(report.get("repeatability", [])) != 8 or any(
            r["loss_delta"] > 1e-6 for r in report.get("repeatability", [])):
        errors.append("repeatability")
    return sorted(set(errors))


def main():
    root = Path("evidence/contact-correction-20261017")
    source = root / "comparison.json"
    destination = root / "verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = json.loads(source.read_text(encoding="utf-8"))
    errors = validate(report)
    for original, expected in report.get("source_sha256", {}).items():
        snapshot = root / "comparison-source" / Path(original).name
        if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
            errors.append("source snapshot identity")
    if len(report.get("source_sha256", {})) != 3:
        errors.append("source manifest completeness")
    receipt = {"passed": not errors, "errors": errors,
               "comparison_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "fd_selection": "central iterates only; paid finite-difference probes not candidate outputs"}
    destination.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
