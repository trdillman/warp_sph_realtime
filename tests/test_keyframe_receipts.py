"""CPU-only negative controls for budget and actual-outcome receipt validation."""

from copy import deepcopy

from scripts.verify_keyframe_results import candidate_errors, qualification_errors, trajectory_errors, trial_errors


def candidate(coverage=1.0, spill=0.0, loss=0.01):
    return {"coverage": coverage, "spill": spill, "loss": loss,
            "success": coverage >= 0.90 and spill <= 0.05, "controls": [0.0] * 8}


def trial():
    first, second = candidate(0.5, 0.25, 0.001), candidate()
    return {"method": "coordinate", "start": [0.0] * 8,
            "history": [
                {"candidate": first, "best": first, "units": 1, "forwards": 1, "backwards": 0,
                 "completed_seconds": 0.5},
                {"candidate": second, "best": second, "units": 2, "forwards": 2, "backwards": 0,
                 "completed_seconds": 1.5}],
            "best": second, "units": 2, "forwards": 2, "backwards": 0,
            "time_cutoff_seconds": 1.0, "best_at_time_cutoff": first,
            "completed_units_at_time_cutoff": 1, "wall_seconds": 1.6}


def test_hard_constraints_outrank_lower_surrogate_loss():
    assert not trial_errors(trial(), 2, 1.0)


def test_rejects_candidate_not_available_before_deadline():
    value = trial()
    value["best_at_time_cutoff"] = value["best"]
    assert any("unavailable" in error for error in trial_errors(value, 2, 1.0))


def test_rejects_uncharged_backward_work():
    value = trial()
    value["method"] = "adam"
    assert any("accounting" in error for error in trial_errors(value, 2, 1.0))


def test_rejects_loss_only_selection_and_false_success():
    value = trial()
    value["history"][1]["best"] = deepcopy(value["history"][0]["candidate"])
    assert any("constraint-ranked" in error for error in trial_errors(value, 2, 1.0))
    bad = candidate(spill=0.125)
    bad["success"] = True
    assert "success contradicts hard constraints" in candidate_errors(bad)


def test_rejects_unpaired_start_and_incomplete_budget():
    value = trial()
    value["start"] = [0.1] * 8
    errors = trial_errors(value, 4, 1.0)
    assert "trial did not evaluate paired start first" in errors
    assert "incomplete nominal budget" in errors


def test_rejects_independent_metric_disagreement_and_unsafe_witness():
    trajectory = {"finite": True, "minimum": [-1, -0.1, 0.05], "maximum": [-0.5, 0.1, 0.5],
                  "hard_floor_clamps": 0, "max_solid_penetration": 0.0,
                  "recomputed_coverage": 1.0, "recomputed_spill": 0.0}
    assert not trajectory_errors(trajectory, candidate())
    trajectory["recomputed_spill"] = 0.25
    assert "independent spill mismatch" in trajectory_errors(trajectory, candidate())
    trajectory["hard_floor_clamps"] = 1
    assert "unsafe trajectory" in trajectory_errors(trajectory, candidate())


def qualification():
    starts = [0.1] * 8
    near = [0.06, 0.01, 0.04, -0.01, 0.02, 0.01, 0, -0.01]
    fixed = [{"controls": controls, "epsilon": 0.0005, "rtol": 0.08, "atol": 0.0005}
             for controls in (starts, near)]
    fixed.extend({"controls": [0.0] * 8, "selected_seed": seed,
                  "epsilon": 0.0005, "rtol": 0.08, "atol": 0.0005} for seed in (0, 4, 7))
    return {"gradient_checks": fixed,
            "trials": [{"method": "adam", "seed": seed, "start": starts, "best": candidate()}
                       for seed in (0, 4, 7)],
            "witness": {"replay_passed": True, "replay_position_max_abs": 0.0, "replay_velocity_max_abs": 0.0}}


def test_prescribed_gradient_probe_identities_and_witness_replay():
    report = qualification()
    assert not qualification_errors(report)
    report["gradient_checks"].pop(3)
    assert "missing or duplicated prescribed gradient probes" in qualification_errors(report)
    report = qualification()
    report["gradient_checks"][3]["selected_seed"] = 0
    assert "missing or duplicated prescribed gradient probes" in qualification_errors(report)
    report = qualification()
    report["gradient_checks"][3]["controls"] = [0.2] * 8
    assert "selected derivative probe controls mismatch" in qualification_errors(report)
    report = qualification()
    report["witness"]["replay_passed"] = False
    assert "witness graph replay failed" in qualification_errors(report)
