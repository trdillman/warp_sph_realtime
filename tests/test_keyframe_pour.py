"""Small contract tests; GPU qualification is retained in the experiment receipt."""

import numpy as np
import pytest

from scripts.benchmark_keyframe_pour import optimize, starting_controls
from warp_sph_realtime.experimental.keyframe_pour import KeyframePour, outcome_rank, segments_hit_box


def test_controls_and_distinct_starts():
    starts = [starting_controls(seed) for seed in range(8)]
    assert all(KeyframePour.validate_controls(x).shape == (8,) for x in starts)
    assert len({tuple(x) for x in starts}) == 8
    for invalid in (np.zeros(2), np.full(8, np.nan), np.full(8, 0.31)):
        with pytest.raises(ValueError):
            KeyframePour.validate_controls(invalid)


def test_constraints_outrank_smooth_loss():
    feasible = {"coverage": 0.95, "spill": 0.03, "loss": 1.0}
    misleading = {"coverage": 0.85, "spill": 0.10, "loss": 0.00001}
    assert outcome_rank(feasible) < outcome_rank(misleading)


def test_segment_detects_skipped_spill_slab():
    a = np.array([[-0.9, 0.0, 0.5], [-1.45, 0, 0.5], [-0.9, 0.8, 0.5]])
    b = np.array([[-0.9, 0.0, 0.2], [-1.45, 0, 0.2], [-0.9, 0.8, 0.2]])
    assert segments_hit_box(a, b, [-1.22, -0.4, 0.3], [-0.68, 0.4, 0.4]).tolist() == [True, False, False]


def test_full_allowed_emission_domain_clears_obstacles():
    # Most rightward/horizontal-extreme nozzle particle is still outside all
    # compliant contact bands. Inter-key path has no emission, so only keys apply.
    corner = np.array([-1.45 + 0.30 + 0.09, 0.30 + 0.09, 0.50])
    sphere_clearance = np.linalg.norm(corner - [0, 0, 0.55]) - 0.55
    pillar_clearance = np.linalg.norm(corner[:2] - [-0.85, 0.55]) - 0.18
    assert min(sphere_clearance, pillar_clearance) > 0.06


@pytest.mark.parametrize("method", ["adam", "coordinate", "cem", "grouped_coordinate"])
def test_work_and_completed_time_accounting(method):
    def evaluate(x, gradient=False):
        return {"loss": float(np.sum(np.square(x))), "coverage": 0.0, "spill": 1.0,
                "success": False, "seconds": 0.001, "gradient": 2 * np.asarray(x) if gradient else None}
    result = optimize(evaluate, np.full(8, 0.1), method, 0, budget=14, time_limit=0)
    assert result["units"] == 14
    assert result["forwards"] + result["backwards"] == 14
    assert result["forwards"] == (7 if method == "adam" else 14)
    assert result["best_at_time_cutoff"] is None
    assert result["completed_units_at_time_cutoff"] == 0
    assert all(row["completed_seconds"] > 0 for row in result["history"])
    if method == "grouped_coordinate":
        original = np.asarray(result["start"]).reshape(4, 2)
        for row in result["history"]:
            controls = np.asarray(row["candidate"]["controls"]).reshape(4, 2)
            assert np.allclose(controls - controls[0], original - original[0])


def test_grouped_bounds_preserve_nonuniform_key_shape():
    start = np.array([-0.28, 0.15, -0.20, -0.10, 0.15, 0.0, 0.28, 0.20])

    def evaluate(x, gradient=False):
        return {"loss": float(np.sum(np.square(x))), "coverage": 0.0, "spill": 1.0,
                "success": False, "seconds": 0.001, "gradient": None}

    result = optimize(evaluate, start, "grouped_coordinate", 0, budget=12)
    original = start.reshape(4, 2)
    for row in result["history"]:
        controls = np.asarray(row["candidate"]["controls"]).reshape(4, 2)
        assert np.allclose(controls - controls[0], original - original[0])
        assert np.all(np.abs(controls) <= 0.30 + 1e-7)
