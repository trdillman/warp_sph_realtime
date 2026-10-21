import numpy as np
import pytest
import warp as wp

from warp_sph_realtime.experimental.differentiable_pour import PourExperiment
from scripts.benchmark_differentiable_pour import optimize


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_retained_rollout_matches_production():
    experiment = PourExperiment(steps=8)
    for graph in (False, True):
        actual = experiment.evaluate([0.7, -0.2], gradient=False)
        positions, velocities = experiment.production_rollout([0.7, -0.2], use_graph=graph)
        np.testing.assert_allclose(actual["positions"], positions, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(actual["velocities"], velocities, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_gradient_matches_finite_difference():
    experiment = PourExperiment(steps=8)
    result = experiment.check_gradient([0.7, -0.2])
    assert result["passed"], result


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="steps"):
        PourExperiment(steps=0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_tape_reuse_does_not_accumulate_gradients():
    experiment = PourExperiment(steps=8)
    first = experiment.evaluate([0.7, -0.2], gradient=True)
    experiment.evaluate([-0.4, 0.6], gradient=True)
    last = experiment.evaluate([0.7, -0.2], gradient=True)
    np.testing.assert_allclose(first["gradient"], last["gradient"], atol=2e-6, rtol=2e-5)
    assert last["loss"] == pytest.approx(first["loss"], abs=2e-6)


@pytest.mark.parametrize("method", ["adam", "random", "coordinate"])
def test_comparison_budget_and_best_observed_result(method):
    calls = []

    def quadratic(controls, gradient=False):
        controls = np.asarray(controls)
        calls.append((controls.copy(), gradient))
        residual = controls - [0.8, -0.4]
        return {"loss": float(residual @ residual), "coverage": 0.0,
                "gradient": 2 * residual, "seconds": 0.0}

    result = optimize(quadratic, [0.0, 0.0], method, seed=7)
    assert sum(1 + int(gradient) for _, gradient in calls) == result["units"] == 24
    assert result["forward_evaluations"] == len(calls)
    assert all(np.all(np.abs(controls) <= 2) for controls, _ in calls)
    assert result["best"]["loss"] == min(row["loss"] for row in result["history"])
    assert result["best"]["loss"] < result["history"][0]["loss"]
    np.testing.assert_array_equal(calls[0][0], [0.0, 0.0])


@pytest.mark.parametrize("controls", [[np.nan, 0], [3, 0], [0], [[0, 0]]])
def test_invalid_controls_rejected(controls):
    with pytest.raises(ValueError, match="controls"):
        PourExperiment.validate_controls(controls)
