import numpy as np
import pytest
import warp as wp

from warp_sph_realtime.sim.sim_core import compliant_contact_accel


@wp.kernel
def sample_contact(d: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = compliant_contact_accel(d[i], 1.0, 0.06, 25.0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_contact_force_is_continuous_with_correct_derivative():
    samples = np.array([0.0601, 0.06, 0.0599, 0.03, -0.002], dtype=np.float32)
    d = wp.array(samples, device="cuda:0", requires_grad=True)
    out = wp.zeros(5, device="cuda:0", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(sample_contact, 5, [d, out], device="cuda:0")
    tape.backward(grads={out: wp.ones(5, device="cuda:0")})
    actual = out.numpy()
    assert actual[0] == actual[1] == 0
    assert 0 < actual[2] < 0.02
    assert actual[4] > actual[3] > actual[2]
    numerical = []
    for delta in (0.00001, -0.00001):
        shifted = wp.array(samples + delta, device="cuda:0")
        value = wp.zeros(5, device="cuda:0")
        wp.launch(sample_contact, 5, [shifted, value], device="cuda:0")
        numerical.append(value.numpy())
    fd = (numerical[0] - numerical[1]) / 0.00002
    np.testing.assert_allclose(d.grad.numpy()[3:], fd[3:], rtol=0.005, atol=0.1)


def test_reachable_fixture_is_available():
    from warp_sph_realtime.experimental.differentiable_pour import PourExperiment
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    experiment = PourExperiment(steps=8, fixture="reachable")
    assert experiment.solver.enable_obstacles == 2


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_contact_rollout_gradient_replay_and_reachability():
    from warp_sph_realtime.experimental.differentiable_pour import PourExperiment
    experiment = PourExperiment(384, "reachable")
    # This previously gave dx=.04607 versus finite difference=.07423. Testing
    # the entire rollout catches dynamic-loop replay errors missed by unit contact tests.
    for epsilon in (0.002, 0.001):
        check = experiment.check_gradient([1.4, 0.4], epsilon)
        assert check["passed"], check
    actual = experiment.evaluate([0.7, -0.2])
    trajectory = experiment.trajectory_diagnostics()
    assert trajectory["contact_particle_steps"] > 0
    assert trajectory["max_solid_penetration"] <= 0.001
    assert trajectory["hard_floor_clamp_particle_steps"] == 0
    for graph in (False, True):
        p, v = experiment.production_rollout([0.7, -0.2], graph)
        np.testing.assert_allclose(p, actual["positions"], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(v, actual["velocities"], rtol=2e-5, atol=2e-6)
    witness = experiment.evaluate([0, 0])
    assert witness["coverage"] >= 0.75


def test_finite_difference_search_charges_all_evaluations():
    from scripts.benchmark_differentiable_pour import optimize
    calls = []

    def objective(controls, gradient=False):
        assert not gradient
        calls.append(controls.copy())
        residual = np.asarray(controls) - [0.3, -0.2]
        return {"loss": float(residual @ residual), "coverage": 0, "seconds": 0}

    result = optimize(objective, [1.0, 0.4], "fd_adam", 0, budget=64)
    assert result["forward_evaluations"] == len(calls) == result["units"] == 60
    assert result["backward_evaluations"] == 0
    assert result["best"]["loss"] < result["history"][0]["loss"]


def test_receipt_gate_rejects_witness_penetration_and_missing_work():
    import copy

    from scripts.verify_contact_results import validate
    trajectory = {"max_solid_penetration": 0, "hard_floor_clamp_particle_steps": 0,
                  "finite_all_states": True, "max_z": 0.8}
    report = {"reachability_witness": {"coverage": 1, "replay_matches": True, "trajectory": trajectory},
              "gradient_checks": [{"passed": True}] * 6, "start_gradient_checks": [{"passed": True}] * 16,
              "parity": [{"passed": True}] * 2, "repeatability": [{"loss_delta": 0}] * 8,
              "trials": []}
    for seed in range(8):
        for method in ("adam", "random", "coordinate", "fd_adam"):
            units = 60 if method == "fd_adam" else 64
            backwards = 32 if method == "adam" else 0
            report["trials"].append({"seed": seed, "method": method, "units": units,
                "forward_evaluations": units - backwards, "backward_evaluations": backwards,
                "replay_passed": True, "production_positions_match": True,
                "production_coverage": 1, "best": {"coverage": 1}, "trajectory": trajectory.copy(),
                "gradient_checks": [{"passed": True}, {"passed": True}]})
    assert validate(report) == []
    bad = copy.deepcopy(report)
    bad["reachability_witness"]["trajectory"]["max_solid_penetration"] = 0.02
    assert "trajectory safety" in validate(bad)
    bad = copy.deepcopy(report)
    bad["trials"].pop()
    assert "trial completeness" in validate(bad)
    bad = copy.deepcopy(report)
    bad["gradient_checks"][0]["passed"] = False
    assert "gradient checks" in validate(bad)
