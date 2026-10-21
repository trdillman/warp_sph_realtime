# Differentiable WCSPH: result and limits

**Result: the proof of concept is implemented; a useful goal-directed pouring advantage is not demonstrated.**

The existing reorder, density and force kernels run inside a retained Warp Tape rollout. Each timestep owns its state and both spatial hash grids. No production physics, interactive runtime, renderer, installer or dependencies were changed. The experiment is headless, on the actual CUDA device, and remains outside the add-on UI.

## Measured results

Five paired starts per fixture, 64 particles, 24 nominal work units per method. Random and coordinate search each get 24 forward evaluations; Adam gets 12 forward/backward pairs. Wall times are synchronized and exclude warmup and validation. These are equal nominal budgets, not strict equal-wall-time deadlines. No parameters were tuned after observing either fixture.

| Fixture / median result | Gradients (Adam) | Random search | Coordinate search |
|---|---:|---:|---:|
| Floor: horizontal distance loss (m², lower better) | 0.087415 | 0.085518 | **0.073124** |
| Floor: endpoint target coverage | 32.8% | 35.9% | 35.9% |
| Floor: search seconds | 2.11 | 2.22 | 2.17 |
| Obstacle-deflected: horizontal distance loss (m²) | **0.578301** | 0.606686 | 0.597867 |
| Obstacle-deflected: endpoint target coverage | **0%** | **0%** | **0%** |
| Obstacle-deflected: search seconds | 2.07 | 2.07 | 2.08 |

Coverage means particles inside x/y ±0.18 m around (0.30,0.20) and z≤0.18 m at the endpoint. It does not mean capture in a cup. Selection minimizes horizontal squared distance, which is intentionally reported separately from actual coverage. The obstacle result illustrates that this objective can improve while failing the intended task.

### Floor fixture: backward plumbing works, but simple search wins

All six central finite-difference checks passed at three controls and two perturbation sizes (0.002 and 0.001 m/s; tolerance 5% relative plus 0.002 absolute). Retained rollout versus production eager and CUDA Graph states had zero observed maximum position and velocity error at the checked control. All 15 selected results reproduced their loss in production CUDA Graph replay.

Coordinate search beat Adam in all five paired runs. Moreover, uniform horizontal velocity over a flat floor mostly translates the same deformation: independent review fitted all 300 evaluations to a quadratic with maximum residual below 2e-6 m². A direct center-of-mass correction would approximately solve this fixture without differentiating the fluid. This is a plumbing and aiming proof, not evidence for inverse fluid shaping.

### Obstacle fixture: forward parity passes, gradient robustness and goal fail

The additional fixture was fixed before running: translate the initial slug by (-0.50,0,0.88) m, enable the production sphere/pillars, and extend to 256 timesteps. Everything else stays fixed. The source snapshot and complete histories are retained.

Forward parity again had zero observed position/velocity difference at the checked control; all 15 chosen losses replayed through the production CUDA Graph. **Three of six gradient checks failed.** At controls (1.4,0.4), the automatic x derivative was about -0.08972, while finite differences were -0.03086 at epsilon 0.002 and +0.13095 at epsilon 0.001. The numerical derivative itself is step-size sensitive. This is consistent with a nonsmooth contact/neighborhood landscape, but does not isolate the cause or prove the automatic branch derivative is mathematically wrong.

Adam's median loss was slightly lower, but gradient validation failed and every tested method had zero endpoint coverage. The receipt therefore records `validation_failed` and the benchmark exits nonzero. No successful pour or robust differentiable-control advantage can be claimed. The target's reachability within these bounds and horizon is unproven; zero coverage is not a proof of infeasibility.

## Verification and reproduction

Environment: Windows, Python 3.14, warp-lang 1.17.0, CUDA toolkit 12.9, RTX 4080 SUPER, driver 610.88. This is not Blender/Python 3.13 runtime certification. Graph/eager tests do not prove viewport performance.

- `python -m pytest -q`: 66 passed, 2 skipped, 8 subtests passed after the new gradient-reset, control-validation and budget tests.
- Initial Pyright found one new optional-value error (fixed) and missing Blender stubs. The repository-pinned fake-bpy-module 20251003 was installed into isolated `.codex/poc-type-stubs`; with that directory in process-local PYTHONPATH, Pyright reported zero errors/warnings. No global package installation or project suppression was added.
- Final Ruff and plotting verification are recorded in the evidence logs.

Run from the repository root, using a fresh output filename to preserve prior receipts:

```powershell
python -m scripts.benchmark_differentiable_pour --fixture floor --output evidence/new-floor/results.json
python -m scripts.benchmark_differentiable_pour --fixture deflected --output evidence/new-deflected/results.json
python -m scripts.plot_differentiable_pour
```

The plotting script reads the retained dated receipts and does not rerun physics. Benchmark commands require CUDA and exact Warp 1.17.0. A failed acceptance gate intentionally returns nonzero, while keeping diagnostic comparisons.

## Evidence and source identity

- `evidence/differentiable-pour-20261017/results.json`: original floor receipt, all seeds, budgets, timings, gradients, parity and source hashes. Its runner predates a type assertion and later evidence/validation hardening; do not represent it as execution of the final runner.
- `evidence/differentiable-pour-20261017/deflected.json`: second fixture receipt, including initial/final particle positions, source hashes and explicit validation failure.
- `evidence/differentiable-pour-20261017/deflected-source/`: exact runner, retained-rollout module and production solver used for the second fixture.
- `evidence/differentiable-pour-20261017/comparison.png`: convergence and particle endpoint plots from receipts.
- `evidence/differentiable-pour-run.log` and `evidence/differentiable-pour-deflected.log`: raw benchmark output.
- `evidence/differentiable-pour-pytest-final.log` and `evidence/differentiable-pour-pyright-with-stubs.log`: checks.

## Engineering decision

Do not make differentiable pouring a product promise on this evidence. The retained rollout establishes that the current kernels can propagate useful gradients in a smooth case, but the collision case has not passed the gradient gate or achieved the target. A subsequent experiment should establish a reachable receiving goal, include vertical/spill terms in its objective, and investigate contact sensitivity before considering more controls or longer horizons. Those changes are not silently substituted into this completed comparison.
