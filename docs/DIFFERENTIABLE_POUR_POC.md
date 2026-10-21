# Differentiable pouring proof of concept

## Contract and implementation plan

Authorized 2026-10-17: use the current solver and compare goal-directed optimization against simpler search. This is a headless CUDA experiment, not Blender UI delivery or commercial qualification.

Architecture: reuse the production reorder, density and force kernels without changing their equations. Retain distinct state arrays and both hash grids per timestep for Warp Tape backward. Rebuild neighborhoods on every forward step; differentiation follows the executed discrete ordering/contact branches. No neural surrogate, ballistic replacement or differentiable renderer.

Files: `experimental/differentiable_pour.py` owns the retained rollout and objective; `scripts/benchmark_differentiable_pour.py` owns the seeded comparison and JSON receipts; `tests/test_differentiable_pour.py` checks forward equivalence, gradients and input validation. Production preview code stays unchanged.

Acceptance:
- [x] Test first, then implement retained execution of the existing kernels.
- [x] Verify positions and velocities against RealtimeWCSPHSolver eager and graph paths with identical initial conditions and hash-grid dimensions.
- [x] Check automatic gradients against central finite differences, including floor contact, before optimization. Floor passes; deflected fixture fails 3/6 checks, so comparisons there are diagnostic only.
- [x] Run bounded two-control optimization and seeded random/coordinate search; report full histories, forward/backward counts and synchronized wall time. Count one backward as one additional rollout-equivalent for the nominal budget, but do not equate this with measured compute time.
- [x] Report endpoint target coverage as well as smooth squared-distance objective. Use shared starting controls and the same bounds for all methods.
- [x] Run pytest, Ruff and Pyright; preserve errors and distinguish baseline failures.

Fixed first workload: 4x4x4 interacting particles, spacing 0.06 m, production material constants, 192 CFL timesteps, floor at z=0. Initial block x=-0.35..-0.17, y=-0.09..0.09, z=0.12..0.30. Optimize uniform initial x/y velocity in [-2,2] m/s. Target floor patch centered at (0.30,0.20), half-width 0.18 m. Loss is mean squared horizontal distance to its center; coverage counts particles within the patch and below z=0.18. This is a finite slug, not continuous emission or a physical receiving cup.

Comparison: five seeds, 24 rollout-equivalent budget per method including the shared initial evaluation. Projected Adam uses fixed step size 0.25; coordinate search starts at step 0.5 and halves it after an unimproved sweep; random search samples uniform bounded controls. Report every seed; do not tune the workload or optimizer after viewing results to manufacture a win. Warm compilation outside timing. Preserve a negative or inconclusive outcome.

Progress: implementation and bounded comparison complete. See `docs/DIFFERENTIABLE_POUR_RESULTS.md`. No useful pouring advantage demonstrated; do not claim successful product behavior. Original checkout main at 617b4ea; pre-existing .gitignore changes, .ignore, plans, reports and research notes are preserved, along with an AGENTS.md update observed during this task. Runtime observed: Python 3.14, Warp 1.17.0, CUDA toolkit 12.9, RTX 4080 SUPER, NVIDIA driver 610.88. Headless Python evidence does not establish Blender Python 3.13 compatibility. Next action: queued tracker synchronization; any follow-on physics experiment should address reachable objectives and contact-gradient sensitivity.

## Evidence-driven extension (fixed before second experiment)

The first floor result passed six gradient checks and exact eager/graph parity, but Adam lost to coordinate search in all five seeds. Independent review fitted all evaluations to a horizontal-translation quadratic with residual below 2e-6: the initial two controls do not materially affect deformation. Preserve `evidence/differentiable-pour-20261017/results.json` as an easy plumbing test, not evidence of inverse fluid shaping.

Add one `deflected` fixture to test this missing mechanism: same 64 particles, parameters, loss, optimizer settings, seeds and budgets; initial block translated by (-0.50,0,0.88) m; production sphere/pillar obstacles enabled; 256 CFL steps. No new collision model or receiving cup. Retain identical gradient tolerances and finite-difference controls. Report failed gradient gates as validation failures, even if an empirical optimizer result improves. Save source snapshots with receipts and final particle positions for visualization. Do not tune parameters after observing this second fixture.

## Authorized contact/reachability correction

User requested fixing contact gradients, proving reachability and repeating the artist-goal comparison. Current critical path: isolate hard contact discontinuity; implement opt-in compliant contact mode shared by eager/graph/retained callers; test its derivatives and collision exclusion; define a clear target and successful reference control; freeze and run matched comparison; review and document results.

Root mechanism under investigation: original collision projection applies finite velocity reflection immediately on crossing the surface, while no reflection occurs just outside. The old target volume intersects the central sphere above its lower surface; the point (0.3,0.2,0) itself is outside the sphere. Keep historical receipts unchanged.

Candidate contract: add obstacle mode 2 using C1 compact-support repulsion and differentiable approach damping. Keep mode 1 intact for existing demos. Mode 2 is compliant forward physics, not a fabricated backward derivative of hard collision. Keep exact domain safety clamps; reject material solid penetration in the experiment. Reuse production kernels and retain both grids per timestep.

New floor target: center (-0.8,0), half-width 0.22 m, z≤0.18 m; outside the sphere and pillars. Use the existing deflected initial slug and 512 CFL steps. Objective is smooth squared distance outside the receiving box in all three dimensions, plus a small center attraction to avoid flat gradients. Require a reference control to place at least 75% of particles in the target before any comparison. The reference is verification only, never supplied to optimizers. The goal is a receiving region, not a meshed cup.

Proposed comparison before measurement: eight paired seeded starts, fixed 64 nominal forward/backward units, unchanged Adam step .25, random search and coordinate search; also measure gradient versus finite-difference gradient descent to distinguish automatic-gradient efficiency from optimizer quality. Report success fraction (≥75% coverage), best loss and wall time. Freeze any final protocol changes before comparison. No post-result tuning. Diagnostic rollouts and derivative checks are outside optimization budgets and saved separately.

Progress: correction and comparison completed; see Delivery status below. Tracker read/update tools remain unavailable and the pending queue is current.

### Final frozen correction candidate (before comparison)

The initial compliant 512-step test did not pass reachability or derivatives; retain diagnostics.json. Lowering the source by 0.4 m and halving CFL to .2 at 384 steps proved reachability but exposed a stable derivative mismatch before floor contact. Isolated density/force VJPs and the Warp 1.17 dynamic-loop documentation identified replay as an additional root cause. Extracting neighbor accumulations into `@wp.func` functions fixed the mismatch without changing the measured forward result. Mode 2 also uses a 1% rest-density C1 clamp transition and a smooth approaching-pair viscosity onset; these are explicit forward regularizations, not fake gradients. Modes 0/1 retain the original constitutive/contact laws.

Frozen candidate: reachable source z=.60..78, 384 timesteps at CFL .2 (~.367 s); target unchanged; eight seeds 0..7 with shared starts vx=1+uniform(-.5,.5),vy=uniform(-.5,.5), RNG seed1000+seed. Adam/random/coordinate/finite-difference Adam share bound[-2,2], objective and64 work-unit cap; FD-Adam costs5 forward calls/iteration and fits60 calls. Adam step.25 and coordinate step.5 unchanged. Witness[0,0] is verification-only and never supplied to optimizers. Warmup2 calls, eight A/A checks; elapsed times include synchronization and readback, exclude setup/validation. Record actual success≥75%, time to first success, replay, gradients at starts and selected outputs, and full-trajectory penetration≤1mm/no hard clamps. Nominal-budget comparison, not a strict wall-time speed certification. No comparison outcomes were observed before this freeze.

Diagnostics before freeze: all6 derivative checks pass at original controls/epsilons; witness100%coverage; zero observed solid penetration and hard clamps. Root reference: https://raw.githubusercontent.com/NVIDIA/warp/v1.17.0/docs/user_guide/differentiability.rst (Dynamic Loops).

### Delivery status

Correction and comparison completed. `CONTACT_CORRECTION_RESULTS.md` is the detailed report. Automatic gradients and coordinate search both succeed8/8; random6/8; FD-Adam selected output7/8. Median coverage100% for gradients/coordinate, no advantage over coordinate search. All86 gradient checks,32 selected replays and independent receipt validation passed. The old modes0/1 match the preserved solver exactly in4 eager/graph regression cases. Failed exploratory candidates remain in evidence. No installation, UI, release or broad fluid-shaping qualification is claimed.

Artifacts: `evidence/contact-correction-20261017/` contains measured source snapshots, comparison JSON, independent verification, legacy parity, charts and retained-state animation. Final runner adds a stricter witness gate and explicit FD selection metadata after measurement; numerical source remains identical to the measured candidate. Remaining limitation: target is a two-control finite-slug floor region, and the smooth model is opt-in. Tracker update is pending connector access; no further runtime retry is required for this completed bounded comparison.

Final checks: 71 tests passed, 2 skipped, 8 subtests passed; Ruff passed; Pyright passed using the isolated pinned Blender stubs; `git diff --check` passed. Final read-only review and evidence reconciliation accompany delivery.
