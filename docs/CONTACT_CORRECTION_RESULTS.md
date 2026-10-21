# Contact correction and reachable artist goal

**Result: the corrected gradient optimizer achieves the artist's receiving goal in all eight runs, but does not beat coordinate search on this two-control task.** Median initial coverage was 47.7%; both methods reached 100% median final coverage. The default solver modes also match the preserved pre-change solver exactly in the measured regression cases.

| Selected output at fixed budget | Automatic gradients | Coordinate search | Random search | FD gradients (central iterates) |
|---|---:|---:|---:|---:|
| Runs with ≥75% target coverage | 8/8 | 8/8 | 6/8 | 7/8 |
| Median final coverage | 100% | 100% | 95.3% | 100% |
| Median objective, lower better | 0.00083334 | **0.00082256** | 0.00124954 | 0.00085542 |
| Median complete search time | 12.42 s | 12.61 s | 13.12 s | 12.05 s |
| Median work units to first ≥75% coverage | 5 | **3** | 12 among 6 successes | 12.5 |

Automatic gradients and coordinate search both found successful outputs in every run. Coordinate search reached the threshold with fewer nominal work units and selected a lower-loss result in every pair. Automatic gradients require fewer rollouts per update than numerical finite differences, but that does not establish an advantage over coordinate search for two controls. FD-Adam visited a ≥75% central iterate in all eight runs, then selected a lower-loss but lower-coverage iterate in one run; the table reports the actual selected output. This remaining objective/coverage mismatch is explicit.

All 32 selected trajectories avoided obstacle contact (floor contact remained). Contact-gradient correctness was exercised separately by the fixed-control regression cases. The successful outputs therefore prove automatic correction away from obstacles, not optimization of a successful obstacle-deflected splash.

All **86/86** prescribed gradient checks passed (six fixed controls/epsilons, sixteen start checks, sixty-four selected-output checks). All **32/32** selected results replayed through production CUDA Graph execution. The selected trajectories and reference witness had zero measured solid penetration or hard floor clamps, and stayed below the pillar top fade. Checked mode-2 eager/graph position and velocity differences were zero. The independent receipt verifier passed.

Timing is descriptive: A/A ratios ranged from 0.818 to 1.066 on the shared desktop. The roughly 1.5% difference in median total runtime between gradients and coordinate search is not a meaningful speed claim. The evidence supports successful automatic aiming in this fixture, not a market-wide differentiability advantage or arbitrary fluid art direction.

## What changed

The initial proof confused several distinct issues. Hard obstacle restitution has a discontinuous onset; the old receiving volume partly intersected the sphere; and the solver's dynamic neighbor accumulations were not safe for Warp reverse-mode replay. The latter was a real gradient implementation error, not merely numerical noise at contact.

`particle_density` and `particle_forces` now isolate the neighbor accumulations in `@wp.func` functions. Warp replays these functions before differentiating their consumers. This follows the pinned [Warp 1.17 dynamic-loop guidance](https://raw.githubusercontent.com/NVIDIA/warp/v1.17.0/docs/user_guide/differentiability.rst). The density and force kernels remain shared by production eager, CUDA Graph and retained differentiable execution; no surrogate solver or invented backward derivative is used.

An opt-in obstacle mode 2 adds C1 compliant contact for sphere, pillar sides and domain walls, a 1% rest-density clamp transition and a smoothed approaching-pair artificial viscosity term. This is a changed forward model, consistently used by every comparison method. The density regularization yields 1002.5 kg/m³ at a raw density of 1000 kg/m³, so it must not be presented as physically identical to the original law. Modes 0 and 1 retain the original constitutive and contact equations. Existing Blender demos still select modes 0/1; this experiment is not an installed UI feature.

The pillar height fade has no solid cap. This experiment stays below the fade/top region; it does not qualify capped-cylinder collision. The compliant band is 0.06 m, equal to particle spacing. Solid penetration and hard domain-clamp use are checked over complete retained trajectories, rather than inferred from endpoints.

## Root-cause evidence

- `evidence/contact-correction-20261017/diagnostics.json`: first compliant long-horizon attempt, still failed gradients and only 18.75% witness coverage. Preserved, not counted as acceptance.
- `diagnostics-refined.json`: lower source and smaller timestep made the target reachable (96.875%), but a stable derivative mismatch remained before floor contact.
- `horizon.json` and `vjp.json`: prefix and separate density/force/contact derivative investigations. Isolated single-particle contact derivatives agreed; the interacting path required further investigation.
- `diagnostics-regularized.json`: smoothing constitutive transitions alone did not fix the remaining error.
- `diagnostics-replay-fix.json`: moving the accumulations into replayable functions fixed it. At controls (1.4, 0.4), the x derivative changed from 0.046074 to 0.074420; central finite differences remained approximately 0.074232. All six prescribed checks passed without loosening tolerances. Witness coverage became 100% in the regularized model. These are separate exploratory candidates, not repeated measurements of one candidate.

## Frozen artist task and comparison

Deliver a finite liquid slug into a clear floor region at a specified time. The target is centered at (-0.8, 0), half-width 0.22 m, with z≤0.18 m. The loss penalizes distance outside this box in all three dimensions, with 0.05 times horizontal squared center distance to guide motion inside it. Coverage independently counts actual particles within the box.

The source spans x[-.85,-.67], y[-.09,.09], z[.60,.78]. There are 64 particles, 384 timesteps at CFL 0.2, and approximately 0.367 seconds of simulated motion. Controls are uniform initial horizontal velocities, each bounded [-2,2] m/s. The target is below the initial slug; this tests correction of mis-aimed launches, not a goal requiring the successful solution to bounce around an obstacle. A zero-horizontal-velocity witness establishes feasibility and is never supplied to optimizers.

Eight paired starts use RNG seeds 1000..1007, vx=1+uniform(-.5,.5), vy=uniform(-.5,.5). Every method uses the same model, target, controls and start. The cap is 64 nominal forward/backward work units. Adam uses 32 forward/backward pairs; random and coordinate search use 64 forwards. Finite-difference Adam uses 12 iterations costing 60 forwards. Its candidate selection is **central iterates only**; the four paid derivative probes per iteration are not selected as output or counted toward first-success time. Do not claim this secondary baseline returns the best of every point it evaluates.

Success means at least 75% coverage. Adam step 0.25, coordinate initial step 0.5 and all random seeds were fixed before measuring this comparison. Each trial resets initial state, and order rotates across methods. Kernel compilation/setup and correctness checks are excluded from search timing; synchronization and readback are included. Eight same-input A/A checks record repeatability. Times are descriptive on a shared desktop, not a strict equal-wall-time or isolated hardware speed certificate.

## Verification and evidence boundaries

The definitive comparison receipt is `evidence/contact-correction-20261017/comparison.json`; its `comparison-source/` directory contains exact benchmark, experiment and solver snapshots. The independent `verify_contact_results.py` checks trial completeness, gradients, work accounting, replay, coverage and trajectory safety including the witness. Source and receipt hashes bind its verification receipt.

After the measured run, the runner's witness gate was hardened to reject penetration, clamps, non-finite state and pillar-top entry, and FD selection metadata was made explicit. No numerical algorithm, solver or experiment changed after measurement. The saved measurement snapshot remains the executed runner identity; the independent verifier applied the stronger gate to the saved receipt successfully. The delivered solver and experimental module still match their measured snapshots.

`legacy-parity.json` compares the preserved pre-correction solver with the delivered solver over 384 timesteps, for modes 0 and 1 and both eager/graph execution. All four rows have exactly zero measured position and velocity differences. `verification.json` records the successful independent acceptance check and hashes. `comparison.png`, `pour-endpoints.png` and `pour-comparison.gif` visualize the receipt; the animation uses actual retained particle states, slowed for inspection, not a Blender viewport recording.

Final checks: **71 tests passed, 2 skipped, 8 subtests passed**. Ruff and Pyright passed; Pyright used the previously isolated, repository-pinned Blender stubs through process-local PYTHONPATH. `git diff --check` passed. Logs are `evidence/contact-correction-pytest-final.log`, `contact-correction-ruff-final.log` and `contact-correction-pyright-final.log` under the same evidence directory root.

Final independent read-only review recomputed the 86 numerical gradient checks, reconciled all 32 trial receipts and source hashes, and found no remaining blocker for this stated experiment.

This remains headless Windows/Python 3.14/Warp 1.17.0/CUDA 12.9 on an RTX 4080 SUPER with driver 610.88. It does not prove Blender 5.2/Python 3.13 installation, viewport use, continuous emission, a physical cup, larger fluid systems or reliable long-horizon optimization. Old receipts and plots under `evidence/differentiable-pour-20261017/` remain historical results.

Reproduce from the repository root, choosing a fresh output path:

```powershell
python -m scripts.benchmark_differentiable_pour --fixture reachable --output evidence/new-contact-run/comparison.json
python -m scripts.verify_contact_results
python -m scripts.plot_contact_correction
```

The verifier and plotting commands operate on the retained dated receipt; they do not rerun simulation. Current scoped task record: `docs/DIFFERENTIABLE_POUR_POC.md`. Tracker synchronization remains queued in `docs/TRACKER_UPDATE_PENDING.md` because Page read/update tools are unavailable.
