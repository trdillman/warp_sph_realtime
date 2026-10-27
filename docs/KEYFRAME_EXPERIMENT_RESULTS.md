# Multi-keyframe emitter: gradients help reliability, simple correction wins the short budget

The completed experiment found a useful but bounded advantage for automatic
gradients. At 64 nominal simulation-work units, gradients produced 8/8 outputs
meeting the artist's target and spill requirements; the strongest simple search
produced 5/8. At an actual 10-second cutoff, the simple whole-path translation
search produced 5/8 successful outputs and gradients produced 4/8. There is no
blanket winner across the two budgets.

This supports keeping differentiable control as one research avenue within the
profitable-addon design. It does not establish that differentiability is a
commercial differentiator, that artists would pay for it, or that the addon is
ready to sell. The useful product concept is assistance toward an intended fluid
outcome, with inexpensive correction available before a more costly optimizer.
A combined workflow is a recommendation to test, not a feature demonstrated here.

## What actually ran

The current WCSPH density, force and sorting kernels advanced 64 particles on the
RTX 4080 SUPER. Four 16-particle bursts were released at four emitter-position
keyframes. Eight controls adjusted the four x/y positions. Each release inherited
the emitter's secant-derived keyframe velocity and a fixed 2 m/s downward nozzle
velocity. Unreleased particles were absent from neighbor grids; released fluid
received no steering force. This is a discrete-burst emitter, not a continuous
stream or an animated Blender emitter feature.

The artist's goal was at least 90% of particles through a receiving aperture and
at most 5% ever entering a protected nearby volume. The target uses each particle's
first downward crossing at z = 0.35 m, with its crossing position interpolated between
SPH steps. Spill uses segment/box intersections over the entire simulation,
including after delivery, and unique persistent particle IDs. Captured fluid
continues to evolve below the aperture; the experiment does not model retained
basin filling. A per-particle nonarrival penalty prevents avoiding the goal by
staying airborne.

The 64-particle rollout lasted 0.611957 s at dt = 0.000956183 s. Burst final ages were
0.611957, 0.520163, 0.428370 and 0.336576 s. Each forward evaluated 640 timesteps,
31,744 particle updates. The zero-offset feasibility witness delivered 100% with
0% spill and was never supplied to the optimizers. None of the eight starting
animations met both constraints: median starting target coverage was 68.75% and
median spill was 33.59375%.

Eight paired starts were fixed before the main comparison. Adam, independent
coordinate search, cross-entropy search and grouped coordinate search used the
same evaluator, bounds, starting controls and candidate selection. Grouped search
translates all four keys together, preserving the path's relative shape and
inherited velocities. It is an important baseline because the initial errors
contain a substantial common offset and smaller individual key errors.

Selection gives a constraint-satisfying candidate priority over every infeasible
candidate; otherwise it minimizes summed hard-constraint violations, then the
smooth optimization score. A lower smooth loss therefore cannot hide a failed
artist goal. The exact geometry, algorithms, tolerances and pre-measurement
decisions are in [the frozen contract](KEYFRAME_EXPERIMENT_CONTRACT.md).

## Measured artist outcomes

Success means at least 58/64 particles through the aperture and at most 3/64 spilled.
The table gives observed paired outcomes, not a prediction for new scenes.

| Method | Successful outputs at 64 work units | Successful outputs by 10 s | Median target at 64 units | Median spill at 64 units |
|---|---:|---:|---:|---:|
| Automatic gradients (Adam) |8/8|4/8|100%|2.34375%|
| Independent coordinate search |3/8|0/8|91.40625%|6.25%|
| Cross-entropy search |4/8|2/8|96.875%|5.46875%|
| Translate whole path |5/8|5/8|100%|3.90625%|

Gradients solved the three paired starts where whole-path search still failed
within 64 units. In those grouped-search outputs, target coverage reached 100%,
but spill remained above the allowed 5%; coordinating the individual keys helped
meet the second requirement. This is stronger evidence than simply lowering
a loss value or moving a single initial velocity toward a target. It is an
advantage over each single tested baseline, not a unique capability: across
the three simple methods, at least one solved every starting case. An adaptive
or combined simple-search schedule under the same total budget was not tested.

Whole-path search reached its first successful output in a median 3.31 s among
the five cases it solved. Gradients reached first success in a median 9.05 s across
all eight cases; individual coordinate search took 27.10 s among three successes,
and cross-entropy 16.55 s among four. These conditional medians must be read with
their success counts. The common 10-second result is the clearer comparison for
an artist wanting a quick answer: 5/8 for whole-path search versus 4/8 for gradients.

The full 64-unit searches took median 33.14 s for gradients, 32.34 s for coordinate,
32.40 s for cross-entropy and 30.00 s for whole-path search. One forward and one
reverse each cost one nominal unit: Adam performed 32 forwards plus 32 reverses;
the other methods performed 64 forwards. Equal nominal work does not imply equal
execution time. Time-limited selections include only fully completed evaluations,
including gradient/reverse work and result readback; unfinished work receives
no credit. All fixed-budget traces extended past 10 s.

Measured outcomes (`evidence/keyframe-pour-20261017/main/outcomes.png`)

Actual particle animation (`evidence/keyframe-pour-20261017/main/artist-comparison.gif`)
shows seed 0 before correction, after gradients and after whole-path search at
the 64-unit budget. The top row shows x/y and the bottom x/z. The green aperture,
red protected volume and gray obstacle projections remain visible. All particle
motion comes from saved SPH states; the caption explains that target counts a
crossing rather than final floor positions. This seed illustrates a case both
methods ultimately solve; the complete eight-pair results are retained separately.

## Verification and measurement limits

Five complete eight-component centered finite-difference checks passed: the
seed 0 start, a near-witness control vector and selected Adam outputs for
seeds 0, 4, 7. All 40 scalar components meet the frozen 8% relative plus 0.0005 absolute
tolerance at perturbation 0.0005 m. Both the 64-unit and 10 s selected outputs in each
of 32 trials, plus the witness, passed eager versus shared-kernel CUDA Graph
replay. Every position and velocity comparison was exactly equal in these runs.
This verifies per-step captured density/force kernels with grids outside capture;
production step() does not implement staged emission yet.

For every selected output and the witness, a separate NumPy implementation
recomputed hard spill intersections and first-crossing target counts from all
retained states and IDs, exactly matching GPU observations. Birth/input and
output positions and velocities were finite and bounded, with zero hard floor
clamps and zero observed solid penetration. The entire allowed emission-position
box clears the compliant obstacle-contact band. The protected box overlaps a
narrow approximately 3 cm projected fringe of the existing pillars near y = ±0.4 m;
it is not wholly solid-free. No selected trajectory penetrated those obstacles.

Eight A/A pairs had timing ratios 0.945..1.118 (median 0.994), inside the
predeclared 0.8..1.2 envelope. Measurements used one warmed persistent process with
four-method order rotated across seeds. This deliberately departs from fresh
process performance qualification and does not control the entire desktop/GPU
environment. Results are exploratory time-to-outcome observations, not a
certified speedup or statistical proof of superiority.

All methods use the same eager evaluator retaining per-step states and grids
for differentiation and inspection. A specialized forward-only simple search
could reuse buffers/grids or capture more work; that implementation is unmeasured.
Therefore these time results do not establish the fastest possible shipping
implementation of either approach. Small 64-particle bursts, eight starts, one
goal geometry, one set of optimizer settings and scripted constraints do not
establish general fluid shaping, artist satisfaction, scaling to production
particle counts, Blender UI acceptance or commercial demand.

Environment: Windows 11 build 26200, Python 3.14.3, NumPy 2.3.5, Warp 1.17.0,
CUDA toolkit 12.9, NVIDIA GeForce RTX 4080 SUPER 16 GB, driver 610.88. This is headless
CUDA evidence; it does not qualify Blender's bundled Python 3.13 or installed UI.

Final repository checks: **87 passed, 2 skipped, 8 subtests passed** (18.77 s),
Ruff passed, pinned Pyright 1.1.408 reported zero errors/warnings, and
`git diff --check` passed. Logs are `evidence/keyframe-pytest-final.log`,
`evidence/keyframe-ruff-final.log` and `evidence/keyframe-pyright-final.log`.
The stricter independent receipt audit passed all 32 trials and 5 gradient
vectors; its receipt is verification.json (`evidence/keyframe-pour-20261017/main/verification.json`).
After freezing the numerical candidate, tests were strengthened for grouped-key
bounds and the independent verifier was tightened; runner, experiment and solver
numerics were unchanged. Original measured source snapshots remain preserved;
the final audit records its own verifier hash.

Comparison SHA-256: `48bf49a71f6a8e043ca11af83c97e11f62f12efbff6bdf5e88f69174f726b50e`.
Final verifier SHA-256: `5cb0c9f7ec550d561724c3d0d13de2de43f78a0af5f82601f7cb6c599658fcd0`.

Independent final review passed artifact correctness and task intent: all 65
selected-output/witness proof records, all 40 derivative components and measured
source/artifact hashes were checked, with no outstanding findings. This is
acceptance of the bounded experiment, not release or commercial acceptance.

## Evidence and reproduction

The `evidence/` paths below are the local evidence archive; it is gitignored and not published with the repository.

- Measured receipt (`evidence/keyframe-pour-20261017/main/comparison.json`): all 32 trials, hard metrics, every candidate, work counts, completion timestamps, gradients, A/A samples, environment and source hashes.
- Machine-readable summary (`evidence/keyframe-pour-20261017/main/summary.json`): per-method/per-seed outcomes and censored first-success measurements.
- Measured source snapshots (`evidence/keyframe-pour-20261017/main/source/`): exact runner, experiment, current solver, frozen contract and original audit/test pins.
- Plot provenance (`evidence/keyframe-pour-20261017/main/plots.json`): receipt, plot-source and generated artifact SHA-256 values.
- Raw main log (`evidence/keyframe-pour-comparison.log`): actual CUDA module/device execution and trial outputs.
- Failed pilots and diagnostics (`evidence/keyframe-pour-20261017/`): initial floor-retention failures, downward-nozzle fallback, aperture contact-sensitive FD failure, epsilon-convergence evidence and qualifying translated pilot.

Executed main command from the repository root:

```text
python -m scripts.benchmark_keyframe_pour --output evidence/keyframe-pour-20261017/main
```

The runner refuses to overwrite an existing receipt. Reproduction requires a
new output directory and the pinned source/environment. Preserve the current
receipts; the source snapshots identify what was actually measured.

The measured advantage is greater reliability with the larger budget, while
simple whole-path correction is more effective under the short cutoff. Keep
both results in the addon opportunity assessment. The commercial case remains
to be established alongside other ways the addon can stand out.
