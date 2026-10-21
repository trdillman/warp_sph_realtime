# Multi-keyframe emitter experiment

Authorized 2026-10-17. Intended outcome: determine whether automatic gradients help an artist achieve a receiving target while respecting a spill constraint, compared with simpler search at matched simulation-work and elapsed-time budgets.

## Product context

The primary goal is a profitable Blender add-on. This is one bounded investigation within a wider search for useful differentiation, not a commitment to build the product around differentiability. Numerical success is not evidence of demand, willingness to pay, retention, or profitability. Preserve a simple optimizer if it gives comparable results with lower implementation/support cost.

Other opportunities remain hypotheses: faster setup of common liquid shots; dependable preview-to-bake/render workflow; predictable revisions and reusable presets; goal-directed control. This experiment tests only the last. No competitor or market superiority is claimed without research and customer evidence.

Decision rule before seeing the new results: recommend further gradient-specific investment only if it gives more constraint-satisfying outputs under a common budget, or reaches the same outcomes materially sooner than the strongest simple baseline. A lower smooth loss alone is insufficient. A small eight-start experiment can justify a follow-up, not a general superiority claim. If simple search is competitive, keep goal-directed control as a possible workflow feature and deprioritize differentiation-specific engineering. Regardless of outcome, installation, reliable authoring, bake/render delivery and customer demand remain separate commercial gates.

## Critical path and acceptance

1. Define a physically honest emitter animation with multiple keyframes, a reachable target, and an explicit spill constraint. Reuse existing SPH kernels; do not steer released fluid as if it were emitter motion.
2. Prove feasible reference controls independently; do not seed optimizers with that answer. Validate automatic derivatives, finite states and replay before trusting results.
3. Freeze bounds, starts, scoring/selection, budgets, synchronization and timing exclusions before the measured comparison. Select outputs using actual constraints, not only a surrogate loss.
4. Run eight paired starts with gradients, coordinate search and a credible population search. Retain full candidate histories, forward/backward counts and completion times; compare the best eligible output under each budget. Preserve failed attempts and source identities.
5. Report target delivery, spill, success frequency, time/work to success and visual trajectories. Distinguish bounded headless evidence from Blender UI and commercial acceptance.
6. Independently review evidence, run regression/lint/type checks, and update the queued tracker entry.

## Ownership and state

Base HEAD: `617b4eabfb24c89618122523eff3f057d3e3f58a`. Existing dirty core and untracked prior POC files are preserved. No new worktree, commit, publication or installation requested.

Experiment lane: `keyframe_experiment`, deep_worker, declared `gpt-6.1-sol/high`; owns new experiment, runner, tests, contract, results and dated evidence. Sole GPU owner during implementation/measurement. Parent owns integration, this record, product context, tracker queue and final checks. No overlapping runtime tests.

Status: complete and independently reviewed. Frozen contract: `KEYFRAME_EXPERIMENT_CONTRACT.md`; results: `KEYFRAME_EXPERIMENT_RESULTS.md`; raw evidence: `evidence/keyframe-pour-20261017/`.

Pilot findings before measurement: same-place planar bursts caused nozzle interactions and failed the witness. A downward-nozzle fallback reduced that problem but fluid accumulated/spread on the uncontained floor, so final retained coverage remained infeasible. Both failed pilots are retained. Evidence-backed fixture refinement: first downward passage through an aperture at z=0.35 with an adjacent protected slab. Emission remains four finite bursts, no post-release steering, no imaginary basin/capture. Later spill still counts. This tests directed delivery, not retained container fill; the comparison must say so. Parent CPU-only receipt controls passed 5 tests (deadline eligibility, backward accounting, constraint-first selection, paired starts and completeness); these are not GPU qualification.

Additional pre-measurement diagnosis: aperture seed 0 began intersecting the central sphere. Finite-difference epsilon convergence supported local gradient sensitivity rather than an event-mask implementation bug; read-only review confirmed coherent per-step masks and persistent particle IDs. Translating emitter, target and protected slab left removed the accidental overlap. Clear-space pilot: witness 100% coverage/0% spill, both eight-component gradient vectors passed. Exploratory 32-unit seed-0 results favored gradients but are excluded from the main comparison. Freeze the main run at 64 nominal forward/reverse work units, 10-second completed-output cutoff, eight paired starts, Adam/coordinate/CEM, with initial and selected trajectory evidence. Parent receipt tests now 6 passed, Ruff passed; verifier is ready for snapshot. Both work-best and time-best outputs require independent event metrics and replay validation.

Pre-measurement fairness amendment: add grouped-coordinate search (shared x/y translation of all four keyframes), making four methods and 32 trials. Starts mainly share a translation error; changing each key separately can unfairly handicap the simple baseline and introduce unwanted inherited velocities. Shared moves must preserve keyframe differences within bounds. The independent reviewer agreed this is necessary before interpreting gradient value. No main measurement had started at the amendment; final frozen contract/snapshots govern the run. This start family remains narrower than general eight-control shaping.

Measured comparison started with frozen source/contract/verifier/tests under `evidence/keyframe-pour-20261017/main/source/`; raw run log `evidence/keyframe-pour-comparison.log`. Four-method focused tests: 14 passed; Ruff and pinned Pyright 1.1.408 passed. GPU owned solely by experiment lane. Parent final full regression and read-only evidence review remain pending.

Frozen-code review found no material defect in staged emission, IDs, event differentiation, search algorithms or budget handling. Before final acceptance, strengthen/audit the verifier for exactly five prescribed derivative-vector identities and witness replay, in addition to existing gates; retain the frozen verifier snapshot and identify the stricter final audit separately. No numerical experiment changes are required. Known geometry caveat: a small fringe of the protected slab overlaps the side of existing pillars; visualize and disclose it. No wholly unobstructed protected-volume claim.

## Verified outcome

All 32 frozen trials completed without a retry. At 64 nominal work units: gradients 8/8 successes, independent coordinate 3/8, cross-entropy 4/8, whole-path translation 5/8. By 10 seconds: 4/8, 0/8, 2/8 and 5/8 respectively. No initial animation met both constraints. Gradients improved observed reliability at the larger budget; whole-path correction was better under the short deadline. The union of simple-method successes covers every start, but an adaptive schedule at a common total budget was not tested. No claim that gradients uniquely enable these outcomes or establish profitability.

All five prescribed eight-component gradient checks passed. All 65 witness/work-best/time-best validations had independently matching hard metrics, finite/bounded states, zero observed penetration/hard clamps, and exact shared-kernel eager/graph replay. Independent read-only review passed artifact correctness and bounded intent coverage. Main comparison SHA-256: `48bf49a71f6a8e043ca11af83c97e11f62f12efbff6bdf5e88f69174f726b50e`.

Final checks: 87 passed, 2 skipped, 8 subtests passed; Ruff, pinned Pyright and `git diff --check` passed. The stricter receipt verifier passed and records its own hash in `main/verification.json`. Post-freeze verifier/negative-test strengthening is explicit; numerical sources and frozen contract still match measured snapshots. Actual trajectory animation and outcome plots are under `main/`. No Blender integration, package install, commit, publication or commercial acceptance is implied.

Tracker access: no Page read/update tool is exposed; `docs/TRACKER_UPDATE_PENDING.md` remains the delivery queue. Authorized experiment is complete. Next product decision: weigh a goal-directed workflow with inexpensive correction and optional gradient refinement against core authoring/bake/render improvements and customer demand; any new experiment or implementation should receive its own bounded contract.
