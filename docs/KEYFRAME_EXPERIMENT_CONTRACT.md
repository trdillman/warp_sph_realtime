# Frozen multi-keyframe emitter experiment

This is one bounded investigation within the profitable-addon opportunity search.
The decision is whether automatic gradients produce more useful artist outcomes
than credible simple search at the same simulation-work and elapsed-time limits.
It is neither a Blender UI feature nor evidence of willingness to pay.

## Pre-measurement fixture and acceptance

The unchanged current WCSPH density/force/reorder kernels run on CUDA:0 with
Warp1.17.0, float32, h0.12m, spacing0.06m, mass from lattice_rest_mass,
rest density1000, stiffness90000, viscosity0.15, CFL0.2, compliant contact mode2.
The domain is [-2.5,-2.5,0]..[2.5,2.5,5].

Four sixteen-particle planar4x4 bursts are emitted at steps0,96,192,288.
Only emitted particles exist in the grids: counts16,32,48,64. Each particle
retains its ID across sorting, emission and all trajectory observations. Eight
controls are x/y offsets of four emitter-position keyframes, each in [-0.30,0.30]m.
Burst center is (-1.45+offset_x,offset_y,0.50). Particles inherit the one-sided
endpoint or centered interior secant velocity of the keyframe positions, plus
nozzle vertical velocity -2m/s. A cubic Hermite emitter path through the keys
with those tangents defines the between-key motion; there is no between-key
emission. Released fluid receives no further control force.

The timestep is0.2*0.12/sqrt(630), approximately0.000956183s.640steps give
approximately0.611957s total, with burst final ages640,544,448,352 timesteps.
This is a finite burst emitter, not continuous fluid emission.

Target: first downward crossing of the horizontal aperture at z0.35, with
x in[-1.67,-1.23],y in[-0.22,0.22]. Crossing x/y uses the exact linear segment
interpolation between retained SPH positions and differentiates through its
crossing-time fraction. A per-step immutable crossing mask fixes the first
event identity during reverse replay. No arrival receives an explicit final
distance/height penalty. The target is an aperture, not an absorbing basin;
fluid continues to evolve after crossing and may spread on the floor.

Spill: any line-segment intersection with protected box
[-1.22,-0.40,0.30]..[-0.68,0.40,0.40] at any time, including after delivery.
Count unique persistent IDs. GPU slab intersections and an independent NumPy
recomputation inspect every retained step; neither endpoint sampling alone nor
final spill alone is accepted. The hard artist success criterion is at least90%
target and at most5% spill, hence at least58/64 delivered and at most3/64 spilled.

Smooth objective: mean squared x/y distance outside target plus0.02times
squared horizontal center distance at the first downward crossing, final
distance/height penalty for nonarrival, and time-integrated squared penetration
into a smooth margin around the protected box (weight200/(64*640)). The margin
has x halfwidth0.30,y0.45,z0.09 centered(-0.95,0,0.35). This is an optimization
surrogate; hard success and candidate selection use the independent exact metrics.

Common candidate selection is lexicographic: feasible before infeasible, then
sum of target shortfall below0.90 and spill excess above0.05, then smooth loss.
Every evaluated candidate is eligible; the witness is never provided to searches.
Zero offsets witness must achieve success, finite/bounded all states, no hard
floor clamps, no solid penetration greater than1e-5m, and matching graph replay.
The entire permitted emission-position box clears all solids/contact bands:
nearest pillar surface distance approximately0.084m >0.06m. Selected entire
trajectories are independently checked for solids, boundaries and finite velocities.

## Frozen comparison

Eight paired starts use numpy.default_rng(seed+163), seed0..7, translation x
uniform[0.14,0.22],y uniform[-0.09,0.09], then independent0.018m Gaussian
keyframe perturbations, clipped[-0.28,0.28]. No starts are selected after results.

Algorithms: Adam (step0.025,beta1=0.9,beta2=0.999,epsilon1e-8); coordinate
search (step0.06,all16 plus/minus coordinate probes per sweep, halve step if
no constraint-rank improvement); cross-entropy search (population12,elite4,
initial sigma0.08,0.7new/0.3old updates,minimum sigma0.012,seed+719); grouped
coordinate search (step0.06,four plus/minus common x/y translations of all four
keys per sweep,halve step on no constraint-rank improvement). Grouped translation
is the obvious artist operation for these mostly-shared-offset starting errors;
omitting it would unfairly handicap simpler methods. It cannot correct keyframe
shape errors independently, and its result limits any claim about needing eight
independent controls for this particular fixture.
All methods evaluate the identical paired start first and use the same bounds.

Each trial has64work units. One full staged forward=1; one full reverse=1,
so Adam has32forward+32reverse and others64forward. This is explicit nominal
simulation-work accounting, not a claim equal forward/reverse execution time.
Separate actual elapsed-time comparison uses only candidates whose complete
evaluation, including reverse pass and CPU metric readback when requested,
finishes within10seconds. The long fixed-budget trace must extend past10s;
otherwise time comparison is censored and must be labelled accordingly.

Wall timing starts before each algorithm's first evaluation; includes control
uploads, launches, grid builds, gradient work, host readback, synchronize and
optimizer Python overhead. Setup/JIT/allocations and one forward+reverse warmup
are excluded and reported separately. Replay, diagnostics and FD qualification
are outside search budget/time. GPU synchronizes before and after evaluation.
Runs serialize one persistent warmed process; the four-method order rotates by
seed (Adam/coordinate/CEM/grouped,coordinate/CEM/grouped/Adam,and so on).
This departs from
fresh-process performance qualification to bound desktop experimentation and
represent repeated optimizations on an allocated rollout; it is exploratory
timing, not certified performance or a statistical speedup claim.

Eight identical forward A/A pairs precede comparisons. If any ratio is outside
[0.8,1.2], timing is explicitly noise-limited; no samples are removed or retried.
Retain all raw samples, curves, work counts, completion timestamps and failures.
Report per-method paired successes, target/spill fractions, first-success work
and time, full search timing and the strongest simple baseline at both budgets.
No false precision or pooling correlated iterations as independent trials.

Full8-axis centered FD checks (h0.0005,rtol0.08,atol0.0005) run at seeded start0
and a near-witness vector, then selected Adam outputs for seeds0,4,7. Failure
blocks validated status. Both budget-best and time-best selected outputs get
full-trajectory CPU hard-metric/safety checks and eager-vs-shared-kernel CUDA
Graph replay (atol/rtol1e-5). Per-step grids stay outside capture. This verifies
the same kernels; production step() does not yet implement staged emission.

## Preserved pre-measurement failures

1. `pilot/pilot.json`: stationary zero-velocity source; accumulated interacting
bursts produced3.125% final target,12.5%spill and nozzle jets. Unreachable witness.
2. `pilot-downward/pilot.json`: fixed downward nozzle cleared the source but
floor spreading still gave21.875% final target,28.125%spill. No basin retained it.
3. `pilot-aperture/pilot.json`: aperture witness100%/0%, but seeded source
intersected central sphere; coarse FD failed. `contact-epsilon.json` shows first
axis FD converging to AD as step shrinks; no gradient defect was inferred.
4. `pilot-clearance/pilot.json`: translated emitter/aperture/protected slab left,
qualifying reachability and both8-axis derivatives.32unit pilot costs informed
64unit/10s bounded main budgets. Pilot outcomes are excluded from main evidence.

No physics kernel was changed to make an optimizer look better. No post-main
algorithm tuning, input filtering or hidden witness initialization is permitted.
