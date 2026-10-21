# Physics validation (P2.6)

What is measured, against what, and how far to trust it. Code: `warp_sph_realtime/sim/validation.py`
(headless, no bpy). Tests: `tests/test_validation.py`. Sweep with plot and JSON:
`python -m scripts.validate_dambreak` (writes `evidence/physics-validation-<date>/`, gitignored).

All numbers below are from the real `RealtimeWCSPHSolver` at its default physics (stiffness 90000, c = 25.1 m/s,
viscosity 0.15, artificial-viscosity alpha 0.1, CFL 0.4, 2 substeps), Warp 1.17.0, RTX 4080 SUPER (`cuda:0`)
unless marked CPU. Nothing was tuned to pass.

## 1. Dam-break front vs Martin & Moyce (1952)

**Reference.** J. C. Martin and W. J. Moyce, "Part IV. An experimental study of the collapse of liquid columns on a
rigid horizontal plane", Phil. Trans. R. Soc. Lond. A 244 (1952) 312-324, doi:10.1098/rsta.1952.0006. Water
column of width `a` and height `n^2 a` with `n^2 = 2`; surge front position `Z = x/a` (from the column's back wall,
so `Z = 1` at release) against `T = t sqrt(2g/a)`.

**Provenance of the numbers (read this).** The paper is behind a publisher paywall and could not be opened from
the development environment, so the values were *not* read from the primary source. The 15 (T, Z) pairs in
`MARTIN_MOYCE_T/Z` are the ones the open-source Ocellaris CFD code embeds in `scripts/plot_dam_break.py` under
the comment "Data from Martin and Moyce (1952), Table 2 and 6". They were cross-checked against two independent
digitisations of the paper's Figure 3 for the same `n^2 = 2` case (PySPH, `pysph/examples/db_exp_data.py`, columns
`a = 1.125 in` and `a = 2.25 in`): interpolated onto the 15 times, the digitisations differ from the pairs by
`|dZ| <= 0.10` (about 3 %), consistent with reading a plot. A third table (TrixiParticles' `exp_surge_front.csv`)
has the same Z values but a time column that does not match any simple rescaling, so it was not used. Anyone with
the paper should confirm the table before this is quoted externally.

**Setup.** A quasi-2D slab: `a = 0.25 m`, `n^2 = 2` (height 0.5 m), `n` particles across `a` (spacing `a/n`,
`h = 2 spacing`), 4 particles across the slab (y), tank 5a long, free-slip walls (the solver has no wall friction).
The lattice is released at `t = 0` with the right wall removed (the tank is simply long). The domain walls sit half
a spacing outside the fluid, so outermost particle centres start one spacing (the width of the solver's soft wall
layer) from a wall; see the limitations for why.

**Front measure.** `x` of the leading edge along the floor: mean of the 4 foremost particle centres among the two
lowest lattice layers, plus half a spacing. Z is that distance over `a`; Z(0) = 1.0 exactly. The simulated Z(T)
is interpolated linearly at the 15 reference times. Error = Z_sim - Z_ref (positive: ahead of the experiment).

**Measured (GPU, thickness 4).**

| cells across a | particles | mean error | mean abs | RMS | max abs | max abs / Z | error at T = 0.41 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6  | 288    | -0.270 | 0.270 | 0.316 | 0.530 | 13.6 % | -0.084 |
| 8  | 512    | -0.190 | 0.190 | 0.227 | 0.390 | 10.0 % | -0.077 |
| 12 | 1152   | -0.101 | 0.109 | 0.131 | 0.246 | 6.3 %  | -0.068 |
| 16 | 2048   | -0.055 | 0.072 | 0.086 | 0.175 | 5.8 %  | -0.064 |
| 24 | 4608   | +0.011 | 0.046 | 0.053 | 0.091 | 6.3 %  | -0.056 |
| 32 | 8192   | +0.054 | 0.063 | 0.072 | 0.115 | 7.9 %  | -0.054 |
| 48 | 18432  | +0.103 | 0.110 | 0.115 | 0.166 | 9.0 %  | -0.053 |

What this says, without smoothing it:

- The front is within about 6-10 % of the experiment (relative to Z) at every resolution of 8 cells or more, and the
  front-to-front RMS is 0.05-0.13 Z at 12-48 cells.
- It is **not a converging approach to the experiment.** Coarse runs lag (-0.27 at 6 cells); the mean error
  crosses zero near 24 cells and the finest runs are 0.10 ahead of the experiment (+0.10 at 48 cells). The small
  error at 24 cells is where a lagging coarse bias and a leading fine bias cross, not evidence of a converged
  solution.
- **The early point lags at every resolution.** At T = 0.41 the simulated front has moved about 0.05 a against the
  experiment's 0.11 a, an error of -0.05 to -0.08 in Z at all resolutions, i.e. over half of the early travel.
  Possible causes (untested): the front is taken in the two lowest lattice layers next to the soft floor layer;
  the experiment's `T = 0` depends on how fast the gate was lifted; the solver starts from a lattice, not a
  settled column.
- The sawtooth shared by every curve in the error plot (`dambreak.png`) is in the reference points, not the solver.
- Slab thickness matters a little: 8 particles instead of 4 across the slab moves the mean error from -0.101 to
  -0.068 at 12 cells and from +0.011 to +0.060 at 24 cells (free-slip side walls make the slab a thin 3D case).
- The CPU device gives the same front as the GPU: max |Z_cpu - Z_gpu| = 4e-5 at 12 cells (the test allows 0.02
  for cross-hardware float differences).

## 2. Particle conservation and energy

Checked on every run, dam break and rest. A run counts as conserved when the particle count is unchanged, nothing
is NaN or infinite, and no particle ends more than 1e-6 m outside the domain.

| | result |
|---|---|
| particle count | unchanged in all runs |
| NaN / Inf | none |
| distance outside the domain | 0.0 m in all runs (final state; the solver also clamps hard) |
| energy growth (max of KE+PE over the run, relative to the initial value) | 0.0 % in all runs: the initial energy is the maximum |
| dam break, KE+PE at T = 3.33 relative to start | 0.74 (6 cells) rising to 0.88 (48 cells); the loss goes to viscous and artificial-viscosity dissipation, the wall layer and stored compression (not separated here) |
| settling block, KE+PE at 1.0 s relative to start | 0.92; kinetic energy at the end 5e-5 of the initial energy |

PE is `m g z` from the floor plane and KE is `m v^2 / 2` summed over particles. It excludes the elastic energy of the
equation of state and the soft wall layer's potential, so "does not grow" is the claim; exact energy conservation is
not, and a decreasing total is expected, not evidence of accuracy.

## 3. Rest density

A 16 x 16 x 16 block (4096 particles, spacing 0.025 m) is released from the lattice into a closed tank that fits
it, open at the top, and run for 1.0 s. Density is recomputed from the particle positions with an unclamped SPH
sum (the solver's own density is clamped to >= rest density, which would hide any deficiency).

| quantity | measured |
|---|---:|
| mean abs(rho/rho0 - 1), all particles | 0.28 % |
| max abs(rho/rho0 - 1), all particles | 15 % (free-surface or wall-contact particle; kernel deficiency) |
| mean abs(rho/rho0 - 1), bulk (at least `h` from every wall and the surface; 2062 particles) | 0.23 % |
| max abs(rho/rho0 - 1), bulk | 0.52 % |
| mean rho/rho0, all particles | 1.002 |

The smaller CI variant (10 x 10 x 10 = 1000 particles, 0.5 s, CPU device) measures 0.82 % for all particles and
0.19 % for the bulk (342 particles; max 0.35 %). For scale: the hydrostatic compression at the bottom of a 0.4 m
column is about 0.6 % in density (about 0.3 % averaged over its height), so the bulk figure is of the size the
physics asks for rather than merely small.

Test bounds: bulk mean < 1 %, all-particle mean < 2 %.

## 4. What runs where

| | CI (no GPU) | GPU machine |
|---|---|---|
| reference data well-formed, error statistics, lattice helper | yes | yes |
| solver on the Warp CPU device (`device="cpu"`: eager launches, no CUDA graph) | yes | yes |
| dam break at 12 cells (1152 particles): front vs reference, conservation, energy | yes (CPU) | yes (CPU and GPU) |
| rest block 10 cells (1000 particles), 0.5 s: density, energy, kinetic decay | yes (CPU) | yes (also CPU) |
| dam break at 24 cells (4608 particles) with the CUDA graph | skipped | yes |
| rest block 16 cells (4096 particles), 1.0 s | skipped | yes |
| CPU and GPU fronts agree | skipped | yes |
| sweep, plot and JSON (`scripts/validate_dambreak.py`) | not run | manual |

The CPU tests take about 17 s with a cold Warp kernel cache on a development workstation (most of it kernel
compilation). Tolerances are regression guards set above the numbers in this file (for example 10 % of Z at the
12-cell CPU resolution against 6.3 % measured); they are not accuracy claims beyond the measured table.

## 5. Limitations

- The reference table was not checked against the 1952 paper (see Provenance). It is experimental data of one
  column of water at 2 scales; the solver is compared in non-dimensional form at one scale, `a = 0.25 m`. WCSPH with
  a fixed sound speed (25 m/s) is not scale free (Mach number scales with `sqrt(g a)`), so other column sizes
  are untested.
- Quasi-2D slab with free-slip walls and no wall friction; the experiment is a real channel with a wetted floor,
  surface tension and a gate that takes time to clear. The model has no surface tension.
- Only the surge front of the `n^2 = 2` case is compared. Column height over time, pressure, and other aspect
  ratios are not.
- Front deviation depends on resolution and slab thickness and does not converge monotonically to the data (Section 1).
  Differences of a few percent of Z should not be read as better or worse physics.
- The front is a particle-derived edge and the comparison interpolates linearly between 15 sparse points.
- Energy and rest-density checks use a blocky closed-tank case. They do not cover colliders, authored scenes or the
  experimental `enable_obstacles` modes.
- **Fixed after this benchmark was written: the initial fill started inside the wall layer.** A lattice whose
  particle centres start half a spacing from a wall (the old cell-centred `fill_particles` convention when fluid
  touches the domain boundary) is inside the solver's soft wall layer and was expelled violently. In a closed-tank
  probe (12 x 6 x 24 block, spacing 0.021 m) the largest velocity component reached 46 m/s (above the 25 m/s sound
  speed) and the mean particle height ended 24 % above its start (0.25 m to 0.31 m). `fill_particles` now keeps
  every particle at least one spacing from the domain walls and collider surfaces, which is where `wall_accel`'s
  force reaches zero (its layer is `r0 = h/2 = spacing` wide). Re-measured through the real fill path
  (`tests/test_fill_clearance.py`: 8 x 8 x 7 fluid in a 0.5 x 0.5 x 1 m tank, spacing 0.05 m, 100 steps, identical
  on CPU and GPU): peak speed 57.8 m/s and mean height +178 % with the old half-spacing clearance, 2.0 m/s and
  -35 % with the new one. The remaining drop is the fluid spreading into the half spacing of slack that the
  cell-centred lattice leaves (it starts 1.5 spacings from a lower wall; contact is at 1) and then settling; it is
  not a blast. The dam-break, energy and rest-density numbers above are **unchanged by this fix** (re-run: all
  seven fronts and the rest block reproduce to the printed digits), because the benchmark places its own lattice
  one spacing from the walls and never calls `fill_particles`. The two built-in demos place particles with their
  own kernels; they start 0.15 m (Demo 1) and 0.1 m (Demo 2) from the side walls and 0.08 m / 0.05 m above the
  floor, which is already at least one spacing at 1M particles, and now enforce one spacing at coarser settings.
- A lattice start is not a settled hydrostatic state: the run begins with acoustic transients while the lowest
  layers compress into the soft floor layer. The settling run shows them as a brief kinetic-energy bump.
