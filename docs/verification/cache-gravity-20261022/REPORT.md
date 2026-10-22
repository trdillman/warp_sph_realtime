# Gravity-aware bake compatibility, 22 October 2026

Base: main c224a595c45edfea500f72233c6fbe585ba5def2. No subsequent gravity correction was present.

## Change

SimSession.begin_bake uses cache.bake.solver_signature, retaining all prior signature
fields and adding the initialized solver's three effective gravity components as
plain finite Python floats. There is no rounding, string repr, GPU readback or
second scene-unit conversion. Signed zero compares equal through JSON. The same
gravity bound into the solver is checked after restart. Scene gravity edits require
Stop/Start, as other authored-scene changes do.

DiskCache.begin refuses changed gravity or unknown gravity in legacy manifests with
instructions to use Bake Cache for a fresh bake or choose a new folder. Playback
and legacy file decoding stay available; no unknown gravity is invented or migrated.
Fresh bake still clears/replaces the cache. Rejection precedes restore, ring allocation
and frame writes. PR17's GPU capture/restore unit code is untouched.

## Bounded settings audit

- Particle capacity, effective steps/frame, fps, dt, h, stiffness, viscosity, scene
  and demo identity remain in the signature.
- CFL and max_dt are reflected by the effective dt after apply_physics.
- Bake steps override the live substeps in SolverSource.advance; live substeps
  are not an additional bake integration setting.
- Unit scale retains the PR17 convention (absent at 1; explicit otherwise).
- Rest density/mass are not exposed scene controls; current initializers derive
  mass from spacing/h and fixed rest density.
- Domain/fluid geometry, colliders, collider motion and emitter definitions are
  still not fingerprinted. Their existing limitation is now explicit in the
  manual. Start a fresh bake after such edits. Full scene provenance remains a
  follow-up, not a cache rewrite in this patch.
- Graph/eager stepping is tested separately. This is not a promise of cross-device,
  cross-driver or arbitrary long-horizon bitwise reproducibility.

## Verification

Host deterministic regression and cache suites: 62 passed, 6 subtests passed.
These use real disk/job code with fake solver/ring; they are not simulation proof.
Initial red run fails because the signature helper is absent. Raw red/green logs retained.

Actual CUDA: four authored-scene cases (eager/graph x unit scale 1/0.01) pass on
Warp 1.17.0 / CUDA Toolkit 12.9 / RTX 4080 SUPER. Changed gravity rejects and cache
bytes remain intact. Unit-1 unchanged resumes match position/velocity/id bitwise.
At 0.01, ids are exact and positions differ by at most 3.814697265625e-6 scene
units in this fixture; velocities are exact. The existing float32 scene/metre
roundtrip is lossy. Nonunit tests require a 16-float32-epsilon scaled error bound.
The initial two strict nonunit failures are retained, not represented as passes.

Actual background Blender 5.2.0 LTS (fbe6228777e7), Warp 1.17.0, cuda:0:
authored domain/fluid cubes -> scene adapter -> new solver -> real begin_bake.
Cancel after three frames, change Blender Scene Gravity, restart solver, Resume:
actionable rejection and unchanged cache. Restore gravity, restart and Resume:
all eight frame files byte-identical to uninterrupted bake. Legacy manifest:
actionable rejection. Receipt and Blender log retained.

Scope: the runtime probe sets up the session manually and drives the bake job
directly. It does not verify modal operator clicks, viewport transport, UI playback,
installation or release qualification. Its launcher uses the existing showcase
OS lease, other-Blender check, isolated profile paths and factory startup.
Entry point: python scripts/verify_cache_gravity.py --blender <5.2 executable> --out <fresh directory>.

Independent reviewer reran 62 host tests/6 subtests, inspected final product diff,
validated actual Blender receipt against log, and independently compared all eight
uninterrupted/resumed NPZ files. No critical/important issue found. The original
entrypoint coverage gap was closed by actual begin_bake evidence. Reviewer did not
independently rerun CUDA or Blender. Optional timeout-log preservation remains.

Full local suite: 595 passed, 2 skipped, 41 subtests passed; one existing CUDA
STYLISED 0.01 preset failed allocating a 5.64 MiB host array (suite.log). An
isolated retry timed out after 45 seconds. No full-suite-green claim. This
preexisting test exercises physics stepping, not the edited cache signature.
Ruff check . passes. Final Pyright: 0 errors, 0 warnings. Initial Pyright found
two optional-manifest assertions missing in tests; both corrected.

A later attempt to reproduce the unsafe resume on untouched baseline was blocked
before Blender launch: the existing harness tasklist check timed out after ten
seconds. baseline-runner.log is a harness failure, not a baseline physics
reproduction. Actual unsafe baseline cache corruption remains unproven.

## Execution failures retained

Direct local clone initially failed network access; later the tunnel clone succeeded.
Local execution also encountered an invalid sandbox path URI; tunnel tools remained
available. The first focused selection named nonexistent test_cache_gpu.py and ran
no tests; the corrected selection passed. The first probe launcher called the
CSV-parser helper without its argument; corrected to other_blender_running before
any Blender launch. The successful receipt is the first actual Blender invocation.

Final combined focused retry timed out after 45 seconds in the resource-constrained environment; no result claimed. CI remains the clean host integration gate.
