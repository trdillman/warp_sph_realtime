# Showcase verification repair implementation plan

**Goal:** A showcase success must establish its claimed feature; one failed job must preserve the rest of the batch evidence.

**Architecture:** Keep the existing five scenes and their public operator paths. Add strict route checks, a paired stationary/animated paddle experiment and bounded runner fault handling. Serialize runners across checkouts with an OS lease; use a fresh output-owned extension profile.

**Stack:** Python 3.13 / Blender 5.2 LTS+, Warp 1.17.0, CUDA 12.9, cuda:0. Host tests do not qualify GPU or GUI behavior.

**Spec:** Owner request to repair all six PR #17 review findings, preserve assets, obtain independent review, and integrate through GitHub. Baseline main e55dc1dec3de6d4ab92e011fa6f105a281c12536.

## Tasks

- [x] Reproduce corrupt receipt, unhandled launch, unbounded encoding and misleading aggregate failures in focused tests; inspect all six source paths.
- [x] Require the exact backend/mode accelerated transport and separately classify labelled Readback. Test wrong-label, missing-label, and capability mismatch controls.
- [x] Move scene builds inside the protected timer driver for all five demos; retain diagnostics and quit automated Blender after failure.
- [x] Compare stationary and animated paddle caches by stable IDs in the near-paddle pool, after verifying identical initial conditions. Require signed tangential velocity influence at two of three early frames. Reject gravity-only, reordered arrays and reverse-direction synthetic controls.
- [x] Bound job/encoder/media processes, retain individual errors and incrementally publish aggregate JSON, including failed setup and media stages.
- [x] Acquire a cross-checkout lease before mutations; check active Blender before profile installation; use fresh profiles and preserve prior output. Test a real competing process and lease release.
- [x] Run host tests, Ruff and Pyright; retain commands and failures. Full GPU-free Windows/Linux CI passed on initial candidate 30b6deb; PR #18 records final-candidate CI. Final focused controls: 91 passed, 22 subtests.
- [x] Attempt affected isolated Blender 5.2 showcases under the execution guard. BLOCKED by unrelated Blender sessions; all three planned rows retained and no profile installed. Real routes/paired bakes/media remain unverified.
- [x] Obtain independent review, address findings and open PR #18. GitHub review follow-ups reproduced/fixed: frame-rate overflow, paired-bake budget, repeated setup wait and modal teardown. Independent reviewer reran all 40 runner/failure tests successfully.

Integration status is maintained on [PR #18](https://github.com/trdillman/warp_sph_realtime/pull/18) and the linked project tracker. Source/host-test integration does not close blocked GPU acceptance.

## Review focus

Malformed JSON shapes, stale evidence reuse, process-detection failure, setup write failures, and media failures must never become success. Readback cannot qualify acceleration. Paired collider thresholds are fixed before GPU execution and must not be tuned to obtain a pass. Original six showcase jobs remain historical under the old checks.
