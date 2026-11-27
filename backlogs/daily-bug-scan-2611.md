# Daily Bug Scan — 26 November 2026

## Summary

The scan covered the only upstream commit in the 72-hour window (`e3f1103`) and continued backward through the add-on file history until all 75 tracked files under `warp_sph_realtime/` had review coverage. Six concrete bug groups were fixed with focused regressions. One likely OpenGL cleanup leak remains documented for a context-aware fix. The chronological file ledger and working notes are in the git-ignored `.agent/daily-bug-scan-2611-live.md`.

The scan's first fix, committed as `ddcb15b` during this run, made the SDK build and runtime lock checks resilient to Git for Windows CRLF conversion. This follow-up found and fixed a related raw-hash path for the packaged symbol profile.

## Findings by severity

### P1 — CRLF checkouts reject locked add-on text payloads — FIXED

- Evidence: with `core.autocrlf=true`, text files checked out as CRLF no longer matched their LF-based lock digests. The failure affected SDK build/runtime admission and the embedded profile digest at `warp_sph_realtime/render/blender_symbols.py:119`.
- Fix: canonical LF hashing and LF staging for locked `.py`/`.json` payloads in the build and SDK runtime checks; the profile validator now applies the same canonicalization.
- Regression evidence: SDK CRLF payload test plus the profile CRLF regression passed.

### P1 — SSFR own-liquid pass writes zero alpha into additive-premultiplied output — FIXED

- Evidence: `warp_sph_realtime/render/gpu_ssfr.py:195` wrote `vec4(0.98 * own, 0.0)` while the pass uses `ADDITIVE_PREMULT`. Blender documents this blend mode as adding source color premultiplied by source alpha, so alpha zero suppressed the pass contribution ([Blender GPU state API](https://docs.blender.org/api/5.4/gpu.state.html)).
- Fix: emit alpha `1.0`; added a shader regression that rejects the transparent output.
- Runtime limit: no live Blender viewport/GPU render was run in this scan.

### P2 — Emitter-only empty frames cannot be recorded or restored — FIXED

- Evidence: a scene with emitters and no particles yet has a valid zero-particle state, but `VramRing.submit` rejected count zero and `RealtimeWCSPHSolver.restore_state` rejected an empty frame. See `warp_sph_realtime/cache/gpu.py:80` and `warp_sph_realtime/sim/sim_core.py:993`.
- Fix: allow zero-count ring submissions without issuing a copy, and allow empty restore only for emitter-backed solvers. Empty identity restoration sets the next ID to zero.
- Regression evidence: focused ring and solver restore tests passed.

### P2 — Native frame payloads silently coerce invalid channel types — FIXED

- Evidence: `warp_sph_realtime/cache/native_sequence.py:102-111` accepted duplicate channel names and decoded any non-`uint64` identity type or non-`float32` vector type through the fallback dtype. This could silently reinterpret malformed frame bytes.
- Fix: require the exact unique channel set and the declared scalar type for each channel before decoding.
- Regression evidence: malformed scalar type and duplicate-name payload cases now raise validation errors.

### P2 — Malformed cached ID arrays escape cache validation — FIXED

- Evidence: `warp_sph_realtime/cache/disk.py:236` previously called `len(ids)` before checking dimensionality. Scalar IDs raised an uncaught `TypeError`, while a two-dimensional ID array could pass the length check.
- Fix: require a one-dimensional ID vector before checking its length and raise `CacheError` for malformed frames.
- Regression evidence: scalar and two-dimensional ID cases both produce the documented cache error.

### P2 — Reused meshing object keeps a stale point-cloud input — FIXED

- Evidence: `warp_sph_realtime/cache/blender_output.py:147-166` returned an existing Geometry Nodes modifier unchanged when its node group was already current, even when the caller supplied a different point-cloud object.
- Fix: update only the modifier's `Point Cloud` input when reusing the group, preserving other artist-edited inputs.
- Regression evidence: a fake-Blender binding test confirms the object reference changes and other modifier inputs remain untouched.

### P2 — VAO handles from other GL contexts lose their cleanup path — DEFERRED, LIKELY

- Evidence: `warp_sph_realtime/render/interop_bridge.py:618-629` deletes the VAO for the current context, then clears the entire per-context VAO dictionary. The same file documents that VAOs belong to their creating context (`:507`) and can only be deleted from that context (`:625`). If another cached context remains alive, its VAO handle is discarded without deletion.
- Recommendation: retain foreign-context handles and queue deletion for the next time each owning context is current, or add an owning-context teardown callback. This needs a live two-context Blender/OpenGL lifecycle check before changing cleanup behavior.
- Confidence limit: confirmed from the cleanup control flow and ownership comments; not reproduced in a live Blender session during this scan.

## Dismissed suspicions

- The solver's `2^31` next-ID guard is not an off-by-one: the counter is the exclusive next ID, so the largest assigned ID remains `2^31 - 1`.
- The pour emitter's `key * 16` prior-count argument matches its sixteen-particle-per-key schedule. No change was justified.

## Review scope and verification

- Automation last-run marker: `never`; scan window: last 72 hours. The upstream base had one commit in-window, `e3f1103`.
- Per the add-on-wide scope, review continued in descending commit-touch order through the tracked package history until all 75 tracked `warp_sph_realtime/` files were covered (68 Python files plus 7 non-Python files). The seven package files absent from commit path history were reviewed separately. No add-on package path remains unreviewed.
- Focused regressions: **7 passed, 2 subtests passed**.
- Full repository suite after all code changes: **755 passed, 67 skipped, 74 subtests passed**.
- `ruff check .`: passed. `pyright`: 0 errors, 0 warnings, 0 informations. `git diff --check`: passed.
- No Blender GUI or CUDA runtime qualification was run. Source/test evidence does not establish live rendering, GL context teardown, or GPU acceptance.

## Recommendations by ROI

1. Add a bounded live Blender 5.2/CUDA qualification for the SSFR contribution and empty-emitter bake/restore path; it would close the largest remaining runtime evidence gap.
2. Add a two-context GL teardown regression and implement deferred deletion for VAOs that are not owned by the current context.
3. Keep CRLF checkout fixtures in the build and runtime lock suites so text lock boundaries stay consistent on Windows.
