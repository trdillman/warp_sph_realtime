# Daily Bug Scan — 26 November 2026

## Summary

Found and fixed one cross-platform SDK admission bug in commit `e3f110380db9b20f765df33a0f0d0fbee7dea332` (PR #39). Git for Windows had `core.autocrlf=true`; the checked-out pinned SDK therefore had CRLF bytes, while build and runtime admission compared raw SHA-256 values with LF-based locks. On this Windows checkout the exact pinned SDK was rejected, preventing private SDK builds and native snapshot admission.

The build now hashes canonical LF text and stages the SDK bundle with `.py` and `.json` files normalized to LF. Both runtime admission routes use the same canonical hash helper. Added a regression that simulates a CRLF checkout and confirms the staged payload matches the lock.

## Findings by severity

### P1 — Windows line-ending conversion rejects the pinned SDK — FIXED

- Evidence: system Git config reported `core.autocrlf=true`; submodule HEAD was the expected `1b3e4be77e21c6f20437f782207cccf3895dda2a`, but `check_sdk()` rejected its CRLF worktree bytes against `interop_sdk_lock.json`. Snapshot admission likewise rejected `__init__.py`.
- Root cause: the build validator, SDK dependency validator, and snapshot validator independently hashed raw file bytes; the build also copied those converted bytes into the extension.
- Fix: canonical LF hashing in `scripts/build_extension.py` and `warp_sph_realtime/sdk_bundle.py`; normalize staged SDK `.py`/`.json` payloads; route both runtime validators through the shared hash helper.
- Regression: `tests/test_sdk_bundle.py::test_windows_crlf_submodule_is_normalized_to_locked_lf_bytes`.

## Other likely bugs / architecture issues

No additional evidence-backed issue was found in the reviewed SDK build/admission routes or the directly adjacent cache-frame and live-output paths. The current private SDK submodule is pinned and now initialized at the expected commit. No Blender GUI or CUDA qualification was run; this fix addresses byte identity and package admission only.

## Review scope and evidence

- Recent window: last 72 hours (automation last-run marker was `never`); one commit in the window, `e3f1103`.
- Reviewed the latest commit's SDK bundling/build and both runtime consumers, plus adjacent cache frame isolation (`e1e35df`), live-output pacing (`58350c8`), and prior managed-Warp replacement (`7dba10b`). The chronological addon path inventory reaches all 68 tracked add-on files through 25 commits; older unrelated module internals were not re-reviewed line-by-line in this run.
- Focused SDK/bundle/snapshot suite: **26 passed**.
- All SDK-prefixed tests: **54 passed**.
- Full repository suite: **748 passed, 67 skipped, 72 subtests passed**.
- `ruff check .`: passed.
- `pyright`: 0 errors, 0 warnings, 0 informations.
- `git diff --check`: passed before report creation.

## Recommendations by ROI

1. Keep line-ending normalization at both the build boundary and runtime lock checks; the new regression protects the Windows checkout path.
2. Keep running the full private SDK test set on Windows with the pinned submodule populated; the public CI intentionally excludes tests marked `private_sdk`.
3. Treat Blender GUI/CUDA acceptance as a separate qualification; it was not needed to verify this file-identity fix and was not run here.
