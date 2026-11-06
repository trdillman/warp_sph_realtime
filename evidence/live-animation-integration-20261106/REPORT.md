# Live Geometry Nodes integration candidate — 6 November 2026

Prepared locally on `codex/live-animation-integration-20261106` in
`C:/Users/Tyler/.codex/worktrees/live-animation-integration/warp_sph_realtime`.
This is a reviewable integration candidate, not a main merge, publication,
production installation, or new runtime acceptance.

## Source identity and divergence

- Fetched origin before comparison. Local `main` and `origin/main` both resolve
  to `52a383aa45ba14acfba0e5c2c7c86a91f1498181`.
- Verified checkpoint: `3445fab5316d55cabd3459c2fc058cb320945ee9`, branch
  `interop/live-animation-verified-20261106`.
- Main/checkpoint unique commit counts: **0 / 8**. Main is an ancestor; the
  isolated candidate advanced through those eight commits without conflicts.
- Original checkout remains on `feat/p2.9-phase1-foundation` at
  `42855a91c2cad6541c7eab7484ae00ed2d480ac1`, which already contains the checkpoint
  plus unrelated foundation/consolidation work. None of that extra work is
  included here. Its untracked `CLAUDE.md` and other worktrees were preserved.
- Archive SHA-256:
  `c2b707f2730f8d58886270a01a94427e195f60e1e9f54bc38051f7abc04d7719`.
  All **167 source files** match both the manifest and Git checkpoint; all
  **35 assets** match the manifest. See `checkpoint-identity.json` and the
  reproducible `verify_checkpoint.py`.

## Smallest safe scope

The feature cannot be lifted as only its UI and writer guard: main lacks its
shared SDK adapter, pinned dependency admission, stable-ID channels and GPU
drain lifecycle. Preserve the checkpoint's three SDK prerequisite commits
(`21eba92`, `411ce89`, `2d5d681`), feature (`f0d60a5`), budget qualification
(`6a79892`), writer guard (`82ba668`), independent oracle correction
(`424e883`) and evidence (`3445fab`). Keeping their history preserves provenance
and avoids reconstructing a different implementation.

The candidate retains opt-in live GN updates, private-solver animation baking,
bounded native readback, cache integrity checking and metadata validation before
writer-lock/filesystem side effects. Default accelerated publication remains
GPU-only. No solver/physics files differ from main.

Only three small adjustments beyond the checkpoint:

1. Restore main's readable `AGENTS.md` text while retaining the authorized
   native-output exception, excluding incidental encoding damage.
2. Restore main's `≈` speed label in `addon.py`; this is the sole production
   payload difference from the checkpoint.
3. Reuse the foundation branch's `Mapping` assertion in the native-sequence
   test, resolving Pyright's `object is not iterable` diagnostic without
   suppressing checks or changing product behavior.

See `checkpoint-adjustments.patch` and `integration-diff-stat.txt`. The latter
was captured before the final test-only assertion correction.

## Fresh verification

Windows 11, `C:/Python314/python.exe` 3.14.3, NumPy 2.3.5, Warp 1.17.0,
pytest 9.0.1, Ruff 0.14.11 and Pyright 1.1.408. CUDA was explicitly disabled
using `CUDA_VISIBLE_DEVICES=-1` for these host checks.

| Check | Result | Receipt |
| --- | --- | --- |
| Main baseline `python -m pytest -q` | 613 passed, 64 skipped, 72 subtests | `baseline-pytest.log` |
| Eight native/SDK test modules | 49 passed | `focused-pytest.log` |
| Candidate `python -m pytest -q` | 662 passed, 64 skipped, 72 subtests | `candidate-pytest.log` |
| Sequence + admission tests after typing fix | 9 passed | `sequence-final.log` |
| Ruff | passed | `ruff-final.log` |
| Pyright after typing fix | 0 errors, 0 warnings | `pyright-final.log` |

Full suite ran with the final production payload; the subsequent change was
only a test assertion, covered by the nine-test rerun. The initial Pyright
failure is retained in `pyright.log`. No suppressed failures or unrun GPU
checks are counted as passing. `environment.txt` records executable and package
identity. `checkpoint-receipts/` preserves the original manifest and supplied
Linux/Windows logs separately from these new receipts.

## Runtime and acceptance gaps

The checkpoint's historical Windows Blender 5.2.0 build `fbe6228777e7`,
RTX 4080 SUPER/Vulkan evidence establishes its 100,000-particle live updates,
eight-frame independent bake, cancellation, corruption recovery, standalone
reopen and distinct diagnostic renders. Hash verification authenticates the
provided receipts and assets; this task did not repeat those runtime actions.

No new extension package was built or installed. Candidate installed-package,
Blender GUI/native SDK lifecycle, CUDA eager/graph behavior and render acceptance
remain unrun. The SDK is still a separately supplied byte-pinned private wheel;
the feature requires its qualified exact Blender build and Vulkan backend.
Linux evidence covers CPU contracts on the earlier implementation, not the later
guard or Linux Blender/NVIDIA acceptance. No real-time FPS, zero-copy GN,
subframe/motion-blur, other GPU/build, long-animation or artist visual-quality
acceptance is established.

Next: review this branch against main, then qualify a package from this exact
candidate in the isolated approved Blender/SDK environment before promoting its
runtime status. Preserve the separate foundation lane and its outstanding
acceptance requirements. Main and remote branches were not advanced.
