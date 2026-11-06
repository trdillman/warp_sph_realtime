# Phase 1 integration and consolidation — 6 November 2026

## Candidate and scope

Work continues on `feat/p2.9-phase1-foundation`. It initially pointed at the specification-only
commit `82cf500231a2065b8cf3b4a3e26062aaa20acdd4`; the initial checkout was on main `52a383a`.
The owner confirmed Phase 1 integration, then requested consolidation before further work.
The branch was fast-forwarded through main to the existing reviewed native-animation candidate
`3445fab5316d55cabd3459c2fc058cb320945ee9`. No remote branches were changed.

The consolidation preserves other checkouts and their indexes. Eight verified Git bundles,
59 dirty-file copies with SHA-256 checks, binary staged/unstaged patches, and a six-file loose
gravity snapshot are stored under `.codex/context/consolidation-20261106/`. Histories from
independent clones are also reachable through `refs/consolidation/20261106/`. The private
`inventory.json`, `refs.json`, `preservation.json`, and `source-recheck.json` record exact paths,
identities and preservation receipts. Source recheck found no changes to the copied files in
the other checkouts. Ignored files and original directories remain in place; they were not
deleted or moved, and the backup is not a full copy of ignored caches or SDK trees.

## Disposition

| Work | Disposition |
| --- | --- |
| Main, Phase 1 foundation and subsequent main fixes | Integrated by fast-forward; retained newer fixes rather than replaying old drafts. |
| SDK adoption, live Geometry Nodes, native animation and final qualification branch | Integrated through `3445fab`; prior reports remain tied to their original candidates. |
| Standalone solver replay branch | Its solver replay implementation and regression file are identical to the integrated branch; untracked runtime probe preserved. |
| Original dirty interactive worktree and staged publication clone | Earlier foundation drafts preserved, superseded by the integrated Phase 1 implementation. |
| Old installer, SSFR, showcase, review-fix and pre-filter histories | Preserved as bundles/refs; old implementations are not reapplied over their later merged replacements. Pre-filter private history is not merged into the product branch. |
| Native snapshot alternative | Preserved separately. It implements a competing snapshot adapter/UI; merging it wholesale would introduce a second output owner. No deletion or acceptance implied. |
| Cache-provenance candidate `dc9a635` and dirty follow-up | Preserved separately. This is a broader unfinished bake-input freezing design, not the current main boundary fingerprint implementation. Its review and runtime gaps are not closed by this task. |
| Loose gravity snapshot | All six files copied and hashed; partial old source is retained as recovery material rather than overwriting current cache/session modules. |
| Untracked handoffs, runtime probes and SDK snapshot manual | Preserved in per-checkout dirty-file archives; historical claims are not promoted into current acceptance. |

This is preservation and integration of the compatible current work, not a claim that every
alternative branch has been merged or that the original worktrees have been removed.

## Discovery coverage

Everything returned 538 project-name folder matches and 1,178 Git markers on C:, with all
pages consumed. Config inspection found no additional readable project clone beyond the
14 inventoried project checkouts. The broad marker scan encountered 118 unavailable/stale
markers, so this is not proof of exhaustive access to every directory or other volume.

`C:/Temp` contains no indexed Git repository. Its `warp_flow_staged.zip` contains a separate
`warp_flow` extension (205 entries), not this project's package; it was inspected without
extracting or changing it. Windows Temp and user Temp were checked separately: their listed
Git markers were stale/unusable, with access denied at `C:/Windows/Temp/flow-test-v01c4vz4/repo`.
The registered F: worktree path is missing. The original gravity recovery directory retains
only stale Git metadata. No access denial was bypassed and no unavailable path was pruned.

## Phase 1 correction

The animated-collider sync previously saved the last prefetched future pose as the deletion
fallback. It also left the timeline at the sampled frame when key validation raised. Now:

- the timeline returns to the solver's current frame in `finally`;
- only a successful sync refreshes the held-pose fallback, from that restored current frame;
- rejected samples leave the prior held-pose dictionary intact;
- untouched animation key matrices and frame ordering are unchanged.

Two new regressions demonstrated both defects before the correction (`red.log`). This does
not implement Phase 2 live transform following, invalidate already cached animation keys,
remove colliders live, or turn invalid-animation exceptions into recoverable session states.
An imported native-sequence test also now narrows its read-only field mapping for Pyright.

## Verification

Environment: Windows, `C:/Python314/python.exe`, Python 3.14, Warp 1.17.0 from the Python 3.14
user site; RTX 4080 SUPER, driver 621.14. This differs from Blender's bundled Python 3.13.

| Check | Result | Evidence |
| --- | --- | --- |
| New regressions before fix | 2 expected failures | `red.log` |
| Phase 1 focused selection | 68 passed, 13 subtests passed | `focused.log` |
| Full headless suite | 726 passed, 2 skipped, 76 subtests passed | `suite.log` |
| Native sequence after mapping assertion correction | 7 passed | `native-sequence-final.log` |
| Ruff | Passed | `ruff-final.log` |
| Pyright | 0 errors, 0 warnings, 0 informations | `pyright-final.log` |
| Git whitespace | Passed | `git diff --check` |

The full suite preceded the final test-only mapping assertion; the affected test file was
then rerun. No solver kernel changed. GPU hardware presence and headless tests do not establish
live viewport, packaged-extension or performance acceptance.

## Acceptance still required for this combined candidate

- Blender 5.2 isolated-profile verification on OpenGL and Vulkan: undo/redo object identity,
  collider rename/delete fallback, modal transform continuity, and timeline restoration.
- Current-candidate gravity recapture/eager-versus-graph validation inside Blender.
- Reset at 1M particles: at most 2 seconds and faster than Stop + Start on the same scene.
- Both backends at 100k/1M: the owner-restated editing throughput gate (at least 90% of the
  same interaction with change handling disabled), change handling at most 1 ms/tick for
  32 objects, plus the headline interaction capture.
- Recheck save/reopen, bake isolation and combined native-output lifecycle if qualifying
  this consolidated candidate for delivery; build/install validation has not run here.

The 26 October Phase 1 acceptance report and the 6 November native-animation report remain
historical exact-candidate evidence. They are not reruns of this combined working tree.
Phase 2 live transforms and later set/shape/take phases remain outside the confirmed scope.
