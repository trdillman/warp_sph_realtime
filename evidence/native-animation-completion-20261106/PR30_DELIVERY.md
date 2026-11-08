# PR30 delivery — 7 November 2026

PR: https://github.com/trdillman/warp_sph_realtime/pull/30
Merged at 20:01:27 UTC; main merge commit
`c6b37db315592b7c2768827af79e9a8c06c7659d`.
Reviewed PR head: `c951966a6497e3394954680dd885c4c424fdc334`.

The lock fix and all preceding runtime/source evidence were preserved.
CodeRabbit identified an unsupported standalone-receipt claim of disk-cache
independence. The harness and receipt now use null (unverified), matching the
existing report's stated limit. Original receipts remain in Git history.
Ruff and Python compilation passed for the harness correction. Production
package bytes were unchanged.

CI run 42408746323 on the corrected head passed Windows, Ubuntu and Blender 5.2
Ubuntu checks. CodeRabbit passed its initial review and returned neutral/skipped
on the small correction. Its sole thread was resolved/outdated before merge.
Codex review completed on 861ae06 with no findings. GitHub reported CLEAN before
the merge, which was guarded against head changes with --match-head-commit.

The worktree fast-forwarded to the verified remote main after merge. This
delivery-only receipt follows the merge on the preserved integration branch.
No tag, public release, production installation or unrelated worktree mutation.
