# PR31 merge receipt

PR: https://github.com/trdillman/warp_sph_realtime/pull/31
Merged 7 November 2026 at22:14:14 America/Toronto (8 November03:14:14 UTC).
Main merge:01d034efae201323c73b73cd8df65ac3ad7352d7.
Final checked head:321584c3defecc0d70a36078e2fa3bd9caa6e172.

CI run42441697728 passed Windows,Ubuntu and Blender5.2 Ubuntu at that head.
CodeRabbit returned neutral/skipped for the followups. Its initial review and
Codex review findings were inspected and fixed, including the duplicate completed
step finding. All four GitHub review threads were resolved before merge.
Independent follow-up source review PASS at product commit2487104; seven timeline
tests independently passed. The final head differs from2487104 only by the
review-disposition documentation.

Review repairs preserve accepted scene FPS until Reset (including real demo
solver's legacy fps=24), clear satisfied step requests, and hold failed replay
without repeated rebuild/report until explicit scene Reset. See PR31_REPAIRS.md
and adjacent focused RED/GREEN receipts. Latest affected suite21tests/2subtests
passed. Final-head CI provides full source/static/Blender CPU proof.

Installed GPU receipts remain scoped to their exact recorded earlier product
packages. The PR review repairs have source/CI proof and do not retrospectively
change those package identities. No refreshed GPU, broad platform/performance,
public-release or production-installation claim is made.

Merge used a match-head guard, then origin/main was fetched and this worktree
fast-forwarded to the verified merge. This delivery-only receipt is preserved on
the integration branch after merge. Original evidence and failures remain in Git.
No release tag was created and unrelated worktrees/profiles were preserved.
