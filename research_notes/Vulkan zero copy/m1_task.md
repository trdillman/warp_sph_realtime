# M1 profiles and resolver — completed 2026-10-18

User authorizes M1 implementation and rejection verification, autonomously; preserve M0.
No M2 transport integration or GPU self-test is part of this change.
Branch: codex/vulkan-m1-resolver, base617b4eabfb24c89618122523eff3f057d3e3f58a.
Unrelated dirty changes remain in place. New evidence: evidence/vulkan-zero-copy/20261018-m1/.

Critical path: exact-build profile → pure preflight/PE+PDB validation → isolated bounded
DbgHelp helper → independently validated RVA cache → current module rebasing → rejection
tests → fresh-process candidate proof → independent review and final artifact audit.
Acceptance is C1 and the three M1 bullets in docs/VULKAN_ZERO_COPY_PLAN.md.

Ownership: m1_resolver deep_worker (GPT-6.1 Sol/high) owns new resolver/helper/profile/tests.
Parent owns real-candidate verification, preservation, documents and evidence. Independent
deep_worker review follows integration. No concurrent Blender/GPU lanes.

Decisions: standalone/headless-safe resolver, no eager bpy/Warp imports; no GPU calls or
texture pointer extraction during qualification. Linux returns Unavailable. Missing debug
data cannot be bypassed by a cache hit. Cache is only a hint, shipped profile is independent
authority. Helper uses actual Python, never Blender's sys.executable as a Python CLI.

Delivered: exact-build SHA-pinned profile, PE/RSDS/PDB validation, bounded subprocess
DbgHelp helper, independently checked RVA cache, current-module rebasing and immutable
complete ABI metadata. See docs/VULKAN_RESOLVER.md. No production consumers changed.

Verification: full pytest120 passed,2 skipped,8 subtests in17.18s;33 resolver tests included.
Focused Ruff and Pyright pass. Independent reviewer found two gaps (missing complete
metadata and deeply nested JSON exception); both received regression tests and fixes,
then final review passed. Initial Python3.14 `-m pyright` was unavailable; the installed
Pyright CLI passed. Both command receipts retained.

Real candidate: fresh-a resolves20 callables/four data addresses through helper in19.697s;
fresh-b resolves through cache in2.047s. Both use actual module base0x7ff7c8620000,
distinct from preferred0x140000000, and exit naturally with code0 under owned supervisors.
Both happened to receive the same base; differing-base arithmetic is covered by unit tests.
No native GPU calls or texture pointer extractions. Missing PDB returns pdb_missing;
actual1ms helper deadline returns helper_timeout; in-range wrong RVA is rejected and
repaired by exactly one helper invocation. Unknown profile/layout/enum/signature, identity,
PDB GUID/age, JSON schema/types/depth and bounds are covered by resolver tests.

Preserved failure: first real helper attempt rejected DbgHelp names because NameLen
included a terminal NUL. Diagnostic retained; helper now strips only one terminal NUL
and rejects embedded NUL. Corrected offline and fresh-process runs pass. No GPU retry.

Artifact: Blender extension validate/build pass; package.json records SHA256 and confirms
all three new packaged files equal source; no PDB bundled. Artifact not installed.
Preservation-before verified301 M0 manifest entries. All301 baseline copies retained;
final audit checks264 original evidence files and original manifest unchanged, plus all
pre-existing package source hashes. Windows path classification initially copied all301
files rather than only mutable files; this safe overcopy is retained and documented.

Evidence: evidence/vulkan-zero-copy/20261018-m1/{tests-full.log,ruff-final.log,
pyright-cli-final.log,real-rejection-probes.json,fresh-a/,fresh-b/,package.json,
verification-final.json,manifest-final.json,source-final/}. No commits or publication.
M1 complete; M2 remains outside scope. Exact Windows build only; Linux unavailable.
Product lifetime/fault safety and performance still require later integration qualification.
Tracker remains pending in docs/TRACKER_UPDATE_PENDING.md (empty Page/no updater).
