# Exact-build Vulkan symbol preflight (M1, profile authority since M2)

`warp_sph_realtime/render/blender_symbols.py` is a standalone preflight API. It does
not select a transport, import CUDA memory, bind Blender GPU functions, extract a
GPUTexture pointer or change the existing OpenGL path. M2 integration remains separate.

Call `resolve_current_blender()` inside Blender, at session Start (never in `register()`).
By default the shipped exact-build profile is the RVA authority: the loaded executable's
SHA-256, build hash, Python ABI, architecture and PE/RSDS identity must match it, and
every RVA must lie in a correctly classified section. No PDB, helper or cache is used
(`detail == 'profile'`, about 0.1 s on the candidate). Matching the executable digest
already pins every byte the RVAs refer to, so the PDB adds no runtime safety; it was the
offline source used to build and review the profile.

`resolve_current_blender(verify_debug_symbols=True, cache_path=..., helper_timeout=60.0)`
is the optional deep check: it additionally requires the matching PDB and re-resolves
names through the cache or isolated DbgHelp helper. Use it when building or auditing a
profile. The cache path should be in extension user storage.
It returns an immutable `Resolution`: inspect `available`, `reason`, `detail` and
`resolved`. A successful result contains the exact identity, current module base,
callable/data addresses and recursively immutable ABI metadata. Availability means
symbol preflight only; later device, lifetime and transport-self-test gates still apply.

The lower-level `resolve_symbols(...)` takes a candidate executable/PDB and explicit
base for offline verification. Its injected helper seam is for tests. Runtime callers
must use `resolve_current_blender`, which discovers the loaded module through Windows.
`rebase_rvas(...)` is pure arithmetic, not an approval or validation API.

The packaged profile is pinned by a digest in source, independent of mutable cache
contents. Changing a layout, enum, signature or RVA requires a reviewed profile and
anchor update. The profile records the exact executable and PDB hashes, RSDS GUID/age,
architecture, Python ABI, source commit and M0 provenance. Opaque backend/window layouts
record only the proven read prefix; they are not allocations of the complete C++ type.

With verification enabled, the resolver also verifies PDB MSF identity/digest before
accepting either a cache or helper result. Callables must reside in executable sections; data
symbols are separately classified. The Vulkan wait entry is an executable JMP thunk,
not its IAT slot. `GPU_fence_wait` is deliberately absent because it can wait infinitely.

Cold verified resolution runs DbgHelp in a separate, hidden, time-limited bundled Python process.
It never uses Blender's executable as a Python CLI and does not download symbols. The
default helper deadline is60seconds, configurable up to120seconds. A timeout returns
Unavailable. This is a helper deadline, not a hard bound on all filesystem hashing.

Cache files contain identity and RVAs only. Every value is checked against the shipped
manifest, including well-formed, in-range wrong addresses. An invalid cache causes one
resolution from trusted inputs; failed/missing prerequisites remain Unavailable. In verified mode a
missing PDB cannot be bypassed by a valid cache. Cache-write failure does not invalidate an
otherwise verified result. Linux remains Unavailable until M6 supplies reviewed profiles.

Task results, tests, real-candidate checks and preservation receipts are recorded in
`research_notes/Vulkan zero copy/m1_task.md` and
`evidence/vulkan-zero-copy/20261018-m1/`. M0 reports are historical artifacts; do not rerun
their writers to refresh status after M1 changes the package inventory.
