# WarpSPH live Geometry Nodes and animation — cross-platform qualification
**Date:** 2026-11-06 (UTC)
**Status:** Windows native acceptance PASSED for the reviewed private candidate. Linux CPU acceptance PASSED; Linux Blender/NVIDIA-native acceptance NOT RUN.
**Review branch:** `interop/live-animation-verified-20261106`
**Feature implementation:** `f0d60a5f8f71511e399a2c5fb18772e2482ab386`
**Budget qualification:** `6a798924c1481574fbd032f324c8317b837ba3f0`
**Product guard correction:** `82ba668363018ce561b742a91efa084ebafc90a1`
**Independent test correction:** `424e883f475c500b6f6ba152d92cb249866054d4`

## What is now usable
The opt-in **Live Geometry Nodes** mode updates a native point mesh and its editable GN graph while WarpSPH runs. It deliberately reads back completed frames, and retains the accelerated GPU-only renderer as the default.

**Bake Native Animation** advances a separate copy of the solver, not the artist's live simulation. The sequence has eight verified frames with positions, velocity, density, radius, colour and exact integer particle IDs. It supports timeline seeking, integrity-checked playback, cancellation and a standalone embedded GN animation.

The native `.blend` contains all frames. It was reopened in Blender with WarpSPH, Warp and the interop SDK absent, and with the original disk cache physically withheld.

## Linux: CPU contracts only
The provided Linux environment was **Linux x86_64 / Python 3.13.5**. Its original exact implementation checkout `f0d60a5` passed **24 tracked host tests plus six adversarial tests** (30 total) covering metadata, frames, integrity, cancellation, random access and stable identities. Evidence: `evidence/linux-windows-20261106/linux/linux-acceptance.log` and `linux-environment-results.json`.

No Blender executable, `bpy`, Warp GPU runtime, or NVIDIA device was available in that Linux environment. Network/DNS provisioning failed. **No Linux native Blender, CUDA, Vulkan, graphics, or render certification is claimed.** The new metadata-validation guard has passing Windows-focused tests; the prior 30 Linux tests used the exact earlier implementation commit, not the later guard commit.

## Windows: installed candidate and native Blender
Environment: **Windows, Blender 5.2.0 LTS build fbe6228777e7, Warp 1.17.0, NVIDIA RTX 4080 SUPER cuda:0, Vulkan**. All GUI tests used isolated canonical Blender profiles and serialized GPU ownership. The extension and interop SDK were byte-checked against their selected package artifacts.

- **Live updates:** five distinct frame stamps were observed automatically at a 100,000-particle population. Turning the mode off retained the last native mesh.
- **Independent bake:** all eight requested frames were recorded; positions changed. The original solver's GPU arrays, counters, timeline, and emitter state remained unchanged after bake and cancellation.
- **Cancellation:** a two-frame complete prefix was retained without changing live solver state.
- **Playback and corruption:** nonsequential timeline seeks matched cached native attributes. Deliberate cache corruption hid stale output and was visibly reported; restoring valid bytes recovered the correct output.
- **Standalone:** a new Blender process with *none of the runtime packages installed*, and without the original cache directory present, verified **all eight frames**. Each of 100,000 particle IDs was checked through 12 realized GN vertices, with position centres, radius, velocity, colour and density matching reference arrays. Source-only reference files were used only by the verifier.
- **Render Animation:** `bpy.ops.render.render(animation=True)` produced **eight nonblank, distinct 192×192 diagnostic PNGs**, with visible-pixel counts of **7221, 7259, 7231, 7248, 7381, 7424, 7496, 7559**. Their pixel hashes differ.
- **Resource cleanup:** both owned Blender processes and the TCP listener exited; the shared GPU lease was released. Production Blender profiles and other agents' processes remained untouched.

## Final host, packaging and static checks
- Full source suite on Windows: **662 passed, 64 GPU-disabled skips, 72 subtests passed**, exit 0.
- Installed extension's applicable suite: **640 passed, 56 GPU-disabled skips, 72 subtests passed**, exit 0. All 51 loaded consumer Python modules came from the extracted extension payload, not the source checkout. Four tests explicitly dependent on deliberately unshipped `experimental/` modules were excluded **only from this installed-package suite**, not the full source suite.
- Ruff: passed. Pyright: **0 errors**, 38 host-only warnings from missing Blender source stubs.
- Exact reviewed extension ZIP, interop SDK wheel, native saved scene and rendered images are bound by SHA-256 in the delivery manifest.

## Defects and corrections
1. Metadata containing NaN or an unserializable object could leave `.writer.lock` behind after rejecting a new sequence. Two tests first reproduced this and then passed when signature validation moved before filesystem side effects. This is the **only production payload change** from the approved reviewed feature branch.
2. The original standalone test requested 100,000 realized instances across eight frames under an insufficient 512 MiB memory budget (estimate 563,281,920 bytes). The acceptance test was corrected to use a permitted budget **without weakening the actual memory limit**; a negative budget test confirms rejection without mutation.
3. The independent Blender test used the wrong preferences add-on iteration type. This was corrected in test code only; no product behavior changed.
4. Experimental source-only tests are not valid tests of deliberately excluded installed modules. Their separate source suite still passes. Both failed collection attempts remain in the raw receipts.

## Boundaries and next acceptance
**Not established:** NVIDIA-backed Linux qualification, other Blender versions/GPUs, interactive frame-rate target, zero-copy Geometry Nodes, interpolated subframes, physically correct motion blur, or long production animation beyond the verified eight frames. Diagnostic renders establish actual nonblank, nonidentical outputs, not production shading quality. This remains an **isolated development checkpoint**, not a public release or production installation.

All source code, test revisions and raw receipts are preserved in the named isolated worktree and downloadable local ZIP. Main was not modified or pushed.
