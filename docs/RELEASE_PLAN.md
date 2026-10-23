# Release Plan: v0.1 prototype → paid early access → v1.0

Source analysis: the owner's private commercial-readiness report and notes (kept local, not
in this public repo). Effort figures are
estimates for one focused developer (S < 1 week, M = 1–3 weeks, L = 1–3 months).

## Outcome

Sell a real-time GPU liquid tool for Blender whose pitch is **"art-direct it live, then bake
exactly that"**. It targets small-to-mid pours, splashes and stylised/viscous liquids on NVIDIA
GPUs (Windows/Linux). It does not compete with FLIP Fluids on ocean-scale water.

## Owner decisions

| # | Decision | Status (2026-10-17) | Blocks | Lawyer? |
|---|---|---|---|---|
| D1 | New product name (drop "Warp"/"NVIDIA" from the brand) | **Open.** Choose before any public post; run a trademark + Superhive name search | Phase 0 exit | Yes |
| D2 | Licence | **Open.** Recommended: `GPL-3.0-or-later` + `THIRD_PARTY_NOTICES` (Warp Apache-2.0, NVIDIA libmathdx/CUDA) | Phase 0 exit | Yes |
| D3 | Warp delivery | **Decided.** Keep the user-confirmed, hash-verified PyPI download (no bundled wheel) **and** add a local-installation option the user can point the add-on at (P1.11) | — | Only if bundling is ever considered |
| D4 | GPU scope | **Decided.** NVIDIA CUDA GPUs only for now (no AMD, Intel or Apple GPUs). This implies Windows + Linux only, since current Macs have no NVIDIA GPUs. CUDA 12 / Warp 1.17 for v1.0 | — | No |
| D5 | Certified Blender versions | **Decided.** Blender 5.2 and newer. Every new Blender release is certified on both OpenGL and Vulkan before the product page claims it | — | No |
| D6 | Price and update policy | **Open.** Recommendation kept in private owner notes | Phase 1 listing | Purchase terms: yes |
| D7 | Early-access deal | **Decided.** Early access at ~40% off, v1.0 included for early buyers | — | No |
| D8 | Telemetry | **Decided.** None. Local "Copy diagnostics" only; nothing leaves the machine unless the user pastes it | — | No |

## Phase 0 — Stabilise and add Vulkan support (now → early Dec 2026, ~5–7 weeks)

Goal: nothing breaks or leaks when a reviewer enables, disables, reloads or opens a file, and
the add-on **runs on Blender's Vulkan backend**, which becomes the default in 5.3 (≈10 Nov 2026).

**How Vulkan support works.** Blender's *Python* API exposes no Vulkan memory handles, but
Blender's *C* GPU layer does: `GPU_pixel_buffer_get_native_handle()` returns an exportable
`VkDeviceMemory` as a Win32 handle / Linux fd. Cycles uses it for its own CUDA→Vulkan viewport.
The planned adapter calls internal functions only under the exact-build ABI, address,
device identity and cleanup contracts in `docs/VULKAN_ZERO_COPY_PLAN.md`.

**Retained feasibility evidence, not qualification:** on Blender 5.2.0 build fbe6228777e7,
a Warp 1.16 kernel wrote 4,194,304 vec4 positions into Vulkan memory. Sampled reads and shader
sampling reported zero mismatches. The 1.55 ms median measures texture copy plus GPU_finish
after CUDA writing/synchronization; it excludes pack, waits and publish costs. It does not
establish full transport performance or pinned Warp 1.17 acceptance.

P0.11 defines three adapters sharing publish(positions), positions_texture(), capabilities()
and close(), with explicit count/dimensions/readiness. GLInterop must gain a verified GPU
texture bridge; Readback uploads a texture from host staging and is labelled compatibility
mode. VulkanInterop uses exported pixel buffers. Points/spheres use instanced quads and
texelFetch by instance. Legacy OpenGL SSFR stays explicitly separate until P0.13.

Native Vulkan selection requires a reviewed exact-build ABI profile, independently checked
RVAs, matching physical-device UUIDs before import, proven cleanup context ownership and a
known-answer self-test. Unknown prerequisites fall back only with a usable CUDA context.
Recoverable interop failures close safely then switch to Readback; fatal or uncertain CUDA
context failures stop, retain diagnostics and require restart. Never promise universal
crash-free recovery.

Windows resolution uses matching PDB GUID/age and validated profile/signatures; writable
cache data is not callable authority. Linux runtime uses reviewed exact-build tables,
not live RTTI/string scanning. Offline discovery and a historical prototype do not qualify
Linux. Both platforms require the full ABI, lifecycle and pinned-runtime gates in the
detailed plan. M6 follows Windows qualification; Linux remains unqualified until its own
gates pass. Any upstream API proposal requires owner approval before posting.

**2026-10-17 scope:** documentation reconciliation only; no implementation or M0 execution
authorized by this revision. The detailed Vulkan plan is authoritative for transport
contracts, measurement boundaries and dependency gates.

Subsequent authorized contained M0 continuation: native core, containment and normal lifecycle
cleanup pass the exact-build bounded prerequisites. Final reconciliation and supplemental
limit/cadence/generation/completion probes close the bounded M0 checklist.
The shared DRW context is entered through its
paired locked scope, then the incoming window context is restored. M1 has not started or
been authorized; production timeout safety/performance remain unqualified. The M0 qualification
results are kept in local, untracked research notes (removed from the public repo in #20).

| ID | Work | Files | Size | Acceptance evidence |
|---|---|---|---|---|
| P0.1 | **Lifecycle teardown**: `unregister()` stops the modal run, removes the draw handler and timer, unmaps/frees GPU buffers | `addon.py` | S | New Blender test: start sim → disable add-on → no handler/timer left, VRAM returns to baseline |
| P0.2 | **File-load/undo handlers**: `load_pre` stops the run and frees resources; `load_post`/undo leave no stale state | `addon.py` | S | Test: start → open another .blend ×50, no leaked VBOs, no console errors |
| P0.3 | **Transactional reallocation**: allocate new buffers → swap → free old; failure keeps the previous run intact | `addon.py`, `render/interop_bridge.py` | S | Test with injected allocation failure: sim continues at old size, error shown once |
| P0.4 | **Map/unmap as a context manager** + guarded draw callback with throttled (once-per-error) reporting | `render/interop_bridge.py`, `render/viewport_draw.py` | S | Test: raise inside draw → one report, no per-frame spam, GL state restored |
| P0.5 | **Timer and frame budget**: ~1/60 s timer, sim work capped per tick | `addon.py` | S | UI stays responsive at 4M particles (manual check + ms/frame log) |
| P0.6 | **Backend detection + adapter choice**: `gpu.platform.backend_type_get()` plus the detailed plan's pre-call ABI/address, device and cleanup gates followed by interop self-tests selects `VulkanInterop`, `GLInterop` or `Readback`; panel shows the active mode ("Zero-copy (Vulkan)" / "Zero-copy (OpenGL)" / "Compatible"). Only when Readback is active does it offer a one-click, user-confirmed "Use OpenGL fast mode" button, which sets `preferences.system.gpu_backend = 'OPENGL'`, saves preferences and says a restart is needed. Always decide from `backend_type_get()` (what is running), never from the preference: a 2026-10-17 probe on Blender 5.2 showed the preference reads `OPENGL` while `--gpu-backend vulkan` is running, and that **no OpenGL context is current anywhere** (draw handlers, timers) under Vulkan, so the raw-GL path cannot be used there | `addon.py`, `render/` | S | Start on 5.3 Vulkan uses Readback with no traceback; on OpenGL uses GLInterop; recoverable interop failure falls back to Readback; fatal CUDA failure stops and requires restart |
| P0.7 | **Warp self-test**: version + driver check and a known-answer kernel at activation; a foreign Warp is named in the error | `warp_runtime.py` | S | Test with a fake foreign Warp: specific message naming its version |
| P0.8 | **Ship only what works**: exclude `experimental/` from the built zip; fix `ARCHITECTURE.md`, README claims and the manifest `tags` | `blender_manifest.toml`, `scripts/build_extension.py`, docs | S | Built zip contains no `experimental/`; docs describe only shipped features |
| P0.9 | **Rename + licence switch** (after D1, D2): manifest id/name/tagline, panel labels, operator idnames, README, LICENSE, THIRD_PARTY_NOTICES, "not affiliated with NVIDIA" line | package-wide | S | `blender --command extension validate` passes; grep finds no "WarpSPH" brand use |
| P0.10 | **Regression coverage** for P0.1–P0.7 and P0.11–P0.13 in the env-gated Blender test suite | `tests/` | S | All new tests pass locally on the RTX 4080 SUPER box; failing-first evidence kept |
| P0.11 | **Texture transport seam + Readback**: publish(positions), positions_texture(), capabilities(), close(); explicit count/dimensions/readiness. Add GLInterop texture bridge and preallocated Readback staging; prove stream/lifetime ordering before claiming overlap | render/interop_bridge.py, new render/transport_readback.py | M | Position parity, no readback in accelerated paths, labelled compatibility fallback and per-adapter timings on both backends |
| P0.11b | **VulkanInterop**: execute M0–M5 of docs/VULKAN_ZERO_COPY_PLAN.md after an execution request; complete ABI profiles, independently validated RVAs, UUID equality before import, bounded slots and synchronous cleanup; classify fatal CUDA errors separately | new render/transport_vulkan.py, render/blender_symbols.py, resolver helper/profiles | M (re-estimate after M0) | Pinned Warp 1.17 full parity/lifecycle/fault matrix; zero native calls on invalid ABI/cache, zero imports on UUID mismatch; completed transport p95 ≤2 ms at 4,194,304 particles per M5; Linux separately gated by M6 |
| P0.12 | **Points + spheres on the `gpu` module** (backend-neutral shaders), drawing from any transport | `render/viewport_draw.py` | M | Points/spheres render on 5.2 OpenGL and 5.3 Vulkan; screenshots match within tolerance |
| P0.13 | **SSFR ported to the `gpu` module**: depth pass, narrow-range filter, thickness and composite as `gpu` shaders + `GPUOffScreen`; delete raw ctypes GL calls from the renderer | `render/ssfr_shaders.py`, `render/viewport_draw.py` | L (3–4 weeks est.) | Liquid look works on Vulkan; image matches the current GL SSFR reference screenshots within tolerance; frame time measured on both backends |

Exit criteria: P0.1–P0.13 done; 50 cycles each of enable/disable/reload/file-load while
running with no leaks, on both backends; all three display modes run on Blender 5.2 (OpenGL)
and 5.3 (Vulkan); measured Readback cost published internally; manifest says `GPL-3.0-or-later`
under the new name.

Order inside Phase 0: separately authorized M0 prerequisite probes → P0.1–P0.5 (lifecycle) → P0.11 (texture seam) → P0.11b M1/M2 → P0.6/P0.12 (M3/M4) → M5 qualification → P0.13,
with P0.7–P0.9 in parallel. If 5.3 ships before P0.12 lands, a short interim release shows the
"switch to OpenGL" instruction on Vulkan.

## Phase 1 — Paid early access (Dec 2026 → Jan 2027, ~6–9 weeks)

Goal: a buyer on a default Blender 5.3 install can run it, see what it does, and record
something to use in a render. Product page states exactly what is and is not included.

| ID | Work | Seam / module | Size | Acceptance evidence |
|---|---|---|---|---|
| P1.1 | **`SimSession`**: move all runtime state out of `addon.py` globals into one per-scene session object (`start/stop/tick/dispose`); `addon.py` becomes thin UI | new `session.py` | M | All Phase 0 lifecycle tests pass against the session interface |
| P1.2 | **Readback tuning**: optimize measured staging/texture upload while preserving the shared texture interface; alternative formats or storage-buffer paths need a separately reviewed interface and precision contract. Add a particle-count advisor based on P0.11 measurements | `render/` | S–M | Readback ms/frame improved or documented; advisor warns before a frame-rate cliff |
| P1.3 | **Settings PropertyGroup** replacing 15 flat Scene properties; Advanced sub-panel; VRAM warning; fix stiffness step | `addon.py` → `props.py` | M | Settings survive save/reload of the .blend; old files open without errors |
| P1.4 | **Async Warp install** with progress bar and cancel; first-run hardware check panel (GPU, driver, backend, VRAM) | `warp_installer.py`, UI | M | UI stays responsive during the ~150 MB download; cancel leaves no partial install |
| P1.5 | **Copy diagnostics** button + `logging` to the extension user directory | `warp_runtime.py` | S | One click yields GPU, driver, Warp, cudart, Blender version/backend, last errors |
| P1.6 | **Minimal record & playback**: fixed-timestep record of N frames to a disk cache (npz), playback as a Blender point cloud synced to the timeline, with `velocity` and `id` attributes | new `cache/` (FrameStore: disk adapter + VRAM ring) | M–L | Record 120 frames → scrub timeline → render points in Cycles; replay is deterministic (bit-identical positions on re-read) |
| P1.7 | **CI and release pipeline**: Windows job, real lint/type rules (tighten Ruff and Pyright gradually), `extension validate/build` on tag, artifact upload, version single-sourced from the manifest | `.github/workflows/`, `scripts/` | M | Tag `v0.2.0-ea` produces a validated zip as a release artifact |
| P1.8 | **GPU test box**: self-hosted NVIDIA runner for Blender + interop tests on OpenGL and Vulkan; nightly job against latest Warp as early warning | CI | M | Nightly report; failure on a Warp change opens an issue |
| P1.9 | **Benchmark table**: ms/frame and VRAM at 0.5M/1M/2M/4M on ≥3 GPU generations (e.g. 20xx, 30xx, 40xx) | `docs/` | S–M | Published table backs every performance claim on the product page |
| P1.11 | **Local Warp installation option** (D3). Preferences get a Warp source choice: *Download (default)* / *Install from local wheel file* / *Use existing Warp folder*. **Wheel file:** user picks a `.whl` (offline or studio machines); it must match the SHA-256 lock, then goes through the same verified extraction into private storage. **Existing folder:** user picks a folder containing a `warp` package (e.g. a studio-managed site-packages); the add-on adds it to the import path at activation, never modifies or deletes it, and runs the P0.7 version/driver/known-answer self-test. Versions other than 1.17.x are refused by default, with an "allow uncertified version (unsupported)" checkbox that shows a persistent warning. The foreign-Warp guard accepts only the folder the user chose. Diagnostics (P1.5) report which source is in use | `warp_installer.py`, `warp_runtime.py`, preferences UI | M | Tests: matching wheel installs offline; tampered wheel refused; valid folder used without being touched; folder without `warp`, wrong version or failing self-test gives a specific message; switching back to Download restores the owned install; works with Blender "Allow Online Access" off |
| P1.10 | **Launch kit**: product page, 60–90 s demo video, two example .blend files, quick-start docs, known-limitations list, support channel (Discord or email) | marketing | M | Superhive listing approved; page lists hardware requirements and exclusions |

Exit criteria: Warp can be installed by download, local wheel or an existing folder; runs fully on 5.2+ with both OpenGL and Vulkan; record/playback
renders in Cycles; benchmarks published; listed on Superhive (with Gumroad as a backup) at the early-access price.

## Phase 2 — v1.0 (≈ mid-2027, 4–6 months after early access)

Goal: a user builds a pour or splash from their own objects, bakes it, scrubs it, and renders it
in Cycles/EEVEE without leaving Blender.

| ID | Work | Size |
|---|---|---|
| P2.1 | **Scene seam**: `SceneDescription` from Blender objects (domain box, emitters with inflow, static + animated mesh colliders as SDFs, gravity, scene units); demos become the second adapter; obstacles leave the compiled kernel | L |
| P2.2 | **Full bake/cache**: bake range, resume, invalidate-from-frame, cache directory per domain, size estimate before bake | M–L |
| P2.3 | **Output**: point cloud with velocity/id/age; Geometry Nodes meshing preset (points → volume → mesh); Alembic and/or USD export | M–L |
| P2.4 | *(moved to Phase 0 as P0.13)* | — |
| P2.5 | **Physics controls**: viscosity, unit-aware particle spacing, presets (water, viscous, stylised) | M |
| P2.6 | **Validation**: dam-break front position vs published reference data; mass/energy checks in CI | M |
| P2.7 | **Free demo edition**: unlimited preview, bake/export capped (e.g. 120 frames, 25k particles) | S–M |
| P2.8 | **Docs and education**: full manual, 3–5 video tutorials, example scene library, FAQ for hardware issues | M |
| P2.9 | **Live collider drag**: non-animated colliders follow their live transform while running (speed-capped, no tunnelling); later, record live drags as keyframes so a bake replays them. Spec: [`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md) | M |

Exit criteria: P2.1–P2.8 shipped; certified on 5.2 LTS and current stable on both backends;
refund rate and support load tracked from early access are acceptable to the owner.

## Phase 3 — v1.x (H2 2027 onward)

Whitewater (foam/spray/bubbles), surface tension, a PBF/DFSPH final-bake solver (promotes the
solver seam to a real interface), anisotropic/smoothed meshing, Warp 1.18+/CUDA 13 migration
after auditing integer floor division in the spatial hash, device selection, game outputs
(vertex animation textures, flipbooks), readiness for Blender 5.5 LTS and a possible OpenGL
removal in 6.0, switching `VulkanInterop` to the public Python API once the upstream patch lands.

## Standing rules for every phase

- Feature branch from `main` → PR → green CI → merge. Installer, network or permission changes need owner review.
- Every non-trivial change ships with a failing-first test; GPU/UI claims need runtime evidence on real hardware, not mocks alone.
- `CHANGELOG.md` updated per PR; release tags match the manifest version.
- Keep Warp pinned at 1.17.0 until the Phase 3 migration task is done.

## Risks to watch

| Risk | Trigger | Response |
|---|---|---|
| Blender 5.3 ships Vulkan default before P0.12 lands | 5.3 release (~10 Nov 2026) | Interim release with the "switch to OpenGL" instruction; keep P0.11–P0.13 first in the queue |
| Internal symbols move or ABI changes | Pre-call profile/address checks reject candidate | Readback only with usable CUDA; fatal errors stop/restart. Qualify new profiles before native self-tests; never probe guessed addresses |
| Linux profile cannot be verified | Offline M6 discovery or qualification fails | Linux uses Readback with healthy CUDA until its own full gates pass; no runtime heuristic resolver |
| Readback too slow at 1M+ particles | P0.11 benchmark | Measured staging/texture-upload tuning (P1.2) and particle-count advisor; claim an OpenGL speed advantage only from paired measurements |
| SSFR port loses quality or speed on the `gpu` module | P0.13 screenshot/frame-time comparison | Keep the explicitly separate legacy OpenGL SSFR renderer/resource path until the port matches; do not hide rendering inside the texture transport adapter |
| Blender changes the internal GL buffer access used by interop | Any 5.x minor release | Nightly GPU CI catches it; session falls back to Readback automatically |
| Name/licence advice changes the plan | D1/D2 lawyer review | Rename is cheap before launch; delay listing, not engineering |
| Weak early-access sales | < ~20 sales in the first 60 days | Re-check positioning and demo video before investing in Phase 2 scope |

## Status (2026-10-23, main `5965726`)

Last recorded local checks (main `3e20429`): pytest 585 passed / 2 skipped, ruff clean. Not re-run for this refresh.

Merged since the previous refresh:

- #19: P2.9 live collider drag spec ([`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md)) and this status table.
- #20: `evidence/showcase-repair-20261021/` and `research_notes/` are no longer tracked; the files stay local and nothing in code, tests or CI reads them.
- #21: resumed bakes are rejected when solver gravity changed or is unknown (older caches). Evidence:
  [`verification/cache-gravity-20261022/REPORT.md`](verification/cache-gravity-20261022/REPORT.md), a background
  Blender/CUDA session only, with no UI or viewport acceptance. Domain, collider and emitter edits are still not
  fingerprinted; start a fresh bake after them.

| Phase | Done on main | Open |
|---|---|---|
| 0 | P0.1–P0.8, P0.10–P0.13 (lifecycle, transport seam, GL/Vulkan interop with the latency fix, gpu-module points and SSFR, Warp self-test, experimental excluded) | P0.9 rename/licence (D1/D2); exit soak on Blender **5.3** Vulkan, which needs a new exact-build Vulkan profile, or else it runs Readback |
| 1 | P1.1, P1.3–P1.7, P1.11 | P1.2 Readback tuning + count advisor; P1.8 GPU runner; P1.9 benchmark table; P1.10 launch kit; first tagged release (`release.yml` unproven) |
| 2 | P2.1, P2.2, P2.3, P2.5, P2.6, P2.8 | P2.7 free edition; P2.9 live collider drag; showcase reruns under the strengthened checks; certification on 5.2 LTS + current stable |

Owner decisions D1, D2, D6 and the Vulkan 4M latency target are **not blocking engineering**. Work proceeds with
reversible placeholders: brand and licence strings are kept in one place for P0.9, and the P2.7 cap values are
set in one setting. The 4M target stays recorded as open; nothing is relabelled as passed.

## Immediate next actions

1. Blender 5.3 readiness: qualify an exact-build Vulkan profile for the 5.3 release candidate; verify Readback fallback and the "Use OpenGL fast mode" path on an unprofiled build.
2. P2.9 live collider drag: spike S0, then Phase A ([`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md)).
3. Close live gaps: showcase reruns under the strengthened checks; real Warp download install; undo/redo during a run; particle-radius unit-awareness.
4. P1.2 Readback tuning plus particle-count advisor; P1.9 benchmark table on this GPU, with other generations added when hardware is available.
5. Release dry run: bump to `0.2.0`, tag on a fork or draft to prove `release.yml` (tagging the real repo needs owner approval).
