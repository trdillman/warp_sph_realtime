# M5 protocol — requalification on today's build (pre-registered 2026-11-26, before any counted run)

Scope: plan `docs/VULKAN_ZERO_COPY_PLAN.md` M5, re-run on the integration branch
`integrate/m5-requal-20261126` (origin/main `e3f1103` plus every valuable local/unmerged branch; see the
branch's merge commits). Profiled Windows build: Blender 5.2.0 `fbe6228777e7` (exe SHA-256 `0060916d…6327`,
PDB SHA-256 `ffb7f1e9…c7ae`, both equal to the product's exact-build profile
`render/profiles/blender-5.2.0-fbe6228777e7-windows-amd64-cp313-v1.json`), Warp 1.17.0, cuda:0 RTX 4080 SUPER.
One run per lane against the frozen source in `freeze-manifest.sha256`; a source, harness or criterion change
after the freeze invalidates the runs and requires a new freeze plus one rerun.

This protocol is the v4 protocol (`20261019-latency-fix/M5_PROTOCOL.md`) with exactly one criterion change, the
owner's revised 4M latency threshold. Every other gate, count, sample size and metric definition is unchanged.

## Procedure
1. `m5_build.py`: builds the candidate with the product build script `scripts/build_extension.py` (validate +
   build, pinned shared SDK bundled as `_vendor/blender_vulkan_interop`), copies the zip to `artifact/`, checks
   every zip member against the tree it was built from, and installs the same archive into two isolated Blender
   user profiles (`profiles/vulkan`, `profiles/opengl`). It refuses to build unless blender.exe and blender.pdb
   equal the exact-build profile identity.
2. `m5_run.py vulkan`, then `m5_run.py opengl`: GUI Blender under a KILL_ON_JOB_CLOSE Job, deadlines
   5400 s / 2400 s counted from the probe's first progress, stall 600 s. Every launch goes through blender-gate
   with an **exclusive** lease on this PC (`bgate-report.json` per lane); only a `clean` gate verdict counts, a
   `contaminated` one means rerun. Displays are held on (`SetThreadExecutionState`).
3. Warp is installed through the product operator `wm.warp_sph_install`; only `urlopen` is replaced to stream the
   locally cached locked wheel (product size/SHA-256/host checks run).
4. `m5_verify.py` applies the criteria below and writes `m5-verdict.json`, PNGs under `images/`, and
   `reports/benchmarks/vulkan-zero-copy-m5-20261126-requal.{json,md}`.

## Criteria
| Gate | Pass condition |
|---|---|
| Texture parity | N ∈ {1, 4095, 4096, 4097, 4194304}; interop (Vulkan lane) and Readback (both lanes) textures equal the expected RGBA32F padded extent bit-exactly; padding zero; close `completed`. |
| Images | POINTS 3 px and SPHERES 0.02 into a 512² offscreen with fixed matrices. Same backend (interop vs Readback): ≥99.99 % of pixels with every channel within 1/255. Cross backend (Vulkan vs OpenGL Readback): ≥99.5 % within 2/255. Every image non-blank. |
| Lifecycle | 50 each of start/stop, resize 100k↔200k, disable/enable while in flight, file load while in flight. After every cycle: no owned live objects (resize: ≤1 interop transport), empty quarantine, close `completed`, no restart/unavailable flag. Warm-up = first 3 cycles; final VRAM used ≤ warm max + max(64 MiB, 3 × warm spread); handle-count slope over cycles 10–50 < 0.5/cycle and final ≤ warm max + 32. |
| Fault matrix | As v4: resolver failures fall back to Readback exactly once without calling the Vulkan factory; cache stale/corrupt/wrong-RVA recover through one isolated helper, valid cache gives `cache`; activation faults (UUID mismatch, pixel-buffer failure at calls 1/2/3, import failure CUresult 2, self-test mismatch) fall back once, labelled, balanced ledger, close `completed`. |
| Mocked fatal | CUresult 700 from `cuEventQuery` after activation: FatalTransportError, no further native calls, no Readback fallback, close `failed`, transport quarantined (classification only). |
| Latency | Interop at 4 194 304: completed transport latency (publish entry → Vulkan timeline value of that frame's upload fence reached, observed by a blocking `vkWaitSemaphores` thread) **p95 ≤ 16.67 ms** (one 60 Hz frame; owner target change 2026-11-26, plan M5 "Revised acceptance") over 3000 samples after 300 warm-up, zero drops, no missing completions. Cadence: one publish after the previous frame was drawn. |

Informational, not gated: interop completed p95 ≤ 2 ms at ≤ 1 M particles (the former target, kept as the plan's
informational figure). Reported but not gated, as v4: CPU submit, `positions_texture` CPU time, pack GPU time,
publish→draw latency, frame time, Readback budgets at 0.5M/1M/2M/4.19M. Unmeasured: upload GPU time and Readback
completed latency. Readback is a CPU copy and is never labelled zero-copy. Synthetic producer: `m5_kernels.animate`.

## Harness changes from v4 (harness only; disclosed before the freeze)
- Build via `scripts/build_extension.py` (the product build since the SDK bundling, PR #39) instead of a bare
  `blender --command extension build`; member check maps `_vendor/blender_vulkan_interop/*` to the pinned submodule.
- Blender: the byte-identical Downloads 5.2.0 build (same exe/PDB SHA-256 as the profile). The Program Files 5.2
  install carries a foreign Warp 1.18.0 in Blender's own site-packages; smoke attempt 1
  (`smoke/attempt1-programfiles-warp118`) showed the product refusing to install over it, which is correct product
  behaviour, so that install is not a valid clean host and was left untouched.
- blender-gate exclusive lease, queue allowance before the first progress (7200 s), deadline/stall counted from the
  probe start; M0 Job helpers copied verbatim into `harness/m0_supervisor.py`.
- `m5_run.py --smoke` / `M5_SMOKE=1`: reduced counts for a pre-freeze call-pattern smoke only (`smoke/`); the
  counted runs never set it.
- `m5_probe.py` setup: today's `wm.warp_sph_install` returns FINISHED and installs on a worker thread (v4's was`n  synchronous), so the probe waits for `warp_installer.install_running()` to clear, requires an owned install`n  (`owned_install_status()`, recorded in `probe.json`), then activates the runtime. Found by smoke attempt 2`n  (`smoke/attempt2-async-install`: "A Warp install is still running" at activation, both lanes).`n- `m5_probe.py` add-on state: today's add-on keeps settings in `scene.warp_sph` and one `sim_session.SimSession``n  per scene (v4: flat `scene.warp_sph_*` props and one global `addon.RT`). The probe reads the same fields from`n  the current scene's session (timer, draw handle, transport session, solver), the restart/interop flags from`n  `transport.process_state()`, and `last_close` from the session the cycle started (`RAN`), which is what`n  `RT.last_close` was. `use_shared_sdk` stays at its default (off), so Start uses the product transport path`n  as in v4. Found by smoke attempt 3 (`smoke/attempt3-scene-settings`).`n- `m5_probe.py` Start: today's live run is paced by the timeline and starts paused, so after `wm.warp_sph_start``n  the probe presses Play through the product's `wm.warp_sph_pause` (as a user does), and after `wm.warp_sph_stop``n  it cancels playback. The wait condition (running interop session with 3 uploads) is unchanged; a timeout now`n  records the session state. Found by smoke attempts 4-5 (`smoke/attempt4-start-no-uploads`,`n  `smoke/attempt5-start-paused`: running, `paused: True`, 1 upload).`n- `m5_verify.py`: 4M threshold 16.67 ms; the 2 ms figure reported as `informational_2ms`.

## Pre-freeze smoke (reduced counts, not a gate)
Attempt 6 (`smoke/attempt6-pass`, its own build) completed every step on both lanes: Vulkan setup, parity, cycles,
faults (all pass), bench, mocked fatal; OpenGL setup, parity, bench. Smoke numbers are not reported as results.
The counted runs use a fresh build of the same tree, frozen below.
