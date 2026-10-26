# P2.9 Phase 1 live runtime gates — 2026-10-26

Gates from `docs/INTERACTIVE_SIM_SPEC.md` (§Risks R1–R4, §Acceptance Phase 1 and "Every phase, live Blender"),
measured in real Blender with the harness `scripts/run_p29_runtime_gates.py` → `scripts/p29_runtime_gates.py`.

**Verdict: Phase 1 accepted under the restated throughput gate (owner decision, 2026-10-26).** Every gate passed
except the original throughput gate ("steps/s during continuous editing ≥ 90% of an untouched run"). That one
failed at both sizes and on both backends, and the attribution below traces it to viewport redraw contention that
already existed before P2.9. The owner restated the gate as "≥ 90% of the same interaction with change handling off".
All eight A/B pairs meet it (0.935–1.02, table below). The redraw contention is tracked as separate work.

## Setup

- RTX 4080 SUPER (16 GB), driver 610.88, Windows 11; Blender 5.2.0 (official portable build); Warp 1.17.0.
- Isolated profile (the showcase runner's `evidence/showcase/_profile`) with this checkout's add-on copied in; one
  Blender at a time; `--enable-event-simulate` drives real G-grab modal transforms; window 1600×900, vsync off.
- Custom scene: domain, a 1 m³ fluid block, 32 static colliders, one keyframed collider (so the session calls
  `frame_set` every frame). Render mode Points (OpenGL texture interop / Vulkan interop, no CPU readback).
- Source: product code at `990b11c` (worktree branch, pushed with this report). The full matrix ran at `990b11c`;
  the two attribution runs (`diag1`, `diag2`) only added diagnostic stages to the harness, with no product change.

## Results (full matrix, `full-990b11c/`)

| Gate | OpenGL 100k | OpenGL 1M | Vulkan 100k | Vulkan 1M | Result |
|---|---|---|---|---|---|
| R1 `session_uid` stable across undo/redo (35 objects) | 0 changed | 0 changed | 0 changed | 0 changed | PASS |
| R2 ticks continue during a G-grab (steps/s while grabbing) | 184 | 61 | 185 | 52 | PASS |
| R3 `frame_set` during the grab: transform keeps tracking the mouse | yes | yes | yes | yes | PASS |
| R4 / Reset at 1M ≤ 2 s (median, op → first new step) | — | 0.123 s | — | 0.127 s | PASS |
| Reset faster than Stop + Start (median) | 0.083 vs 0.088 s | 0.123 vs 0.150 s | 0.066 vs 1.138 s | 0.127 vs 1.158 s | PASS |
| Reset applies the pending edits | yes | yes | yes | yes | PASS |
| Gravity: next step's mean Δv vs expected (`substeps·dt·Δg`) | 0.993 | 0.997 | 0.993 | 0.997 | PASS (±5%) |
| Gravity: exactly one graph recapture | 1 | 1 | 1 | 1 | PASS |
| Eager vs CUDA-graph step, max abs velocity difference | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| Undo/redo of a transform, a role change, a delete keeps the run | yes | yes | yes | yes | PASS |
| Change handling, 32 objects moved every 20 ms: p95 per tick | 0.75 ms | 0.56 ms | 0.62 ms | 0.58 ms | PASS (≤ 1 ms) |
| Throughput, G-grab drag ≥ 90% of untouched (original gate) | 84% | 75% | 87% | 66% | FAIL → restated |
| Throughput, gravity drag (50 Hz) ≥ 90% of untouched (original gate) | 65% | 78% | 104% | 64% | FAIL → restated |

Change-handling mean per tick: 0.09–0.25 ms; max 0.85–1.6 ms. Untouched steps/s drift by up to 10% between two
windows of the same run (e.g. OpenGL 1M: 85.0 then 77.0), so single-window ratios are noisy. The attribution runs
below use three interleaved repeats and medians.

Screenshots: `full-990b11c/*-1000000/editing_32_objects.png` and `after_grab.png`.

## Attribution of the throughput failure

`diag1/` — the same edits with the add-on's change handling **on** and **off** (the tracker sees nothing: no change
detection, no live gravity). Median ratio to the untouched window just before each, 3 repeats:

| | grab on | grab off | gravity on | gravity off |
|---|---|---|---|---|
| OpenGL 1M | 0.827 | 0.838 | 0.864 | 0.851 |
| Vulkan 1M | 0.701 | 0.687 | 0.688 | 0.679 |
| OpenGL 100k | 0.889 | 0.951 | 0.799 | 0.788 |
| Vulkan 100k | 0.907 | 0.960 | 0.918 | 0.942 |

Restated gate (on ÷ off): grab 0.987 / 1.020 / 0.935 / 0.945 and gravity 1.015 / 1.013 / 1.014 / 0.975 (rows in
table order): all ≥ 0.90, PASS.

At 1M, turning change handling off does not recover throughput. At 100k it accounts for a few percent at most,
within the noise.

`diag2/` (1M) — no scene edit at all: **orbiting the view** at 60 Hz, with the particles drawn and hidden:

| | grab, shown | grab, hidden | orbit, shown | orbit, hidden |
|---|---|---|---|---|
| OpenGL 1M | 0.826 | 0.910 | 0.830 | 0.942 |
| Vulkan 1M | 0.690 | 0.777 | 0.630 | 0.750 |

Orbiting, which edits nothing, loses as much as grabbing or more. Not drawing the particles recovers part of the loss.

Conclusion: the shortfall comes from extra viewport redraws (Blender's own redraw work, plus the add-on's particle
draw on each one) competing with the simulation on the UI thread and the GPU. It is not P2.9 change detection,
classification or live gravity. Meeting the gate as written needs the simulation's stepping decoupled from viewport
redraws, which is an architecture change outside Phase 1's listed scope. The owner chose to restate the gate and track the decoupling separately.

## Fixed during this qualification

- `live_edits`: a touched object keeps its last shape token (bounds) unless Blender flagged its geometry, since a
  move cannot change a mesh. Change handling for 32 dragged objects went from 1.65 to 0.54 ms mean per tick
  (background profile), and from a 1.29–1.76 ms p95 to 0.56–0.75 ms in the live runs. The regression test is
  `test_a_drag_tick_does_not_re_read_bounds_but_a_geometry_update_does`.
- The early R1 and role-undo "failures" were harness errors: `bpy.ops` called from Python pushes no undo step
  unless `undo=True`. With undo pushed as the UI does, both pass.

## Not covered

- Save / reopen / bake with live edits pending, and a video capture for the launch kit (screenshots only).
- SSFR / Spheres (OpenGL legacy VBO route): only Points was measured.
- Other GPUs.

## Reproduce

```
python scripts/run_p29_runtime_gates.py --blender <blender 5.2 exe> --backends opengl,vulkan --particles 100000,1000000
P29_STAGES=throughput_diag python scripts/run_p29_runtime_gates.py --blender <exe> --particles 1000000,100000
P29_STAGES=redraw_diag python scripts/run_p29_runtime_gates.py --blender <exe> --backends vulkan,opengl --particles 1000000
```
