# Live collider drag: spec (P2.9)

Status: **proposed, not started** (2026-10-22). Backlog item P2.9 in [`RELEASE_PLAN.md`](RELEASE_PLAN.md).

## Problem

While a simulation runs, the fluid only follows colliders that are *animated* (keyframes, drivers, constraints
or an animated parent). A collider that is not animated is fixed at its Start pose. Grabbing it with **G** or
the gizmo while the sim runs does nothing until **Stop** + **Start**. Grabbing an animated collider is
undone on the next frame, because `sync_collider_keyframes` re-evaluates its animation with `scene.frame_set`.

Users expect to push the liquid around by hand. It fits the product pitch, "art-direct it live, then bake
exactly that", and it makes the strongest demo.

## Goal

A collider that is not animated follows its live world transform while the simulation runs. The fluid feels
both its position and its velocity, so dragging a paddle through water pushes the water.

## Non-goals (v1)

- Live edits to the fluid, the domain or emitters (still Stop + Start).
- Deforming colliders (edit mode, modifiers, armatures, shape keys) and live scale or shear. The mesh stays the
  Start-frame shape, as it does for animated colliders now.
- Dragging an *animated* collider. Its animation stays authoritative.
- Live input during a **bake**. A bake must be reproducible, so live colliders hold the pose they had when the
  bake started (see Phase B for recording).
- Two-way coupling. The fluid does not push the collider.

## How it works now (main, 2026-10-22)

- `description_from_blender` (`scene/blender_scene.py`) gives each animated collider a `ColliderMotion`
  holding its Start world matrix. Every collider mesh is stored in world space at the Start frame.
- The solver (`sim/sim_core.py`) keeps a ring of `KEY_RING = 8` per-frame poses (`_key_p`, `_key_q`), indexed by
  scene frame. `collider_frames_needed` lists the frames the next step needs, `set_collider_keyframe` writes
  one, and `_update_colliders` interpolates between two keys every substep. `k_collider_motion` writes
  `collider_xform`, `collider_lin` and `collider_ang`, and the collision kernels read them.
- `SimSession.tick` and the bake loop call `sync_collider_keyframes(scene, solver)` before stepping.
- A collider that is not animated has no motion entry. It keeps its identity pose and zero velocity.

So the collision side already takes moving SDFs with velocity. Only the source of the poses is missing.

## Design

### User-facing

- A new per-object setting **Follow While Running** (`warp_sph.live_follow`) on Collider objects. It
  defaults to **on**, and the panel shows it only for colliders that are not animated. For an animated
  collider it reads "Animated: follows its animation".
- A new scene setting **Max Collider Speed** in the Advanced sub-panel. It is in scene units per second,
  defaults to automatic, and caps how fast a dragged collider moves through the fluid (see below).
- The manual's limits page and authoring page drop "edits take effect on the next Stop + Start" for collider
  transforms and document the cap.

### Scene seam (`scene/`)

- `description_from_blender` also gives a `ColliderMotion` to every collider with `live_follow` on. A new
  field, `ColliderMotion.source`, is either `'ANIMATION'` or `'LIVE'`. The mesh, the pivot and the
  `rigid_key` maths are unchanged.
- A new function, `sync_live_colliders(scene, solver)`, runs on the main thread each tick. It reads
  `matrix_world` for the LIVE colliders without calling `frame_set`, and passes the matrices to the solver.
  Scale or shear is rejected through `rigid_key` as it is today. For live input, the user gets a one-time
  warning and that collider stops following, rather than the run stopping.

### Solver (`sim/sim_core.py`)

The solver gets a second pose source for the LIVE colliders. The ring of frame keys is left unchanged:

- `set_collider_live_pose(matrices)` stores a *target* pose per LIVE collider, for the end of the next
  `step()`.
- Inside `step()`, LIVE colliders move from their current pose towards the target. Their velocity is
  `(target - current) / (substeps * dt)`, and the angular velocity comes from the relative quaternion.
  `_update_colliders` writes LIVE and ANIMATION colliders into the same `collider_xform`, `collider_lin`
  and `collider_ang` arrays, so the collision kernels do not change.
- **Speed cap.** The motion per substep is clamped to `max_disp = 0.5 * h` (translation) and
  `max_rot = 0.25 rad`. The effective velocity is therefore at most `0.5 * h / dt`, and it is also limited by
  the user's Max Collider Speed when that is set. A drag faster than the cap makes the collider lag and catch
  up over the following steps. This guards against two failures:
  - **Tunnelling.** A thin collider jumps past particles and traps them inside the SDF.
  - **Velocity spikes.** A teleport would inject a huge collider velocity and blow up the sim.
- **Pause.** While the sim is paused, poses are not consumed. On resume, the collider catches up at the
  cap. Teleporting would trap particles.
- **CUDA graph.** `_update_colliders` already runs outside the captured graph, before the per-substep force
  graph, and writes into fixed arrays. The LIVE path keeps that pattern: no allocation, sync or print
  during capture (an AGENTS.md invariant). The live pose upload is one small host-to-device write per tick,
  which is a tiny pose array, not a particle readback.

### Session (`sim_session.py`)

- `SimSession.tick` calls `sync_live_colliders` next to `sync_collider_keyframes` for Custom scenes.
- The bake loop does **not** call it. At bake start, live colliders are keyed with their current pose as a
  constant.
- Undo or redo during a run must not leave a stale pose. The next tick simply reads `matrix_world` again.

## Open risk to check first (spike S0)

**Does the sim keep ticking while a G-drag (the modal transform operator) is active?** If Blender holds back
the session's timer events during modal transform, the sim freezes during the drag and only reacts on release.
The spike checks this on Blender 5.2, on OpenGL and Vulkan:

1. Start a Custom scene with one collider. Press G and drag for 3 seconds.
2. Record the steps/s during the drag and `matrix_world` per tick.

If ticks stop during the drag, the fallbacks are:

1. Drive stepping from a `bpy.app.timers` callback that does not depend on modal event routing.
2. If that also stalls, accept "reacts on release" for v1 and document it.

Do not change the 1 ms modal timer (an AGENTS.md invariant).

## Phases

| Phase | Scope | Size |
|---|---|---|
| S0 | Spike: tick during modal transform, and `matrix_world` freshness, on both backends | S (1 day) |
| A | Live follow for colliders that are not animated: setting, scene seam, solver pose source, speed cap, session wiring, docs, tests, live demo | M |
| B | **Record live drags as keyframes**: an option that keys the collider's live pose every frame during the run, so a later bake replays exactly what the user did. This makes "art-direct it live, then bake exactly that" literal | S–M |
| C (later) | Live emitter transforms; dragging animated colliders via a temporary override | M |

## Acceptance (Phase A)

Unit tests (GPU-free where possible, with the solver tests on `cuda:0` where they already are):

- A LIVE target pose moves the collider exactly to the target within one step when under the cap. `collider_lin`
  equals displacement / step time.
- A teleport larger than the cap is clamped per substep and converges over later steps. No substep moves
  more than `0.5 * h`.
- Pause, then a moved pose, then resume: the collider catches up at the cap, with no velocity spike.
- A live scale change produces a single warning and the collider stops following. The run continues.
- The bake path ignores live input. A bake with a moved live collider is bit-identical to a bake with it still.
- An ANIMATION collider is unaffected. The existing animated-collider tests pass unchanged.
- No allocation inside graph capture. The existing graph tests pass.

Live Blender proof (one Blender at a time, isolated profile, as in the existing live harnesses):

- A scripted drag of a paddle collider through a resting pool at 100k and 1M particles, on OpenGL and Vulkan.
  Measure the ID-matched near-paddle velocity response against a still control, using the paired method of the
  showcase paddle check.
- No particles end up inside the collider. Count particles whose SDF is negative inside the collider after the
  drag, and require zero beyond the existing contact tolerance.
- Steps/s during the drag stays within 5% of a run where nothing moves.
- A screen capture of the drag, for the launch kit.

## Files touched (expected)

`scene/description.py`, `scene/blender_scene.py`, `props.py` and the collider panel UI, `sim/sim_core.py`,
`sim_session.py`, `docs/manual/authoring.md`, `docs/manual/limits.md`, `CHANGELOG.md`, and new tests in
`tests/`.
