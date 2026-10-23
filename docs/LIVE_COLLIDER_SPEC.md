# Live colliders: component spec (P2.9)

Status: **proposed, not started** (2026-10-23). This is the collider part of the interactive simulation workflow,
[`INTERACTIVE_SIM_SPEC.md`](INTERACTIVE_SIM_SPEC.md), backlog item P2.9 in [`RELEASE_PLAN.md`](RELEASE_PLAN.md).
That spec owns the shared foundation (change detection, identity, classification, Reset) and the overall phases:
collider phase 1 below is workflow phase 2, collider phases 2–3 are workflow phase 4, and collider phase 4 is part
of workflow phase 5 (takes).

## The feature

**Whatever you do to a collider while the simulation runs, the fluid reacts to it on the next step.** Move it,
rotate it, scale it, reshape it, re-animate it, add it, delete it: the result is what Stop + Start would have built,
without stopping, and the fluid feels the motion, so a paddle swept through water pushes the water.

This fits the product pitch, "art-direct it live, then bake exactly that". Today only one kind of collider change
reaches a running simulation: playback of its animation. Everything else needs **Stop** + **Start**.

## What changes for the user

| Change made while the sim runs | Today (main, 2026-10-23) | With live colliders | Phase |
|---|---|---|---|
| Move or rotate a still collider: G / R, gizmo, typed values in the N panel, snapping, Python, undo / redo | Ignored until restart | Followed; the fluid feels its velocity | 1 |
| Move or rotate an **animated** collider by hand | Snaps back on the next frame | Your hand wins while you hold it; the animation takes over again on release | 1 |
| Edit a collider's animation: keys, curves, drivers, constraints, parent | Up to 8 frames of old poses still used | The next step uses the edited animation | 1 |
| Rename a collider | Stops the run (the sync looks objects up by name) | Keeps working | 1 |
| Delete a collider, unmark it, disable it in viewports, exclude its collection | Delete stops the run; the others are ignored | It leaves the simulation | 2 |
| Mark a new collider, duplicate one, append one | Ignored until restart | It joins the simulation | 2 |
| Scale a collider, by hand or animated | Hand: ignored. Animated: stops the run | Followed | 3 |
| Change its shape: Edit Mode, modifiers, Geometry Nodes, shape keys, armatures, by hand or animated | Frozen at the Start shape | Followed, and the fluid feels the surface's own motion | 3 |
| Bake | Rigid animation only | Every animated change above bakes; live hand input can be recorded as keys first | 3, 4 |

Live edits to the domain, fluid volumes and emitters are in [`INTERACTIVE_SIM_SPEC.md`](INTERACTIVE_SIM_SPEC.md).
Out of scope: two-way coupling (the fluid pushing the collider back) and the built-in demos' analytic obstacles.

## Rules the feature keeps

1. **Same answer as a restart.** Once a change settles, the solver holds the mesh and pose that
   `description_from_blender` would build at that moment. Only the particles differ, because they were already
   there.
2. **No explosions.** Any collider motion is capped per substep (translation `0.5 * h`, rotation `0.25 rad`, any
   vertex `0.5 * h`). A faster change makes the collider lag and catch up. The cap prevents two failures:
   - **Tunnelling:** a thin collider jumps past particles and traps them inside.
   - **Velocity spikes:** a teleport injects a huge collider velocity into the fluid.

   Particles found inside a collider are pushed out at the same capped rate, never in one jump. This matters
   because the current hard projection would move a whole buried volume to the surface in one substep and crush
   it.
3. **Bakes stay reproducible.** A bake reads only the scene's animation. Live hand input never reaches a bake;
   recording turns it into animation first (Phase 4).
4. **Authoring mistakes never kill a run.** If a collider becomes invalid mid-run, it freezes at its last valid
   shape and pose and the panel says why. Examples: an open mesh, a non-mesh object, a mirrored or zero-volume
   shape. GPU and CUDA failures stay fatal, as today.
5. **The interface stays responsive.** Change detection is cheap and runs every tick. Expensive work, mesh
   evaluation and BVH builds, runs only when Blender reports that a collider's geometry changed. Its cost is
   measured and shown.
6. **The CUDA graph invariants hold.** There is no allocation, host sync or print inside capture. A change in the
   number of colliders recaptures the graph through the existing `rebuild_cuda_graph()` path, the same path the
   physics sliders use. A swapped mesh is a new id written into the existing id array, which the graph reads by
   value.
7. **The 1 ms modal timer is not changed** (an AGENTS.md invariant).

## How it works now

- `description_from_blender` (`scene/blender_scene.py`) reads every collider once, at the Start frame, as a
  world-space `MeshData`. Animated colliders also get a `ColliderMotion` (start matrix, pivot).
- `MeshSet` (`sim/colliders.py`) builds one `wp.Mesh` per collider, once. Animated colliders are stored relative
  to their pivot.
- The solver keeps a ring of `KEY_RING = 8` per-frame poses for the animated colliders only. Each tick,
  `sync_collider_keyframes` calls `scene.frame_set` for the frames the next step needs and reads `matrix_world`
  by object **name**. Every substep, `_update_colliders` launches `k_collider_motion`, outside the graph, to
  interpolate into `collider_xform`, `collider_lin` and `collider_ang`. The captured kernel reads those device
  buffers plus `collider_ids` and the **captured value** `num_colliders`.
- Collision applies a soft wall law plus a hard projection to the surface, using the closest point and a
  winding-number check for deep "inside" answers. Collider velocity is rigid only: `lin + ang × r`.

## Design

### 1. Change detection (session, main thread)

- **Transforms** are polled every tick: one `matrix_world` read per live collider. This is cheap, and it catches
  every source: modal transforms, typed values, drivers, Python and undo.
- **Geometry, structure and animation edits** come from a `depsgraph_update_post` handler that is registered only
  while a Custom run is active. It records dirty flags per collider (`is_updated_geometry`, animation-data
  updates) and set changes: role property edits, objects added or removed, visibility. It does no GPU work; the
  tick consumes the flags.
- **Self-caused updates are ignored.** The handler is muted while the session itself calls `frame_set`.
  Otherwise every animated frame read would look like a user edit.
- **Identity by object, not name.** Colliders are tracked by `session_uid`, falling back to the object pointer.
  Names are kept for display only, so renames are harmless.

### 2. Pose sources (solver)

Every collider gets a pose slot, not just the animated ones. Every mesh is stored relative to its pivot, so a
still collider is just a slot that is not moving; its query cost is unchanged. Each slot has one source per tick:

| Source | Used for | Pose for the next step |
|---|---|---|
| `ANIMATION` | Animated colliders, untouched | Key ring interpolation, as today; **not** capped, so existing animation and bakes behave exactly as now |
| `LIVE` | Still colliders | Latest `matrix_world` as the target; capped approach |
| `HELD` | Animated collider while the user is transforming it | Latest `matrix_world`; capped; frame keys for it are not read |
| `RETURN` | Animated collider just released | Animation pose as the target, capped until it is reached, then back to `ANIMATION` |

`k_collider_motion` becomes one launch over all slots. Per slot it computes the target, clamps the step to the
cap, writes the pose, and writes the velocity from the motion actually applied (`displacement / dt`, angular
from the relative quaternion), so a capped collider never reports more velocity than it moved. The new host API
replaces the frame-key setter for live data: `set_collider_targets(slot_matrices)` is one small host-to-device
write per tick. The existing `collider_frames_needed` / `set_collider_keyframe` keep serving `ANIMATION`.

**Detecting HELD.** An animated collider is HELD while a transform modal operator acting on it is running.
`Window.modal_operators` is the candidate signal, to be verified on 5.2 (risk R3). The fallback is: the polled
`matrix_world` differs from the pose the animation gives for the current frame. While any collider is HELD,
the session does not call `frame_set` for that tick, so it never fights the user's transform (risk R2). The
other animated colliders hold their last keys for that tick. Release with auto-keying on inserts a key, and the
animation then agrees with the hand, so RETURN is instant.

**Animation edits** drop that collider's cached keys after the current frame (`invalidate_collider_keys(slot)`),
and the next sync re-reads them.

**Pause.** No targets are consumed while paused. On resume, live colliders catch up at the cap.

### 3. Shape changes (Phase 3)

A collider whose geometry changed (dirty flag) or that deforms by animation (armature, shape-key action,
animated modifier or Geometry Nodes input) moves to the **shape path**:

- The evaluated mesh is re-read with the existing `_world_mesh` maths in pivot-local metres. Scale and shear move
  into the vertices, so the rigid path's scale rejection no longer stops a run; it routes the collider here.
- **Same topology:** new points go into the existing `wp.Mesh` (`points`, then `refit()`). Per-vertex
  velocities go into the mesh's `velocities` array, and collision adds `mesh_eval_velocity` at the closest
  point to the rigid velocity, so a growing or flexing collider pushes the fluid. If any vertex would move more
  than the cap in one substep, the points are interpolated on the GPU over several substeps, with a refit each
  substep.
- **Topology changed:** a new `wp.Mesh` is built outside the graph and its id is written into `collider_ids`. The
  old mesh is kept alive until the last step that used it has finished. There is no surface velocity across a
  topology change, so overlap is handled by the capped push-out (rule 2). The closed-mesh validation runs only
  here, because a same-topology update cannot open a closed mesh.
- **Throttle.** At most one re-evaluation per collider per tick, within a per-tick budget. Over budget, shape
  updates for that collider drop to every Nth tick, and the panel shows "shape updates throttled".
- **Animated deformation** is sampled per scene frame alongside the pose keys, into a points ring, so bakes
  follow it too. Memory cost is at most `KEY_RING × vertices × 12 bytes` per deforming collider.

### 4. Set changes (Phase 2)

- **Joins:** marking an object Collider, duplicating or appending one, re-enabling it, or including its
  collection. The mesh is read and validated, then gets a slot and a `wp.Mesh`; `num_colliders` changes, so the
  graph is recaptured once. An invalid new collider is not added, and the panel says why.
- **Leaves:** deleting the object, setting role None, disabling it in viewports (`hide_viewport`), or excluding
  its collection. Its slot is freed, and its mesh is released after the in-flight step. Hiding it with the eye
  icon only (`hide_get`) keeps it colliding, as in Blender's own physics.
- A new collider placed over fluid expels the buried particles at the capped rate (rule 2). The fluid visibly
  "boils out" over a few steps instead of exploding.

### 5. Settings and panel

- Per collider: **Live** (`warp_sph.live_follow`, default on). Off means hand edits are ignored and the collider
  behaves as it does today.
- Scene, Advanced sub-panel:
  - **Max Collider Speed**: scene units per second. The default is automatic, the cap above.
  - **Live Shape Updates**: On or Off, default On. The per-tick budget is shown when it is throttling.
  - **Record Live Colliders** (Phase 4).
- Status lines appear only when something is abnormal: "Paddle: lagging (speed cap)", "Rock: frozen, mesh is not
  closed", "Blob: shape updates throttled". The "Stop and Start to apply scene changes" hint is no longer shown
  for collider edits; the workflow spec's classification decides what every other edit shows.
- **During a bake** the panel says "Live collider edits are ignored while baking".

### 6. Bake and record (Phase 4)

- The bake loop never reads live input. Colliders follow animation only, including animated deformation once
  Phase 3 lands. A bake with a collider dragged mid-bake is bit-identical to one without the drag.
- **Record Live Colliders** keys location, rotation and scale of LIVE and HELD colliders once per scene frame the
  run advances, with Blender's keyframe API on the main thread, so a later bake replays the take. Live shape edits
  are not recorded in v1: to bake a deformation, animate it.
- Resumed bakes: collider edits since the cache was written are still not fingerprinted (see #21); this spec does
  not change that.

## Risks to check first (S0, inside Phase 1)

| # | Question | If the answer is bad |
|---|---|---|
| R1 | Do session ticks continue during a modal G / R drag, on OpenGL and Vulkan? | Step from a `bpy.app.timers` callback; if that also stalls, ship "reacts on release" for v1 and document it |
| R2 | Does `scene.frame_set` during a modal transform disturb the transform? | Skip `frame_set` while any transform modal is running, not only while a collider is held |
| R3 | Is `Window.modal_operators` available and reliable on 5.2 for detecting HELD? | Use the matrix-differs-from-animation fallback |
| R4 | Does `wp.Mesh.refit()` update the winding-number data (`support_winding_number=True`) in Warp 1.17.0? | Rebuild the mesh on each shape change, or confirm deep "inside" with a different check |
| R5 | What do `evaluated_get().to_mesh()` and a refit cost at 10k and 100k triangles? | Set the throttle defaults from this, recorded before Phase 3 starts |

## Phases

| Phase | Scope | Size |
|---|---|---|
| 1. Live transforms | S0; pose slots for every collider; LIVE / HELD / RETURN sources; speed cap; animation-edit invalidation; identity by `session_uid`; Live setting; Max Collider Speed; docs; live demo | M |
| 2. Live set | Join and leave mid-run; capped push-out of buried particles; one graph recapture per set change | S–M |
| 3. Live shape | Scale, Edit Mode, modifiers, Geometry Nodes, shape keys, armatures, by hand and animated; surface velocity; throttle; deforming colliders in bakes | L |
| 4. Record | Record Live Colliders as keys; bake replays the take | S–M |

Each phase ships on its own and keeps every earlier test passing.

## Acceptance

Measured gates. These are fixed now and are not changed after measurement.

**Phase 1**
- A LIVE target within the cap is reached in one step, and `collider_lin` equals the displacement divided by the
  step time.
- A teleport beyond the cap is clamped every substep: no substep moves more than `0.5 * h` or `0.25 rad`.
- The collider converges on the target, and the reported velocity never exceeds the applied motion.
- HELD: an animated collider follows the hand while held. After release it returns to the animation at the cap,
  then matches the animation exactly. `frame_set` is not called while it is held.
- Editing a key two frames ahead of the current frame changes the pose the solver uses at that frame.
- Renaming a collider mid-run, and undo / redo of a transform, keep the run going with the correct pose.
- Untouched animated colliders and every existing collider test are bit-identical to main.
- Live Blender, one Blender at a time: a scripted paddle drag through a resting pool at 100k and 1M particles,
  on OpenGL and Vulkan.
  - The near-paddle velocity response is ID-matched against a still control, using the paired method of the
    showcase paddle check.
  - After the drag, zero particles are inside the collider beyond the contact tolerance.
  - Steps/s during the drag are at least 95% of the still run.
  - The UI thread spends at most 0.3 ms per tick on collider sync with 32 live colliders.
- A screen capture of the drag, for the launch kit.

**Phase 2**
- Each set change causes exactly one graph recapture, and that tick takes at most 50 ms at 1M particles.
- A collider added over a resting pool expels the buried particles with no particle faster than the speed cap.
- The fluid's peak density settles back within 10% of rest within 1 s of simulated time.
- Deleting a collider mid-run frees its slot, and its mesh is released only after the in-flight step.
- An invalid added collider is refused with a panel message, and the run continues.

**Phase 3**
- Scaling a collider by hand matches, after settling, the mesh that Stop + Start builds: vertices equal within
  float32 tolerance.
- A sphere growing by 2× over 1 s inside the fluid pushes the fluid outward, measured by mesh velocity. No
  particle ends up inside it, and no substep moves a vertex more than `0.5 * h`.
- A 10k-triangle collider deforming every tick at 1M particles keeps steps/s at 80% or more of the rigid run;
  shape work costs at most 2 ms of UI thread per tick.
- A 100k-triangle collider throttles instead of stalling, and the panel says so.
- An armature-deformed collider bakes, and two bakes are bit-identical.

**Phase 4**
- A recorded drag, then baked, is bit-identical to a second bake of the same keys.
- The recorded keys reproduce the live poses within float32 tolerance at each recorded frame.

## Files touched (expected)

- `scene/description.py`, `scene/blender_scene.py`: identity by `session_uid`, a per-collider pivot for all
  colliders, the shape path reader.
- `sim/colliders.py`: mesh slots, swap and deferred release, velocities.
- `sim/sim_core.py`: pose sources, cap, push-out cap, collider count recapture. These edits need owner go-ahead
  per the standing rule on `sim_core.py`.
- `sim_session.py`: depsgraph handler, tick sync, held detection, bake isolation, recording.
- `props.py`, `addon.py`: settings and status lines.
- `docs/manual/authoring.md`, `docs/manual/limits.md`, `CHANGELOG.md`.
- New tests in `tests/`.
