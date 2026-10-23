# Interactive simulation workflow: spec (P2.9)

Status: **proposed, not started** (2026-10-23). Backlog item P2.9 in [`RELEASE_PLAN.md`](RELEASE_PLAN.md).
The collider part is specified in detail in [`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md).

## The update

This turns the add-on from **set up, Start, watch, Stop, edit, Start again** into a fully interactive simulation
workflow. The simulation keeps running while you build the shot. You edit anything in the scene, and every edit
has a visible, defined outcome. When the take looks right, you bake exactly that.

In industry terms this is an **interactive simulation** with **kinematic (one-way coupled) colliders and
sources**. EmberGen's "edit while it simulates" is the reference experience. The user-facing name is
*interactive mode*.

The workflow loop:

```
 author ──► run ──► tweak anything live ──► rewind / reset ──► record the take ──► bake exactly that
              ▲                                   │
              └───────────────────────────────────┘
```

## Rules

1. **Every edit has one of three outcomes, and the panel always says which.**
   - **Live:** it reaches the fluid on the next step.
   - **Reset to apply:** one click restarts at the start frame with the edit, without tearing down the GPU
     runtime.
   - **Not while baking:** it is ignored during a bake, and the panel says so.

   No edit is ever silently ignored.
2. **Same answer as a restart.** Once a live edit settles, the solver holds exactly what Start would build
   from the scene at that moment. Only the particles differ, because they were already there.
3. **No explosions.** Anything that moves through the fluid, or appears inside it, does so at a capped rate.
   This covers colliders, emitters and domain walls. Displacement per substep is at most `0.5 * h`, and particles
   are pushed out of new solids at the same rate, never in one jump.
4. **Authoring mistakes never kill a run.** An invalid object freezes or pauses at its last valid state, with a
   panel message, and the run continues. GPU and CUDA failures stay fatal, as today.
5. **Bakes stay reproducible.** A bake reads only the scene's animation and settings. Live input reaches a bake
   only by being recorded first (Phase 5).
6. **The interface stays responsive.** Change detection is cheap and runs every tick. Expensive work runs only
   when Blender reports that the relevant data changed, and its cost is shown when it throttles.
7. **Existing invariants hold.**
   - No allocation, host sync or print inside CUDA graph capture.
   - Value changes recapture through `rebuild_cuda_graph()`.
   - The 1 ms modal timer is unchanged.
   - Warp stays pinned to 1.17.0.

## Every edit, today and target

"Restart" means Stop + Start. The **Phase** column is the phase that delivers the target.

| Edit while running | Today (main, 2026-10-23) | Target | Phase |
|---|---|---|---|
| Stiffness, time step, viscosity, substeps, fluid preset | Live | Live (unchanged) | — |
| Display: render mode, point size, colours, filter | Live | Live (unchanged) | — |
| Pause, Step One Frame | Yes | Yes (unchanged) | — |
| **Gravity** (Scene > Gravity, on/off) | Restart | Live | 1 |
| Particle spacing, Max Particles, unit scale | Restart | Reset to apply (one click) | 1 |
| Fluid object moved, reshaped, added or removed | Restart | Reset to apply; plus **Add Fluid Now**, which pours the object's volume in live | 1, 3 |
| Collider moved or rotated, still or animated; collider animation edited; collider renamed | Animation playback only; rename stops the run | Live | 2 |
| **Emitter** moved, rotated or animated | Restart | Live: the nozzles follow, and the emitter's own motion adds to the jet | 3 |
| Emitter speed, direction, rate, frame range | Restart | Live | 3 |
| Emitter reshaped, added or removed | Restart | Live: nozzles re-sampled from the new shape | 3 |
| **Domain** moved or resized | Restart | Live: walls move at the cap; fluid outside a shrinking box is herded in, not deleted | 3 |
| Collider added, removed, hidden, scaled or reshaped (Edit Mode, modifiers, Geometry Nodes, shape keys, armatures) | Restart or ignored | Live | 4 |
| Undo / redo of any of the above | Untested (known live gap) | Same as making the edit by hand | 1 |
| Scrub the timeline while running | Not defined | Pauses and rewinds to the nearest snapshot | 5 |
| Keep this take | Not possible | **Record** inputs as keys for a reproducible bake, or **Capture** the live run straight to the cache | 5 |

Out of scope: two-way coupling (the fluid pushing objects back), the built-in demos' analytic obstacles,
multi-GPU, and live solver-type changes.

## Design

### Shared foundation (Phase 1)

- **Change detection.** A `depsgraph_update_post` handler is registered only while a run is active. It records
  dirty flags per object and per kind (transform, geometry, settings), set changes (roles, objects added or
  removed, visibility) and scene changes (gravity, units). It does no GPU work; the tick consumes the flags.
  Self-caused updates, for example the session's own `frame_set`, are muted. Transforms of live objects are also
  polled every tick, so no source of motion is missed.
- **Identity by object.** Every object is tracked by `session_uid`, not by name. After undo / redo, an undo
  handler re-resolves the objects; an object that no longer resolves is treated as removed (risk R1).
- **Change classification.** One table in code maps every detected change to Live, Reset to apply or Not while
  baking. Tests and the panel read the same table, so the UI cannot drift from behaviour.
- **Reset.** This is a new operator. It re-reads the scene, rebuilds the solver state, re-fills and returns to the
  start frame, while keeping the Warp runtime, the CUDA context and the render transport alive. When anything
  "Reset to apply" is pending, the panel shows "Reset to apply: particle spacing changed" with the button.
- **Live gravity.** Gravity is a kernel value. A change recaptures the graph through the existing physics-slider
  path.

### Colliders (Phases 2 and 4)

See [`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md) for the full design:
- pose slots for every collider, with LIVE, HELD, RETURN and ANIMATION sources;
- the speed cap and capped push-out of buried particles;
- set changes with one graph recapture each;
- the shape path: mesh refit with surface velocity, plus a throttle;
- deforming colliders in bakes.

### Emitters (Phase 3)

Today `_setup_emitters` samples nozzle positions once, at Start, into fixed device arrays.
- **Transform.** Nozzles are stored relative to the emitter's pivot and placed by its transform, which is
  interpolated per substep as for colliders. The released particle gets the nozzle velocity rotated with the
  emitter, plus the emitter's own surface velocity. Waving a hose sprays an arc.
- **Settings.** Speed, direction, rate and frame range are host-side schedule values, applied on the next tick.
  The rate clamp against the natural rate is re-derived.
- **Shape, add and remove.** Nozzles are re-sampled outside the graph into arrays sized for a nozzle capacity, so
  the graph keeps its shape. If a new set exceeds the capacity, the arrays are reallocated, followed by one
  recapture.
- **Clearance.** An emitter moved into a wall or collider pauses with a panel message ("Hose: inside a collider,
  paused") and resumes when it is clear. It does not stop the run, as Start validation does today.
- **Pool.** The particle pool stays fixed. "Particle budget reached: emitters stopped" behaves as today.

### Domain (Phase 3)

`domain_min` / `domain_max` are kernel values: a change recaptures the graph. Walls move toward the new box at
the cap, so a shrinking box herds fluid instead of teleporting it through the hard clamp. If the fluid no longer
fits at rest density, the panel warns "Domain too small for the fluid". The fluid then compresses rather than
being deleted. A rotated domain still uses its axis-aligned bounding box, as today.

### Fluid volumes (Phases 1 and 3)

Fluid objects are the **initial condition**, so moving or reshaping one applies on Reset. **Add Fluid Now** is a
new operator that pours the selected Fluid object's volume in live. It activates pool slots on the object's fill
lattice, skipping points within one spacing of existing fluid or solids. It uses the same capped insertion as
emitters, so new fluid never lands on old fluid.

### Rewind (Phase 5)

A ring of GPU snapshots taken every N frames holds positions, velocities, ids, the active count, the clock,
emitter progress and collider state. It reuses the cache `restore` path in `cache/gpu.py`. The default budget is
10% of free VRAM. At 1M particles that is about 28 MB per snapshot, so a 1 GB budget holds 36 snapshots.

Scrubbing the timeline while running pauses the run and restores the nearest snapshot at or before that frame.
Resuming branches from there, and later snapshots are dropped.

### Takes (Phase 5)

There are two ways to keep a take, because they answer different needs:

- **Record (reproducible).** While recording, every live input is keyed once per scene frame the run advances:
  transforms of colliders, emitters and the domain, emitter settings, and gravity. A later bake replays the take
  bit-for-bit from the keys. Live shape edits are not recorded; to bake a deformation, animate it.
- **Capture (exact).** The live run itself is written to the cache as it plays, using the bake job's host-copy
  ring. Resume and rebake are disabled, because the inputs were not recorded. This is the cache's own host copy,
  not a display readback, and its cost per frame is measured.

During a bake, all live input is ignored, and the panel says "Live edits are ignored while baking".

## Risks to check first

| # | Question | If the answer is bad |
|---|---|---|
| R1 | Is `session_uid` stable across memfile undo / redo on 5.2? | Re-resolve by name plus type at the undo handler, and treat a mismatch as removed then added |
| R2 | Do session ticks continue during modal transforms (G / R / S, gizmo), on OpenGL and Vulkan? | Step from `bpy.app.timers`; if that also stalls, ship "reacts on release" for v1 and document it |
| R3 | Does `scene.frame_set` during a modal transform disturb it? | Skip `frame_set` while any transform modal is running |
| R4 | How fast is Reset compared with Stop + Start at 1M particles? | If Reset is not clearly faster, find the cost before Phase 1 ships |
| R5 | Plus the collider risks in [`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md) | Covered there |

## Phases

| Phase | Scope | Size |
|---|---|---|
| 1. Foundation | Change detection, identity by `session_uid`, classification table and panel hints (no silent ignores), Reset, live gravity, undo / redo safety; risks R1–R4 | S–M |
| 2. Live collider transforms | Collider spec phase 1 | M |
| 3. Live emitters, domain and Add Fluid Now | Emitter transform, settings, shape and set changes; domain move / resize; live fluid pour | M |
| 4. Live collider set and shape | Collider spec phases 2–3 | L |
| 5. Takes and rewind | Record inputs as keys, Capture to cache, the snapshot ring with timeline scrub | M–L |

Each phase ships on its own, behind no flag, and keeps every earlier test passing.

## Acceptance

These gates are fixed now and are not changed after measurement. The collider gates are in
[`LIVE_COLLIDER_SPEC.md`](LIVE_COLLIDER_SPEC.md).

**Phase 1**
- Every row of the edit table above is covered by a test that makes the edit mid-run and checks the declared
  outcome. A change missing from the classification table fails the test.
- A gravity change is reflected in the next step's acceleration, with exactly one graph recapture.
- At 1M particles, Reset takes at most 2 s and is faster than Stop + Start on the same scene. Both times are
  recorded.
- Undo / redo of a transform, a role change and a delete mid-run keeps the run going, with the pose and set
  that the scene shows.

**Phase 3**
- A moving emitter's nozzle positions match the ones Start would sample from the emitter at that pose, within
  float32 tolerance.
- A swung emitter's released velocities include the emitter's surface velocity.
- Changing rate or speed mid-run changes the released count and speed from the next tick.
- An emitter moved into a collider pauses with the message and resumes when moved clear, and the run never
  stops.
- Shrinking the domain by 20% on one axis over 0.1 s: no particle is ever outside the new box after the walls
  arrive, and no substep moves a particle more than `0.5 * h` because of the walls.
- Add Fluid Now into a half-full tank adds the object's volume, within 5% of the fill count Start would give for
  the empty part. Peak density settles within 10% of rest within 1 s of simulated time.

**Phase 5**
- Record a 5 s take with a dragged collider, a moved emitter and a gravity change, then bake it twice. The
  bakes are bit-identical to each other, and the recorded keys reproduce the live inputs within float32
  tolerance per frame.
- A Capture writes every displayed frame to the cache in order. The playback matches the live run's frame
  count, and resume is refused.
- Scrub back 3 s, then resume: the state equals the snapshot bit-for-bit, and the run continues with no
  velocity spike.
- The snapshot ring never exceeds its VRAM budget.

**Every phase, live Blender** (one Blender at a time, isolated profile, both backends, 100k and 1M particles):
- Steps/s during continuous editing is at least 90% of an untouched run.
- UI-thread time spent on change handling is at most 1 ms per tick for 32 live objects.
- A screen capture of the phase's headline interaction is recorded for the launch kit.

## Files touched (expected)

- `sim_session.py`: the handler, classification, Reset, takes, rewind.
- `scene/blender_scene.py`, `scene/description.py`, `scene/emitters.py`, `scene/fill.py`.
- `sim/sim_core.py`, `sim/colliders.py`: these edits need owner go-ahead per the standing rule on `sim_core.py`.
- `cache/bake.py`, `cache/gpu.py`, `cache_ui.py`.
- `props.py`, `addon.py`.
- `docs/manual/*.md`, `CHANGELOG.md`.
- New tests in `tests/`.
