# 3. Authoring a scene

Set the **Scene** box source to **Custom**. The simulation is then built from the objects you mark with a WarpSPH role.

## Quick start

Click **Add Fluid Scene**. It adds a domain cube, a fluid block and one tilted collider box (objects named
"WarpSPH Domain", "WarpSPH Fluid", "WarpSPH Collider") and switches the source to Custom. Press **Start Realtime Sim**.
The button refuses to add a second domain ("This scene already has a WarpSPH domain; clear its role first").

## Roles

Select one or more mesh objects and click **Domain**, **Fluid**, **Collider** or **Emitter** under "Set selected
objects as:". **Clear** removes the role. The role is also a setting on **Properties > Physics > WarpSPH Realtime
Fluid**. Domain and Fluid objects are switched to wireframe so the particles stay visible.

| Role | Meaning |
|---|---|
| **Domain** | The simulation box: the object's world-space axis-aligned bounding box (use a cube). Exactly one per scene; marking a new one replaces the old. |
| **Fluid** | A closed mesh volume, filled with particles when you press Start. |
| **Collider** | A closed mesh obstacle the fluid flows around. Its animated location and rotation are followed. |
| **Emitter** | A surface (a plane, for example) that releases particles over a frame range. |

Closed meshes need consistently oriented, outward-facing faces. Start reports problems such as "is not closed
(watertight)" or "encloses no volume". The box lists problems as it goes ("No domain: mark a box as Domain", "No
fluid: mark a closed mesh as Fluid", "Fluid '...' extends outside the domain") and Start refuses an invalid scene
with the reason.

## Emitters

With the Emitter role the Physics panel shows:

* **Direction**: **Normal** (particles leave along the face normals at **Speed**) or **Vector** (they leave with the
  **Velocity** vector). Particles leave the side the normals point to; flip the normals to reverse the flow.
* **Speed** or **Velocity**, in the scene's velocity units.
* **Rate (particles/s)**: 0 means a full stream (one particle per spacing of travel); higher values are capped at
  the full stream.
* **Start Frame** and **End Frame**.

The emitter must sit in open space, at least one particle spacing from the domain walls and colliders; otherwise
Start reports "emitter '...' is outside the domain, inside a collider, or closer than ... m ...".

With emitters, **Max Particles** (default 1,000,000, range 1,000 to 4,000,000) is the budget reserved on the GPU at
Start. The panel shows "Particles: N of Max", and when the budget is full it shows "Particle budget reached:
emitters stopped". Raise **Max Particles** and restart for a longer pour. A scene without emitters uses exactly its
fill.

## Animated colliders

Keyframes, drivers, constraints and animated parents on a collider's location and rotation move the collider and
push the fluid. While a Custom scene with animated colliders runs, the timeline follows the simulation's frame.
Limits: the collider keeps the shape it had at the Start frame (armatures, shape keys and animated modifiers are not
followed), an animated scale or shear stops the run with an error, and rotation is interpolated along the shortest
arc, so spins of more than half a turn per frame are not supported.

## Particle spacing

**Particle Spacing** (default 0.02, range 0.0001 to 1000) sets the resolution: smaller means more particles, about
fluid volume divided by spacing cubed. It is a length in scene units, so you can type `2cm` or `0.02`. The panel
estimates the particle count and warns if it will not fit in GPU memory. The initial fill keeps one spacing of
clearance from the domain walls and collider surfaces, so fluid modelled against a wall starts one and a half
spacings away and settles.

## Units and gravity

The scene's **Unit Scale** (Scene Properties > Units) converts Blender units to metres, so a scene modelled in
centimetres with Unit Scale 0.01 simulates at real size. Spacing and emitter speeds follow the scene's units. With Unit System None, Blender ignores Unit Scale and so does the simulation (1 unit = 1 m).
Gravity is the scene's **Gravity** (Scene Properties > Gravity).

## Applying changes

The simulation keeps running while you edit the scene. Each change either applies live or waits for **Reset**:

| Change | While running |
| --- | --- |
| Scene **Gravity** (or turning it off) | Live, on the next step |
| Viscosity, presets, stiffness, time step, sub-steps, display settings | Live, as before |
| Animated colliders | Follow their animation (the object panel says "Follows its animation while running") |
| Moving, reshaping or changing the settings of a domain, fluid, still collider or emitter | On **Reset** |
| Adding or removing objects, or changing a role | On **Reset** |
| Particle spacing, Max Particles, Unit Scale, frame rate, scene source | On **Reset** |

Changes that wait are listed in the Scene box as "Reset to apply: ...". **Reset** (next to Pause and Step)
rebuilds the simulation from the scene at the frame it started on, keeping the GPU runtime and the viewport
connection when the particle count is unchanged. Reset performance still needs runtime qualification.
If the edited scene cannot be built (for example no domain),
Reset reports why and the current run continues.

Undo and redo no longer stop the simulation. During a bake, edits are ignored until the bake ends ("Live edits are
ignored while baking; they apply when the bake ends"), and an undo cancels the bake, keeping the frames already
saved. An animated collider that is deleted while running stays at its last position until Reset.
After undo or redo, a mesh counts as reshaped only if its vertices or edges differ from the running state. A mesh
with modifiers or shape keys cannot be compared that way, so it may show a conservative "reshaped" hint until Reset.
A paused Reset stays paused; if the viewport connection is busy, its initial particle frame is retried on later
ticks without advancing the simulation. A rejected Reset restores the previous timeline frame.

Scripts that build a scene are in [`examples/`](../../examples/README.md).
