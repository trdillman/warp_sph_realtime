# 4. Physics settings and presets

All of these apply live while the simulation runs.

## Presets

The **Fluid** box has three preset buttons. A preset sets stiffness, time step, viscosity and sub-steps together; it
never changes particle spacing or display settings. The button whose values are all currently set is highlighted;
changing any one value un-highlights it.

| Preset | Stiffness | Time step (CFL) | Viscosity | Sub-steps | Character |
|---|---:|---:|---:|---:|---|
| **Water** | 90 (90,000) | 0.4 | 0.15 | 2 | The tuned default: free-flowing. |
| **Viscous** | 90 | 0.4 | 8 | 2 | Thick and slow like honey or syrup; splashes die quickly. |
| **Stylised** | 60 | 0.4 | 1.5 | 3 | Softer and a little blobby; larger effective time step and 3 sub-steps per redraw, so it moves faster per frame. |

Each preset was run on both demos at 1,000,000 and 100,000 particles and on a drop-into-pool scene at 0.02 m and
0.01 m spacing (GPU, with the CUDA graph; and CPU at coarse spacing) with no NaN, nothing leaving the domain and no
energy growth. A hard impact can still throw a few spray particles faster than the sound speed.

## Viscosity

**Viscosity** (kg/s per particle, default 0.15, range 0 to 20) is how strongly each particle's velocity is pulled
toward its neighbours' average. Higher means thicker, slower flow and quicker settling. It is applied explicitly, so
it is capped automatically where it would overshoot; the panel then shows "Limited to N at this particle size
(stability)". The effect scales with 1 / particle mass, so the same number is thicker at a finer spacing and the cap
is reached sooner. If a preset feels too thin or too thick after you change the spacing, adjust **Viscosity**.

## Particle spacing

See [Authoring a scene](authoring.md#particle-spacing). Halving the spacing multiplies the particle count by about
eight; check the VRAM warning before you start.

## Simulation Speed vs. Stability

* **Sub-steps / Frame** (1 to 10, default 2): physics steps per viewport redraw. More is faster simulated time per
  frame at a lower FPS.
* **Time Step (CFL)** (0.1 to 1.0, default 0.4): the step as a fraction of the stability limit. Higher is faster but
  less stable. **Values above 0.4 can blow up on hard impacts at high particle counts** (0.45 did at 1,000,000
  particles). Stay at 0.4 or lower for large runs.
* **Fluid Stiffness (x1000)** (under **Advanced**; default 90, range 10 to 400): sets the sound speed. Lower allows
  larger steps and faster simulation but is squishier and less stable.
* **Warp 1.17 CUDA Graph Capture** (under **Advanced**, default on): runs the density, force and copy-back work as
  one captured CUDA graph.

The panel shows the current `dt` and the approximate simulated speed relative to real time.

Hold **Shift** while dragging a value for 10x finer control, or hover a value and use **Ctrl + mouse wheel** to step
it.

## What the solver is validated against

Weakly compressible SPH with a fixed sound speed, no surface tension and no wall friction. The dam-break front,
conservation, energy and rest density measurements, and their limits, are in [`VALIDATION.md`](../VALIDATION.md).
The dam-break front is within roughly 6 to 10 percent of the Martin and Moyce experiment at 8 or more cells across,
but it does not converge monotonically, so small differences should not be read as better or worse physics.
