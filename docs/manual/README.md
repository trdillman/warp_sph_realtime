# WarpSPH Realtime manual

WarpSPH Realtime runs a GPU SPH fluid inside Blender 5.2 using NVIDIA Warp, shows it live in the 3D viewport, and
can bake it to a cache that plays back on the timeline, renders in Cycles and exports as a meshed USD or Alembic
sequence.

| Page | What it covers |
|---|---|
| [1. Install](install.md) | Hardware check, the three ways to get Warp, repair |
| [2. First run](first-run.md) | The two demos and the run controls |
| [3. Authoring a scene](authoring.md) | Domain, fluid, collider and emitter roles, animated colliders, units and gravity |
| [4. Physics settings](physics.md) | Presets, viscosity, particle spacing, time step (CFL), stiffness |
| [5. Viewport display](display.md) | Render modes and the three transports |
| [6. Record, bake, playback and export](cache-output.md) | Bake, resume, invalidate, point cloud, age, mesh preset, USD / Alembic |
| [7. Known limits](limits.md) | What the solver and the add-on do not do |
| [FAQ and troubleshooting](faq.md) | GPU, driver and Warp install problems, backend and Readback, VRAM, performance |

Runnable example scenes (Python scripts that build a scene for you) are in [`examples/`](../../examples/README.md).

Panel and button names in this manual are written exactly as they appear in Blender. The panel is in the 3D
Viewport sidebar (press `N`), tab **WarpSPH**; the add-on settings are in
`Edit > Preferences > Add-ons > WarpSPH`.

Background reading for people who want the details: [`ARCHITECTURE.md`](../ARCHITECTURE.md) (how the modules fit
together), [`VALIDATION.md`](../VALIDATION.md) (what the solver was measured against and how far to trust it) and
[`WARP_INSTALLATION.md`](../WARP_INSTALLATION.md) (the Warp installer's rules).
