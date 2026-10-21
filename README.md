# WarpSPH Realtime (`warp_sph_realtime`)

**Real-time viewport WCSPH fluid solver for Blender 5.2 LTS+ powered by NVIDIA Warp 1.17, CUDA Graph capture, and GPU-resident OpenGL / Vulkan CUDA interop (no per-frame CPU readback).**

---

## Key Features
* **GPU-resident viewport rendering (no CPU readback):**
  * **OpenGL (legacy VBO interop):** maps a Blender OpenGL Core VAO/VBO to a CUDA device pointer (`cudaGraphicsGLRegisterBuffer` $\rightarrow$ `wp.from_ptr`). Draw modes: raw points, spheres, and the 5-pass SSFR (screen-space fluid) renderer.
  * **OpenGL Points (`render/transport_gl.py`):** the GLInterop texture adapter, chosen when Start is pressed with Points mode: GPU copy into an RGBA32F `GPUTexture` via `cudaGraphicsGLRegisterImage`, no CPU readback, drawn by `render/gpu_points.py`. If it cannot start it falls back once to a labelled **Readback**; later runs in that Blender session use the legacy VBO interop.
  * **Vulkan (`render/transport_vulkan.py`):** GPU copy, no CPU readback ("zero host-copy"). Available only on the exact certified Blender 5.2.0 build; any other Vulkan build runs a labelled **Readback** fallback. Draws points, spheres and SSFR liquid through Blender's `gpu` module (`render/gpu_points.py`, `render/gpu_ssfr.py`).
* **Coalesced SoA Memory + $O(n)$ Spatial Hash:** Uses 12-byte `float3` Structure of Arrays (SoA) buffers with cell-sorted reordering (`k_reorder_soa`) to eliminate GPU L2 cache thrashing during turbulent 27-cell neighbor queries.
* **Warp 1.17 CUDA Graph Execution:** Captures density summation, WCSPH symmetric pressure/viscosity forces, SDF obstacle collisions, and copy-back into a single CUDA Graph (`wp.ScopedCapture`).
* **One-Click Interactive Demo Scenes:**
  * **Demo 1 (`DOUBLE_DAM_PILLARS`):** Opposing double dam-break crashing into a central sphere and 4 vertical cylindrical pillars.
  * **Demo 2 (`BLOCK_FLOP_SPLASH`):** Heavy suspended fluid cube dropping into a resting shallow puddle basin.
* **Record, bake and play back (P1.6, P2.2):** bake a frame range to a per-scene cache folder (one `.npz` per frame with `position`, `velocity` and a stable particle `id`), watch it in the viewport while it runs, cancel or resume it, invalidate from any frame, and scrub the result on the timeline as a point cloud object that renders in Cycles. The bake runs in small timer-driven chunks, so the UI stays responsive, and the GPU-to-host copy is asynchronous (pinned memory, second CUDA stream); it is an explicit bake action and never part of the live-view path.
* **Live GPU Telemetry & Dynamic Buffer Scaling:** Adjust particle counts on the fly from `100,000` to `4,000,000` particles while monitoring smoothed FPS, frame time (ms), VRAM footprint (MB), and throughput (particles/ms).

---

## System Requirements

* Blender 5.2.0 or newer. OpenGL supports all draw modes including SSFR; Vulkan draws all modes including SSFR; Vulkan GPU interop requires the exact certified Blender 5.2.0 build, and other Vulkan builds use the labelled Readback fallback.
* NVIDIA CUDA GPU. Certified Warp 1.17 PyPI uses CUDA Toolkit 12.9 and requires driver 525 or newer. OpenGL interop separately requires dynamic CUDA 12.9 runtime (cudart64_12.dll on Windows or libcudart.so.12 on Linux).
* Certified candidate: warp-lang 1.17.0, exact SHA-256-locked CUDA 12 PyPI wheel. Versions 1.16 and 1.18 / CUDA 13 are not automatically accepted.
* OpenGL and Warp cuda:0 must refer to the same NVIDIA GPU.

## Installation

1. Build the zip with `python scripts/build_extension.py` (writes `dist/warp_sph_realtime-<version>.zip`) and install it via `Edit > Preferences > Get Extensions > Install from Disk...`, **or** point a Local Extension Repository at the inner `warp_sph_realtime/` package folder. It registers even without NVIDIA Warp.
2. In Add-on Preferences (Edit > Preferences > Add-ons > WarpSPH), check the **Hardware Check** box (GPU, driver, CUDA support, VRAM, graphics backend; it works before Warp is installed), then pick a **Warp source**:
   * **Download** (default): click Download and Install. Enable Blender Online Access first. The ~150 MB download runs in the background with a progress bar and a Cancel button; Cancel leaves nothing behind.
   * **Install from local wheel file**: choose the certified `warp_lang-1.17.0` `.whl` for your platform and click Install. No network access is used. A file that does not match the SHA-256 lock is refused.
   * **Use existing Warp folder**: choose the folder that contains the `warp` package (for example a studio-managed site-packages). WarpSPH only reads it, never modifies or deletes it, and checks the version, the NVIDIA driver and a known-answer GPU kernel when you press Start. Only Warp 1.17.x is accepted unless you tick "Allow uncertified version (unsupported)", which keeps a warning visible. Switching back to Download or a local wheel uses the install WarpSPH already owns.
3. The download and local-wheel paths verify the exact wheel size, SHA-256, metadata and file paths, and place it only in the extension's private user storage. No pip or Blender bundled Python modification is required.
4. For repairs involving a loaded native runtime, click Repair Warp and restart Blender to safely apply the staged wheel. Changing the Warp source after Warp has loaded needs a Blender restart.
5. Start the simulation after the panel reports an installed runtime. Start checks CUDA and graphics-device compatibility and shows the active transport mode.

Foreign and unmanaged Warp installations are preserved, never deleted. If another extension loaded its own Warp, use a separate Blender process with only WarpSPH enabled. Blender's official Extensions repository may reject downloadable Python modules; this flow is intended for self-hosted packages.

See [`docs/WARP_INSTALLATION.md`](./docs/WARP_INSTALLATION.md) and [`docs/INSTALLER_QUALIFICATION_20261014.md`](./docs/INSTALLER_QUALIFICATION_20261014.md) for qualification, reproducibility and limitations.

---

## Quick Start (Running the Demos)
1. Press `N` in the 3D Viewport to open the Sidebar and click the **WarpSPH** tab.
2. In the **Scene** box (source **Demo**), click either:
   * **Demo 1: Double Dam + Pillars**
   * **Demo 2: Block Flop Crown Splash**
3. Drag the **Particles** slider (`100,000` – `4,000,000`) or toggle **Warp 1.17 CUDA Graph Capture** to benchmark real-time viewport throughput.
4. Click **Pause** to freeze the physics (display settings stay live on the frozen frame), **Step** to advance one frame, and **Resume** to continue.
5. Press `ESC` or click **Stop** to release GPU buffers.

### Your Own Scene (domain, fluid, colliders, emitters)
1. In the **Scene** box, choose **Custom**, then click **Add Fluid Scene** for a ready-made domain box, fluid block and
   collider, or build your own: select a mesh and click **Domain**, **Fluid**, **Collider** or **Emitter** (**Clear**
   removes the role). The role is also on **Properties > Physics > WarpSPH Realtime Fluid**.
   * **Domain**: exactly one; the simulation box is the object's world bounding box.
   * **Fluid**: closed meshes, filled with particles when you press Start.
   * **Collider**: closed meshes the fluid flows around. Animated **location and rotation** (keyframes, drivers,
     constraints, parents) move the collider and push the fluid; see the limits below.
   * **Emitter**: a surface (for example a plane) that releases particles. Its settings are on the Physics panel:
     **Velocity** along the face normal (**Speed**) or a fixed **Vector**, **Rate** in particles per second
     (0 = a full stream at that speed), and **Start**/**End** frames. Particles leave the side the normals point to.
     The emitter must sit in open space, at least one particle spacing from the domain walls and colliders.
2. **Particle Spacing** sets the resolution (smaller = more particles); the panel shows the particle count after Start.
   It is a length in the scene's units (type `2cm` or `0.02` like any Blender length), so it follows the Unit Scale.
   The initial fill keeps one spacing of clearance from the domain walls and collider surfaces (that is where the
   solver's wall force is zero), so fluid modelled against a wall starts one and a half spacings from it and settles.
   With emitters, **Max Particles** (default 1,000,000) is the particle budget reserved on the GPU at Start; the panel
   shows "Particles: N of Max". When the budget is full the emitters stop and Blender shows a warning; raise
   Max Particles and restart for a longer pour. Problems (no domain, no fluid or emitter, fluid outside the domain)
   are listed in the box; Start refuses an invalid scene with the reason.
3. Editing the objects takes effect on the next **Stop** + **Start** (no live re-meshing). While a Custom scene with
   animated colliders runs, the timeline follows the simulation's frame.
4. Units and gravity: the scene's **Unit Scale** (Scene Properties > Units) converts Blender units to metres, so a
   scene modelled in centimetres (Unit Scale 0.01) simulates at real size. **Particle Spacing** is in scene units, like
   the objects; emitter speeds follow the scene's units like any Blender velocity. Gravity is the scene's **Gravity** (Scene Properties >
   Gravity).

Collider limits: a collider keeps the shape it has at the Start frame. Deforming colliders (armatures, shape keys,
modifiers that animate) are not followed, and an animated **scale** or shear stops the run with an error. Rotation
between frames is interpolated along the shortest arc, so very fast spins (more than half a turn per frame) are
not supported.

### Panel Controls (apply live while running)
| Section | Control | Effect |
|---|---|---|
| Fluid | **Water / Viscous / Stylised** | Sets stiffness, time step, viscosity and sub-steps together (table below). The preset whose values are all set is highlighted; changing any one of them un-highlights it. Particle spacing and display settings are never touched. |
| | **Viscosity** | How strongly each particle's velocity is pulled toward its neighbours' average (kg/s per particle, default `0.15`, range 0 to 20). Higher = thicker, slower flow and quicker settling. See the note below. |
| Simulation Speed vs. Stability | **Sub-steps / Frame** | More physics steps per redraw: faster sim, lower FPS. |
| | **Time Step (CFL)** | Step size as a fraction of the stability limit (default `0.4`). Higher = faster, but values above `0.4` can blow up on hard impacts at high particle counts (0.45 did at 1M). |
| | **Fluid Stiffness** (Advanced) | Sets sound speed (default `90000`). Lower = larger steps and faster sim, but squishier and less stable. |
| Viewport Display | **Show Particles** | Hide/show the particle overlay. |
| | **Point Size** | Screen-space point size, 1–20 px (Raw Points mode). |
| | **Particle Radius (m)** | Sphere radius for SSFR / Sphere modes; arrows move it 0.5 mm per click. |
| | **Particle Color** | RGBA tint; alpha < 1 enables blending. |
| | **Hide Behind Objects** | Off = particles draw on top of scene geometry. |

The panel also shows the current `dt` and approximate simulated-time speed relative to real time.

| Preset | Stiffness | Time step (CFL) | Viscosity | Sub-steps | Character |
|---|---:|---:|---:|---:|---|
| Water | 90 (90,000) | 0.4 | 0.15 | 2 | The tuned default: free-flowing, what every earlier version used. |
| Viscous | 90 | 0.4 | 8 | 2 | Thick and slow like honey or syrup: splashes die quickly, flow creeps. |
| Stylised | 60 | 0.4 | 1.5 | 3 | Softer and a little blobby, 22 % larger time step and 3 sub-steps per redraw, so it moves faster per frame. |

All three were run on both demos at 1,000,000 and 100,000 particles and on a drop-into-pool scene at 0.02 m and
0.01 m spacing, on the GPU with the CUDA graph, and on the CPU at coarse spacing: no NaN, nothing leaves the domain,
the energy never grows. A hard impact still throws a few spray particles faster than the sound speed (a property of
this solver at its default stiffness too).

Viscosity note: the viscous force is applied explicitly, so it is capped automatically where it would overshoot
(the panel says "Limited to ..." while running). Its strength per kg/s scales with `1 / particle mass`: the same
number is thicker at a finer spacing, and the cap is reached sooner there. If a preset feels too thin or too thick
after changing the spacing, adjust Viscosity.

Fine adjustment: hold **Shift** while dragging any value for 10× finer control, or hover a value and use **Ctrl + mouse wheel** to step it.

### Record / Bake Cache
Open **Record / Bake Cache** under the main panel (the simulation need not be running: **Bake** starts it).
1. **Cache Folder**: leave empty for the default, `<blend folder>/<name>_warp_sph_cache/<scene>` next to the saved
   .blend (a temporary folder while the file is unsaved), or pick a folder. Each scene has its own.
2. Set **Start** / **End** (frame 1 is the initial state) and **Steps / Frame** (0 = real time: the steps that make one
   frame last `1 / frame rate` seconds of simulated time). The panel shows the size before you bake: 28 bytes per
   particle per frame, and warns if the drive is too small.
3. **Bake** simulates from the initial state and saves each frame; a progress bar and **Cancel** show while it runs, and
   the viewport follows. Cancel (or Esc, or a failure) keeps every frame saved so far. **Resume** continues after the
   last saved frame (or extends **End**); it refuses a cache baked with different particle count or solver settings.
   **Invalidate From Current Frame** deletes that frame onward so **Resume** re-simulates it; **Delete** removes the cache.
4. **Show Cache as Point Cloud** adds an object that follows the timeline (frames outside the cached range show the
   nearest cached frame). It has `velocity`, `id` and `radius` point attributes and renders in Cycles as spheres of the
   particle radius.
5. **Output** (below the cache buttons). The point cloud's `age` attribute is the time in seconds since each particle first
   appears in the cache (use it in a material through an Attribute node, or in Geometry Nodes). **Add Mesh Preset** adds a
   *WarpSPH Mesh* object that turns the points into a surface with Geometry Nodes (Points to Volume, Volume to Mesh, then
   smoothing): edit **Radius**, **Voxel Size** (smaller = finer and heavier), **Threshold**, **Smooth Iterations** and
   **Material** in the panel or on the modifier. The defaults come from the particle spacing (radius 1.5x, voxel 0.5x);
   the point cloud is hidden from renders while the mesh exists. **Export USD** / **Export Alembic** ask for a file and
   write the mesh over the baked frames only (adding the preset first if needed). Blender 5.2 exports the meshed object;
   it does not export point clouds. A 24-frame, 12,000-particle bake is about 65 MB as USD and 35 MB as Alembic.
The solver state is the particle positions and velocities, so a resumed bake continues bit for bit like an uninterrupted one
on the same GPU and driver. Changing the scene's objects after a bake is not detected: invalidate from frame 1.

---

## Running the Test Suite
The repository includes a `pytest` / `unittest` suite under `tests/` that validates physics kernels, mass/bounds conservation, CUDA Graph equivalence, demo scene initializers, the transports, the installer, and the build file list (running headless on CUDA without requiring an active Blender UI window):

```bash
pip install numpy "warp-lang==1.17.0" "fake-bpy-module==20251003" "pytest>=8" "ruff>=0.6" "pyright==1.1.408"
pytest                      # run from the repository root
ruff check .
pyright
# or using unittest directly:
python -m unittest discover -s tests -v
```

---

## Project Layout
```
warp_sph_realtime/          Blender extension package (the folder that is zipped)
  blender_manifest.toml     Extension metadata
  __init__.py               register()/unregister(); lazily imports addon
  addon.py                  Operators, panels, modal timer (thin UI)
  props.py                  scene.warp_sph settings group, old-file migration, GPU memory estimate
  physics_presets.py        Water / Viscous / Stylised preset values (no bpy)
  sim_session.py            Per-scene SimSession: runtime state, start/tick/stop/dispose, timer-driven bake
  cache_ui.py               Record / Bake Cache panel and operators
  cache/                    Frame cache: disk format, VRAM ring, writer thread, bake planning, playback point cloud,
                            particle age, meshing preset and USD / Alembic export
  sim/                      WCSPH solver (sim_core)
  scene/                    Regime presets, demo scene initialisers
  render/                   OpenGL and Vulkan transports, gpu-module points/spheres/SSFR, legacy GL SSFR shaders, viewport drawing
tests/                      Headless test suite (mocks bpy/gpu where needed)
docs/                       Architecture, research notes, spec and roadmap
scripts/build_extension.py  Validate + build (+ optional install) via Blender
```

`warp_sph_realtime/experimental/` stays in the repository for research and tests but is excluded from the built zip (see below).

---

## Experimental (not shipped)
The `experimental/` package is excluded from the built extension via `paths_exclude_pattern` in `blender_manifest.toml`. It is not wired into the UI or runtime and carries no support promise. It contains: implicit position-based fluids (IPBF), analytical/SDF collider kernels, whitewater classification, a neighbor-search abstraction, a Fourier-neural-operator DLPack bridge, particle-upsample detection, a SplashSurf meshing wrapper, and differentiable-pour, keyframe and compliant-contact studies.

---

## Documentation Index
* [`docs/manual/`](./docs/manual/README.md) — User manual (install, first run, authoring, physics, display, bake and export, limits) and the [FAQ / troubleshooting](./docs/manual/faq.md).
* [`examples/`](./examples/README.md) — Example scenes as runnable Blender Python scripts.
* [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — Mermaid system diagram and module connection guide.
* [`docs/RESEARCH.md`](./docs/RESEARCH.md) — Literature comparison and linked bibliography.
* [`docs/SPEC_AND_PLAN.md`](./docs/SPEC_AND_PLAN.md) — Technical specification; shipped scope plus an experimental, not-shipped roadmap.
* [`AGENTS.md`](./AGENTS.md) — Instructions and invariants for AI coding agents working in this repository.
* [`CONTRIBUTING.md`](./CONTRIBUTING.md) — Developer setup, coding standards, and pull request workflow.
* [`CHANGELOG.md`](./CHANGELOG.md) — Notable changes by release.
