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

### Your Own Scene (domain, fluid, colliders)
1. In the **Scene** box, choose **Custom**, then click **Add Fluid Scene** for a ready-made domain box, fluid block and
   collider, or build your own: select a mesh and click **Domain**, **Fluid** or **Collider** (**Clear** removes the role).
   The role is also on **Properties > Physics > WarpSPH Realtime Fluid**.
   * **Domain**: exactly one; the simulation box is the object's world bounding box.
   * **Fluid**: closed meshes, filled with particles when you press Start.
   * **Collider**: closed meshes the fluid flows around (static).
2. **Particle Spacing** sets the resolution (smaller = more particles); the panel shows the particle count after Start.
   Problems (no domain, no fluid, fluid outside the domain) are listed in the box; Start refuses an invalid scene with
   the reason.
3. Moving or editing the objects takes effect on the next **Stop** + **Start** (no live re-meshing). Units: 1 Blender unit = 1 m.

### Panel Controls (apply live while running)
| Section | Control | Effect |
|---|---|---|
| Simulation Speed vs. Stability | **Sub-steps / Frame** | More physics steps per redraw: faster sim, lower FPS. |
| | **Time Step (CFL)** | Step size as a fraction of the stability limit (default `0.4`). Higher = faster; above ~`0.5` impacts can blow up. |
| | **Fluid Stiffness** | Sets sound speed (default `90000`). Lower = larger steps and faster sim, but squishier and less stable. |
| Viewport Display | **Show Particles** | Hide/show the particle overlay. |
| | **Point Size** | Screen-space point size, 1–20 px (Raw Points mode). |
| | **Particle Radius (m)** | Sphere radius for SSFR / Sphere modes; arrows move it 0.5 mm per click. |
| | **Particle Color** | RGBA tint; alpha < 1 enables blending. |
| | **Hide Behind Objects** | Off = particles draw on top of scene geometry. |

The panel also shows the current `dt` and approximate simulated-time speed relative to real time.

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
  sim_session.py            Per-scene SimSession: runtime state, start/tick/stop/dispose, timer-driven bake
  cache_ui.py               Record / Bake Cache panel and operators
  cache/                    Frame cache: disk format, VRAM ring, writer thread, bake planning, playback point cloud
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
* [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — Mermaid system diagram and module connection guide.
* [`docs/RESEARCH.md`](./docs/RESEARCH.md) — Literature comparison and linked bibliography.
* [`docs/SPEC_AND_PLAN.md`](./docs/SPEC_AND_PLAN.md) — Technical specification; shipped scope plus an experimental, not-shipped roadmap.
* [`AGENTS.md`](./AGENTS.md) — Instructions and invariants for AI coding agents working in this repository.
* [`CONTRIBUTING.md`](./CONTRIBUTING.md) — Developer setup, coding standards, and pull request workflow.
* [`CHANGELOG.md`](./CHANGELOG.md) — Notable changes by release.
