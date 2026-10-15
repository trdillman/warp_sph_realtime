# WarpSPH Realtime (`warp_sph_realtime`)

**Multi-million particle real-time viewport fluid solver for Blender 5.2 LTS+ powered by NVIDIA Warp 1.16+, CUDA Graph capture, and Zero-Copy OpenGL-CUDA interoperability.**

---

## Key Features
* **Zero-Copy Viewport Rendering:** Maps Blender 5.2 OpenGL Core Profile VAO/VBO buffers directly to CUDA device pointers (`cudaGraphicsGLRegisterBuffer` $\rightarrow$ `wp.from_ptr`) with **zero per-frame CPU readback**.
* **Coalesced SoA Memory + $O(n)$ Spatial Hash:** Uses 12-byte `float3` Structure of Arrays (SoA) buffers with cell-sorted reordering (`k_reorder_soa`) to eliminate GPU L2 cache thrashing during turbulent 27-cell neighbor queries.
* **Warp 1.16+ CUDA Graph Execution:** Captures density summation, WCSPH symmetric pressure/viscosity forces, SDF obstacle collisions, and copy-back into a single CUDA Graph (`wp.ScopedCapture`).
* **One-Click Interactive Demo Scenes:**
  * **Demo 1 (`DOUBLE_DAM_PILLARS`):** Opposing double dam-break crashing into a central sphere and 4 vertical cylindrical pillars.
  * **Demo 2 (`BLOCK_FLOP_SPLASH`):** Heavy suspended fluid cube dropping into a resting shallow puddle basin.
* **Live GPU Telemetry & Dynamic Buffer Scaling:** Adjust particle counts on the fly from `100,000` to `4,000,000` particles while monitoring smoothed FPS, frame time (ms), VRAM footprint (MB), and throughput (particles/ms).

---

## System Requirements
* **Blender:** `5.2.0 LTS` or newer (running with the **OpenGL** display backend: `Edit > Preferences > System > Display Graphics > OpenGL`).
* **GPU:** NVIDIA CUDA-capable GPU (RTX 20/30/40/50 series recommended; CUDA 12.x or 13.x runtime).
* **Python Dependency:** `warp-lang >= 1.16.0` installed in Blender 5.2's bundled Python 3.13 environment.

---

## Installation

1. **Install NVIDIA Warp into Blender 5.2 Python:**
   ```bash
   # Windows
   "C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe" -m pip install "warp-lang>=1.16.0"

   # Linux
   /path/to/blender-5.2/5.2/python/bin/python3.13 -m pip install "warp-lang>=1.16.0"
   ```
2. **Switch Blender Display Backend to OpenGL:**
   * Open Blender 5.2 $\rightarrow$ `Edit > Preferences > System > Display Graphics` $\rightarrow$ select **OpenGL**.
3. **Install Extension:**
   * Zip the `warp_sph_realtime` folder and install via `Edit > Preferences > Get Extensions > Install from Disk...`, **or** point a Local Extension Repository directly to this folder ([Blender Extension Docs](https://developer.blender.org/docs/handbook/extensions/addon_dev_setup/)).

---

## Quick Start (Running the Demos)
1. Press `N` in the 3D Viewport to open the Sidebar and click the **WarpSPH** tab.
2. Under **Quick-Run Demo Scenes**, click either:
   * **Demo 1: Double Dam + Pillars**
   * **Demo 2: Block Flop Crown Splash**
3. Drag the **Particles** slider (`100,000` – `4,000,000`) or toggle **Warp 1.16+ CUDA Graph Capture** to benchmark real-time viewport throughput.
4. Click **Pause** to freeze the physics (display settings stay live on the frozen frame), **Step** to advance one frame, and **Resume** to continue.
5. Press `ESC` or click **Stop** to release GPU buffers.

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

---

## Running the Test Suite
The repository includes a comprehensive `pytest` / `unittest` suite under `tests/` that validates physics kernels, mass/bounds conservation, CUDA Graph equivalence, whitewater neighbor classification, and demo scene initializers (running headless on CUDA without requiring an active Blender UI window):

```bash
pytest tests/ -v
# or using unittest directly:
python -m unittest discover -s tests -v
```

---

## Documentation Index
* [`ARCHITECTURE.md`](./ARCHITECTURE.md) — Mermaid system diagram and module connection guide.
* [`RESEARCH.md`](./RESEARCH.md) — Literature comparison and linked bibliography.
* [`SPEC_AND_PLAN.md`](./SPEC_AND_PLAN.md) — Technical specification and 4-phase engineering roadmap.
* [`AGENTS.md`](./AGENTS.md) — Instructions and invariants for AI coding agents working in this repository.
* [`CONTRIBUTING.md`](./CONTRIBUTING.md) — Developer setup, coding standards, and pull request workflow.
