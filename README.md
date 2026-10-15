# WarpSPH Realtime (`warp_sph_realtime`)

**Multi-million particle real-time viewport fluid solver for Blender 5.2 LTS+ powered by NVIDIA Warp 1.17, CUDA Graph capture, and Zero-Copy OpenGL-CUDA interoperability.**

---

## Key Features
* **Zero-Copy Viewport Rendering:** Maps Blender 5.2 OpenGL Core Profile VAO/VBO buffers directly to CUDA device pointers (`cudaGraphicsGLRegisterBuffer` $\rightarrow$ `wp.from_ptr`) with **zero per-frame CPU readback**.
* **Coalesced SoA Memory + $O(n)$ Spatial Hash:** Uses 12-byte `float3` Structure of Arrays (SoA) buffers with cell-sorted reordering (`k_reorder_soa`) to eliminate GPU L2 cache thrashing during turbulent 27-cell neighbor queries.
* **Warp 1.17 CUDA Graph Execution:** Captures density summation, WCSPH symmetric pressure/viscosity forces, SDF obstacle collisions, and copy-back into a single CUDA Graph (`wp.ScopedCapture`).
* **One-Click Interactive Demo Scenes:**
  * **Demo 1 (`DOUBLE_DAM_PILLARS`):** Opposing double dam-break crashing into a central sphere and 4 vertical cylindrical pillars.
  * **Demo 2 (`BLOCK_FLOP_SPLASH`):** Heavy suspended fluid cube dropping into a resting shallow puddle basin.
* **Live GPU Telemetry & Dynamic Buffer Scaling:** Adjust particle counts on the fly from `100,000` to `4,000,000` particles while monitoring smoothed FPS, frame time (ms), VRAM footprint (MB), and throughput (particles/ms).

---

## System Requirements

* Blender 5.2.0 or newer with the OpenGL display backend.
* NVIDIA CUDA GPU. Certified Warp 1.17 PyPI uses CUDA Toolkit 12.9 and requires driver 525 or newer. OpenGL interop separately requires dynamic CUDA 12.9 runtime (cudart64_12.dll on Windows or libcudart.so.12 on Linux).
* Certified candidate: warp-lang 1.17.0, exact SHA-256-locked CUDA 12 PyPI wheel. Versions 1.16 and 1.18 / CUDA 13 are not automatically accepted.
* OpenGL and Warp cuda:0 must refer to the same NVIDIA GPU.

## Installation

1. Install the WarpSPH extension first. It registers even without NVIDIA Warp.
2. In Add-on Preferences or the WarpSPH viewport sidebar, click Install NVIDIA Warp. Enable Blender Online Access before downloading.
3. The extension verifies the exact wheel size, SHA-256, metadata and file paths, and places it only in its private extension user storage. No pip or Blender bundled Python modification is required.
4. For repairs involving a loaded native runtime, click Repair Warp and restart Blender to safely apply the staged wheel.
5. Select the OpenGL display backend and start the simulation after the panel reports an installed runtime. Start additionally checks CUDA and GL device compatibility.

Foreign and unmanaged Warp installations are preserved, never deleted. If another extension loaded its own Warp, use a separate Blender process with only WarpSPH enabled. Blender's official Extensions repository may reject downloadable Python modules; this flow is intended for self-hosted packages.

See docs/WARP_INSTALLATION.md and docs/INSTALLER_QUALIFICATION_20261014.md for qualification, reproducibility and limitations.

---

## Quick Start (Running the Demos)
1. Press `N` in the 3D Viewport to open the Sidebar and click the **WarpSPH** tab.
2. Under **Quick-Run Demo Scenes**, click either:
   * **Demo 1: Double Dam + Pillars**
   * **Demo 2: Block Flop Crown Splash**
3. Drag the **Particles** slider (`100,000` – `4,000,000`) or toggle **Warp 1.17 CUDA Graph Capture** to benchmark real-time viewport throughput.
4. Press `ESC` or click **Stop Sim** to release GPU buffers.

### Panel Controls (apply live while running)
| Section | Control | Effect |
|---|---|---|
| Simulation Speed vs. Stability | **Sub-steps / Frame** | More physics steps per redraw: faster sim, lower FPS. |
| | **Time Step (CFL)** | Step size as a fraction of the stability limit (default `0.4`). Higher = faster; above ~`0.5` impacts can blow up. |
| | **Fluid Stiffness** | Sets sound speed (default `90000`). Lower = larger steps and faster sim, but squishier and less stable. |
| Viewport Display | **Show Particles** | Hide/show the particle overlay. |
| | **Point Size** | Screen-space point size, 1–20 px. |
| | **Particle Color** | RGBA tint; alpha < 1 enables blending. |
| | **Hide Behind Objects** | Off = particles draw on top of scene geometry. |

The panel also shows the current `dt` and approximate simulated-time speed relative to real time.

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
