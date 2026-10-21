# Technical Specification & 4-Phase Engineering Plan (`SPEC_AND_PLAN.md`)

## 1. Executive Objective
Deliver a real-time, multi-million-particle fluid simulation extension for **Blender 5.2 LTS+** using **NVIDIA Warp 1.17.0** (SHA-256-locked CUDA 12.9 wheel), prioritizing **interactive 3D viewport performance** via GPU-resident CUDA interop (OpenGL VBO path, and a Vulkan path on the exact certified Blender 5.2.0 build; no per-frame CPU readback). Implicit stability, whitewater and neural acceleration are experimental and not shipped (see section 3).

---

## 2. Core Technical Specifications

### 2.1 Memory & Data Layout Specification
| Buffer Name | Type | Bytes / Particle | Residency | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| `vbo` (`pos_vbo_wp`) | `wp.vec3` (`float3`) | 12 B | OpenGL VBO $\leftrightarrow$ CUDA | GPU-resident viewport draw buffer (OpenGL path) |
| `pos_sim` | `wp.vec3` (`float3`) | 12 B | CUDA (`cuda:0`) | Primary simulation positions |
| `vel` | `wp.vec3` (`float3`) | 12 B | CUDA (`cuda:0`) | Primary simulation velocities |
| `pos_sorted` | `wp.vec3` (`float3`) | 12 B | CUDA (`cuda:0`) | Cell-sorted positions for coalesced L2 reads |
| `vel_sorted` | `wp.vec3` (`float3`) | 12 B | CUDA (`cuda:0`) | Cell-sorted velocities for coalesced L2 reads |
| `density_sorted` | `float` (`float32`) | 4 B | CUDA (`cuda:0`) | Cell-sorted SPH density ($\rho_i$) |
| `grid` (`wp.HashGrid`) | `uint32` / `uint64` | ~12 B | CUDA (`cuda:0`) | $128^3$ spatial hash & prefix scan offsets |

* **Total SoA Footprint:** ~76 bytes/particle (~72.5 MB at 1M particles; ~290.0 MB at 4M particles).

### 2.2 Mathematical Formulation
* **Smoothing Kernel:** 3D Cubic Spline $W(\mathbf{x}_i - \mathbf{x}_j, h)$ with compact support radius $h = 0.04\text{ m}$.
* **WCSPH Tait Equation of State:**
  $$p_i = k \left( \left( \frac{\rho_i}{\rho_0} \right)^\gamma - 1 \right), \quad \gamma = 7, \quad \rho_0 = 1000\text{ kg/m}^3$$
* **IPBF Local Variational Update (experimental, not shipped; `experimental/ipbf_solver.py`):**
  $$C_i(\mathbf{x}) = \max\left(\frac{\rho_i}{\rho_0} - 1, 0\right), \quad \lambda_i = \frac{-C_i}{\sum_j \|\nabla_{\mathbf{x}_j} C_i\|^2 + \epsilon}$$
* **Whitewater Classification (experimental, not shipped; `experimental/whitewater.py`):**
  * **Spray:** $n_f \le 5$ (ballistic gravity trajectory)
  * **Foam:** $5 < n_f < 15$ (advected with local fluid surface velocity)
  * **Bubbles:** $n_f \ge 15$ (submerged drag + upward buoyancy $+2.5\text{ m/s}^2$)

---

## 3. Phased Execution Plan & Status

Only Phase 1 is shipped. Phases 2-4 are an **Experimental (not shipped)** roadmap: the code lives in `warp_sph_realtime/experimental/`, which is excluded from the built zip and not wired into the UI or runtime.

### Phase 1: Zero-Copy GPU Viewport & Coalesced WCSPH Core ✅ [COMPLETED]
* [x] `blender_manifest.toml` configured for Blender 5.2+ and Python 3.13.
* [x] `ZeroCopyGLBuffer52` OpenGL 4.5+ Core Profile VAO/VBO allocation and `cudaGraphicsGLRegisterBuffer` pointer mapping (`wp.from_ptr`).
* [x] $O(n)$ spatial hashing (`wp.HashGrid`), `k_reorder_soa` cell sorting, and fused WCSPH pressure/viscosity + copy-back inside a Warp 1.17 CUDA Graph (`wp.ScopedCapture`).
* [x] Dynamic buffer reallocation slider (`100k`–`4M` particles), live GPU telemetry readout, and two one-click demo scenes (`DOUBLE_DAM_PILLARS`, `BLOCK_FLOP_SPLASH`).
* [x] Vulkan transport (`render/transport_vulkan.py`, exact-build Blender 5.2.0 only; GPU copy, no CPU readback) with a labelled Readback fallback, and points/spheres drawing through `render/gpu_points.py`.
* [x] Backend-neutral SSFR through the `gpu` module (`render/gpu_ssfr.py`) for transport sessions on Vulkan and OpenGL.
* [x] Hash-locked Warp installer and runtime gate.

### Phase 2 (experimental, not shipped): Unconditional Stability (IPBF) & Two-Way Mesh SDF Collisions 🔄 [SCAFFOLDED]
* [x] Core kernels written in `experimental/ipbf_solver.py` and `experimental/collider_sdf.py`.
* [ ] Wire UI solver selector (`WCSPH` vs. `IPBF`) into `RealtimeWCSPHSolver.step()`.
* [ ] Convert arbitrary selected Blender `bpy.types.Mesh` objects into `wp.Mesh` BVH/SDF colliders with lock-free atomic impulse readback (6 `uint` values per rigid body) to drive Blender object transforms.
* [ ] Velocity color ramps in the viewport.

### Phase 3 (experimental, not shipped): Real-Time GPU Whitewater (Spray, Foam, Bubbles) 🔄 [SCAFFOLDED]
* [x] Neighbor-count classification and advection kernel written in `whitewater.py`.
* [ ] Add in-kernel trapped-air spawning during the WCSPH/IPBF force pass and stream compaction into a secondary `ZeroCopyGLBuffer52` for multi-color viewport rendering.

### Phase 4 (experimental, not shipped): Hybrid 3D Neural Operators (`FNO` / `FC_PINO`) & Cycles Meshing 🔄 [SCAFFOLDED]
* [x] Zero-copy DLPack bridge (`neural_fno.py`), surface particle detector (`particle_upsample.py`), and `SplashSurf` CLI wrapper (`mesher_io.py`) implemented.
* [ ] Load pre-trained 3D FNO / `FC_PINO` checkpoints to predict coarse velocity fields on a $64^3$ grid and guide large-timestep IPBF iterations.
* [ ] Add one-click background `SplashSurf` mesh sequence baking and automatic Cycles Glass/Liquid BSDF material assignment.
