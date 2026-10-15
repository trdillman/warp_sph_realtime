# AGENTS.md — AI Coding Agent Guidelines for `warp_sph_realtime`

This file provides strict architectural invariants, build/test commands, and hardware rules for AI coding agents (Claude Code, Gemini, Cursor, GitHub Copilot, etc.) modifying this repository.

---

## 1. Target Stack & Environment Constraints
* **Blender Target:** Blender `5.2.0 LTS+` (Extension system using `blender_manifest.toml`, schema `1.0.0`, bundled Python `3.13`).
  * **NEVER** use deprecated `bgl` module calls (removed in Blender 4.0+).
  * **NEVER** rely on legacy `bl_info` dictionaries in `__init__.py`; metadata belongs exclusively in `blender_manifest.toml`.
* **Compute Target:** `warp-lang == 1.17.0 (pinned PyPI CUDA 12.9)` on `cuda:0` (CUDA 12.x / 13.x runtime).
* **Viewport Graphics API:** OpenGL 4.5+ Core Profile (`gpu.platform.backend_type_get() == 'OPENGL'`).
  * Raw `glDrawArrays` calls **must** bind a valid OpenGL Vertex Array Object (`VAO`) and explicitly upload `ModelViewProjectionMatrix` to `gpu.shader.from_builtin('UNIFORM_COLOR')`.

---

## 2. Non-Negotiable Performance Invariants
1. **Zero Per-Frame CPU Readback:**
   * Never call `.numpy()`, `wp.to_torch().cpu()`, or `bpy.data.meshes.foreach_set()` inside the modal `TIMER` step (`WM_OT_warp_sph_start.modal`).
   * Particle positions must move strictly on the GPU from `sim_core.py` (`pos_sim`) into the mapped OpenGL VBO (`ZeroCopyGLBuffer52.map_to_warp()`).
2. **Structure of Arrays (SoA) & Coalesced Sorting:**
   * All per-particle GPU buffers must remain flat 1D SoA arrays (`wp.vec3` or `float`).
   * Never remove `k_reorder_soa` before the density and force passes; unsorted neighbor queries thrash the GPU L2 cache by up to 88% once the fluid becomes turbulent.
3. **Warp 1.17 CUDA Graph Compatibility:**
   * Any kernel added inside `RealtimeWCSPHSolver.rebuild_cuda_graph()` (`wp.ScopedCapture`) **must not** perform dynamic memory allocation (`wp.zeros`, `wp.empty`), host synchronization (`wp.synchronize_device`), or CPU print statements.
   * Note that `wp.HashGrid.build()` uses internal radix sort / prefix scan state and must remain **outside** `wp.ScopedCapture` unless using fixed-allocation graph-safe hash grid APIs.

---

## 3. Module Ownership Map
* `blender_manifest.toml` — Extension metadata & permissions.
* `__init__.py` — Blender `Operator`, `Panel`, `Scene` properties, and modal timer lifecycle.
* `interop_bridge.py` — `ctypes` OpenGL VAO/VBO creation and `cudaGraphicsGLRegisterBuffer` mapping.
* `viewport_draw.py` — Blender `gpu.shader` draw callbacks.
* `sim_core.py` — Primary `RealtimeWCSPHSolver`, cubic spline kernels, SoA reordering, and CUDA Graph capture.
* `demo_scenes.py` — GPU particle initializers and Blender wireframe/collider helper meshes for Demo 1 & Demo 2.
* `scene_config.py` — Physical regime presets (`WATER`, `MILK`, `SHOCKWAVE`) and domain bounding box extraction.
* `ipbf_solver.py` — Implicit Position-Based Fluids (IPBF) variational prediction and local $3 \times 3$ Hessian/Jacobi iteration kernels.
* `collider_sdf.py` — Analytical and SDF collider kernels.
* `whitewater.py` — Secondary spray ($n_f \le 5$), foam ($5 < n_f < 15$), and bubble ($n_f \ge 15$) classification and advection.
* `neighbor_rt.py` — Abstraction layer for `wp.HashGrid` and future OptiX BVH RT-core search.
* `neural_fno.py` — Zero-copy `wp.to_torch` DLPack bridge for 3D Fourier Neural Operators (`neuraloperator` / `FC_PINO`).
* `particle_upsample.py` — Free-surface particle detection for dynamic splitting.
* `mesher_io.py` — Disk export and `SplashSurf` CLI reconstruction wrapper.

---

## 4. Verification & Testing Protocol for Agents
Whenever you modify any physics kernel or module:
1. Run the headless unit test suite:
   ```bash
   python -m unittest discover -s tests -v
   ```
2. Verify that all tests in `tests/test_sim_core.py`, `tests/test_submodules.py`, and `tests/test_blender_mock.py` pass with zero NaNs, strict domain boundary enforcement, and identical results between eager kernel launches (`use_graph=False`) and CUDA Graph launches (`use_graph=True`).
