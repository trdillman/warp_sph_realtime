# AGENTS.md — AI Coding Agent Guidelines for `warp_sph_realtime`

This file provides strict architectural invariants, build/test commands, and hardware rules for AI coding agents (Claude Code, Gemini, Cursor, GitHub Copilot, etc.) modifying this repository.

---

## 1. Target Stack & Environment Constraints
* **Blender Target:** Blender `5.2.0 LTS+` (Extension system using `blender_manifest.toml`, schema `1.0.0`, bundled Python `3.13`).
  * **NEVER** use deprecated `bgl` module calls (removed in Blender 4.0+).
  * **NEVER** rely on legacy `bl_info` dictionaries in `warp_sph_realtime/__init__.py`; metadata belongs exclusively in `warp_sph_realtime/blender_manifest.toml`.
* **Compute Target:** `warp-lang == 1.17.0` (pinned PyPI CUDA 12.9 wheel) on `cuda:0`.
* **Viewport Graphics API:** OpenGL 4.5+ Core Profile (`gpu.platform.backend_type_get() == 'OPENGL'`).
  * Raw `glDrawArrays` calls **must** bind a valid OpenGL Vertex Array Object (`VAO`) and explicitly upload `ModelViewProjectionMatrix` to `gpu.shader.from_builtin('UNIFORM_COLOR')`.

---

## 2. Non-Negotiable Performance Invariants
1. **Zero Per-Frame CPU Readback:**
   * Never call `.numpy()`, `wp.to_torch().cpu()`, or `bpy.data.meshes.foreach_set()` inside the modal `TIMER` step (`WM_OT_warp_sph_start.modal`).
   * Particle positions must move strictly on the GPU from `sim/sim_core.py` (`pos_sim`) into the mapped OpenGL VBO (`ZeroCopyGLBuffer52.map_to_warp()`).
2. **Structure of Arrays (SoA) & Coalesced Sorting:**
   * All per-particle GPU buffers must remain flat 1D SoA arrays (`wp.vec3` or `float`).
   * Never remove `k_reorder_soa` before the density and force passes; unsorted neighbor queries thrash the GPU L2 cache by up to 88% once the fluid becomes turbulent.
3. **Warp 1.17 CUDA Graph Compatibility:**
   * Any kernel added inside `RealtimeWCSPHSolver.rebuild_cuda_graph()` (`wp.ScopedCapture`) **must not** perform dynamic memory allocation (`wp.zeros`, `wp.empty`), host synchronization (`wp.synchronize_device`), or CPU print statements.
   * Note that `wp.HashGrid.build()` uses internal radix sort / prefix scan state and must remain **outside** `wp.ScopedCapture` unless using fixed-allocation graph-safe hash grid APIs.

---

## 3. Module Ownership Map
All paths are relative to the `warp_sph_realtime/` package folder. Use relative imports only
(Blender loads the package as `bl_ext.<repo>.warp_sph_realtime`). Keep `bpy` imports out of
`sim/` and `experimental/` so they stay importable headless.

* `blender_manifest.toml` — Extension metadata & permissions.
* `__init__.py` — `register()` / `unregister()` only; lazily imports `addon`.
* `addon.py` — Blender `Operator`, `Panel`, `Scene` properties, and modal timer lifecycle. Imports Warp and CUDA-dependent modules only inside `_activate_runtime()`.
* `warp_installer.py` / `warp_runtime.py` / `warp_wheels.lock.json` — SHA-256-locked Warp wheel install into extension user storage and the runtime gate. Kept at the package root because storage is keyed by `__package__`.
* `render/interop_bridge.py` — `ctypes` OpenGL VAO/VBO creation and `cudaGraphicsGLRegisterBuffer` mapping.
* `render/viewport_draw.py` — Blender `gpu.shader` draw callbacks and SSFR passes.
* `render/ssfr_shaders.py` — SSFR GLSL sources and CPU reference math.
* `sim/sim_core.py` — Primary `RealtimeWCSPHSolver`, cubic spline kernels, SoA reordering, and CUDA Graph capture.
* `scene/demo_scenes.py` — GPU particle initializers and Blender wireframe/collider helper meshes for Demo 1 & Demo 2.
* `scene/scene_config.py` — Physical regime presets (`WATER`, `MILK`, `SHOCKWAVE`) and domain bounding box extraction.
* `experimental/ipbf_solver.py` — Implicit Position-Based Fluids (IPBF) variational prediction and local $3 \times 3$ Hessian/Jacobi iteration kernels.
* `experimental/collider_sdf.py` — Analytical and SDF collider kernels.
* `experimental/whitewater.py` — Secondary spray ($n_f \le 5$), foam ($5 < n_f < 15$), and bubble ($n_f \ge 15$) classification and advection.
* `experimental/neighbor_rt.py` — Abstraction layer for `wp.HashGrid` and future OptiX BVH RT-core search.
* `experimental/neural_fno.py` — Zero-copy `wp.to_torch` DLPack bridge for 3D Fourier Neural Operators (`neuraloperator` / `FC_PINO`).
* `experimental/particle_upsample.py` — Free-surface particle detection for dynamic splitting.
* `experimental/mesher_io.py` — Disk export and `SplashSurf` CLI reconstruction wrapper.

---

## 4. Verification & Testing Protocol for Agents
Whenever you modify any physics kernel or module:
1. Run the headless unit test suite:
   ```bash
   pytest          # from the repository root; or: python -m unittest discover -s tests -v
   ruff check .
   pyright
   ```
2. Verify that all tests in `tests/` pass with zero NaNs, strict domain boundary enforcement, and identical results between eager kernel launches (`use_graph=False`) and CUDA Graph launches (`use_graph=True`).
