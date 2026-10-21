# AGENTS.md — AI Coding Agent Guidelines for `warp_sph_realtime`

<!-- BEGIN PROJECT_TRACKER_EVIDENCE_POLICY -->
## Project tracker and durable evidence

Project tracker: https://chatgpt.com/space/page_00fe12a2595881919c5f4a4a926008a7

This is the canonical agent rule for tracker maintenance. It supplements this
repository's existing local records and preserves its goal, scope, canonical
truth chain, task-selection rules, ownership, and approval/acceptance boundaries.
The tracker reports progress; it does not replace those authorities.

1. Before work, read this project's linked tracker and the required local records.
2. Update the tracker after meaningful progress, blockers, decisions, tests,
   source/package identity changes, or release-state changes, and before handoff
   or completion. State verified status, remaining work, next step, and evidence
   references. Keep existing required local records current as well.
3. Distinguish implementation, merge, and publication from runtime acceptance.
   Never record an unrun or blocked test as passed, or broaden a bounded result
   into platform, GPU, release, or product acceptance.
4. Save required logs, reports, screenshots, and gate receipts durably using this
   repository's established evidence/artifacts convention. When none applies,
   use `evidence/<dated-task>/`. Reference repo-relative paths and the exact
   source/package candidate and environment when relevant. Chat and temporary
   or external files must not be the sole progress or evidence record.
5. If the tracker cannot be read or updated, record a clearly marked pending
   tracker update in the existing pending-update queue, or otherwise in
   `docs/TRACKER_UPDATE_PENDING.md`. This is a delivery queue, not a competing
   status ledger or roadmap. Record the intended update, blocker, and evidence
   references; report the blocker at handoff/completion. Reconcile the actual
   tracker when access returns and mark the queued entry reconciled. Never claim
   the tracker is synchronized while an update remains pending.

These are agent instructions, not automated enforcement or additional authority
to change code, run tests, publish, or bypass existing approval boundaries.
### Tracker access fallback

Discover and use authenticated Pages tools when available. A web-reader failure
alone does not establish a Page permission denial. If authenticated tools are
unavailable or fail, read `docs/PROJECT_TRACKER_SNAPSHOT.md` and the required local
records. The snapshot records its Page URL, sequence and fetch time; it is a stale
read-only cache, not live authority, and does not change project holds.

Write proposed updates to `docs/TRACKER_UPDATE_PENDING.md`, never into the
snapshot. Notify dot with the project, evidence paths, Page URL and expected Page
or snapshot sequence. Dot reconciles against the live Page and acknowledges
publication. Preserve each pending entry until its publication receipt is
verified; local queuing alone is not synchronization.

<!-- END PROJECT_TRACKER_EVIDENCE_POLICY -->

This file provides strict architectural invariants, build/test commands, and hardware rules for AI coding agents (Claude Code, Gemini, Cursor, GitHub Copilot, etc.) modifying this repository.

---

## 1. Target Stack & Environment Constraints
* **Blender Target:** Blender `5.2.0 LTS+` (Extension system using `blender_manifest.toml`, schema `1.0.0`, bundled Python `3.13`).
  * **NEVER** use deprecated `bgl` module calls (removed in Blender 4.0+).
  * **NEVER** rely on legacy `bl_info` dictionaries in `warp_sph_realtime/__init__.py`; metadata belongs exclusively in `warp_sph_realtime/blender_manifest.toml`.
* **Compute Target:** `warp-lang == 1.17.0` (pinned PyPI CUDA 12.9 wheel) on `cuda:0`.
* **Viewport Graphics API:** the existing native path targets OpenGL 4.5+ Core Profile. Planned Vulkan/Readback support follows `docs/VULKAN_ZERO_COPY_PLAN.md`; this guidance change does not certify or implement it. Detect the running backend with `gpu.platform.backend_type_get()`.
  * Raw GL calls are permitted only on `OPENGL`. Raw `glDrawArrays` calls **must** bind a valid OpenGL Vertex Array Object (`VAO`) and explicitly upload `ModelViewProjectionMatrix` to `gpu.shader.from_builtin('UNIFORM_COLOR')`.
  * Vulkan native interop requires an approved exact-build ABI profile, independently verified callable addresses, matching Vulkan/CUDA device UUIDs before import, and a proven synchronous cleanup route. Unknown prerequisites mean unavailable, not a speculative native self-test.
  * Keep compute on `cuda:0`; never silently change devices. Fatal CUDA/context failures stop the session and require restart. Readback is a fallback only while the CUDA context is usable.

---

## 2. Non-Negotiable Performance Invariants
1. **Zero Per-Frame CPU Readback in accelerated transports:**
   * GLInterop and VulkanInterop must not use `.numpy()`, `wp.to_torch().cpu()`, or CPU mesh uploads in their per-frame path. The existing GL path moves `pos_sim` into its mapped VBO; the planned texture seam must preserve GPU-only position transport in both accelerated adapters.
   * The explicitly labelled `Readback` compatibility adapter is the sole exception: it may copy positions into preallocated host staging and upload a GPU texture. Confine this transfer to the adapter, outside solver kernels and CUDA graph capture; modal code delegates to the transport instead of performing direct readback or mesh updates. Measure its limits and never label it zero-copy.
   * One-time self-tests and separately identified correctness probes may read back; these do not qualify the no-readback steady-state performance lane. A GPU-to-GPU copy and a host wait are distinct from CPU position readback.
2. **Structure of Arrays (SoA) & Coalesced Sorting:**
   * Solver per-particle GPU buffers remain flat 1D SoA arrays (`wp.vec3` or `float`). The rendering transport may pack a separate flat `wp.vec4` buffer into an RGBA32F texture; this does not change solver layout. Initialize padded texels and draw exactly the logical particle count.
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
* `addon.py` — Thin UI: Blender `Operator`, `Panel`, `Scene` properties and the modal timer, which call the scene's `SimSession`. Imports Warp and CUDA-dependent modules only inside `_activate_runtime()`.
* `sim_session.py` — `SimSession` (one per scene): all runtime state and the start/tick/stop/dispose lifecycle; reads Warp bindings from `sim_session.runtime`, which only `addon._activate_runtime()` fills.
* `warp_installer.py` / `warp_runtime.py` / `warp_wheels.lock.json` — SHA-256-locked Warp wheel install into extension user storage and the runtime gate. Kept at the package root because storage is keyed by `__package__`.
* `render/interop_bridge.py` — `ctypes` OpenGL VAO/VBO creation and `cudaGraphicsGLRegisterBuffer` mapping; cudart/GL seam (`GLTextureNative`) for the texture adapter.
* `render/transport_gl.py` — GLInterop texture adapter (`cudaGraphicsGLRegisterImage`, GPU copy, no CPU readback) behind the P0.11 transport seam.
* `render/viewport_draw.py` — Blender `gpu.shader` draw callbacks and SSFR passes.
* `render/ssfr_shaders.py` — SSFR GLSL sources and CPU reference math.
* `sim/sim_core.py` — Primary `RealtimeWCSPHSolver`, cubic spline kernels, SoA reordering, and CUDA Graph capture.
* `scene/demo_scenes.py` — GPU particle initializers and Blender wireframe/collider helper meshes for Demo 1 & Demo 2.
* `sim/colliders.py` — Static mesh colliders: one `wp.Mesh` per collider (winding-number signed distance), mesh ids passed to kernels as a uint64 array.
* `scene/description.py` — Pure-NumPy `SceneDescription`/`MeshData` and validation (no bpy, no Warp).
* `scene/fill.py` — Lattice particle fill of fluid volumes (`fill_particles`); `RealtimeWCSPHSolver.from_scene` builds a solver from a description.
* `scene/blender_scene.py` — Duck-typed Blender adapter (`description_from_blender`); never imported by `sim/`.
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
