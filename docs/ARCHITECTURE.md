# WarpSPH Realtime: Architecture

This document describes what the built extension ships. Planned and research modules live in
`warp_sph_realtime/experimental/` and are listed separately at the end; they are not part of the
built zip.

---

## 1. Shipped architecture

```mermaid
graph TD
    subgraph Host ["Blender 5.2+ host and UI"]
        Manifest["blender_manifest.toml<br/>(metadata, permissions, build exclusions)"]
        Init["__init__.py<br/>(register / unregister, lazy addon import)"]
        Addon["addon.py<br/>(operators, panels, modal timer)"]
        SimSession["sim_session.py<br/>(per-scene SimSession: start, tick, stop, dispose)"]
        Props["props.py<br/>(scene.warp_sph settings, migration, VRAM estimate)"]
        Installer["warp_installer.py / warp_runtime.py<br/>(hash-locked Warp install, runtime gate)"]
        SceneCfg["scene/scene_config.py + demo_scenes.py<br/>(regime presets, Demo 1 and Demo 2)"]
    end

    subgraph Solver ["Warp GPU solver (cuda:0)"]
        SimCore["sim/sim_core.py<br/>(SoA buffers, hash grid, k_reorder_soa, WCSPH, CUDA Graph)"]
    end

    subgraph GL ["OpenGL path (legacy VBO interop)"]
        Interop["render/interop_bridge.py<br/>(VAO/VBO registered with CUDA)"]
        ViewDraw["render/viewport_draw.py + ssfr_shaders.py<br/>(points, spheres, 5-pass SSFR)"]
    end

    subgraph VK ["Vulkan path (GPU copy, no CPU readback)"]
        Session["render/session.py<br/>(adapter selection and fallback)"]
        Transport["render/transport.py<br/>(contract, labelled Readback fallback)"]
        VkTransport["render/transport_vulkan.py<br/>(exact-build Blender 5.2.0 interop)"]
        GpuPoints["render/gpu_points.py<br/>(points / spheres via gpu module)"]
    end

    Manifest -.-> Init
    Init --> Addon
    Addon --> Installer
    Addon --> SimSession
    Addon --> Props
    SimSession --> Props
    SimSession --> SceneCfg
    SimSession --> Session
    SimSession --> Interop
    SceneCfg --> SimCore
    SimCore -->|"wp.copy into mapped VBO"| Interop
    Interop --> ViewDraw
    SimCore -->|"flat wp.vec3 positions"| Session
    Session --> VkTransport
    Session --> Transport
    VkTransport --> GpuPoints
    Transport --> GpuPoints
```

Transport selection follows `gpu.platform.backend_type_get()` (what is running). Terminology: the
OpenGL and Vulkan paths keep particle data in GPU memory with no CPU readback ("zero host-copy").
They are not zero GPU copy; see the top of [`VULKAN_ZERO_COPY_PLAN.md`](./VULKAN_ZERO_COPY_PLAN.md).

## 2. Shipped modules

| Module | Role |
|---|---|
| `blender_manifest.toml` | Extension metadata (schema 1.0.0, Blender 5.2.0+), `files` and `network` permissions, tags, build exclusions (`experimental/` is not zipped). |
| `__init__.py` | `register()` / `unregister()` only; imports `addon` lazily so the extension registers without Warp. |
| `addon.py` | Thin UI: operators, panels (main and Advanced) and the modal timer, which call the scene's `SimSession`; registers `scene.warp_sph` and the `load_post` migration; imports CUDA modules only on Start (`_activate_runtime`). |
| `props.py` | `WARPSPH_SceneSettings` (`scene.warp_sph`, stiffness stored in thousands), the one-time migration of the old flat `scene.warp_sph_*` properties, and the GPU memory estimate and warning. |
| `sim_session.py` | `SimSession`, one per scene: all runtime state (solver, transport session or legacy VBO, draw handler, timer, telemetry) with `start` / `tick` / `stop` / `dispose`, transactional reallocation, quarantine on fatal errors. One run per process; `process_state()` flags survive add-on reloads. |
| `warp_installer.py`, `warp_runtime.py`, `warp_wheels.lock.json` | SHA-256-locked `warp-lang==1.17.0` (CUDA 12.9) install into extension storage and the runtime gate. `plan_install` (main thread, reads Blender) and `run_plan` (no `bpy`, cancellable: temp download or verified local-wheel copy, then atomic promotion) split the install; the Warp source (download / local wheel / existing folder, P1.11) selects the single directory Warp may be imported from, and the foreign-Warp guard accepts only that one. |
| `install_job.py` | Pure-Python worker thread for `run_plan` with a lock-protected progress snapshot and a cancel flag (P1.4). |
| `install_ui.py` | Preferences UI: Warp source, progress bar and Cancel (polled by a `bpy.app.timers` timer), first-run hardware check panel, install/cancel operators. The only `bpy` consumer of `install_job`. |
| `diagnostics.py` | Diagnostics report, log setup, and the pre-Warp hardware probe (`nvidia_smi_info`, `hardware_summary`). |
| `sim/sim_core.py` | `RealtimeWCSPHSolver`: SoA buffers, `wp.HashGrid`, `k_reorder_soa`, cubic-spline WCSPH, SDF obstacle collisions, CUDA Graph capture. |
| `scene/scene_config.py`, `scene/demo_scenes.py` | Regime presets (`WATER`, `MILK`, `SHOCKWAVE`) and the two demo scenes (`DOUBLE_DAM_PILLARS`, `BLOCK_FLOP_SPLASH`). |
| `render/interop_bridge.py` | OpenGL VAO/VBO creation and `cudaGraphicsGLRegisterBuffer` mapping (legacy VBO path), plus the cudart/GL seam for the GLInterop texture adapter. |
| `render/transport_gl.py` | GLInterop texture adapter for Points on the OpenGL backend: pack kernel, `cudaGraphicsGLRegisterImage`, GPU copy into a `GPUTexture`, no CPU readback; falls back once to labelled Readback. |
| `render/viewport_draw.py`, `render/ssfr_shaders.py` | Legacy OpenGL VBO draw callbacks: points, spheres and the raw-GL SSFR liquid renderer. |
| `render/session.py`, `render/transport.py` | Transport session owner and contract; Vulkan tries the exact-build adapter and OpenGL (Points mode) the GLInterop texture adapter; each falls back once to the labelled Readback transport. |
| `render/transport_vulkan.py`, `render/blender_symbols*.py`, `render/profiles/` | Vulkan CUDA interop for the exact certified Blender 5.2.0 build, with symbol resolution profiles. |
| `render/gpu_points.py` | Backend-neutral points/spheres draw via Blender's `gpu` module, and the scene-to-`DrawSettings` mapping. |
| `render/gpu_ssfr.py` | Backend-neutral SSFR via the `gpu` module (depth impostors, narrow-range filter, half-res thickness, depth-tested composite); `ViewportRenderer` is the session default and dispatches SSFR vs points/spheres. |

## 3. Experimental (not shipped)

These modules are in the repository under `warp_sph_realtime/experimental/` and are excluded from
the built zip. None is wired into the UI or runtime.

| Module | Intent |
|---|---|
| `ipbf_solver.py` | Implicit position-based fluids (IPBF) kernels. |
| `collider_sdf.py` | Analytical / SDF collider kernels. |
| `whitewater.py` | Spray, foam and bubble classification and advection. |
| `neighbor_rt.py` | Neighbor-search abstraction (`wp.HashGrid`; OptiX BVH is a future idea). |
| `neural_fno.py` | DLPack bridge for 3D Fourier neural operators. |
| `particle_upsample.py` | Free-surface particle detection for splitting. |
| `mesher_io.py` | Disk export and SplashSurf reconstruction wrapper. |
| `differentiable_pour.py`, `keyframe_pour.py` | Differentiable-pour and keyframe studies (results in `DIFFERENTIABLE_POUR_*.md`, `KEYFRAME_*.md`, `CONTACT_CORRECTION_RESULTS.md`). |

Research background for these ideas is in [`RESEARCH.md`](./RESEARCH.md).
