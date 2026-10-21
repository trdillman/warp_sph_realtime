# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- `RealtimeWCSPHSolver` takes an optional `device` (default `"cuda:0"`, unchanged for the add-on). `"cpu"` launches
  the kernels eagerly with no CUDA graph; it exists for headless validation in CI.
- CI and release pipeline: the check job now also runs on `windows-latest`; Ruff additionally enforces `B`, `UP` and
  `I` (violations fixed, `UP018` documented as incompatible with Warp kernels); the version is single-sourced from
  `blender_manifest.toml` (Blender qualification tests derive the zip name from it, `tests/test_version_consistency.py`
  guards it); `scripts/build_extension.py` selects the zip by manifest version, prints its SHA-256 and takes
  `--tag` to refuse a tag that differs from the manifest. A `v*` tag now runs `.github/workflows/release.yml`:
  CI, pinned Blender 5.2.0 `extension validate` and `build`, workflow artifact, and a GitHub Release.
- Runtime state moved from `addon.py` globals into a per-scene `SimSession` (`sim_session.py`) with
  start/tick/stop/dispose; `addon.py` is now thin UI. Behaviour is unchanged for a single scene. One run per
  Blender process: Start in a second scene while another runs reports which scene to stop first, and the
  panel there says so. A run whose scene is deleted stops with a warning.
- Scene settings moved from 17 flat `scene.warp_sph_*` properties into one settings group, `scene.warp_sph`
  (`props.py`); the per-object role stays on objects. Files saved by earlier versions are migrated once when
  opened (or when the add-on is enabled with such a file open): saved values are copied and the old ones
  removed; a missing or unusable value keeps the default; an old value never overwrites one already set in
  the group.
- Fluid stiffness is shown and stored in thousands (default 90 = 90,000; range 10 to 400), so each slider step
  is 1,000 instead of the previous 500,000 step over a 10,000 to 400,000 range. Old values are divided by
  1,000 on migration; the solver receives the same value as before.
- Stiffness, CUDA Graph capture and the SSFR filter and absorption settings moved to a new, collapsed
  **Advanced** sub-panel.

### Added
- Physics validation (P2.6): `sim/validation.py` runs the real solver headless on any Warp device and measures
  the quasi-2D dam-break surge front against Martin & Moyce (1952, `n^2 = 2`), particle conservation (count, NaN,
  domain containment), total-energy growth and rest density (unclamped SPH sum). `tests/test_validation.py` runs
  coarse variants on Warp's CPU device in CI (about 17 s cold) and higher-resolution plus CUDA-graph variants on
  GPU machines; `scripts/validate_dambreak.py` writes a plot and JSON summary under `evidence/`. Measured front
  deviation is 6-10 % of Z at 8-48 cells per column width, lagging when coarse and leading when fine, with no
  energy growth and 0.2 % bulk rest-density error; `docs/VALIDATION.md` has the method, numbers and limitations,
  including that the reference table could not be checked against the paywalled 1952 paper.
- Warp install runs in the background (P1.4): the download, checksum check and extraction run on a worker thread
  polled by a Blender timer, so the UI stays responsive. Add-on Preferences show a progress bar and a Cancel
  button; the wheel is downloaded to a temporary file, verified and only then extracted and promoted, so Cancel
  or a failure leaves no partial install. The worker never touches `bpy`.
- First-run **Hardware Check** in Add-on Preferences (P1.4): GPU, NVIDIA driver, supported CUDA version, VRAM
  and Blender graphics backend, read from `nvidia-smi` so it works before Warp is installed (it switches to
  Warp's own values once Warp is active), with a plain verdict such as "update the NVIDIA driver".
- **Warp source** choice in Add-on Preferences (P1.11): *Download* (default), *Install from local wheel file*
  (offline; must match the SHA-256 lock, then the same verified extraction into private storage) or *Use existing
  Warp folder* (a folder containing `warp`, for example a studio site-packages). An existing folder is added to
  the import path at Start and is never modified or deleted (no `__pycache__` is written there); Start runs the
  driver and known-answer self-test and names the folder if it fails. Only 1.17.x is accepted unless
  "Allow uncertified version (unsupported)" is ticked, which shows a persistent warning. The foreign-Warp guard
  accepts only the chosen folder, switching back to Download restores the owned install, and only Download needs
  Blender's Allow Online Access. Copy Diagnostics reports which source is in use.
- GPU memory warning under the particle count (Demo) and spacing (Custom): about 150 bytes per particle plus
  32 MB, measured on Warp 1.17; it warns above half of the GPU's memory (read once after the first Start) or
  above 4 GB while that is unknown. The Custom estimate is an upper bound from the fluid objects' bounding
  boxes.
- Scene authoring: mark meshes as **Domain**, **Fluid** or **Collider** (object `warp_sph.role`, sidebar buttons and
  a Properties > Physics panel), choose **Custom** as the scene source, set the particle spacing, and press Start;
  the fluid volumes are filled and the colliders built from the evaluated meshes. **Add Fluid Scene** creates a
  starter domain, fluid block and collider. The panel lists objects by role and warns about a missing domain or
  fluid and fluid outside the domain; Start refuses an invalid scene with the reason and leaves no GPU state. The
  Demo source is unchanged and remains the default; the early `warp_sph_role` custom property is still read.
- Copy Diagnostics button (panel and preferences): one click copies add-on, Blender/backend, Warp, driver, cudart,
  GPU, transport, last error/close and recent log lines to the clipboard, with home paths redacted. The add-on now
  writes a small rotating log (256 KB x 2) to the extension user directory (`logs/warpsph.log`).
- Verified Warp installer: SHA-256-locked `warp-lang==1.17.0` (CUDA 12.9) wheel installed on
  user confirmation into extension-owned storage; the extension registers without Warp and imports
  CUDA modules only on Start. Adds `files` and `network` manifest permissions.
- Vulkan GPU interop transport (GPU copy, no CPU readback) for the exact certified Blender 5.2.0
  build, a labelled Readback fallback for other Vulkan builds, and a `gpu`-module points/spheres
  renderer.
- GLInterop texture transport (`render/transport_gl.py`, GPU copy, no CPU readback) for Points mode on
  Blender's OpenGL backend. It registers an RGBA32F `GPUTexture` with CUDA (`cudaGraphicsGLRegisterImage`),
  publishes positions on the solver stream, is drawn by the `gpu`-module points renderer through the
  same `TransportSession` seam as Vulkan, and falls back once to the labelled Readback transport. The route
  is chosen when Start is pressed; SSFR liquid and sphere impostors started on OpenGL keep the legacy
  OpenGL VBO path. The Blender Python `GPUTexture` has no GL name, so the adapter identifies it by content
  once at setup and verifies the copy path with a 4x4 known-answer self-test.
- SSFR liquid on Vulkan and on OpenGL transport sessions: `render/gpu_ssfr.py` ports the depth-impostor,
  narrow-range filter, half-res thickness and depth-tested composite passes to Blender's `gpu` module
  (per-renderer offscreen targets, reallocated on resize). Differences from the legacy raw-GL renderer: no
  refraction offset (the scene behind the liquid is attenuated in place) and scene occlusion only at composite.
  The legacy OpenGL VBO path keeps its raw-GL SSFR.
- Manifest `tags = ["Physics"]`.
- Experimental differentiable-pour, keyframe and compliant-contact studies (not shipped).
- Pause / single-step simulation controls and fine-step particle radius.
- `pyproject.toml` with pytest, ruff, and pyright configuration and a `dev` dependency group.
- `scripts/build_extension.py` to validate, build, and optionally install the extension zip.
- GitHub Actions CI running ruff, pyright, and pytest (CUDA tests skip on hosted runners).

### Fixed
- Legacy OpenGL path: the particle VBO is always unmapped from CUDA, even when a solver step fails while it is mapped (`ZeroCopyGLBuffer52.mapped()`); after a sticky CUDA error the step error is kept and the unmap failure is attached to it.
- SSFR: matching-format MSAA depth resolve before half-res scaling, exception-safe GL state
  restoration, cleanup after failed shader / partial renderer initialization, and rejection of
  unsupported default-framebuffer depth formats.

### Changed
- Repository restructured: the extension now lives in the `warp_sph_realtime/` package folder,
  split into `sim/`, `scene/`, `render/`, and `experimental/` subpackages.
- Blender UI and runtime moved from `__init__.py` to `addon.py`; `__init__.py` registers lazily
  so simulation modules import without Blender.
- Design documents moved to `docs/`.
- `experimental/` is excluded from the built extension zip (`paths_exclude_pattern`); README,
  ARCHITECTURE and SPEC_AND_PLAN now describe only shipped features and label the rest
  "Experimental (not shipped)". The build script fails if the zip contains `experimental/`.
- Tests import the package directly; per-file `sys.path` edits and try/except import fallbacks removed.

### Removed
- `pyrightconfig.json` (settings moved to `pyproject.toml`, each suppression documented).

## [0.1.0]
- Initial real-time WCSPH viewport solver with zero-copy OpenGL-CUDA interop, CUDA Graph capture,
  demo scenes, and the 5-pass screen-space fluid renderer.
