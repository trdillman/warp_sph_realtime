# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- User manual in `docs/manual/` (install, first run, authoring a scene, physics settings and presets, viewport
  display and transports, record / bake / export, known limits) and an FAQ for GPU, driver, Warp install, backend,
  Readback, VRAM and performance problems, quoting the add-on's own messages (P2.8).
- `examples/`: four Blender Python scripts that build a Custom scene (pour into a box, dam break, moving paddle,
  viscous drop); `tests/test_examples.py` checks that every setting, operator and enum value they use exists.
- `examples/showcase/`: five feature demos (viewport display and transports, dam-break validation, emitter with a
  rotating collider, bake / mesh / USD export, physics presets side by side) and `scripts/render_showcase.py`,
  which runs them in Blender one at a time and collects screenshots, renders, a contact sheet and a summary.
- The Time Step (CFL) tooltip now says what the README says: above 0.4 hard impacts can blow up at high particle
  counts.

### Changed
- **Licence is now GPL-3.0-or-later** (was MIT in the manifest, with no licence file). `LICENSE` and
  `THIRD_PARTY_NOTICES.md` ship inside the add-on zip (release plan D2).
- **Particle Spacing** is now a length in scene units (`subtype='DISTANCE'`, `unit='LENGTH'`, range 0.0001 to 1000),
  shown and typed with the scene's units and Unit Scale like the objects, instead of a number of metres. It is
  multiplied by the Unit Scale for the solver. With the default Unit Scale of 1 nothing changes, and old files keep
  their value (migrated as saved); a file saved by a development build with another Unit Scale reads the same number
  as scene units.
- `RealtimeWCSPHSolver` takes an optional `device` (default `"cuda:0"`, unchanged for the add-on). `"cpu"` launches
  the kernels eagerly with no CUDA graph; it exists for headless validation in CI.
- Custom scenes use Blender's scene **Unit Scale** (Blender units to metres) and scene **Gravity** (P2.1).
  Before, 1 Blender unit was 1 m and gravity was fixed at -9.81 m/s^2.
- Both demos now run through the collider path (P2.1): the pillars and sphere are one closed mesh collider
  instead of analytic shapes. Visuals and behaviour match: at 1M particles the centre of mass after the
  comparison run differs by under 1 mm, no particle penetrates the obstacles, and the step time is within
  about 2.5% of the analytic path.
- Mesh collider signed distance takes the sign from the closest face's pseudo-normal and confirms deep
  "inside" answers with the winding number, instead of using the winding number on every query near a
  surface (which doubled the Demo 1 step time at 1M particles).
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
- Output (P2.3). The playback point cloud has a per-particle **`age`** attribute (seconds since the particle first
  appears in the cache; emitters release particles over time, so ages differ). It is derived from the cache, whose
  format is unchanged, so existing caches work: `cache/age.py` finds each id's first cached frame in one incremental
  pass over the id arrays and keeps it while you scrub, jump or play (a re-baked frame rolls the scan back, detected by
  file timestamps). **Add Mesh Preset** adds a mesh object with a Geometry Nodes modifier (point cloud -> Points to
  Volume -> Volume to Mesh -> Blur Attribute smoothing -> smooth shading -> Set Material) whose radius, voxel size,
  threshold, smoothing and material are modifier inputs, defaulting to 1.5 / 0.5 times the particle spacing. **Export
  USD** / **Export Alembic** write that mesh over exactly the baked frame range (re-imported in Blender 5.2: the same
  vertex counts per frame). New `cache/age.py`, `cache/output.py` (pure) and `cache/blender_output.py` (`bpy`).
  Blender 5.2 note: Geometry Nodes modifier inputs are set through `modifier.properties.inputs.<identifier>.value`
  (ID-property access no longer works) and a Python edit needs `object.update_tag()` to re-evaluate. Point cloud
  objects do not export to USD or Alembic in Blender 5.2, so the export is of the meshed object.
- Physics controls (P2.5): a **Fluid** box with **Water**, **Viscous** and **Stylised** presets (stiffness, time
  step, viscosity and sub-steps together; the active one is highlighted) and a **Viscosity** setting (kg/s per
  particle, default 0.15 as before, range 0 to 20). A change applies live: the CUDA graph is recaptured once, as for
  stiffness. The explicit viscous force is capped where it would overshoot (`RealtimeWCSPHSolver.effective_viscosity`);
  the panel shows when the cap is active. `from_scene` takes a `viscosity`.
- Particle ids for emitted particles (P1.6 with P2.1): each particle an emitter releases gets a fresh unique id, so
  cached frames keep one id per particle as the pool fills. A bake of an emitter scene records the live particle count
  per frame (frames grow), resume restores the emitters' progress and the clock, and bakes now feed animated-collider
  keyframes the way the live run does.
- Physics validation (P2.6): `sim/validation.py` runs the real solver headless on any Warp device and measures
  the quasi-2D dam-break surge front against Martin & Moyce (1952, `n^2 = 2`), particle conservation (count, NaN,
  domain containment), total-energy growth and rest density (unclamped SPH sum). `tests/test_validation.py` runs
  coarse variants on Warp's CPU device in CI (about 17 s cold) and higher-resolution plus CUDA-graph variants on
  GPU machines; `scripts/validate_dambreak.py` writes a plot and JSON summary under `evidence/`. Measured front
  deviation is 6-10 % of Z at 8-48 cells per column width, lagging when coarse and leading when fine, with no
  energy growth and 0.2 % bulk rest-density error; `docs/VALIDATION.md` has the method, numbers and limitations,
  including that the reference table could not be checked against the paywalled 1952 paper.
- Record, bake and playback (P1.6, P2.2). New `cache/` package: `FrameStore` (a VRAM ring of device snapshots with a
  second CUDA stream and pinned host buffers, so the readback is asynchronous and never stalls the solver, plus a
  writer thread and a disk adapter) writes one plain `.npz` per frame with `position`, `velocity` and `id`.
  **Record / Bake Cache** sub-panel: bake a frame range from the initial state in timer-driven chunks (never a blocking
  loop) with progress and Cancel; **Resume** after cancel, failure or a closed session; **Invalidate From Current Frame**;
  a cache folder per scene (default next to the saved .blend, temp when unsaved, `scene.warp_sph.cache_dir`); a size
  estimate (28 bytes per particle per frame) and free-disk check before baking. Frames are written atomically and in
  order, so cancel or a crash leaves a consistent prefix; resume refuses a cache with different settings. **Show Cache as
  Point Cloud** adds a `PointCloud` object synced to the timeline by a `frame_change_post` handler, with `velocity`,
  `id` and `radius` attributes; it renders in Cycles. The readback exists only in the explicit bake action, never in the live
  transports. `cache/` is `bpy`-free except `cache/blender_adapter.py`. Measured on an RTX 4080 SUPER
  (`scripts/benchmark_cache_record.py`, median of 3): recording adds about 7 to 8 ms per frame at 66 steps per frame (real
  time at 24 fps) for 91k and 1M particles; when the solver outruns the disk (about 500 to 650 MB/s here, 28 MB per frame
  at 1M particles) the disk sets the pace and the ring applies backpressure.
- Stable particle id (`RealtimeWCSPHSolver.pid`): every step rewrites `pos_sim` / `vel` in spatial-sort order, so the array
  index did not identify a particle. `k_reorder_soa_id` (the existing reorder plus the id) replaces the reorder launch in `step()`; the id arrays are
  not in the CUDA graph. Measured at 1M particles: 5.94 ms per substep against 5.86 ms before (within run-to-run
  noise); the telemetry footprint grows by 8 bytes per particle.
- **Emitter** role (P2.1): a mesh surface releases particles along its face normals (Speed) or a fixed velocity
  Vector, at a Rate (0 = a full stream) over a Start/End frame range. The particle pool is allocated once at
  Start from **Max Particles** (`scene.warp_sph.max_particles`, default 1,000,000); emission writes into it with
  no reallocation and no allocation in CUDA graph capture. When the budget is full the emitters stop, Blender
  shows a warning and the panel shows "Particle budget reached". Emitted particles start at least one spacing
  (plus one step of travel) from domain walls and colliders; Start refuses an emitter that is outside the
  domain, inside a collider or too close to either.
- Animated colliders (P2.1): a collider whose location or rotation is animated (keyframes, drivers,
  constraints or a parent) moves the fluid. Its world matrix is read once per Blender frame on the main thread;
  the pose is interpolated per substep on the GPU and the mesh is never rebuilt. The collider's velocity
  (linear and angular) drives the wall response, so a moving paddle pushes fluid. While such a scene runs, the
  timeline follows the simulation frame. Not supported: deforming colliders (they keep their Start-frame
  shape) and animated scale or shear (the run stops with an error).
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
- Bake with a Unit Scale other than 1: the first baked frame, the viewport during a bake and a resumed bake used
  metres where the cache and the viewport use scene units (frame 1 of a centimetre scene was 100x too small and a
  resume restarted from the wrong place). Every cached frame is now in scene units and a resume converts back.
- Add Mesh Preset with a Unit Scale other than 1 sized the splat radius and voxels from the spacing in metres. The
  bake signature now records the Unit Scale (only when it is not 1, so existing caches still resume) and the
  preset converts the spacing to scene units.
- The initial particle fill of a custom scene started half a spacing from the domain walls and colliders, inside the
  solver's one-spacing boundary layer, so the layer threw the fluid out (a resting box peaked at 58 m/s and its mean
  height rose 178 %). The fill now keeps one spacing from walls and collider surfaces (where the boundary force is
  zero); the same box peaks at 2 m/s and does not rise. The built-in demos keep the same clearance at coarse
  particle counts (unchanged at 1M).
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
