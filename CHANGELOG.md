# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
