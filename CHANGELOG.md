# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Copy Diagnostics button (panel and preferences): one click copies add-on, Blender/backend, Warp, driver, cudart,
  GPU, transport, last error/close and recent log lines to the clipboard, with home paths redacted. The add-on now
  writes a small rotating log (256 KB x 2) to the extension user directory (`logs/warpsph.log`).
- Verified Warp installer: SHA-256-locked `warp-lang==1.17.0` (CUDA 12.9) wheel installed on
  user confirmation into extension-owned storage; the extension registers without Warp and imports
  CUDA modules only on Start. Adds `files` and `network` manifest permissions.
- Vulkan GPU interop transport (GPU copy, no CPU readback) for the exact certified Blender 5.2.0
  build, a labelled Readback fallback for other Vulkan builds, and a `gpu`-module points/spheres
  renderer. SSFR remains OpenGL-only; on Vulkan it shows spheres with a note.
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
