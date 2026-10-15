# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Verified Warp installer: SHA-256-locked `warp-lang==1.17.0` (CUDA 12.9) wheel installed on
  user confirmation into extension-owned storage; the extension registers without Warp and imports
  CUDA modules only on Start. Adds `files` and `network` manifest permissions.
- Pause / single-step simulation controls and fine-step particle radius.
- `pyproject.toml` with pytest, ruff, and pyright configuration and a `dev` dependency group.
- `scripts/build_extension.py` to validate, build, and optionally install the extension zip.
- GitHub Actions CI running ruff, pyright, and pytest (CUDA tests skip on hosted runners).

### Fixed
- SSFR: matching-format MSAA depth resolve before half-res scaling, exception-safe GL state
  restoration, cleanup after failed shader / partial renderer initialization, and rejection of
  unsupported default-framebuffer depth formats.

### Changed
- Repository restructured: the extension now lives in the `warp_sph_realtime/` package folder,
  split into `sim/`, `scene/`, `render/`, and `experimental/` subpackages.
- Blender UI and runtime moved from `__init__.py` to `addon.py`; `__init__.py` registers lazily
  so simulation modules import without Blender.
- Design documents moved to `docs/`.
- Tests import the package directly; per-file `sys.path` edits and try/except import fallbacks removed.

### Removed
- `pyrightconfig.json` (settings moved to `pyproject.toml`, each suppression documented).

## [0.1.0]
- Initial real-time WCSPH viewport solver with zero-copy OpenGL-CUDA interop, CUDA Graph capture,
  demo scenes, and the 5-pass screen-space fluid renderer.
