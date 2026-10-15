# Context

**Current Task**: Repo restructured into the `warp_sph_realtime/` package (sim, scene, render, experimental) with pyproject tooling, CI and build script; remote SSFR repairs `5d8a5ff` + `c674e38` ported onto it. Pushed to `origin/main`.

**Key Decisions**
- Extension source is the inner `warp_sph_realtime/` folder; build with `python scripts/build_extension.py [--install]`. Relative imports only inside the package.
- `__init__.py` registers lazily via `addon.py`, so `sim/` and `experimental/` import without bpy.
- Live-Blender probe scripts (`tests/run_ssfr_gui.py`, `tests/runtime_ssfr_*.py`) are excluded from pyright; pyright suppressions are documented in `pyproject.toml`.

**Verification**: pytest 35 passed, ruff clean, pyright 0 errors; Blender 5.2 extension build/install/enable checked headless. SSFR GUI probes (controls + workflow) rerun on the package layout: PASS.

**Next Steps**
- Retire superseded branches `feat/ssfr-renderer` (local + origin).
- Port `origin/feat/isolated-warp-installer` (old root layout) into the package before merging.
