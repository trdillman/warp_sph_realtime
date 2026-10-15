# Context

**Current Task**: Repo restructured into the `warp_sph_realtime/` package (sim, scene, render, experimental) with pyproject tooling, CI and build script; remote SSFR repair `5d8a5ff` ported onto it. On local `main`, not pushed.

**Key Decisions**
- Extension source is the inner `warp_sph_realtime/` folder; build with `python scripts/build_extension.py [--install]`. Relative imports only inside the package.
- `__init__.py` registers lazily via `addon.py`, so `sim/` and `experimental/` import without bpy.
- Live-Blender probe scripts (`tests/run_ssfr_gui.py`, `tests/runtime_ssfr_*.py`) are excluded from pyright; pyright suppressions are documented in `pyproject.toml`.

**Verification**: pytest 33 passed, ruff clean, pyright 0 errors; Blender 5.2 extension build/install/enable checked headless. SSFR GUI evidence (`docs/verification/SSFR_REVIEW_REPAIR.md`) predates the restructure; GUI probes not rerun since.

**Next Steps**
- Push `main` after owner approval; then retire `feat/ssfr-renderer` / `refactor/repo-structure` / `port/ssfr-repair`.
- Port `origin/feat/isolated-warp-installer` (old root layout) into the package before merging.
- Rerun the GUI probes (`tests/run_ssfr_gui.py --case controls|workflow`) on the new layout.
