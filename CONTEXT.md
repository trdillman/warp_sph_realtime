# Context

**Current Task**: Repo restructured into the `warp_sph_realtime/` package (sim, scene, render, experimental) with pyproject tooling, CI, and build script; landed on local `main` (not yet pushed).

**Key Decisions**
- Blender extension source is the inner `warp_sph_realtime/` folder; build with `python scripts/build_extension.py [--install]`. Only relative imports inside the package.
- `__init__.py` registers lazily via `addon.py`, so `sim/` and `experimental/` import without bpy and are testable headless.
- Pyright suppressions are kept (460 baseline errors are bpy/warp stub false positives); each is documented in `pyproject.toml`.

**Verification (main @ restructure commit)**: pytest 25 passed, unittest 25 OK, ruff clean, pyright 0 errors; Blender 5.2 extension validate/build/install/enable verified headless. No GPU sim or GUI run since restructure.

**Next Steps**
- Push `main` (and decide fate of `feat/ssfr-renderer`, `refactor/repo-structure`) after owner approval.
- Port `origin/feat/isolated-warp-installer` (old root-path layout) into `warp_sph_realtime/` before merging.
- Remaining SSFR review findings: GL-state restoration, multi-context cleanup churn, perspective point diameter, orthographic view/filter scaling, off-center projection.
