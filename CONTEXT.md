# Context

**Current Task**: Repo restructured into the `warp_sph_realtime/` package (on `origin/main`). Verified Warp installer (`origin/feat/isolated-warp-installer`) ported onto the package layout and merged via PR #3.

**Key Decisions**
- Extension source is the inner `warp_sph_realtime/` folder; build with `python scripts/build_extension.py [--install]`. Relative imports only inside the package.
- `warp_installer.py` / `warp_runtime.py` / lock stay at the package root (storage keyed by `__package__`); `addon._activate_runtime()` lazily imports Warp + render/sim/scene and is idempotent.
- Installer adds network/files permissions, so it merges via reviewed PR, not direct push.

**Verification**: pytest + ruff + pyright clean; env-gated Blender tests (missing-Warp registration, locked-wheel CUDA/GL) pass; SSFR GUI probes PASS with Warp in the owned directory.

CI installs `fake-bpy-module` stubs and pins pyright/warp-lang; green on `main`.

**Next Steps**
- Only `main` remains; start new work on feature branches from it.
