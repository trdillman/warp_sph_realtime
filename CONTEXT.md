# Context

**Current Task**: Preserve Blender's independent read/draw framebuffer bindings during SSFR allocation and resize; verified in Blender 5.2.0 LTS.

**Key Decisions & Robustness Fixes**
- `_resize_fbos()` saves both incoming framebuffer bindings before allocation and restores each target separately, so the first draw and viewport resize retain Blender's scene-capture/composite target. Queries run only during allocation/resize.
- Added `ssfr_shaders.py` with GLSL 450 Core shaders and CPU reference math (`compute_projected_filter_sigma`, `narrow_range_filter_1d_cpu`, `compute_beer_lambert_transmittance`, `compute_schlick_fresnel`).
- Fixed Narrow-Range Filter boundary handling and silhouette preservation (rejecting foreground samples beyond the delta neighborhood to prevent bleeding across depth discontinuities).
- Added camera projection auto-detection for both Perspective and Orthographic views in point sizing (`SSFR_DEPTH_VERT`), depth scaling (`SSFR_NARROW_RANGE_FRAG`), and normal reconstruction (`SSFR_COMPOSITE_FRAG`).
- Upgraded `interop_bridge.py` (`_GLInterop`) with FBO, Texture2D, packed depth-stencil (`GL_DEPTH24_STENCIL8` with `GL_DEPTH_STENCIL_ATTACHMENT`), `glGetError`, and GLSL shader program compilation bindings.
- Upgraded `viewport_draw.py` with `SSFRViewportRenderer` supporting 3 selectable viewport modes (`SSFR_LIQUID`, `SPHERES`, `POINTS`) and per-context offscreen resources.
- Fixed point-sprite Y normal inversion in `SPHERE_IMPOSTOR_FRAG` to ensure correct upward lighting in OpenGL eye space.
- Added depth-tested obstacle occlusion to Pass 2 (half-res optical thickness) so fluid particles occluded behind solid scene geometry do not add optical thickness.
- Eliminated FBO reallocation thrashing in `draw_spheres()` by preserving existing offscreen textures when switching render modes.
- Implemented context-safe deferred OpenGL resource cleanup in `destroy()` and `draw_particles_viewport()` to avoid cross-context GL handle leaks.
- Preserved all `eb02964` physics stability controls (`lattice_rest_mass`, `ARTIFICIAL_VISCOSITY_ALPHA`, `wall_accel`, `stiffness = 90000.0`, `cfl_factor = 0.4`).

**Framebuffer Fix Verification**
- Full headless suite: 25 tests passed, including allocation/resize/default-framebuffer binding regressions. Separate standalone and both import-order checks verified test-module isolation.
- Blender 5.2.0 LTS / NVIDIA OpenGL 4.6: actual SSFR passes using a controlled GL VBO retained distinct read/draw targets at 256x192 and after resize to 512x384; background capture and composite pixels matched expectations with zero GL errors. This is renderer proof, not a full CUDA simulation run.
- Local loop receipts and rendered image are under `.codex/context/`; temporary probe scripts are removed after verification.

**Next Steps**
- Rebuild + reinstall after changes: `blender --command extension build`, then `extension install-file -r user_default`.
- Run `python -m unittest discover -s tests -v` for the full headless suite.
- The other six PR-review findings remain outside this framebuffer fix: broader GL-state restoration, multi-context cleanup churn, perspective point diameter, orthographic view direction/filter scaling, and off-center projection reconstruction.
