# Context

**Current Task**: 5-Pass Screen-Space Fluid Renderer (SSFR) with Truong & Yuksel Narrow-Range Depth Filter, Beer-Lambert Absorption, and Schlick Fresnel shading integrated on top of commit `eb02964`.

**Key Decisions & Robustness Fixes**
- Added `ssfr_shaders.py` with GLSL 450 Core shaders and CPU reference math (`compute_projected_filter_sigma`, `narrow_range_filter_1d_cpu`, `compute_beer_lambert_transmittance`, `compute_schlick_fresnel`).
- Fixed Narrow-Range Filter boundary handling and silhouette preservation (rejecting foreground samples beyond the delta neighborhood to prevent bleeding across depth discontinuities).
- Added camera projection auto-detection for both Perspective and Orthographic views in point sizing (`SSFR_DEPTH_VERT`), depth scaling (`SSFR_NARROW_RANGE_FRAG`), and normal reconstruction (`SSFR_COMPOSITE_FRAG`).
- Upgraded `interop_bridge.py` (`_GLInterop`) with FBO, Texture2D, packed depth-stencil (`GL_DEPTH24_STENCIL8` with `GL_DEPTH_STENCIL_ATTACHMENT`), `glGetError`, and GLSL shader program compilation bindings.
- Upgraded `viewport_draw.py` with `SSFRViewportRenderer` supporting 3 selectable viewport modes (`SSFR_LIQUID`, `SPHERES`, `POINTS`), saving and restoring Blender's bound framebuffer (`GL_DRAW_FRAMEBUFFER_BINDING`) and OpenGL states directly.
- Fixed point-sprite Y normal inversion in `SPHERE_IMPOSTOR_FRAG` to ensure correct upward lighting in OpenGL eye space.
- Added depth-tested obstacle occlusion to Pass 2 (half-res optical thickness) so fluid particles occluded behind solid scene geometry do not add optical thickness.
- Eliminated FBO reallocation thrashing in `draw_spheres()` by preserving existing offscreen textures when switching render modes.
- Implemented context-safe deferred OpenGL resource cleanup in `destroy()` and `draw_particles_viewport()` to avoid cross-context GL handle leaks.
- Preserved all `eb02964` physics stability controls (`lattice_rest_mass`, `ARTIFICIAL_VISCOSITY_ALPHA`, `wall_accel`, `stiffness = 90000.0`, `cfl_factor = 0.4`).

**Next Steps**
- Rebuild + reinstall after changes: `blender --command extension build`, then `extension install-file -r user_default`.
- Run `python -m unittest discover -s tests -v` to execute all 23 tests across 4 modules.
