# Context

**Current Task**: Viewport display + sim speed controls done and verified in Blender 5.2 (commit d96d251).

**Key Decisions**
- Particles drawn via raw GL; point size/depth/blend set directly in `draw_points` (Blender's `gpu.state` is lazy) and restored after.
- Speed vs stability exposed as Sub-steps, Time Step (CFL, default 0.4, unstable >~0.5), Fluid Stiffness (default 90000); changes apply live via `solver.update_time_step()` + graph rebuild.
- h/mass/dt derive from spawn spacing via `solver.set_particle_spacing()`.

**Next Steps**
- Rebuild + reinstall after changes: `blender --command extension build`, then `extension install-file -r user_default`.
- Docs drift: README/ARCHITECTURE still say "Warp 1.16+".
- Alpha <1 looks near-opaque at 1M particles; consider round/soft point shader if needed.
