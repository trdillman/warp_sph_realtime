# Context

**Current Task**: Demo stabilized, rendering in Blender 5.2, extension installed (commits 6d9caf7, 9db5d25).

**Key Decisions**
- Sound speed ~25 m/s (stiffness 90000) + Monaghan AV + soft walls; dt = CFL 0.4*h/c, so sim runs ~half the old per-frame speed.
- h/mass/dt derive from demo spawn spacing via `solver.set_particle_spacing()`.
- Points VAO created lazily per GL context in `interop_bridge.py`.

**Next Steps**
- Rebuild + reinstall after changes: `blender --command extension build` then `extension install-file -r user_default`.
- Docs drift: README/ARCHITECTURE still say "Warp 1.16+" and list planned modules.
- Splash demo droplets peak ~48 m/s; tune AV alpha if splash looks too energetic.
