# Using the WarpSPH shared-SDK checkpoint

## Open the saved result without installation

Open the supplied `WarpSPH-Actual-Solver-GN-Snapshot-2026-10-31.blend` in Blender 5.2. It contains100,000 particles from the real solver, native point attributes and an editable Geometry Nodes modifier. The scene was verified with WarpSPH, Warp and the SDK absent. Inspect the original mesh in the Spreadsheet for velocity, density, radius, colour and the low/high stable-ID words. The embedded text explains the static capture.

## Developer preview of the new workflow

The private candidate requires the exact Blender 5.2 build fbe6228777e7, Warp 1.17.0 and the pinned SDK wheel from the checkpoint, on the qualified Windows CUDA/Vulkan route. The package is provided for isolated development/qualification, not automatic production installation. No packages are downloaded or installed by the new switch.

In a correctly prepared development profile, select **Shared SDK Output (Experimental)** in WarpSPH's **Native Particles / Geometry Nodes** subpanel while stopped. Start the simulation, then Pause after at least one step. Set the snapshot object name and press **Snapshot to Geometry Nodes**. Wait for the panel to show COMPLETED, then Stop and edit the resulting native object. A later snapshot updates the same scene-owned object while retaining its GN modifier. Move/scale the resulting object only after capture; transformed replacement targets are explicitly rejected instead of silently double-transforming positions.

Normal viewport use stays GPU-based. Snapshot and opt-in Live Geometry Nodes perform bounded readback under the Native Memory Budget. Live updates are throttled and may skip intermediate frames. Bake Native Animation runs an independent solver copy and records every admitted frame; cancellation preserves its completed prefix. Stop the simulation before loading that cache for timeline playback, or embed standalone native animation for playback without WarpSPH or the SDK. These native-output workflows are not zero-copy. Nothing changes in the default route when the experimental option is off.

The solver keeps its existing int32 identity range; the adapter widens those stable values to exact uint64. In the native mesh, `particle_id_low` and `particle_id_high` retain the two bit words. Exact GN filtering compares both words; it does not select by current array position.
