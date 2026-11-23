# Native SDK snapshot preview

This is a development-candidate action in the actual WarpSPH add-on, not a replacement solver or a new live renderer. It requires the exact separately installed shared SDK identified by sdk_snapshot.lock.json. Normal startup, simulation, viewport, cache/bake and export remain unchanged.

## Artist workflow

Start the real simulation, let it advance, then Pause. Open WarpSPH > Record / Bake Cache > Native Snapshot (SDK Preview) and choose Snapshot Paused Simulation. The dialog names the output object/collection and lets you create Geometry Nodes instances or just native points. Choose Update Existing Snapshot only to replace a snapshot created by the same running session. The object and existing generated GN modifier are retained; a different simulation cannot silently replace an older snapshot.

The output contains actual solver positions, velocities and stable particle IDs in the solver's current sorted order. IDs are promoted losslessly from the solver's current nonnegative int32 IDs to the SDK's uint64 representation; this does not change the solver's own ID allocator width. Native Blender stores the exact bits in particle_id_low/high. Radius and colour are the current artist display settings, not inferred physical measurements. Density is the last substep's force-evaluation density; it is omitted before the first real step rather than exporting an unmeasured zero buffer.

Resume the simulation and take another paused snapshot to update the native result. The snapshot operation neither advances the solver nor changes its arrays/time. Reset creates a new snapshot identity epoch; disposing/recreating the entire session creates a new producer namespace so old IDs are not silently reused.

## Limits and rejection behavior

This action intentionally synchronizes and reads back the active particle prefix. It is not zero-copy/live Geometry Nodes and is never called from normal playback. The default host capture and native allocation budgets bound explicit work; large GN realization can exceed the native budget. Uncheck Geometry Nodes Instances for native points only. User-edited GN graphs can have their own evaluation costs beyond the default graph estimate.

The action rejects running/unpaused, pending-step, baking, wrong-scene, invalid-ID, wrong-SDK and over-budget cases. An unknown CUDA synchronization/readback error marks the existing session restart-required, instead of continuing with uncertain GPU ownership. Failed native/GN creation rolls back its own staged output. Existing unrelated objects, profiles and scenes are never selected as implicit replacement targets.

No actual-solver performance, collider/emitter A/B, full live SDK renderer adoption, animation bake, export or render-engine parity follows from this paused snapshot sub-slice. Read the qualification report for completed native cases and precise limitations. The saved native output can be inspected without the live solver once clean-profile reopening is verified.
