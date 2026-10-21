# 7. Known limits

**Platform and runtime**

* NVIDIA CUDA GPUs only, on `cuda:0`. Only Warp 1.17.x is accepted (an existing folder of another version needs the
  unsupported opt-in).
* One simulation run per Blender process.
* Vulkan GPU interop works only on the exact certified Blender 5.2.0 build; other Vulkan builds use the labelled
  Readback fallback. OpenGL supports every display mode.
* After a fatal GPU or CUDA error the run stops and Blender must be restarted.
* Particle counts of the demos range from 100,000 to 4,000,000; a Custom scene is bounded by **Max Particles**
  (1,000 to 4,000,000) and by GPU memory.

**Scenes**

* Edits to objects take effect only on the next **Stop** + **Start**.
* A collider keeps the shape it has at the Start frame. Deforming colliders are not followed, an animated scale or
  shear stops the run with an error, and spins of more than half a turn per frame are not supported.
* The domain is the object's world axis-aligned bounding box.
* Emitters must be at least one particle spacing from walls and colliders.

**Physics**

* Weakly compressible SPH: no surface tension and no wall friction. A hard impact can throw a few spray particles
  faster than the sound speed.
* A **Time Step (CFL)** above 0.4 can blow up on hard impacts at high particle counts.
* Viscosity is explicit and is capped where it would be unstable; the same value is thicker at a finer spacing.
* Accuracy claims and their limits: [`VALIDATION.md`](../VALIDATION.md).

**Cache and output**

* A bake stores positions, velocities and ids (28 bytes per particle per frame); there is no surface mesh in the
  cache, the mesh preset is built on playback.
* Changing scene objects after a bake is not detected.
* USD and Alembic export the meshed object only. Blender 5.2 does not export point clouds.
* The `experimental/` package is excluded from the built add-on and has no support promise.
