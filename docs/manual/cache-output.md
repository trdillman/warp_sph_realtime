# 6. Record, bake, playback and export

Open **Record / Bake Cache** under the main panel (it is collapsed by default). The simulation does not need to be
running: **Bake** starts it. Baking is an explicit action; the live view never reads particles back to the CPU.

## Bake

1. **Cache Folder**: leave empty for the default, `<blend folder>/<name>_warp_sph_cache/<scene>` next to the saved
   .blend (a temporary folder while the file is unsaved), or choose a folder. Each scene has its own.
2. Set **Start** and **End** frames (frame 1 is the initial state) and **Steps / Frame** (0 means real time: the
   steps that make one frame last 1 / frame rate seconds of simulated time).
3. The panel shows the size before you bake ("Size: ... (N frames x ...)"): 28 bytes per particle per frame, and
   warns "Only ... free on this drive" if the drive is too small.
4. Click **Bake** (the record button). A progress bar and **Cancel Bake** appear; the viewport follows. The bake runs
   in small timer-driven chunks, so Blender stays responsive. Each frame is one `.npz` file with `position`,
   `velocity` and a stable particle `id`; the GPU-to-host copy is asynchronous.

## Resume, invalidate, delete

* Cancel, Esc or a failure keeps every frame saved so far.
* **Resume** continues after the last saved frame (or extends **End**). It refuses a cache baked with a different
  particle count or different solver settings.
* **Invalidate From Current Frame** deletes the current frame onward so **Resume** re-simulates it.
* **Delete** (**Delete Cache**) removes every cached frame of this scene.

The solver state is positions and velocities, so a resumed bake continues bit for bit like an uninterrupted one on
the same GPU and driver. Changing the scene's objects after a bake is not detected: invalidate from frame 1.

## Playback

**Show Cache as Point Cloud** adds an object that follows the timeline (frames outside the cached range show the
nearest cached frame). It has `velocity`, `id` and `radius` point attributes and renders in Cycles as spheres of the
particle radius. **Hide Cache** removes it (and the mesh preset object, which reads it).

## Output

* **Age.** The point cloud also has an `age` attribute: the time in seconds since each particle first appears in the
  cache. Use it in a material through an Attribute node, or in Geometry Nodes.
* **Add Mesh Preset** adds a *WarpSPH Mesh* object that turns the points into a surface with Geometry Nodes (Points
  to Volume, Volume to Mesh, then smoothing). Edit **Radius**, **Voxel Size** (smaller is finer and heavier),
  **Threshold**, **Smooth Iterations** and **Material** in the panel or on the modifier. Defaults come from the
  particle spacing (radius 1.5x, voxel 0.5x). The point cloud is hidden from renders while the mesh exists.
  **Remove Mesh** removes it.
* **Export USD** / **Export Alembic** ask for a file and write the mesh over the baked frames only (adding the
  preset first if needed). Blender 5.2 exports the meshed object, not point clouds. A 24-frame, 12,000-particle bake
  measured about 65 MB as USD and 35 MB as Alembic.
