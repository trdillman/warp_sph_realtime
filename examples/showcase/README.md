# Showcase demos

One script per major feature. Each builds its scene with the add-on's own properties and operators, runs or bakes
it, and checks the result. Run by `scripts/render_showcase.py`, it also renders its media, writes a `summary.json`
and quits. Unlike the [example scenes](../README.md), which only build a scene for you to start, these drive the
whole workflow. They need Blender 5.2 with the add-on enabled, Warp installed (see
[the manual](../../docs/manual/install.md)), an NVIDIA GPU and a window (not `-b`: the simulation draws in the 3D
viewport).

| Script | Feature (PR) | What it does and checks |
|---|---|---|
| [`display.py`](display.py) | Viewport display and transports (#6) | Runs the Double Dam demo in **Points**, **Spheres** and **SSFR Liquid**, and screenshots each with an overlay naming the transport and GPU backend. On Vulkan every mode uses Vulkan interop. On OpenGL, Points use the OpenGL texture interop and the other modes the legacy VBO interop. All of these are GPU copies with no CPU readback. If the add-on falls back to **Readback** (a CPU copy), the overlay says so. |
| [`validation.py`](validation.py) | Dam-break validation (#11) | Bakes a 0.5 m dam-break column (quasi-2D slab, about 50k particles) and keyframes a red marker at the Martin & Moyce (1952) surge front, using the reference table from `sim/validation.py`. It measures the simulated front of every cached frame with the same `front_position` / `compare_front` as the headless validation and checks the mean error. |
| [`emitter_mixer.py`](emitter_mixer.py) | Emitters and moving colliders (#13) | Uses a centimetre scene (Unit Scale 0.01). An emitter pours into a pool while a keyframed collider paddle turns two full turns (under half a turn per frame, rotation only). It bakes, then adds the mesh preset. Checks: the emitter adds particles, the bake records the unit scale, the paddle stirs the pool and the mesh radius is in centimetres. |
| [`bake_mesh_export.py`](bake_mesh_export.py) | Bake, playback, mesh preset (#12) and export (#14) | Bakes a splash, then runs **Show Cache** and **Add Mesh Preset** with a clear water material (Principled BSDF, transmission). It renders in Cycles (16 samples, denoised), plus a still of the point cloud coloured by particle age. It exports the meshed sequence to USD and reads the file back to check that every baked frame was written. |
| [`presets.py`](presets.py) | Physics presets (#15) | Bakes the same spill off a shelf with **Water**, **Viscous** and **Stylised** (`wm.warp_sph_physics_preset`) into three cache folders, one after another. It meshes and renders the same frames of each into a labelled side-by-side comparison, and checks that the viscous fluid moves slower than water. |

`_showcase.py` is the shared helper (scene building, viewport overlay, renders, image labels and grids, the
summary). It is not a demo.

## Running one in Blender

```
blender --python examples/showcase/validation.py
```

The script builds the scene in the current file, deleting the objects named `Showcase ...` from an earlier run.
It then runs or bakes and leaves Blender open on the result: scrub the timeline to play the cache. Caches go to a
temporary folder (`warpsph_showcase/<demo>` in the system temp folder). For the display demo's OpenGL routes, add
`--gpu-backend opengl` before `--python`. You can also open a script in the Text editor and press Run Script.

## Running all of them

```
python scripts/render_showcase.py --blender PATH/TO/blender.exe
python scripts/render_showcase.py --demos display --backends opengl
```

The runner starts one Blender per demo, one at a time. If another Blender is already running, it waits for it to
exit. The display demo runs once per backend (Vulkan, then OpenGL); the other demos use the first backend.

Each Blender gets an isolated user profile with this checkout's add-on copied into it. Warp comes from
`--warp-folder` (a folder that contains the `warp` package, used as the add-on's **Local Folder** Warp source),
else from the `WARPSPH_SHOWCASE_WARP_FOLDER` environment variable, else from the Python running the runner.
`--timeout` limits each Blender run (default 20 minutes). Each demo takes a few minutes on an RTX-class GPU.

Outputs go to `evidence/showcase/<timestamp>/`, which is gitignored:

* one folder per demo, with its media, `summary.json` and Blender's log
* `summary.json` for the whole run: pass or fail per demo, particle counts, timings, transport
* `contact_sheet.png` with one key frame per demo
* an mp4 per frame sequence when `ffmpeg` is on `PATH`; otherwise the PNG sequences stay as they are

The exit code is 0 only when every demo passes and the contact sheet was made.

`tests/test_examples.py` checks every setting, operator and enum value the demos use. `tests/test_showcase_runner.py`
tests the runner's argument parsing, plan, summary and contact-sheet layout without Blender.
