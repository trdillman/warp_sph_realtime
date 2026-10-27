# FAQ and troubleshooting

When something goes wrong, click **Copy Diagnostics** (in the WarpSPH panel or in the add-on preferences) and paste
the result into your bug report. It lists the add-on, Blender, OS, Warp, CUDA driver, GPU, active transport, last
error and the tail of the log (`warpsph.log`), with your user folder replaced by `~`.

## Hardware

**What GPU do I need?** An NVIDIA GPU with a driver that supports CUDA 12 or newer (driver 525 or newer). WarpSPH
uses `cuda:0`. AMD, Intel and Apple GPUs are not supported. In the preferences, **Hardware Check** shows the GPU,
driver, CUDA support and VRAM before Warp is installed.

**The Hardware Check says something is wrong.**

| Message | Meaning and fix |
|---|---|
| `nvidia-smi not found: no NVIDIA driver installed?` | Install the NVIDIA driver (the check runs `nvidia-smi`). |
| `nvidia-smi failed: ...` | The driver tool ran but failed; reinstall or update the driver, then **Check Again**. |
| `No NVIDIA GPU found: WarpSPH needs one` | No NVIDIA GPU was reported. If the machine has several GPUs, make Blender use the NVIDIA one. |
| `Driver only supports CUDA 11.x; Warp needs 12.x: update the NVIDIA driver` | Update the driver. |
| `GPU found; the driver did not report its CUDA version` | Not blocking; the real check happens on Start. |

**What does the Start-time check say when it fails?**

* `NVIDIA driver supports CUDA X.Y but Warp 1.17.0 needs CUDA 12.x: update the NVIDIA driver`
* `Warp CUDA is unavailable: check NVIDIA GPU, driver and Warp CUDA 12.9 runtime`
* `CUDA device cuda:0 is unavailable: ...` and `WarpSPH requires CUDA device cuda:0`
* `Warp CUDA self-test failed to run (driver CUDA X.Y): ...` or `... returned [...], expected [...]; the GPU or
  driver is not computing correctly`: a known-answer kernel ran on the GPU and failed. Update or reinstall the
  driver; if it persists, report it with the diagnostics.

**Blender and Warp use different GPUs.** OpenGL and Warp `cuda:0` must be the same NVIDIA GPU. On a multi-GPU
machine, run Blender on the GPU that is `cuda:0`.

## Installing Warp

| Message | Meaning and fix |
|---|---|
| `Online access required to install Warp. Enable online access in System preferences` | Turn on Blender's online access, or use **Install from local wheel file**. |
| `Online access required to install Warp. Launch Blender without --offline-mode` | Blender was started with `--offline-mode`; restart it without, or use the local wheel. |
| `NVIDIA Warp is not installed for WarpSPH` | Use **Download and Install Warp 1.17.0 (~150 MB)**, a local wheel or an existing folder. |
| `Choose a Warp wheel (.whl) file first` | Set **Wheel File** before clicking **Install from This Wheel (offline)**. |
| A wheel is refused | It does not match the locked SHA-256 or file list. Use the exact `warp_lang-1.17.0` CUDA 12 wheel for your platform. |
| `Warp X is already installed in Blender's user site-packages (...). Remove that copy, then press Install Warp. WarpSPH will not change it.` | Another Warp is on Blender's path. Remove it, or choose it as an **Existing Warp folder** if it is 1.17.x. |
| `Warp X is already loaded from ..., which WarpSPH does not own. Remove that copy, then press Install Warp.` | Another extension loaded its own Warp. Use a separate Blender process with only WarpSPH enabled. |
| `Warp X is already loaded from the existing folder ... Restart Blender to switch the Warp source.` | Changing the Warp source after Warp has loaded needs a restart. |
| `Warp repair staged; restart Blender to apply it` | Restart Blender. |
| `Unmanaged Warp files; preserved but unavailable` | Files WarpSPH did not create are in its storage. They are never overwritten; remove them yourself. |
| `Existing Warp folder not found: ...`, `No 'warp' package in ...`, `Choose the folder that contains the 'warp' package` | Pick the folder that contains `warp`, not the `warp` folder. |
| `Warp X is not certified: WarpSPH supports Warp 1.17.x. Use a 1.17.x folder, or tick 'Allow uncertified version (unsupported)' ...` | Use 1.17.x. Newer releases (including CUDA 13 builds) are not qualified. |
| `Stop the simulation before changing the Warp runtime` | Stop first. |

Cancelling an install leaves nothing behind. WarpSPH never changes Blender's bundled Python.

## Graphics backend: Vulkan or OpenGL?

Blender 5.2 can run on Vulkan or OpenGL. Check the **Graphics backend** row of the Hardware Check or the
**Transport** line in the panel.

* On **OpenGL**, the particles reach the screen with a GPU copy and no CPU readback in every display mode.
* On **Vulkan**, that GPU path exists only on the exact certified Blender 5.2.0 build. Any other Vulkan build runs
  **Readback (CPU copy)**, with a reason such as `Vulkan interop unavailable: ...`.

To switch, use **Use OpenGL fast mode** in the panel (shown when Readback is active on Vulkan), or set the backend in
`Edit > Preferences > System`, and restart Blender.

## What does "Readback" mean?

The transport line reads `Readback (CPU copy)` when the particles are copied through CPU memory each frame instead
of staying on the GPU. It is a fallback, always labelled, and slower. Typical reasons:

* Vulkan on a Blender build that is not the certified one.
* An interop setup error. The panel then shows `<reason> (Readback until restart)`; WarpSPH does not retry interop
  until Blender restarts.
* `OpenGL interop unavailable: ...`, for example when `CUDA 12 runtime cudart64_12.dll not found; install CUDA 12
  runtime for GPU/OpenGL interop.` (Linux: `libcudart.so.12`).
  When only another CUDA 12 version is installed the reason lists each copy found and its version
  (`... no CUDA runtime matched: <path>: reports 12040 ...`); install the CUDA 12.9 runtime.

The simulation itself is identical; only the display path differs.

## "GPU transport failed: restart Blender to run WarpSPH again"

A fatal CUDA or transport error happened (for example `solver step failed; CUDA state unknown: ...`). WarpSPH stops
using CUDA for the rest of the process because its state is unknown. Restart Blender. A line "Last GPU cleanup: ...
(retained until exit)" means GPU objects could not be released safely and are kept until Blender exits.

## Memory and size

**How much VRAM do I need?** The panel estimates about 150 bytes per particle plus about 32 MB. It warns
"Needs about X GB of this GPU's Y GB: may not fit" when the estimate exceeds half of the GPU's memory (Blender and
other applications share it). One million particles is roughly 0.17 GB by that estimate. The **SoA VRAM
Footprint** line shows the measured solver footprint while running.

**It says it needs too much memory.** Increase **Particle Spacing** (halving it is about 8 times the particles),
lower **Particles** (demos) or **Max Particles** (emitters), and close other GPU-heavy applications.

**A bake needs a lot of disk.** 28 bytes per particle per frame. The Record / Bake Cache panel shows the size and
"Only ... free on this drive".

## Performance tips

* Use **Raw Points (Max FPS)** to see the solver's own speed; **SSFR Liquid** costs extra for rendering.
* Keep **Warp 1.17 CUDA Graph Capture** on (under **Advanced**).
* Lower the particle count or raise **Particle Spacing** until the FPS is acceptable. For a final result, bake with
  **Steps / Frame** left at 0 and watch it with **Show Cache as Point Cloud**.
* More **Sub-steps / Frame** advance more simulated time per redraw, at lower FPS.
* Do not raise **Time Step (CFL)** above 0.4 to gain speed on large runs: it can blow up on hard impacts.
* On a machine with more than one GPU, make sure Blender runs on the NVIDIA GPU that is `cuda:0`.
* Check the Transport line: Readback costs frame time.

## Scene problems

| Message | Fix |
|---|---|
| `No domain: mark a box as Domain` | Mark a cube as **Domain**. |
| `No fluid: mark a closed mesh as Fluid` | Mark a closed mesh as **Fluid** or add an **Emitter**. |
| `Fluid '...' extends outside the domain` | Move or scale the fluid inside the domain box. |
| `... is not closed (watertight)` / `... encloses no volume` | Make the mesh closed with outward-facing normals. |
| `no lattice point fits in the fluid volumes at spacing ...; decrease the spacing` | The fluid is thinner than the particle spacing. |
| `emitter '...' is outside the domain, inside a collider, or closer than ... m ...` | Move the emitter into open space, at least one spacing from walls and colliders. |
| `... speed must be above 0 m/s (flip the normals to reverse it)` | Use positive speed and flip the normals instead. |
| Run stops on an animated collider | Animated scale or shear is not supported; animate location and rotation only. |

## The fluid explodes or looks wrong

* Return to the **Water** preset, and keep **Time Step (CFL)** at 0.4 or lower.
* Check that the fluid does not start inside a collider and that the domain is large enough.
* Very fast spins (more than half a turn per frame) of a collider are not supported.
* If it persists, report it with **Copy Diagnostics** and the scene.
