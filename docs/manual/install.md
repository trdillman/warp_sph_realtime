# 1. Install

## Requirements

* Blender 5.2.0 or newer.
* An NVIDIA CUDA GPU. The certified Warp 1.17 wheel uses CUDA Toolkit 12.9 and needs NVIDIA driver 525 or newer.
* OpenGL interop additionally needs the CUDA 12.9 runtime library (`cudart64_12.dll` on Windows, `libcudart.so.12`
  on Linux).
* Blender's OpenGL context and Warp's `cuda:0` must be the same NVIDIA GPU.
* Warp 1.17.0 (the exact, SHA-256-locked CUDA 12 wheel). Warp 1.16 and 1.18 / CUDA 13 are not accepted
  automatically.

## Install the add-on

Build the zip with `python scripts/build_extension.py` (it writes `dist/warp_sph_realtime-<version>.zip`) and use
`Edit > Preferences > Get Extensions > Install from Disk...`. Alternatively point a Local Extension Repository at the
inner `warp_sph_realtime/` package folder. The add-on registers even when Warp is not installed yet.

## Hardware check

Open `Edit > Preferences > Add-ons > WarpSPH`. The **Hardware Check** box lists **GPU**, **NVIDIA driver**,
**Driver supports CUDA**, **GPU memory (VRAM)** and **Graphics backend**, and ends with a verdict such as
"Ready for Warp: the driver supports CUDA 12.x (needs 12.x or newer)". It reads `nvidia-smi`, so it works before
Warp is installed. **Check Again** re-reads it. If it reports a problem, see the [FAQ](faq.md).

## Get Warp

Choose a **Warp Source** in the same preferences box.

1. **Download** (the default). Enable Blender's online access first (the preferences say so if it is off), then click
   **Download and Install Warp 1.17.0 (~150 MB)**. The download runs in the background with a progress bar and a
   **Cancel** button; cancelling keeps nothing.
2. **Install from local wheel file**. Pick the certified `warp_lang-1.17.0` `.whl` for your platform in **Wheel File**
   and click **Install from This Wheel (offline)**. No network is used. A file that does not match the locked
   SHA-256 is refused.
3. **Use existing Warp folder**. Pick the folder that contains the `warp` package (not the `warp` folder itself), for
   example a studio-managed site-packages. WarpSPH only reads it. It accepts Warp 1.17.x; another version needs
   **Allow uncertified version (unsupported)**, which keeps a warning visible. The CUDA driver and a known-answer
   GPU kernel are checked when you press Start.

The download and local-wheel routes verify the wheel's size, SHA-256, metadata and file paths and put it only in the
extension's own user storage. Blender's bundled Python is never modified. Switching back to Download or a local wheel
uses the install WarpSPH already owns.

## Repair and restarts

**Repair Warp** (labelled **Repair Warp (restart required)** in the preferences) re-installs the locked wheel. If a
native runtime is already loaded the wheel is staged, and the panel says "Warp repair staged; restart Blender to
apply it". Changing the Warp source after Warp has loaded also needs a Blender restart. An existing Warp that
WarpSPH did not install is preserved, never deleted: if another extension or the user site-packages already provides
Warp, WarpSPH tells you where it is and does not touch it.

## Check it worked

In the 3D Viewport sidebar, tab **WarpSPH**, the status box shows `Runtime: Installed (GPU check on Start)`. The GPU
and CUDA checks run the first time you press **Start Realtime Sim**. **Copy Diagnostics** (in the panel and in the
preferences) copies a report of GPU, driver, Warp, Blender and recent errors to the clipboard for bug reports.
