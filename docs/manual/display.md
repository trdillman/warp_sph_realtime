# 5. Viewport display

## Render modes

**Viewport Display > Mode**:

| Mode | What it is |
|---|---|
| **SSFR Liquid (Narrow-Range + Beer-Lambert)** (default) | A 5-pass screen-space fluid renderer with a narrow-range depth filter and Fresnel shading. Its tuning controls (**Narrow-Range Filter Passes**, **Filter Sigma (px)**, **Beer-Lambert Absorption (RGB)**) are under **Advanced**. |
| **Shaded Sphere Impostors** | Single-pass shaded 3D spheres with per-fragment depth correction. Size comes from **Particle Radius (m)**. |
| **Raw Points (Max FPS)** | The fastest mode, for benchmarking. Size comes from **Point Size** (1 to 20 px). |

Other display controls: **Show Particles**, the colour swatch (**Particle Color**, RGBA; alpha below 1 enables
blending) and **Hide Behind Objects** (off draws particles on top of scene geometry).

## Transports: how particles reach the screen

The solver keeps particle data on the GPU. The status box shows the active transport on the **Transport** line:

* **Vulkan interop (GPU copy, no CPU readback)**: on the Vulkan backend. Particle positions are copied on the GPU
  into a texture Blender draws; nothing is read back to the CPU each frame. This is available only on the exact
  certified Blender 5.2.0 build.
* **OpenGL interop (GPU copy, no CPU readback)**: on the OpenGL backend. Chosen when you press Start in
  **Raw Points (Max FPS)** mode: the particles are copied on the GPU into an RGBA32F texture
  (`cudaGraphicsGLRegisterImage`) and drawn by Blender's `gpu` module. For the other modes OpenGL uses the older
  VBO interop, which maps a Blender vertex buffer to a CUDA pointer and also has no CPU readback.
* **Readback (CPU copy)**: the labelled fallback. Positions are copied through host memory and uploaded each
  frame. It works everywhere CUDA does but is slower. It is used when interop cannot start, for example on any
  Vulkan build other than the certified one, or after an OpenGL interop failure. The reason is shown under the
  Transport line; after a recoverable interop failure Readback is used until Blender restarts.

The accelerated transports avoid CPU readback; they still copy data on the GPU, so they are not "zero GPU copy".

On the Vulkan backend with Readback active the panel offers **Use OpenGL fast mode**. It asks for confirmation,
saves Blender's graphics backend preference as OpenGL, and takes effect after you restart Blender.

If a GPU transport fails fatally the panel says "GPU transport failed: restart Blender to run WarpSPH again". See the
[FAQ](faq.md) for causes.
