# 2. First run

1. Press `N` in the 3D Viewport and open the **WarpSPH** tab.
2. In the **Scene** box leave the source on **Demo** and click either
   * **Demo 1: Double Dam + Pillars** (two opposing fluid columns crash into a central sphere and four pillars), or
   * **Demo 2: Block Flop Crown Splash** (a heavy fluid cube drops into a shallow puddle).

   Clicking a demo starts it. (**Start Realtime Sim** at the bottom starts whichever demo is selected.)
3. Drag **Particles** (100,000 to 4,000,000) to change the size of the run. The panel warns when the estimate needs
   more than half of the GPU's memory. While the demo runs, the same change resizes it.
4. While running, the bottom row shows **Pause** / **Resume**, **Step** (advance exactly one frame, then stay
   paused) and **Stop (ESC)**. Pause freezes the physics; display settings stay live on the frozen frame. Stop
   releases the GPU buffers.
5. The **GPU Performance Telemetry** box shows status, FPS, frame time, VRAM footprint and throughput.

The **Transport** line in the status box tells you how the particles reach the screen. See
[Viewport display](display.md) if it says Readback.

Only one run exists per Blender process. Starting a second scene while another runs reports which scene to stop
first.

Next: [build your own scene](authoring.md).
