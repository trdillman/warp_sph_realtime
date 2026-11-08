# Timeline-bound live simulation — 7 November 2026

User correction: Pause, Single Step and Reset must bind to Blender's timeline.
Implemented at source `3c3b90d`, separate from merged main `c6b37db`.

Native playback and frame changes now supply solver targets. Initial state is at
scene.frame_start. Pause toggles Blender playback, Step pauses and moves one
timeline frame, and Reset pauses playback and returns physics/playhead to start.
Native frame-offset and scrubbing work while paused. Backward seeks rebuild and
replay current settings; historical live edits are not recorded/reproduced.
One solver step per modal tick bounds catchup; backpressure preserves elapsed
time. Actual step duration includes substeps. Overlarge steps warn and pause
instead of reporting unchanged physics as a completed frame. Collider sampling
restores requested playhead. Demo count rebuilds preserve the reset origin.
Legacy bake exit invalidates/rebuilds the timeline clock; native bake retains its
independent routing. Start/Pause alone does not qualify every bake workflow.

Independent source review found origin, bake-clock and overlarge-step gaps;
focused regressions and repairs received follow-up PASS. Full host suite before
the final demo-rebuild origin repair: 678 passed, 64 CUDA-disabled skips,
72 subtests. The final affected suite passed 21 tests and two subtests; reviewer
independently ran seven timeline tests. Final Ruff and Pyright pass.

Built ZIP SHA256:
`88c955ec155fa44d0136a296fb7657307107e01594b3edf1a1a376fa9afdd7e5`.
Installed through Blender's extension installer in canonical isolated PID24556,
port9879, module `bl_ext.timeline_demo.warp_sph_realtime`; installed files match
the ZIP byte-for-byte. Blender5.2 fbe6228777e7 / Vulkan / Warp1.17 / SDK a2 /
RTX4080SUPER cuda:0. All runtime proof uses the installed package.

Verified: Start from playhead10 resets to scene start1; paused initial solver
remains at zero steps; native Play advances frames and actual physics; addon
Pause cancels native playback; addon Reset returns frame1/zero solver steps;
addon Single Step and native frame_offset each advance frame1→2 and settle at
13 solver steps / 0.042848699 seconds for 24FPS; backward scrub returns initial
state; held frame2 remains at the same step count; Viscous preset reaches the
live solver (viscosity8), then Water/Reset restores a ready demo.

Setup failure: first Start without viewport override rejected GPU context;
correct 3D viewport override passed. No speculative admission bypass used.
Older demo PID33448 was paused and preserved. New demo PID24556 remains open,
installed/enabled, running session paused at frame1; Space/Play drives it.

Limits: 100K physics can lag playback; panel reports target catchup. No real-time
FPS claim. Time accuracy is within one full solver step. Rewind uses current
settings and can be expensive. No historical-input capture, fresh bake GPU
acceptance, reverse-playback performance, Linux GPU or release acceptance.
No push, main merge, public release or production-profile mutation for this
new feature. Receipts and initial failed-probe context are preserved here.
