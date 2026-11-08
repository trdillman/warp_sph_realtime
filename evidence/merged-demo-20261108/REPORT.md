# Installed merged-main demo — 8 November 2026

PASS for the requested live timeline controls on Windows Blender 5.2.0 LTS
build fbe6228777e7, Vulkan, RTX 4080 SUPER cuda:0, Warp 1.17.0 and pinned private
interop SDK 0.1.0a2. Source product files match remote main
01d034efae201323c73b73cd8df65ac3ad7352d7; local HEAD before evidence was
46699321f06f18c91f804765645bf988ad7e8184 (documentation only beyond main).
No product changes were made.

Fresh validated extension ZIP SHA256:
3db5a2a1013064e26e5df32610cdad15bda01aa036ab75027c832ce6e13af44b.
All 60 archive files byte-match the actual installed extension in owned isolated
session 934d0d1301c64b0698d978b9ad9530c6. Module
bl_ext.demo.warp_sph_realtime is enabled and in preferences. PID 48636 owns
listener 9877 and a visible window. Paired launcher gate and final
status/file_state/scene_info/execute_code succeeded; durable process streams usable.
The regular Blender profile and other worktrees were preserved.

The Double Dam + Pillars live demo runs 100,000 particles (the property minimum;
the initial requested 10,000 value was clamped). Shared SDK and Live Geometry
Nodes are enabled. This is live solver output, not cache playback.

`runtime.json` records sixteen samples from public native/add-on operators:

- Start remains paused at frame 1, zero steps.
- Native Play advances solver time, steps and positions.
- Native Pause with backlog holds frame 12, 7 steps, time 0.02307237647 and all
  particle positions unchanged across samples 1.2 seconds apart.
- Native Jump to Start restores frame 1, zero time/steps and original positions.
- Native frame offset evaluates frame 2: 13 steps, time 0.04284869916; a later
  sample stays identical. Add-on Step evaluates frame 3: 26 steps.
- Add-on Pause with backlog holds frame 10, 31 steps, time 0.10217766723 and
  identical particle positions across samples 1.2 seconds apart. Resume advances.
- Public Viscous preset and Reset produce solver viscosity 8.0; Water and Reset
  restore approximately 0.15. Stop and Start succeed; live GN has 100,000 points.

`verify_live.py` uses deferred timer callbacks and public operators, with one-time
explicit correctness readbacks. Initial harness attempts are preserved: Blender
animation_cancel returns PASS_THROUGH despite successfully pausing, and a fixed
1.2-second wait was too short for a full single-frame evaluation. The harness was
corrected to accept that operator result and wait for target completion under a
90-second deadline. These were verification-harness errors; no product fixes.
The log also retains a later JSON serialization error for a diagnostic set, fixed
by serializing it as a list. No product error or draw error remained at handoff.

The startup splash was dismissed with a mouse click. An attempted Escape stopped
the live modal operator; it was explicitly restarted through public Start and
native Play afterward. `handoff.json` and `demo.png` prove the surviving visible
session running at frame 289, 221 solver steps/time 0.7284278857, with editable
WarpSPH controls exposed. The saved demo .blend is a reusable scene configuration;
live GPU execution itself is process state.

This bounded test does not establish real-time speed (the solver lags the playhead),
Linux GPU, long-run stability, all replay-failure cases or full release acceptance.
No public release/tag or regular-profile installation was performed. Demo remains
open and running for the user.
