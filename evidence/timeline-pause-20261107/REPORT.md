# Pause correction — 7 November 2026 America/Toronto

The previous timeline candidate incorrectly kept integrating its backlog after
timeline playback stopped. The prior ready/accepted wording did not cover the
user's actual pause-while-behind workflow and is superseded for that behavior.

Source f56999f cancels playback catchup on the playing-to-paused transition,
including a final frame arriving together with Pause. The addon Pause operator
explicitly cancels native playback and sets a cancellation request. Resume uses
native playback. Explicit Single Step retains a seek request so it is not lost
when it intentionally cancels playback. Paused frame seeks still evaluate their
requested target; unchanged paused frames do not keep advancing a cancelled take.
Status reports playhead and simulated seconds separately when paused behind.

Reproduced regression failed before the fix. Final focused suite:21 passed,
2 subtests, covering native Pause under backlog, addon Pause/Resume, Single Step
while playing and post-bake cancellation. Independent follow-up review PASS;
reviewer independently ran seven timeline tests. Full host suite before final
post-bake local-state correction:678 passed,64 CUDA-disabled skips,72 subtests.
Final affected suite passed afterward. Ruff/Pyright passed; no solver kernel edits.

Exact built ZIP SHA256:
30ae8c234aaf82a44d6cc80438fe9d3358b5ddbc7ff51295344aafa3d4fd58ec.
Installed files byte-match the package in canonical isolated Blender PID55680,
port9878,module bl_ext.demo.warp_sph_realtime,Blender5.2 fbe6228777e7/Vulkan,
Warp1.17,private SDK a2,RTX4080SUPER cuda:0,10K particles.

Actual GPU paused-state receipts compare frame, solver-step count, solver time
and SHA256 of all particle positions in samples one second apart after retiring
already-submitted GPU work. Native Pause at playhead357 with physics backlog:
66steps/time0.21753954958/positions unchanged. Addon Pause after successful addon
Resume at playhead698:128steps/time0.42189488404/positions unchanged. Both passed.
This tests the reported failure directly, rather than pausing a fully caught-up
simulation. Native timeline Jump to Start returned frame1/zero steps/time0.

No claim of real-time performance or cached historical rewind. When paused behind,
the solver freezes at its actual frontier rather than continuing toward playhead.
An explicit seek/step is a new evaluation request. One already-submitted GPU step
can finish at the pause boundary; no new steps are queued by paused backlog.
Regular Blender profiles and other windows were preserved. The new corrected
installed demo remains open for the user; prior candidate held paused.
New correction is committed locally; not pushed,merged or released.

Reference: Blender Simulation Zone manual states evaluation is driven by animation
frame changes: https://docs.blender.org/manual/en/5.0/modeling/geometry_nodes/simulation/simulation_zone.html
