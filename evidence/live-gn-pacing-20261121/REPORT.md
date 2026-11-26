# Live Geometry Nodes pacing — 21 November 2026

The installed candidate advanced 2.61 times as much simulation time per wall-clock
second as the previous capture rule in a bounded playback comparison. Median
ratios were 0.049× and 0.128× real time. The scene still lags wall-clock playback.

Automatic Live Geometry Nodes capture now coalesces solver publications within
the same simulated timeline frame during Play. The existing UI rate remains an
upper bound. Pause allows publication of the latest partial frame, and reset
begins a fresh output epoch. Physics, substeps, mesh converter and snapshot
ownership rules remain as before. Manual snapshots and animation bake use their
existing routes.

The candidate is commit 58350c80042d7f9448d991c39532f0ced5806ccd on main base
e736bae. The archive SHA256 is
bab5497d3d79893db1023466c1c34e03de78bf02fc5a8b0198d0559bc2d497bc.
All 64 archive files matched source and installed files. The installed Windows
GUI ran Blender 5.2.0 LTS, Warp 1.17.0, private SDK 0.1.0a2, RTX 4080 SUPER
on cuda:0 and Vulkan. Exclusive Blender Gate lease bb3adcf9a38a finished clean
with exit 0. Logs and identity are in gate-comparison.json and runtime.json.

Four roughly five-second Play trials alternated baseline, paced, paced, baseline
with Reset and initial live snapshot completion between trials. Baseline disabled
only frame coalescing in the installed candidate for this diagnostic. Solver,
scene, particle count of 100,000 and Live Geometry Nodes stayed the same. This
measures playback throughput over short trials in one process. It does not
establish equal end state for unequal simulated work or general hardware speedup.

The probe tested Play, add-on Pause with stable frame/steps/time, Reset and
native one-frame advance to frame 2 with 13 solver steps and live GN points.
A fresh installed review process opened WarpSPH-Live-Output-Demo.blend and
advanced the live solver. See open-demo.json. Its dedicated GUI lease is bounded
to 20 minutes. No remote merge or release was requested.

Focused pytest: 39 passed and 2 subtests passed. Ruff and targeted Pyright passed.
The first broad host attempt lacked the private SDK. The second supplied the SDK
via PYTHONPATH, but the cold-start child deliberately ignores that variable and
could not import NumPy. The configured suite installs NumPy in the Python 3.13
test environment so that the child can import it. The configured CPU-masked suite
finished with 738 passed, 67 skipped and 72 subtests passed. Both setup attempts
are retained.

To repeat locally, set BLENDER_USER_RESOURCES to a fresh profile under this
evidence directory and use Blender Gate with Blender 5.2.0, Vulkan, and
--python-exit-code 1. Run demo_probe.py under an exclusive lease for the
comparison, or open_demo.py under a bounded demo lease for live review.
