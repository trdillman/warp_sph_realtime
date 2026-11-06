# Installed Windows native runtime qualification — 6 November 2026

Product source: `48cac96034ead6d9c52d25559dc8b198588f9160`.
Extension ZIP SHA256: `002e6f902bbcbe0ed43c59c20cd7dd357e113a525252db6563df7a36aadacdb4`.
Private SDK a2 wheel SHA256: `24a4e65a290a4f95e9cba7ee54e31f3cecd066c17e2897c8e6cf4ddf8e7e7bda`.
The SDK wheel is not redistributed in this repository.

Blender 5.2.0 build `fbe6228777e7`, Vulkan, Warp 1.17.0, RTX 4080 SUPER.
Canonical Blender MCP launcher created isolated profiles. The extension ZIP was
installed with Blender's `extensions.package_install_files` into a new local
repository inside the owned session. All 59 installed package files matched the
ZIP. Installed module: `bl_ext.roi_extensions.warp_sph_realtime`; PID 75984,
port 9877. Dependencies were imported from a task-owned, version-pinned directory.
Public existing-folder preferences selected that Warp directory without saving
the regular user profile. The shared Windows GPU lease serialized native probes.

## Results

| Probe | Result and scope |
|---|---|
| Installed adoption | 21 checks passed, 100,000 particles: Start/Pause/Reset/Step/Stop, exact IDs and native attributes, Geometry Nodes snapshots, reset identity epoch and completed cleanup |
| Live GN and animation | 14 checks passed: live updates, disabled-live retention, independent eight-frame solver bake, preserved live state, cancellation prefix, nonmonotonic scrub, corrupt-cache hiding/recovery, standalone memory-budget rejection |
| Authored scene effects | 704 checks passed through public Reset/Step: collider/emitter A/B and output identity |
| Running disable | SSFR running with 694 solver steps and 693 draws; extension disable completed native drain, removed callbacks and released solver; re-enable and next start succeeded |
| In-flight file load | Queued 64 actual solver steps. Drain observed readiness `[false, true]` before and `[true, true]` after. File load completed cleanup with no active consumer or fatal SDK state |
| Fresh standalone reopen | Separate PID 55624, no Warp, SDK or WarpSPH. Frames 1/4/8 each evaluated 100,000 unique IDs and 1.2 million mesh vertices; geometry changed with time |
| Independent standalone oracle | All eight frames passed exact IDs and native-field/geometry tolerances with original cache physically absent. Oracle reference files were held separately; cache restored after verification |
| Native renders | Frames 1 and 8 passed, respectively 7,202 and 7,543 visible pixels; image and pixel hashes differ |

The fresh-process verifier initially failed because it iterated Blender Addon
objects as names. The one-line correction uses `.keys()`; the real runtime check
then passed. Independent review approved that correction. Product package bytes
were not changed during testing.

## Setup failures and cleanup

The initial launch rejected an SDK path mismatch because the reused Warp
dependency directory also contained SDK a3. A clean dependency directory fixed
the mismatch. A legacy-module staging probe then correctly failed extension
storage admission; the subsequent fresh process used the actual ZIP installer.
Neither failed setup is counted as product acceptance.

Owned processes 46884, 47488, 75984 and 55624 exited. The pre-existing user
Blender process 60364 was not controlled. GPU leases were explicitly released.
Launcher profiles and dependency staging are isolated from the regular profile;
retained local evidence includes the blend, cache, receipts and logs. Compact
JSON and rendered endpoint evidence are committed alongside this report.

## Limits

This is bounded installed Windows correctness and lifecycle evidence. It is not
real-time FPS, zero-copy Geometry Nodes, Linux GPU, arbitrary Blender/device,
long-animation, subframe/motion-blur or production artistic-quality acceptance.
Private pinned SDK availability remains a deployment prerequisite. Merge and
release are separate from this draft PR delivery.
