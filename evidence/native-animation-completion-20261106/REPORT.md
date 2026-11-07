# Native animation completion — 7 November 2026

Source frozen at `c43a31f` after independent source review and follow-up PASS.
The lock cleanup repair releases owned reservations on write, short-write and
close failures; exclusive-open failure preserves a competing writer's lock.
Three regressions failed before the repair. Consolidated invariants: 42 passed.
Full source: 671 passed, 64 CUDA-disabled skips, 72 subtests. Ruff and Pyright pass.

Built with the repository extension builder and Blender 5.2.0. Package:
`dist/warp_sph_realtime-0.1.0.zip`, SHA256
`1f4e382b36345287d5493aa807c3b37aee06206377661780b234f1b89c7c0402`.
All installed files matched that ZIP in isolated canonical launcher PID 5216,
port 9877, module `bl_ext.completion_extensions.warp_sph_realtime`.
Dependencies: Warp 1.17.0 and byte-pinned SDK a2 from the retained development
dependency directory. Blender build fbe6228777e7, Vulkan, cuda:0.

Installed actual 100,000-particle animation probe: **14 checks passed**, covering
live native updates, independent eight-frame baking, live-state preservation,
cancellation prefix, scrub, corrupt-cache hiding/recovery and standalone budget
rejection. See `animation-final/live-animation.json`.
Endpoint renders passed: frame 1 had 7,282 visible pixels, frame 8 had 7,572;
image and pixel hashes differ. This is diagnostic rendering evidence.

Fresh canonical Blender process PID 32700 reopened the saved standalone blend
without Warp, SDK or consumer loaded. Frames 1/4/8 each evaluated 100,000 unique
IDs and 1.2 million vertices, with changing geometry. See standalone-reopen.json.
This rerun did not physically withhold the disk cache or repeat the independent
eight-frame attribute oracle; those stronger historical checks remain bound to
their original package. Both owned processes were requested to quit after proof.

Setup failures are preserved in animation/ and animation-context/: Start rejected
the inherited extra temporary window. Closing that owned temporary window made
the public Start workflow pass. An initial extension-repository call used the
wrong API keyword and was corrected before installation; no regular profile was
modified. Existing unrelated Blender processes were preserved.

## Release disposition

PR29 is already merged at main 15a57c4. This follow-up lock fix is committed on the
integration branch; it has not been merged or published. Package/native animation
qualification above is bounded Windows evidence. Running disable/file-load and
authored-scene probes were not repeated; prior receipts remain historical.
No Linux NVIDIA, other builds/devices, FPS, long-animation or motion-blur claim.

The wider release plan still has owner branding/pricing decisions, private SDK
deployment availability, Linux/native and performance gates, and an unproven first
tagged release. `docs/RELEASE_PLAN.md` explicitly requires owner approval to tag
the real repository. No release tag, public release or production install was made.
Engineering scope beyond this native-animation continuation, including P2.9 in
the separate foundation lane, remains governed by its own plan and ownership.
