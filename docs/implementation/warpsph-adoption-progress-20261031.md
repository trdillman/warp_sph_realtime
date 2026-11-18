# Ledger:warpsph-adoption-plan-20261031.md
Started 2026-10-31T22:40:12.714662+00:00
SDK owns example/qualification; consumer owns lazy capture/operator. Both isolated branches claimed by this session. Existing parallel integration branches and sim_core.py are read-only.

Capture16 initialbehavioraltests and4atomic/UIregressions observed RED->GREEN. Added2CUDA-sync/readback failure regressions RED->GREEN; errors now mark existing session fatal, never resume uncertain CUDA state. Focusedcapture/UI/registration23passed. Initialfullsource633passed/64skipped/72subtests; finalinstalledrerun pending. ExactSDKidentitylock33payloadfiles; no source-copy bundling.

Review found identity namespace reuse after dispose/new-session on same Blender sceneUID. Added failing regression, then unique per-session producer namespace while preserving namespace and epoch across repeated same-session snapshots. Old snapshots cannot be accidentally replaced by a new run after disable/enable. This changes metadata only, not solver IDs or physics.
