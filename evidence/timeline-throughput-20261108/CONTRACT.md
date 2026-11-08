# Bounded development comparison

Operation: public native timeline single-frame advance, including completion of
the resulting Live Geometry Nodes point mesh, in the installed Double Dam demo.
100,000 particles; WATER preset; dt/substeps, scene FPS24, graph mode, render and
Live GN settings unchanged. Start/reset at frame1 then frame_offset(delta=1).
Complete only when solver reaches1/24second and latest native snapshot commits.
Time includes timer scheduling, unchanged physics, GPU publication/readback,
native mesh commit. No profiler or per-step synchronization during timing.
After timing synchronize once and hash all positions and stable IDs; finite
positions and matching baseline results required.13steps are expected.

Baseline: merged main01d034e package3db5a2a1013064e26e5df32610cdad15bda01aa036ab75027c832ce6e13af44b.
Candidate: only physics batching per publication, keeping the same solver calls,
dt, inputs, validation, memory/ownership safeguards and native output route.

One untimed warmup and16 measured reset trials per installed process. Baseline
adjacent trial pairs provide an A/A noise observation; candidate matched by trial
index. Save every attempt and fail on a20second deadline. Keep GUI/GPU lanes
serialized. Baseline retained paused while candidate runs in a fresh installed
process. This is a narrow development experiment with process/order effects;
it is not independent counterbalanced hardware qualification or release proof.
Report raw trial values/ranges and medians, no general speedup confidence claim.
Improvement requires lower median wall time with identical final solver state,
all controls passing, and no unreported trial failures.

Execution deviation, recorded8November2026: the baseline was held paused after
its trials, then explicitly stopped and its owned process closed before launching
the candidate. Evidence and saved scene were retained. This avoids competing
viewport GPU work; the earlier paragraph describes the initial plan. No measured
trial, workload, timing boundary or exclusion policy changed.
