"""Record overhead of the frame cache (P1.6): ms per frame with and without recording.

    python scripts/benchmark_cache_record.py [--particles 100000 1000000] [--steps 2 66] [--frames 30]
        [--repeat 3] [--cooldown 8] [--out FILE]

For each (particles, steps per frame) it times
  * solver only: ``frames`` frames stepped with no recording, one host sync at the end;
  * recorded: the same frames through ``BakeJob`` (device snapshot, pinned async copy, writer thread, one ``.npz`` per
    frame in a temporary folder), driven the way the add-on drives it (small time-budgeted chunks);
and reports the difference (the record overhead), the time the driving thread spends inside ``run_chunk`` (what it
costs the Blender UI thread), and the disk throughput. Frames are 28 bytes per particle.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp

from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import DiskCache
from warp_sph_realtime.cache.frame_store import FrameStore
from warp_sph_realtime.cache.gpu import SolverSource, VramRing
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver

SPACING = 0.02


def make_solver(particles: int, steps: int) -> RealtimeWCSPHSolver:
    side = round(particles ** (1 / 3))
    count = side**3
    axis = (np.arange(side, dtype=np.float32) - side / 2) * SPACING
    x, y, z = np.meshgrid(axis, axis, axis + side * SPACING / 2 + 0.2, indexing="ij")
    solver = RealtimeWCSPHSolver(num_particles=count, h=2.0 * SPACING, substeps=steps, enable_obstacles=0)
    solver.pos_sim.assign(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32))
    solver.vel.zero_()
    return solver


def solver_only(solver, frames: int) -> float:
    out = wp.empty(solver.num_particles, dtype=wp.vec3, device="cuda:0")
    solver.step(out)  # warm-up (kernel load, graph)
    wp.synchronize_device("cuda:0")
    start = time.perf_counter()
    for _ in range(frames):
        solver.step(out)
    wp.synchronize_device("cuda:0")
    return (time.perf_counter() - start) / frames * 1000.0


def recorded(solver, steps: int, frames: int, folder: Path) -> dict:
    disk = DiskCache(folder)
    signature = {"particles": solver.num_particles, "steps_per_frame": steps}
    plan = bake.plan_bake(disk, signature, 1, frames + 1, resume=False)
    store = FrameStore(disk, VramRing(wp, solver.num_particles, 3))
    job = bake.BakeJob(store, SolverSource(wp, solver, steps), plan)
    in_chunks, start = 0.0, time.perf_counter()
    while job.active:
        t = time.perf_counter()
        job.run_chunk(0.004)
        in_chunks += time.perf_counter() - t
        time.sleep(0.001)  # the add-on's timer period
    total = time.perf_counter() - start
    store.close()
    # Frame 1 is the initial state (no solver step), so ``frames`` advances plus one extra snapshot: divide by
    # ``frames`` to compare with the solver-only time (this slightly overstates the overhead by 1 / frames).
    return {"ms_per_frame": total / frames * 1000.0, "run_chunk_ms_per_frame": in_chunks / frames * 1000.0,
            "mb_per_s": store.bytes_written / total / 2**20, "frames_recorded": frames + 1}


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--particles", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 66])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=3, help="runs per case; the median is reported")
    parser.add_argument("--cooldown", type=float, default=8.0,
                        help="seconds to wait between runs so the previous run's disk write-back does not "
                             "throttle the next one")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    wp.init()
    rows = []
    for particles in args.particles:
        for steps in args.steps:
            runs = []
            for _ in range(args.repeat):
                solver = make_solver(particles, steps)
                base = solver_only(solver, args.frames)
                solver = make_solver(particles, steps)
                with tempfile.TemporaryDirectory() as tmp:
                    rec = recorded(solver, steps, args.frames, Path(tmp) / "cache")
                runs.append({"particles": solver.num_particles, "steps_per_frame": steps, "solver_only_ms": base, **rec,
                             "overhead_ms_per_frame": rec["ms_per_frame"] - base})
                del solver
                time.sleep(args.cooldown)
            runs.sort(key=lambda r: r["overhead_ms_per_frame"])
            row = {**runs[len(runs) // 2], "runs": runs}
            rows.append(row)
            overheads = ", ".join(f"{r['overhead_ms_per_frame']:+.1f}" for r in runs)
            print(f"{row['particles']:>9,} particles, {steps:>3} steps/frame: solver {row['solver_only_ms']:7.2f} "
                  f"ms/frame, recorded {row['ms_per_frame']:7.2f}, overhead {row['overhead_ms_per_frame']:+7.2f} ms "
                  f"(median of {len(runs)}: {overheads}), run_chunk {row['run_chunk_ms_per_frame']:6.2f} ms/frame, "
                  f"disk {row['mb_per_s']:5.0f} MB/s", flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
