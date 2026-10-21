"""Compare modes 0/1 with the preserved pre-correction solver on actual CUDA."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver


def main():
    source = Path("evidence/differentiable-pour-20261017/deflected-source/sim_core.py")
    output = Path("evidence/contact-correction-20261017/legacy-parity.json")
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location("legacy_pour_solver", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Legacy snapshot unavailable")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    initial = np.array([[-0.85 + i * 0.06, -0.09 + j * 0.06, 0.6 + k * 0.06]
                        for i in range(4) for j in range(4) for k in range(4)], dtype=np.float32)
    velocities = np.tile(np.array([1.4, 0.4, 0], dtype=np.float32), (64, 1))
    report = {"steps": 384, "particles": 64, "rtol": 2e-5, "atol": 2e-6,
              "legacy_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(Path("warp_sph_realtime/sim/sim_core.py").read_bytes()).hexdigest(),
              "rows": []}
    for mode in (0, 1):
        for graph in (False, True):
            results = []
            for solver_class in (legacy.RealtimeWCSPHSolver, RealtimeWCSPHSolver):
                solver = solver_class(64, h=0.12, substeps=1, enable_obstacles=mode)
                solver.set_particle_spacing(0.06)
                solver.cfl_factor = 0.2
                solver.update_time_step()
                solver.grid = wp.HashGrid(16, 16, 16, device="cuda:0")
                solver.rebuild_cuda_graph()
                solver.pos_sim.assign(initial)
                solver.vel.assign(velocities)
                vbo = wp.empty_like(solver.pos_sim)
                for _ in range(384):
                    solver.step(vbo, use_graph=graph)
                results.append((solver.pos_sim.numpy(), solver.vel.numpy()))
            p0, v0 = results[0]
            p1, v1 = results[1]
            report["rows"].append({"mode": mode, "graph": graph,
                "position_max_abs": float(np.max(np.abs(p1 - p0))),
                "velocity_max_abs": float(np.max(np.abs(v1 - v0))),
                "passed": bool(np.allclose(p1, p0, atol=2e-6, rtol=2e-5)
                               and np.allclose(v1, v0, atol=2e-6, rtol=2e-5))})
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(row["passed"] for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
